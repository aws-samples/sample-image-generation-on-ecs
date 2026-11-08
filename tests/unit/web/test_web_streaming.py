from __future__ import annotations

import json
import struct

from streaming import FrameBuilder, FrameParser, FrameType, parse_frame_header


class TestFrameType:
    def test_intermediate_value(self):
        assert FrameType.INTERMEDIATE == 1

    def test_final_value(self):
        assert FrameType.FINAL == 2

    def test_error_value(self):
        assert FrameType.ERROR == 4

    def test_complete_value(self):
        assert FrameType.COMPLETE == 3

    def test_metadata_value(self):
        assert FrameType.METADATA == 5

    def test_is_int_enum(self):
        assert int(FrameType.FINAL) == 2
        assert FrameType.ERROR + 1 == 5


class TestFrameBuilder:
    def test_build_frame_header_format(self):
        payload = b"\x89PNG\r\n\x1a\nfake"
        frame = FrameBuilder.build_frame(FrameType.INTERMEDIATE, step=3, payload=payload)
        assert len(frame) == 12 + len(payload)
        ft, step, plen = struct.unpack(">III", frame[:12])
        assert ft == 1
        assert step == 3
        assert plen == len(payload)
        assert frame[12:] == payload

    def test_build_intermediate(self):
        png = b"fake-png-data"
        frame = FrameBuilder.build_intermediate(step=5, png_data=png)
        ft, step, plen = struct.unpack(">III", frame[:12])
        assert ft == FrameType.INTERMEDIATE
        assert step == 5
        assert frame[12:] == png

    def test_build_final(self):
        png = b"final-image-bytes"
        meta = {"seed": 42, "latency_s": 3.5}
        frame = FrameBuilder.build_final(png_data=png, metadata=meta)
        ft, step, plen = struct.unpack(">III", frame[:12])
        assert ft == FrameType.FINAL
        assert step == 8
        payload = frame[12:]
        assert payload.startswith(png)
        meta_bytes = payload[len(png) :]
        assert json.loads(meta_bytes) == meta

    def test_build_error(self):
        msg = "Something went wrong"
        frame = FrameBuilder.build_error(msg)
        ft, step, plen = struct.unpack(">III", frame[:12])
        assert ft == FrameType.ERROR
        assert step == 0
        assert frame[12:] == msg.encode("utf-8")

    def test_build_complete(self):
        meta = {"request_id": "abc-123", "frames_received": 9}
        frame = FrameBuilder.build_complete(meta)
        ft, step, plen = struct.unpack(">III", frame[:12])
        assert ft == FrameType.COMPLETE
        assert step == 0
        payload = frame[12:]
        assert json.loads(payload) == meta

    def test_build_frame_empty_payload(self):
        frame = FrameBuilder.build_frame(FrameType.INTERMEDIATE, step=0)
        ft, step, plen = struct.unpack(">III", frame[:12])
        assert plen == 0
        assert len(frame) == 12


class TestFrameParser:
    def test_parse_single_frame(self):
        payload = b"hello-frame"
        raw = struct.pack(">III", FrameType.INTERMEDIATE, 2, len(payload)) + payload
        parser = FrameParser()
        frames = list(parser.feed(raw))
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.INTERMEDIATE
        assert frames[0].step == 2
        assert frames[0].payload == payload

    def test_parse_multiple_frames_in_one_feed(self):
        data = b""
        for i in range(3):
            p = f"payload-{i}".encode()
            data += struct.pack(">III", FrameType.INTERMEDIATE, i, len(p)) + p
        parser = FrameParser()
        frames = list(parser.feed(data))
        assert len(frames) == 3
        for i, f in enumerate(frames):
            assert f.step == i
            assert f.payload == f"payload-{i}".encode()

    def test_parse_partial_data_buffered(self):
        payload = b"full-payload-data"
        raw = struct.pack(">III", FrameType.FINAL, 8, len(payload)) + payload
        parser = FrameParser()
        frames = list(parser.feed(raw[:12]))
        assert frames == []
        frames = list(parser.feed(raw[12:15]))
        assert frames == []
        frames = list(parser.feed(raw[15:]))
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.FINAL
        assert frames[0].payload == payload

    def test_parse_complete_frame_extracts_metadata(self):
        meta = {"request_id": "xyz", "frames_received": 5}
        meta_bytes = json.dumps(meta).encode("utf-8")
        raw = struct.pack(">III", FrameType.COMPLETE, 0, len(meta_bytes)) + meta_bytes
        parser = FrameParser()
        frames = list(parser.feed(raw))
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.COMPLETE
        assert frames[0].metadata == meta
        assert frames[0].payload == b""

    def test_parse_metadata_frame(self):
        meta = {"key": "value"}
        meta_bytes = json.dumps(meta).encode("utf-8")
        raw = struct.pack(">III", FrameType.METADATA, 0, len(meta_bytes)) + meta_bytes
        parser = FrameParser()
        frames = list(parser.feed(raw))
        assert frames[0].metadata == meta

    def test_reset_clears_buffer(self):
        parser = FrameParser()
        list(parser.feed(b"\x00" * 6))
        assert len(parser.buffer) > 0
        parser.reset()
        assert parser.buffer == b""


class TestRoundTrip:
    def test_intermediate_round_trip(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        frame_bytes = FrameBuilder.build_intermediate(step=4, png_data=png)
        parser = FrameParser()
        frames = list(parser.feed(frame_bytes))
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.INTERMEDIATE
        assert frames[0].step == 4
        assert frames[0].payload == png

    def test_error_round_trip(self):
        msg = "Timeout exceeded"
        frame_bytes = FrameBuilder.build_error(msg)
        parser = FrameParser()
        frames = list(parser.feed(frame_bytes))
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.ERROR
        assert frames[0].payload == msg.encode("utf-8")

    def test_complete_round_trip(self):
        meta = {"request_id": "test-123", "backend": "neuron"}
        frame_bytes = FrameBuilder.build_complete(meta)
        parser = FrameParser()
        frames = list(parser.feed(frame_bytes))
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.COMPLETE
        assert frames[0].metadata == meta


class TestParseFrameHeader:
    def test_valid_header(self):
        header = struct.pack(">III", FrameType.FINAL, 8, 1024)
        result = parse_frame_header(header)
        assert result == (FrameType.FINAL, 8, 1024)

    def test_short_data_returns_none(self):
        assert parse_frame_header(b"short") is None
        assert parse_frame_header(b"") is None
        assert parse_frame_header(b"\x00" * 11) is None

    def test_extra_bytes_ignored(self):
        header = struct.pack(">III", FrameType.ERROR, 0, 5) + b"extra"
        result = parse_frame_header(header)
        assert result == (FrameType.ERROR, 0, 5)
