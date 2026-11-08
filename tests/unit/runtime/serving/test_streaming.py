from runtime.lib.contracts.enums import FrameType
from runtime.lib.streaming.protocol import decode_frames, encode_frame


class TestStreamingProtocol:
    def test_encode_decode_roundtrip(self) -> None:
        payload = b"hello world"
        encoded = encode_frame(FrameType.INTERMEDIATE, step=3, payload=payload)
        assert len(encoded) == 12 + len(payload)

        frames = decode_frames(encoded)
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.INTERMEDIATE
        assert frames[0].step == 3
        assert frames[0].payload == payload

    def test_multiple_frames(self) -> None:
        data = b""
        data += encode_frame(FrameType.INTERMEDIATE, 0, b"step0")
        data += encode_frame(FrameType.INTERMEDIATE, 1, b"step1")
        data += encode_frame(FrameType.FINAL, 2, b"final_png_data")
        data += encode_frame(FrameType.COMPLETE, 3, b"")

        frames = decode_frames(data)
        assert len(frames) == 4
        assert frames[0].frame_type == FrameType.INTERMEDIATE
        assert frames[3].frame_type == FrameType.COMPLETE
        assert frames[3].payload == b""

    def test_empty_data(self) -> None:
        frames = decode_frames(b"")
        assert frames == []

    def test_partial_header_ignored(self) -> None:
        frames = decode_frames(b"\x00\x00\x00")
        assert frames == []
