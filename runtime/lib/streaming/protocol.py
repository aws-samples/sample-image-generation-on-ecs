from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from runtime.lib.contracts.enums import FrameType

HEADER_SIZE = 12
HEADER_FMT = ">III"


@dataclass
class Frame:
    frame_type: FrameType
    step: int
    payload: bytes
    metadata: dict[str, Any] | None = None


def encode_frame(frame_type: FrameType, step: int, payload: bytes) -> bytes:
    header = struct.pack(HEADER_FMT, int(frame_type), step, len(payload))
    return header + payload


def parse_frame_header(data: bytes) -> tuple[int, int, int] | None:
    if len(data) < HEADER_SIZE:
        return None
    ft, step, plen = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    return ft, step, plen


def decode_frames(data: bytes) -> list[Frame]:
    frames: list[Frame] = []
    pos = 0
    while pos + HEADER_SIZE <= len(data):
        ft, step, plen = struct.unpack(HEADER_FMT, data[pos : pos + HEADER_SIZE])
        payload = data[pos + HEADER_SIZE : pos + HEADER_SIZE + plen]
        metadata = None
        if ft in (FrameType.METADATA, FrameType.COMPLETE):
            try:
                metadata = json.loads(payload.decode("utf-8"))
                payload = b""
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        frames.append(
            Frame(frame_type=FrameType(ft), step=step, payload=payload, metadata=metadata)
        )
        pos += HEADER_SIZE + plen
    return frames


class FrameParser:
    def __init__(self) -> None:
        self.buffer = b""

    def feed(self, data: bytes) -> Iterator[Frame]:
        self.buffer += data
        while len(self.buffer) >= HEADER_SIZE:
            ft, step, plen = struct.unpack(HEADER_FMT, self.buffer[:HEADER_SIZE])
            total = HEADER_SIZE + plen
            if len(self.buffer) < total:
                break
            payload = self.buffer[HEADER_SIZE:total]
            self.buffer = self.buffer[total:]
            metadata = None
            if ft in (FrameType.METADATA, FrameType.COMPLETE):
                try:
                    metadata = json.loads(payload.decode("utf-8"))
                    payload = b""
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            yield Frame(frame_type=FrameType(ft), step=step, payload=payload, metadata=metadata)

    def reset(self) -> None:
        self.buffer = b""


class FrameBuilder:
    @staticmethod
    def build_frame(
        frame_type: FrameType,
        step: int,
        payload: bytes = b"",
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        if metadata is not None:
            payload = payload + json.dumps(metadata).encode("utf-8")
        return encode_frame(frame_type, step, payload)

    @staticmethod
    def build_intermediate(step: int, png_data: bytes) -> bytes:
        return FrameBuilder.build_frame(FrameType.INTERMEDIATE, step, png_data)

    @staticmethod
    def build_final(png_data: bytes, metadata: dict[str, Any]) -> bytes:
        return FrameBuilder.build_frame(FrameType.FINAL, 8, png_data, metadata)

    @staticmethod
    def build_complete(metadata: dict[str, Any]) -> bytes:
        return FrameBuilder.build_frame(FrameType.COMPLETE, 0, b"", metadata)

    @staticmethod
    def build_error(message: str) -> bytes:
        return FrameBuilder.build_frame(FrameType.ERROR, 0, message.encode("utf-8"))
