from __future__ import annotations

from enum import IntEnum


class FrameType(IntEnum):
    INTERMEDIATE = 1
    FINAL = 2
    COMPLETE = 3
    ERROR = 4
    METADATA = 5


class Resolution(IntEnum):
    RES_512 = 512
    RES_768 = 768
    RES_1024 = 1024
