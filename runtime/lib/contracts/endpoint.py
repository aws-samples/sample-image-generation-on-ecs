from __future__ import annotations

from pydantic import BaseModel, Field

from runtime.lib.contracts.enums import Resolution


class InferenceRequest(BaseModel):
    prompt: str
    resolution: Resolution = Resolution.RES_512
    steps: int = Field(default=8, ge=1, le=50)
    seed: int = Field(default=42, ge=0)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    denoise_viz: bool = False
    request_id: str = ""


class InferenceResponse(BaseModel):
    request_id: str
    resolution: int
    steps: int
    seed: int
    total_time_ms: float
    status: str = "success"
