from __future__ import annotations

import os
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    _PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _default_prefix() -> str:
    env = os.environ.get("ZIMAGE_ENV", "dev")
    return f"{env}-zimg"


def _default_region() -> str:
    return os.environ.get("CDK_DEFAULT_REGION", os.environ.get("AWS_REGION", ""))


def _default_gallery_bucket() -> str:
    prefix = _default_prefix()
    account = os.environ.get("CDK_DEFAULT_ACCOUNT", "")
    region = _default_region()
    return f"{prefix}-gallery-{account}-{region}" if account else ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZIMAGE_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint_512: str = Field(default_factory=lambda: f"{_default_prefix()}-endpoint-512")
    endpoint_768: str = Field(default_factory=lambda: f"{_default_prefix()}-endpoint-768")
    endpoint_1024: str = Field(default_factory=lambda: f"{_default_prefix()}-endpoint-1024")
    endpoint_gpu: str = Field(default_factory=lambda: f"{_default_prefix()}-endpoint-gpu")

    gpu_enabled: bool = Field(default=True, description="Whether GPU backend is deployed")

    inferentia_enabled: bool = Field(
        default=True, description="Whether Inferentia2 backend is deployed"
    )

    @model_validator(mode="after")
    def at_least_one_backend(self) -> Self:
        if not self.gpu_enabled and not self.inferentia_enabled:
            raise ValueError(
                "At least one inference backend must be enabled: "
                "set gpu_enabled or inferentia_enabled to true"
            )
        return self

    gpu_max_pixels: int = Field(default=1048576)
    gpu_min_dimension: int = Field(default=256)
    gpu_dimension_step: int = Field(default=8)

    default_backend: str = Field(default="gpu")

    gallery_bucket: str = Field(default_factory=_default_gallery_bucket)
    gallery_table: str = Field(default="", description="DynamoDB gallery table name")

    region: str = Field(default_factory=_default_region)
    prefix: str = Field(default_factory=_default_prefix)
    local: bool = Field(default=False, description="Local development mode")
    port: int = Field(default=8000, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")  # nosec B104

    websocket_url: str = Field(default="", description="WebSocket API endpoint URL")

    cognito_user_pool_id: str = Field(default="", description="Cognito User Pool ID")
    cognito_app_client_id: str = Field(default="", description="Cognito App Client ID")
    cognito_domain: str = Field(default="", description="Cognito Hosted UI domain")

    ws_heartbeat_interval: float = Field(
        default=30.0, description="WebSocket heartbeat interval in seconds"
    )
    ws_max_message_size: int = Field(
        default=10 * 1024 * 1024, description="Max WebSocket message size (10MB)"
    )

    sm_streaming_timeout: int = Field(
        default=60, description="SageMaker streaming timeout in seconds"
    )
    sm_oneshot_timeout: int = Field(
        default=120, description="SageMaker non-streaming (GPU) request timeout in seconds"
    )
    sm_max_concurrent_streams: int = Field(
        default=10, description="Max concurrent streaming requests"
    )

    gallery_page_size: int = Field(default=20, description="Number of images per gallery page")
    gallery_presigned_url_expiry: int = Field(
        default=3600, description="Presigned URL expiry in seconds"
    )

    @property
    def is_lambda(self) -> bool:
        return bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

    @property
    def is_production(self) -> bool:
        return not self.local

    def pick_endpoint(self, resolution: int, backend: str = "gpu") -> tuple[str, bool]:
        backend = (backend or self.default_backend).lower()
        if backend not in ("gpu", "neuron"):
            raise ValueError(f"unknown backend {backend!r}; expected 'gpu' or 'neuron'")

        if backend == "gpu":
            if not self.gpu_enabled:
                raise ValueError("backend='gpu' requested but GPU is disabled")
            return self.endpoint_gpu, False

        if not self.inferentia_enabled:
            raise ValueError("backend='neuron' requested but Inferentia is disabled")

        neuron_map = {
            512: self.endpoint_512,
            768: self.endpoint_768,
            1024: self.endpoint_1024,
        }
        if resolution not in neuron_map:
            raise ValueError(
                f"resolution={resolution} not supported by Neuron; "
                f"valid: {sorted(neuron_map.keys())}"
            )
        return neuron_map[resolution], True


settings = Settings()

if settings.local and settings.is_lambda:
    from loguru import logger as _cfg_logger

    _cfg_logger.warning(
        "ZIMAGE_LOCAL=true detected inside Lambda — "
        "auth bypass is suppressed for safety. Unset ZIMAGE_LOCAL in production."
    )
