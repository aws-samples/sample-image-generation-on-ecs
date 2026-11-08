from __future__ import annotations

import os
import re
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class NetworkConfig(BaseModel):
    vpc_cidr: str = "10.20.0.0/16"
    nat_count: int = Field(default=1, ge=1, le=3)
    az_count: int = Field(default=3, ge=2, le=6)


class GpuEndpointConfig(BaseModel):
    enabled: bool = True
    instance_type: str = "g5.2xlarge"
    max_pixels: int = Field(default=1048576, ge=65536)
    min_dimension: int = Field(default=256, ge=64)
    dimension_step: int = Field(default=8)
    min_capacity: int = Field(default=1, ge=0)
    max_capacity: int = Field(default=2, ge=1)
    startup_timeout_s: int = Field(default=1800, ge=60, le=3600)
    root_volume_gb: int = Field(default=80, ge=30)
    memory_mib: int = Field(default=30720, ge=4096)
    cpu: int = Field(default=8192, ge=1024)

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.max_capacity < self.min_capacity:
            raise ValueError(
                f"max_capacity ({self.max_capacity}) < min_capacity ({self.min_capacity})"
            )
        return self


class InferentiaEndpointConfig(BaseModel):
    enabled: bool = True
    instance_type: str = "inf2.8xlarge"
    min_capacity: int = Field(default=1, ge=0)
    max_capacity: int = Field(default=4, ge=1)
    warmup_min_capacity: int = Field(default=1, ge=0)
    warmup_max_capacity: int = Field(default=2, ge=1)
    root_volume_gb: int = Field(default=200, ge=50)
    health_check_grace_s: int = Field(default=1800, ge=60)
    alb_idle_timeout_s: int = Field(default=300, ge=60)
    container_port: int = 8080

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.max_capacity < self.min_capacity:
            raise ValueError(
                f"max_capacity ({self.max_capacity}) < min_capacity ({self.min_capacity})"
            )
        return self


class SecurityConfig(BaseModel):
    findings_email: str = "admin@example.com"

    @field_validator("findings_email")
    @classmethod
    def reject_placeholder_email_in_prod(cls, v: str) -> str:
        env = os.environ.get("ZIMAGE_ENV", "dev")
        if env != "dev" and v.endswith("@example.com"):
            raise ValueError(
                f"findings_email must be a real address in '{env}' environment, "
                f"got placeholder: {v}"
            )
        return v


class WebConfig(BaseModel):
    cognito_domain_prefix: str = ""

    @field_validator("cognito_domain_prefix")
    @classmethod
    def validate_domain_prefix(cls, v: str) -> str:
        if not v:
            return v
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", v) or len(v) > 63:
            raise ValueError(
                f"cognito_domain_prefix must be 1-63 chars of lowercase letters, "
                f"digits and non-leading/trailing hyphens, got: {v!r}"
            )
        for reserved in ("aws", "amazon", "cognito"):
            if reserved in v:
                raise ValueError(
                    f"cognito_domain_prefix must not contain {reserved!r} "
                    f"(reserved by Cognito), got: {v!r}"
                )
        return v


class TagConfig(BaseModel):
    project: str = "z-image"
    owner: str = "ml-inference-team"
    cost_center: str = "ml-inference"


class DeployConfig(BaseModel):
    model_config = {"extra": "forbid"}

    network: NetworkConfig = NetworkConfig()
    inferentia_endpoint: InferentiaEndpointConfig = InferentiaEndpointConfig()
    gpu_endpoint: GpuEndpointConfig = GpuEndpointConfig()
    security: SecurityConfig = SecurityConfig()
    web: WebConfig = WebConfig()
    tags: TagConfig = TagConfig()

    @model_validator(mode="after")
    def at_least_one_backend(self) -> Self:
        if not self.inferentia_endpoint.enabled and not self.gpu_endpoint.enabled:
            raise ValueError(
                "At least one inference backend must be enabled: "
                "set inferentia_endpoint.enabled or gpu_endpoint.enabled to true"
            )
        return self
