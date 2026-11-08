from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from infra.utils.contracts import DeployConfig, WebConfig
from infra.utils.enums import Environment

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

CONFIG_DIR = Path(__file__).parent


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class DeploySettings:
    def __init__(self, env_override: str | None = None) -> None:
        env_name = env_override or os.environ.get("ZIMAGE_ENV", "dev")
        self.environment = Environment(env_name)

        self.account = os.environ["CDK_DEFAULT_ACCOUNT"]
        self.region = os.environ["CDK_DEFAULT_REGION"]

        raw = self._load_merged_config()
        config = DeployConfig(**raw)

        self.network = config.network
        self.inferentia_endpoint = config.inferentia_endpoint
        self.gpu_endpoint = config.gpu_endpoint
        self.security = config.security
        self.web = config.web
        self.tags = config.tags

    def _load_merged_config(self) -> dict[str, Any]:
        defaults = _load_yaml(CONFIG_DIR / "defaults.yaml")
        env_overrides = _load_yaml(CONFIG_DIR / "environments" / f"{self.environment.value}.yaml")
        return _deep_merge(defaults, env_overrides)

    @property
    def prefix(self) -> str:
        return f"{self.environment}-zimg"

    @property
    def cognito_domain_prefix(self) -> str:
        override = self.web.cognito_domain_prefix
        resolved = override or f"{self.prefix}-auth-{self.account}"
        try:
            WebConfig(cognito_domain_prefix=resolved)
        except ValidationError as e:
            source = "web.cognito_domain_prefix" if override else "computed default"
            raise ValueError(
                f"Invalid Cognito domain prefix {resolved!r} ({source}) for env "
                f"{self.environment.value!r}: must be 1-63 chars of lowercase letters, "
                f"digits and inner hyphens, and must not contain 'aws', 'amazon' or "
                f"'cognito'. Set web.cognito_domain_prefix in "
                f"infra/config/environments/{self.environment.value}.yaml to a valid value."
            ) from e
        return resolved

    def cdk_env(self):
        import aws_cdk as cdk

        return cdk.Environment(account=self.account, region=self.region)

    def __repr__(self) -> str:
        return (
            f"DeploySettings(env={self.environment}, account={self.account}, region={self.region})"
        )
