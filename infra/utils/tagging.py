from __future__ import annotations

from infra.utils.contracts import TagConfig
from infra.utils.enums import Environment


def build_tags(env: Environment, tag_config: TagConfig) -> dict[str, str]:
    return {
        "App": f"z-image-inferentia-{env.value}",
        "Project": tag_config.project,
        "Environment": env.value,
        "Owner": tag_config.owner,
        "CostCenter": tag_config.cost_center,
        "ManagedBy": "CDK",
    }
