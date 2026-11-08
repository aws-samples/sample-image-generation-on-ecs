from __future__ import annotations

from infra.utils.enums import Environment

MAX_RESOURCE_NAME_LENGTH = 63
PROJECT_SHORT = "zimg"


class ResourceNamer:
    def __init__(self, env: Environment) -> None:
        self.env = env

    def name(self, resource: str) -> str:
        result = f"{self.env}-{PROJECT_SHORT}-{resource}"
        if len(result) > MAX_RESOURCE_NAME_LENGTH:
            raise ValueError(
                f"Resource name '{result}' is {len(result)} chars, "
                f"exceeds AWS limit of {MAX_RESOURCE_NAME_LENGTH}"
            )
        return result

    def stack_name(self, stack: str) -> str:
        return f"{self.env}-{PROJECT_SHORT}-{stack}"
