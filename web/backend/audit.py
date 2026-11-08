from __future__ import annotations

from typing import Any

from loguru import logger

from metrics import emit_admin_action


def audit(
    *,
    actor_sub: str,
    actor_email: str,
    action: str,
    resource: str = "",
    outcome: str = "success",
    **extra: Any,
) -> None:
    payload = {
        "event": "admin_action",
        "actor.sub": actor_sub,
        "actor.email": actor_email,
        "action": action,
        "resource": resource,
        "outcome": outcome,
        **extra,
    }
    logger.bind(**payload).info(f"[audit] {action} resource={resource} outcome={outcome}")
    emit_admin_action(action)
