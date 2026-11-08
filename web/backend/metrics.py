from __future__ import annotations

import json
import sys
import time
from typing import Any

_NAMESPACE = "ZImageWeb"


def emit_metric(metric_name: str, value: float, unit: str = "Count", **dims: str) -> None:
    dim_keys = list(dims.keys())
    payload: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": _NAMESPACE,
                    "Dimensions": [dim_keys] if dim_keys else [],
                    "Metrics": [{"Name": metric_name, "Unit": unit}],
                }
            ],
        },
        metric_name: value,
        **dims,
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def emit_auth_failure(reason: str) -> None:
    emit_metric("AuthFailures", 1, reason=reason)


def emit_admin_action(action: str) -> None:
    emit_metric("AdminActions", 1, action=action)


def emit_generation_latency(ms: float, backend: str, resolution: int) -> None:
    emit_metric(
        "GenerationLatencyMs",
        ms,
        unit="Milliseconds",
        backend=backend,
        resolution=str(resolution),
    )
