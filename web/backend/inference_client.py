from __future__ import annotations

import json
import os

import urllib3
from loguru import logger

ALB_URL = os.environ.get("ZIMAGE_INFERENCE_ALB_URL", "")

_http = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=10, read=120),
    retries=urllib3.Retry(total=0),
)


def invoke_json(resolution: int | str, payload: dict) -> dict:
    resp = _http.request(
        "POST",
        f"{ALB_URL}/invocations",
        body=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Target-Resolution": str(resolution),
        },
    )
    if resp.status != 200:
        raise RuntimeError(f"Inference returned {resp.status}: {resp.data[:200]}")
    return json.loads(resp.data)


def invoke_streaming(resolution: int | str, payload: dict):
    resp = _http.request(
        "POST",
        f"{ALB_URL}/invocations",
        body=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
            "X-Target-Resolution": str(resolution),
        },
        preload_content=False,
    )
    try:
        if resp.status != 200:
            raise RuntimeError(f"Inference returned {resp.status}")
        yield from resp.stream(4096)
    finally:
        resp.release_conn()


def health_check(resolution: int | str) -> dict:
    try:
        resp = _http.request(
            "GET",
            f"{ALB_URL}/ping",
            headers={"X-Target-Resolution": str(resolution)},
            timeout=urllib3.Timeout(connect=5, read=5),
        )
        if resp.status == 200:
            return {"status": "healthy", "resolution": resolution}
        return {"status": "unhealthy", "code": resp.status}
    except Exception as e:
        logger.debug(f"health check {resolution} failed: {e}")
        return {"status": "error", "error": str(e)[:100]}
