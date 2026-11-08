from __future__ import annotations

import json
import logging
import os
from pathlib import Path

ALL_TRACED_NEFFS = [
    "transformer_512.pt",
    "encoder_256.pt",
    "vae_512.pt",
    "vae_768.pt",
    "vae_1024.pt",
]
CACHE_768_PREFIX = "cache_768"
CACHE_1024_PREFIX = "cache_1024"
COMPILED_MODELS_DIR = "compiled_models"
EFS_MOUNT_PATH = "/mnt/efs"

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

EFS_MOUNT = Path(os.environ.get("EFS_MOUNT_PATH", EFS_MOUNT_PATH))

DEFAULT_EXPECTED_FILES = ALL_TRACED_NEFFS
DEFAULT_MIN_BYTES = 1024
ZIP_MAGIC = b"PK\x03\x04"
EOCD_MAGIC = b"PK\x05\x06"


def _verify(expected, min_bytes, key_prefix="compile"):
    verified = []
    failures = []

    base_dir = EFS_MOUNT / key_prefix

    for fname in expected:
        file_path = base_dir / fname
        key = f"{key_prefix}/{fname}"

        if not file_path.exists():
            failures.append({"key": key, "reason": "not found"})
            continue

        size = file_path.stat().st_size

        if size < min_bytes:
            failures.append({"key": key, "reason": f"size {size} < min_bytes {min_bytes}"})
            continue

        if not fname.endswith(".pt"):
            verified.append({"key": key, "size_bytes": size, "magic_ok": True})
            continue

        try:
            with open(file_path, "rb") as f:
                head_bytes = f.read(4)
        except OSError as e:
            failures.append({"key": key, "reason": f"read header failed: {e}"})
            continue
        magic_ok = head_bytes == ZIP_MAGIC
        if not magic_ok:
            failures.append(
                {"key": key, "reason": f"first 4 bytes {head_bytes!r} != {ZIP_MAGIC!r}"}
            )
            continue

        try:
            with open(file_path, "rb") as f:
                f.seek(max(0, size - 64))
                tail_bytes = f.read(64)
        except OSError as e:
            failures.append({"key": key, "reason": f"read tail failed: {e}"})
            continue
        if EOCD_MAGIC not in tail_bytes:
            failures.append(
                {"key": key, "reason": f"EOCD signature {EOCD_MAGIC!r} not in last 64 bytes"}
            )
            continue

        verified.append(
            {
                "key": key,
                "size_bytes": size,
                "size_gb": round(size / 1e9, 2),
                "magic_ok": True,
                "eocd_ok": True,
            }
        )
        LOG.info("ok %s (%.2f GB)", key, size / 1e9)

    return verified, failures


MINIMUM_CACHE_NEFFS = {
    768: 7,
    1024: 1,
}


def _count_neffs(prefix: str) -> int:
    cache_dir = EFS_MOUNT / COMPILED_MODELS_DIR / prefix.rstrip("/")
    if not cache_dir.is_dir():
        return 0
    count = 0
    for f in cache_dir.rglob("*.neff"):
        if f.is_file():
            count += 1
    return count


def _cache_complete(prefix: str, resolution: int) -> bool:
    required = MINIMUM_CACHE_NEFFS.get(resolution, 8)
    count = _count_neffs(prefix)
    LOG.info("cache check: prefix=%s count=%d required=%d", prefix, count, required)
    return count >= required


def handler(event, context):
    expected = event.get("expected_files") or DEFAULT_EXPECTED_FILES
    min_bytes = int(event.get("min_bytes") or DEFAULT_MIN_BYTES)
    key_prefix = event.get("key_prefix", "compile")
    check_only = event.get("check_only", False)
    require_pjrt_cache = event.get("require_pjrt_cache", False)
    cache_version = event.get("cache_version", "")

    LOG.info(
        "check_artifacts starting efs=%s check_only=%s prefix=%s expected=%s pjrt=%s ver=%s",
        str(EFS_MOUNT),
        check_only,
        key_prefix,
        expected,
        require_pjrt_cache,
        cache_version,
    )
    verified, failures = _verify(expected, min_bytes, key_prefix)

    pjrt_ok = True
    if require_pjrt_cache:
        if cache_version:
            prefix_768 = f"{CACHE_768_PREFIX}/{cache_version}"
            prefix_1024 = f"{CACHE_1024_PREFIX}/{cache_version}"
        else:
            prefix_768 = CACHE_768_PREFIX
            prefix_1024 = CACHE_1024_PREFIX
        ok_768 = _cache_complete(prefix_768, 768)
        ok_1024 = _cache_complete(prefix_1024, 1024)
        pjrt_ok = ok_768 and ok_1024
        LOG.info(
            "pjrt cache complete=%s (768=%s [%s], 1024=%s [%s])",
            pjrt_ok,
            ok_768,
            prefix_768,
            ok_1024,
            prefix_1024,
        )

    LOG.info("check_artifacts done: %d verified, %d failures", len(verified), len(failures))

    if check_only:
        return {
            "all_cached": len(failures) == 0 and pjrt_ok,
            "verified": len(verified),
            "missing": len(failures),
            "pjrt_cache_ok": pjrt_ok,
        }

    if failures:
        raise RuntimeError(f"NEFF verification failed: {json.dumps(failures, default=str)}")
    return {"verified": verified, "failures": failures}
