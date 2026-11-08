from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from loguru import logger

LOCAL_CACHE_BASE = Path(os.environ.get("NEURON_LOCAL_CACHE_DIR", "/tmp/neuroncc"))


def _is_writable(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=path):
            return True
    except OSError:
        return False


def resolve_compile_cache(efs_cache: Path, subdir: str) -> Path:
    if _is_writable(efs_cache):
        return efs_cache

    local_cache = LOCAL_CACHE_BASE / subdir
    local_cache.mkdir(parents=True, exist_ok=True)
    if efs_cache.is_dir():
        for item in efs_cache.iterdir():
            dest = local_cache / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            elif not dest.exists():
                shutil.copy2(item, dest)
        logger.info(f"[neuron_cache] mirrored read-only cache {efs_cache} -> {local_cache}")
    else:
        logger.warning(
            f"[neuron_cache] shipped cache {efs_cache} missing; using empty writable "
            f"local cache {local_cache} (this may trigger an on-device compile)"
        )
    return local_cache
