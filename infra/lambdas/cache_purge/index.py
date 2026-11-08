from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

EFS_MOUNT_PATH = os.environ.get("EFS_MOUNT_PATH", "/mnt/efs")
DEFAULT_PREFIXES = ["cache_1024/", "cache_768/"]


def handler(event: dict, context) -> dict:
    LOG.info("event: %s", json.dumps(event))
    prefixes = event.get("cache_prefixes", DEFAULT_PREFIXES)
    efs_root = Path(EFS_MOUNT_PATH)

    if not efs_root.exists():
        LOG.info("EFS mount %s not found — skipping purge", EFS_MOUNT_PATH)
        return {"purged_count": 0, "kept_count": 0, "purged_sample": []}

    purged: list[str] = []
    kept = 0

    for prefix in prefixes:
        scan_dir = efs_root / prefix
        if not scan_dir.exists():
            LOG.info("prefix dir %s does not exist — skipping", scan_dir)
            continue

        LOG.info("scanning %s", scan_dir)
        failed_dirs: list[Path] = []

        for log_file in scan_dir.rglob("compile_cache_log.json"):
            try:
                data = json.loads(log_file.read_text())
                if data.get("compile_status") == "failed":
                    failed_dirs.append(log_file.parent)
                    LOG.info("failed cache dir: %s", log_file.parent)
            except Exception as exc:
                LOG.warning("failed to parse %s: %s", log_file, exc)

        for d in failed_dirs:
            entries = list(d.iterdir())
            shutil.rmtree(d, ignore_errors=True)
            purged.extend(str(f.relative_to(efs_root)) for f in entries)
            LOG.info("purged %s (%d entries)", d.name, len(entries))

        for entry in scan_dir.iterdir():
            if entry.is_dir() and entry not in failed_dirs:
                kept += 1

    LOG.info("purged %d entries, kept %d dirs", len(purged), kept)
    return {"purged_count": len(purged), "kept_count": kept, "purged_sample": purged[:20]}
