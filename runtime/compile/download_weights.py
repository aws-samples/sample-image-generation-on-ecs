from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from lib.paths import EFS_ARTIFACTS_PATH, WEIGHTS_DIR


def main():
    repo_id = os.environ["HF_REPO_ID"]
    artifacts_dir = Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", EFS_ARTIFACTS_PATH))
    output_dir = artifacts_dir / WEIGHTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.iterdir()) if output_dir.is_dir() else []
    if len(existing) > 5:
        logger.info(f"Weights already present at {output_dir} ({len(existing)} files), skipping.")
        return

    revision = os.environ.get("HF_MODEL_REVISION")
    if not revision:
        raise RuntimeError("HF_MODEL_REVISION env var required for reproducible downloads")

    logger.info(f"Downloading {repo_id}@{revision} to {output_dir}")

    from huggingface_hub import snapshot_download

    snapshot_download(  # nosec B615 - repo_id from trusted env var, not user input
        repo_id,
        revision=revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    files = sorted(output_dir.iterdir())
    logger.info(f"Done. {len(files)} files downloaded.")
    for f in files[:20]:
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
