from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from lib.paths import COMPILED_MODELS_DIR, EFS_ARTIFACTS_PATH


def main():
    trace_script = os.environ["TRACE_SCRIPT"]
    expected_output = os.environ["EXPECTED_OUTPUT_FILE"]
    output_mode = os.environ.get("COMPILE_OUTPUT_MODE", "batch").lower()
    root = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
    compiled_dir = root / COMPILED_MODELS_DIR
    compiled_dir.mkdir(parents=True, exist_ok=True)

    efs_artifacts_dir = Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", EFS_ARTIFACTS_PATH))
    efs_compile_dir = efs_artifacts_dir / COMPILED_MODELS_DIR
    efs_compile_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"trace_script={trace_script} expected_output={expected_output} "
        f"root={root} mode={output_mode} efs_dir={efs_artifacts_dir}"
    )

    script_path = root / "compile" / trace_script
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)

    rc = subprocess.run([sys.executable, str(script_path)], check=False).returncode
    if rc != 0:
        logger.warning(
            f"{trace_script} returned exit code {rc} — treating as advisory; "
            f"will check for {expected_output} on disk before deciding."
        )

    output_path = compiled_dir / expected_output
    if not output_path.exists():
        logger.error(
            f"Output not produced: {expected_output} (rc={rc}). "
            f"This is a real failure; trace did not generate the NEFF."
        )
        sys.exit(rc or 1)

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Trace output present: {output_path} ({size_mb:.1f} MB)")

    efs_dest = efs_compile_dir / expected_output
    if not efs_dest.exists() or efs_dest.stat().st_size != output_path.stat().st_size:
        shutil.copy2(output_path, efs_dest)
        logger.info(f"Copied to EFS: {efs_dest} ({size_mb:.1f} MB)")
    else:
        logger.info(f"EFS already has {expected_output} (same size), skipping copy")

    logger.info("Done")


if __name__ == "__main__":
    main()
