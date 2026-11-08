from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

CODE_ROOT = Path("/opt/ml/code")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace XLA graphs for warmup compilation.")
    parser.add_argument("--resolution", type=int, required=True, choices=[768, 1024])
    args = parser.parse_args()

    resolution = args.resolution
    logger.info(f"warmup_collect: tracing graphs for resolution={resolution}")

    sys.path.insert(0, str(CODE_ROOT))
    sys.path.insert(0, str(CODE_ROOT / "src"))

    if resolution == 768:
        from serving.production.pipeline_768 import ZImagePipeline768

        pipe = ZImagePipeline768.from_default_artifacts(verbose=True)
    else:
        from serving.production.pipeline_1024 import ZImagePipeline1024

        pipe = ZImagePipeline1024.from_default_artifacts(verbose=True)

    logger.info("tracing non-streaming graph (generate)...")
    pipe.generate("warmup", seed=0)
    logger.info("non-streaming graph traced")

    logger.info("tracing streaming graph (generate_streaming)...")
    for _frame in pipe.generate_streaming(
        "warmup", seed=0, num_inference_steps=8, denoise_viz=True
    ):
        pass
    logger.info("streaming graph traced")

    if hasattr(pipe, "close"):
        pipe.close()

    logger.info("warmup_collect: all graphs traced successfully")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:
        logger.opt(exception=True).error(f"warmup_collect FATAL: {type(exc).__name__}: {exc}")
        rc = 1
    sys.exit(rc)
