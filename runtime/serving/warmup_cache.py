from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from lib.paths import (
    CACHE_768_PREFIX,
    CACHE_1024_PREFIX,
    COMPILED_MODELS_DIR,
    DEFAULT_ARTIFACTS_DIR,
    NEFF_ENCODER_256,
    NEURON_NEFFS,
    WEIGHTS_DIR,
)

CODE_ROOT = Path("/opt/ml/code")


def _verify_efs_artifacts(artifacts_dir: Path, resolution: int) -> None:
    weights_dir = artifacts_dir / WEIGHTS_DIR
    compiled_dir = artifacts_dir / COMPILED_MODELS_DIR

    if weights_dir.is_dir() and any(weights_dir.iterdir()):
        n_files = sum(1 for _ in weights_dir.rglob("*") if _.is_file())
        logger.info(f"EFS weights present: {weights_dir} ({n_files} files)")
    else:
        logger.warning(f"EFS weights MISSING: {weights_dir}")

    neffs = NEURON_NEFFS.get(resolution, [NEFF_ENCODER_256])

    for neff in neffs:
        p = compiled_dir / neff
        if p.exists():
            logger.info(f"EFS NEFF present: {p} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.warning(f"EFS NEFF missing: {p}")


def _build_endpoint_env(resolution: int) -> dict[str, str]:
    env = os.environ.copy()

    env["PJRT_DEVICE"] = "NEURON"
    env["NEURON_RT_VISIBLE_CORES"] = "0,1"
    env["NEURON_FUSE_SOFTMAX"] = "1"
    env["NEURON_RT_EXEC_TIMEOUT"] = "120"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "16"
    env["MKL_NUM_THREADS"] = "16"

    env["ZIMAGE_DEFAULT_RESOLUTION"] = str(resolution)
    env["ZIMAGE_ALLOWED_RESOLUTIONS"] = str(resolution)

    env["USE_TRACED_VAE"] = "1"
    if resolution == 1024:
        env["VAE_ON_CPU"] = "1"
    else:
        env["VAE_ON_CPU"] = "0"

    env["ZIMAGE_ENCODER_MODE"] = "subprocess"

    env["NEURON_CC_FLAGS"] = "--retry_failed_compilation"

    env["ZIMAGE_EAGER_LOAD"] = "false"

    return env


def _run_neuron_parallel_compile(
    command: str,
    resolution: int,
    env: dict[str, str],
    timeout: int = 7200,
) -> bool:
    collect_script = str(CODE_ROOT / "serving/warmup_collect.py")
    cmd = [
        "neuron_parallel_compile",
        "--command",
        command,
        sys.executable,
        collect_script,
        "--resolution",
        str(resolution),
    ]
    logger.info(f"running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout)
        if result.returncode == 0:
            logger.info(f"neuron_parallel_compile --command {command} succeeded")
            return True
        else:
            logger.error(
                f"neuron_parallel_compile --command {command} failed with rc={result.returncode}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"neuron_parallel_compile --command {command} timed out ({timeout}s)")
        return False
    except FileNotFoundError:
        logger.error(
            "neuron_parallel_compile not found — "
            "falling back to NEURON_EXTRACT_GRAPHS_ONLY approach"
        )
        return _run_fallback_compile(command, resolution, env, timeout)


def _run_fallback_compile(
    command: str,
    resolution: int,
    env: dict[str, str],
    timeout: int = 7200,
) -> bool:
    collect_script = str(CODE_ROOT / "serving/warmup_collect.py")
    cmd = [sys.executable, collect_script, "--resolution", str(resolution)]
    fallback_env = env.copy()
    if command == "collect":
        fallback_env["NEURON_EXTRACT_GRAPHS_ONLY"] = "1"
    elif command == "compile":
        fallback_env["NEURON_PARALLEL_COMPILE"] = "1"
    else:
        logger.error(f"unknown command: {command}")
        return False

    logger.info(f"fallback {command}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=fallback_env, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"fallback {command} timed out ({timeout}s)")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True, choices=[768, 1024])
    args = parser.parse_args()

    attempt = int(os.environ.get("AWS_BATCH_JOB_ATTEMPT", "1"))
    logger.info(f"starting warmup: resolution={args.resolution} attempt={attempt}/3")
    logger.info(f"CPU={os.cpu_count()} PJRT_DEVICE={os.environ.get('PJRT_DEVICE')}")

    artifacts_dir = Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _verify_efs_artifacts(artifacts_dir, args.resolution)

    os.environ["ZIMAGE_ALLOWED_RESOLUTIONS"] = str(args.resolution)
    os.environ["ZIMAGE_DEFAULT_RESOLUTION"] = str(args.resolution)

    sys.path.insert(0, str(CODE_ROOT))
    t0 = time.perf_counter()

    if args.resolution == 768:
        from serving.production.pipeline_768 import ZImagePipeline768

        logger.info("loading ZImagePipeline768...")
        pipe = ZImagePipeline768.from_default_artifacts(verbose=True)
    else:
        from serving.production.pipeline_1024 import ZImagePipeline1024

        logger.info("loading ZImagePipeline1024...")
        pipe = ZImagePipeline1024.from_default_artifacts(verbose=True)

    logger.info("running pipe.warmup() (compiles generate + streaming paths)...")
    try:
        pipe.warmup()
        logger.info("warmup complete")
    except (MemoryError, RuntimeError) as exc:
        exc_str = str(exc).lower()
        is_oom = "memory" in exc_str or "tensor_data" in exc_str or "allocate" in exc_str
        if is_oom:
            logger.warning(f"OOM during warmup (expected for 1024): {type(exc).__name__}: {exc}")
            logger.info("continuing to S3 sync (NEFFs were compiled)")
        else:
            logger.error(f"warmup FAILED: {type(exc).__name__}: {exc}")
            logger.error("[warmup_cache] returning 1 (warmup failed)")
            return 1

    total_time = time.perf_counter() - t0
    logger.info(f"warmup complete in {total_time:.1f}s")

    cache_version = os.environ.get("NEURON_CACHE_VERSION", "")
    cache_prefix = CACHE_768_PREFIX if args.resolution == 768 else CACHE_1024_PREFIX
    subdir = f"{cache_prefix}/{cache_version}" if cache_version else cache_prefix

    local_cache = artifacts_dir / COMPILED_MODELS_DIR / subdir
    if local_cache.is_dir():
        n_neffs = sum(1 for f in local_cache.rglob("*.neff") if f.is_file())
        logger.info(f"cache on EFS: {local_cache} ({n_neffs} NEFFs)")
    else:
        logger.warning(f"cache directory not found on EFS: {local_cache}")

    logger.info("done; all subgraphs compiled successfully (EFS-backed)")
    logger.info("[warmup_cache] returning 0")
    return 0


if __name__ == "__main__":
    logger.info("[warmup_cache] entered __main__")
    try:
        rc = main()
        logger.info(f"[warmup_cache] main() returned rc={rc}")
    except Exception as exc:
        logger.opt(exception=True).error(f"FATAL: {type(exc).__name__}: {exc}")
        rc = 1
        logger.error(f"[warmup_cache] main() raised; rc={rc}")

    import threading

    alive = [t.name for t in threading.enumerate() if t.is_alive()]
    logger.info(f"[warmup_cache] live threads at exit: {alive}")
    logger.info(f"[warmup_cache] calling os._exit({rc}) NOW")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
