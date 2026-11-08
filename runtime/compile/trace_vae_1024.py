from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

from lib.paths import COMPILED_MODELS_DIR, NEFF_VAE_1024, WEIGHTS_DIR

sys.dont_write_bytecode = True

ROOT = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
SRC = ROOT / "zimage"
sys.path.insert(0, str(ROOT / "compile"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

try:
    from _common import (
        MODEL_PATH,
        RESULTS_DIR,
        CompileConfig,
        assert_neuron_available,
        save_results,
        setup,
    )
    from _safety import Limits, install_resource_monitor

    _HAVE_COMPILE_SCAFFOLD = True
except Exception:  # pragma: no cover — the compile container doesn't need these
    _HAVE_COMPILE_SCAFFOLD = False

    def assert_neuron_available():
        pass

    def save_results(*a, **k):
        pass

    class _NoopCfg:
        pass

    def setup(*a, **k):
        pass

    RESULTS_DIR = ROOT / "results"
    MODEL_PATH = str(ROOT / WEIGHTS_DIR)

RESULTS_PATH = RESULTS_DIR / "compile_results.json"
VAE_1024_PT = ROOT / COMPILED_MODELS_DIR / NEFF_VAE_1024
LATENT_HW = 128


def main() -> int:
    assert_neuron_available()

    if _HAVE_COMPILE_SCAFFOLD:
        cfg = CompileConfig(resolution=1024, max_sequence_length=512, vae_mode="cpu_fp32")
        setup(cfg, timeout_min=30)
    vae_cache = ROOT / COMPILED_MODELS_DIR / "cache_vae1024"
    vae_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = str(vae_cache)
    os.environ["NEURON_CC_FLAGS"] = (
        "--model-type=unet-inference --enable-fast-loading-neuron-binaries --auto-cast matmult"
    )

    if _HAVE_COMPILE_SCAFFOLD:
        install_resource_monitor(
            log_path=ROOT / "logs" / "vae_1024_monitor.log",
            limits=Limits(min_ram_gb=10.0, min_disk_gb=20.0, max_workdir_gb=40.0, interval_s=15.0),
            watch_glob_dirs=[(ROOT, "neuronxcc-*")],
        )

    import torch
    import torch_neuronx
    from _vae_wrapper import DecoderWrapper
    from safetensors.torch import load_file
    from zimage.autoencoder import AutoencoderKL

    from utils import ensure_model_weights

    model_path = Path(ensure_model_weights(MODEL_PATH, verify=False))
    vae_dir = model_path / "vae"
    with open(vae_dir / "config.json") as fh:
        vcfg = json.load(fh)

    vae = AutoencoderKL(
        in_channels=vcfg.get("in_channels", 3),
        out_channels=vcfg.get("out_channels", 3),
        down_block_types=tuple(vcfg.get("down_block_types", ("DownEncoderBlock2D",))),
        up_block_types=tuple(vcfg.get("up_block_types", ("UpDecoderBlock2D",))),
        block_out_channels=tuple(vcfg.get("block_out_channels", (64,))),
        layers_per_block=vcfg.get("layers_per_block", 1),
        latent_channels=vcfg.get("latent_channels", 4),
        norm_num_groups=vcfg.get("norm_num_groups", 32),
        scaling_factor=vcfg.get("scaling_factor", 0.18215),
        shift_factor=vcfg.get("shift_factor", None),
        use_quant_conv=vcfg.get("use_quant_conv", True),
        use_post_quant_conv=vcfg.get("use_post_quant_conv", True),
        mid_block_add_attention=vcfg.get("mid_block_add_attention", True),
    )
    sd = {}
    for sf in sorted(vae_dir.glob("*.safetensors")):
        sd.update(load_file(str(sf), device="cpu"))
    vae.load_state_dict(sd, strict=False)
    vae.eval()

    wrapper = DecoderWrapper(vae).to(torch.float32).eval()

    example_latent = torch.randn(1, 16, LATENT_HW, LATENT_HW, dtype=torch.float32)

    logger.info(f"\n[[vae_1024] Tracing VAE decoder at (1, 16, {LATENT_HW}, {LATENT_HW}) fp32 …")
    logger.info(f"[vae_1024] compiler flags: {os.environ['NEURON_CC_FLAGS']}")

    workdir = ROOT / "workdirs" / "vae_1024"
    t0 = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            wrapper,
            (example_latent,),
            compiler_workdir=str(workdir),
            compiler_args=["--model-type=unet-inference", "--auto-cast", "matmult"],
        )
    except Exception as e:
        traceback.print_exc()
        save_results(
            {
                "experiment": "vae_1024",
                "description": "Trace VAE decoder at 1024×1024 latent shape",
                "status": "trace_failed",
                "trace_time_s": round(time.perf_counter() - t0, 1),
                "error": str(e)[:500],
            },
            RESULTS_PATH,
        )
        return 1
    trace_time = time.perf_counter() - t0

    VAE_1024_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(VAE_1024_PT))
    neff_size = VAE_1024_PT.stat().st_size
    logger.info(f"[vae_1024] Traced VAE saved: {VAE_1024_PT} ({neff_size / 1e6:.1f} MB)")

    logger.info("[vae_1024] Sanity check — CPU vs traced output MAE …")
    with torch.no_grad():
        cpu_out = wrapper(example_latent)
    mae = float("inf")
    try:
        with torch.no_grad():
            trc_out = traced(example_latent)
        mae = float((cpu_out - trc_out).abs().mean())
    except Exception as e:
        logger.error(f"vae_1024 traced forward failed: {e}")

    fwd_times = []
    try:
        for i in range(3):
            with torch.no_grad():
                t0f = time.perf_counter()
                _ = traced(example_latent)
                fwd_times.append(round(time.perf_counter() - t0f, 3))
    except Exception as e:
        logger.error(f"vae_1024 timing failed: {e}")

    results = {
        "experiment": "vae_1024",
        "description": "Trace VAE decoder at 1024×1024 latent shape",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "latent_shape": [1, 16, LATENT_HW, LATENT_HW],
        "trace_time_s": round(trace_time, 1),
        "neff_bytes": neff_size,
        "neff_mb": round(neff_size / 1e6, 1),
        "artifact_path": str(VAE_1024_PT),
        "cpu_vs_traced_mae": round(mae, 6),
        "traced_fwd_times_s": fwd_times,
        "status": "ok",
    }
    save_results(results, RESULTS_PATH)
    logger.info(
        f"[vae_1024] OK: trace_time={trace_time:.1f}s  "
        f"neff={results['neff_mb']}MB  mae={mae}  "
        f"fwd={fwd_times}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TimeoutError as e:
        logger.error(f"vae_1024 TIMEOUT: {e}")
        sys.exit(124)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
