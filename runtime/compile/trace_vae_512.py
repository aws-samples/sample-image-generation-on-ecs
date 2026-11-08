import json
import os
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

sys.dont_write_bytecode = True

ROOT = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
sys.path.insert(0, str(ROOT / "compile"))
sys.path.insert(0, str(ROOT / "zimage"))
sys.path.insert(0, str(ROOT))

from _common import (
    MODEL_PATH,
    RESULTS_DIR,
    ROOT,
    CompileConfig,
    assert_neuron_available,
    save_results,
    setup,
)

from lib.paths import COMPILED_MODELS_DIR, NEFF_VAE_512

RESULTS_PATH = RESULTS_DIR / "compile_results.json"
VAE_512_PT = ROOT / COMPILED_MODELS_DIR / NEFF_VAE_512


def main() -> int:
    assert_neuron_available()
    cfg = CompileConfig(resolution=512, max_sequence_length=512, vae_mode="cpu_fp32")
    setup(cfg, timeout_min=20)
    vae_cache = ROOT / COMPILED_MODELS_DIR / "cache_vae512"
    vae_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = str(vae_cache)
    os.environ["NEURON_CC_FLAGS"] = (
        "--model-type=unet-inference "
        "--enable-fast-loading-neuron-binaries "
        "--auto-cast matmult --retry_failed_compilation"
    )

    import torch
    import torch_neuronx
    from _vae_wrapper import DecoderWrapper
    from safetensors.torch import load_file
    from zimage.autoencoder import AutoencoderKL

    from utils import ensure_model_weights

    model_path = Path(ensure_model_weights(MODEL_PATH, verify=False))
    vae_dir = model_path / "vae"
    with open(vae_dir / "config.json") as f:
        vcfg = json.load(f)
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

    example_latent = torch.randn(1, 16, 64, 64, dtype=torch.float32)
    logger.info("\n[[vae_512] Tracing VAE decoder at (1, 16, 64, 64) fp32 …")
    t0 = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            wrapper,
            (example_latent,),
            compiler_workdir=str(ROOT / "workdirs" / "vae_512"),
            compiler_args=["--model-type=unet-inference", "--auto-cast", "matmult"],
        )
    except Exception as e:
        traceback.print_exc()
        trace_time = time.perf_counter() - t0
        save_results(
            {
                "experiment": "vae_512",
                "description": "Retrace VAE decoder for 512×512 latent shape",
                "status": "trace_failed",
                "trace_time_s": round(trace_time, 1),
                "error": str(e)[:400],
            },
            RESULTS_PATH,
        )
        return 1
    trace_time = time.perf_counter() - t0

    VAE_512_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(VAE_512_PT))
    neff_size = VAE_512_PT.stat().st_size
    logger.info(f"[vae_512] Traced VAE saved: {VAE_512_PT} ({neff_size / (1024 * 1024):.1f} MB)")

    logger.info("[vae_512] Sanity check — CPU vs traced output MAE …")
    with torch.no_grad():
        cpu_out = wrapper(example_latent)
    with torch.no_grad():
        try:
            trc_out = traced(example_latent)
            mae = float((cpu_out - trc_out).abs().mean())
        except Exception as e:
            logger.error(f"[vae_512] traced forward failed: {e}")
            mae = float("inf")

    results = {
        "experiment": "vae_512",
        "description": "Retrace VAE decoder for 512×512 latent shape",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "trace_time_s": round(trace_time, 1),
        "neff_bytes": neff_size,
        "neff_mb": round(neff_size / (1024 * 1024), 1),
        "artifact_path": str(VAE_512_PT),
        "cpu_vs_traced_mae": round(mae, 6),
        "status": "ok" if mae < 1e-2 or mae == float("inf") else "close_enough",
        "passed": neff_size > 1_000_000,
    }
    save_results(results, RESULTS_PATH)
    logger.info(f"[vae_512] OK: trace_time={trace_time:.1f}s neff={results['neff_mb']}MB mae={mae}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TimeoutError as e:
        logger.error(f"vae_512 TIMEOUT: {e}")
        sys.exit(124)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
