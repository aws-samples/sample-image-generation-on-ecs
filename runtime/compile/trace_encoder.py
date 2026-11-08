from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

from lib.paths import COMPILED_MODELS_DIR, NEFF_TRANSFORMER_512, NEFF_VAE_512

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

ROOT = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
SRC = ROOT / "zimage"
sys.path.insert(0, str(ROOT / "compile"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

MAX_SEQ_LEN = 256
LATENT_H = LATENT_W = 64
IN_CH = 16

TRACED_ENCODER = (
    Path(os.environ.get("ZIMAGE_OUTPUT_DIR", str(ROOT / COMPILED_MODELS_DIR)))
    / f"encoder_{MAX_SEQ_LEN}.pt"
)


def _run_full_trace() -> int:
    from _common import (
        GOLDEN_512,
        MODEL_PATH,
        OUTPUTS_DIR,
        PROMPT,
        RESULTS_DIR,
        SEED,
        STEPS,
        CompileConfig,
        assert_neuron_available,
        compare_to_golden,
        image_stats,
        save_results,
        setup,
    )
    from _safety import Limits, cleanup_stale_neuronxcc_dirs, install_resource_monitor

    RESULTS_PATH = RESULTS_DIR / "compile_results.json"
    WORKDIR = ROOT / COMPILED_MODELS_DIR / "workdir"

    TRACED_TRANSFORMER = ROOT / COMPILED_MODELS_DIR / NEFF_TRANSFORMER_512
    TRACED_VAE_512 = ROOT / COMPILED_MODELS_DIR / NEFF_VAE_512

    assert_neuron_available()
    cleanup_stale_neuronxcc_dirs(ROOT, max_age_min=5)

    cfg = CompileConfig(resolution=512, max_sequence_length=512, vae_mode="neuron_traced_512")
    setup(cfg, timeout_min=45)

    install_resource_monitor(
        log_path=ROOT / "logs" / "encoder_monitor.log",
        limits=Limits(min_ram_gb=10.0, min_disk_gb=20.0, max_workdir_gb=90.0, interval_s=20.0),
        watch_dirs=[WORKDIR],
        watch_glob_dirs=[(ROOT, "neuronxcc-*")],
    )

    import atexit

    atexit.register(
        lambda: [
            __import__("shutil").rmtree(d, ignore_errors=True) for d in ROOT.glob("neuronxcc-*")
        ]
    )

    import numpy as np
    import torch
    import torch_neuronx
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    from zimage.pipeline import calculate_shift, retrieve_timesteps
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler

    from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT
    from neuron.runtime import apply_patches, pad_caption
    from utils import ensure_model_weights

    torch.set_num_threads(16)
    apply_patches(attention_backend="native")

    model_path = Path(ensure_model_weights(MODEL_PATH, verify=False))

    tok = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"), trust_remote_code=True)  # nosec B615 - local path, no remote download
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    probe_inputs = tok([formatted], padding=False, truncation=True, return_tensors="pt")
    n_tok = probe_inputs.input_ids.shape[1]
    logger.info(
        f"[encoder] fixed prompt = {n_tok} tokens; tracing at "
        f"max_seq_len={MAX_SEQ_LEN} with inline_weights_to_neff=False"
    )
    if n_tok > MAX_SEQ_LEN:
        save_results(
            {
                "experiment": "encoder",
                "status": "prompt_too_long",
                "n_tokens": n_tok,
                "max_seq_len": MAX_SEQ_LEN,
            },
            RESULTS_PATH,
        )
        return 1

    compile_time_s = None
    if TRACED_ENCODER.exists():
        logger.info("[encoder] cached NEFF present, attempting Neuron load…")
        try:
            traced_enc = torch.jit.load(str(TRACED_ENCODER))
        except Exception as e:
            logger.error(f"[encoder] load failed: {e}")
            save_results(
                {
                    "experiment": "encoder",
                    "status": "load_failed",
                    "error": str(e)[:500],
                    "neff_size_bytes": TRACED_ENCODER.stat().st_size,
                },
                RESULTS_PATH,
            )
            return 1
    else:
        logger.info("[encoder] loading Qwen3 on CPU bf16 for trace…")
        te = (
            AutoModel.from_pretrained(
                str(model_path / "text_encoder"),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,  # nosec B615 - local path, no remote download
            )
            .to("cpu")
            .eval()
        )

        class _EncoderWrapper(torch.nn.Module):
            def __init__(self, te):
                super().__init__()
                self.te = te

            def forward(self, input_ids, attention_mask):
                out = self.te(
                    input_ids=input_ids,
                    attention_mask=attention_mask.bool(),
                    output_hidden_states=True,
                    use_cache=False,
                )
                return out.hidden_states[-2].bfloat16()

        wrapper = _EncoderWrapper(te).eval()

        dummy = tok(
            ["hello world"],
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        ids_ex = dummy.input_ids
        mask_ex = dummy.attention_mask.long()

        WORKDIR.mkdir(parents=True, exist_ok=True)
        logger.info("[encoder] torch_neuronx.trace (~10-25 min) — inline_weights_to_neff=False…")
        t0 = time.perf_counter()
        try:
            traced_enc = torch_neuronx.trace(
                wrapper,
                (ids_ex, mask_ex),
                compiler_workdir=str(WORKDIR),
                compiler_args=[
                    "--model-type=transformer",
                    "--auto-cast",
                    "matmult",
                    "--optlevel=1",
                    "--enable-fast-loading-neuron-binaries",
                ],
                inline_weights_to_neff=False,
            )
        except Exception as e:
            traceback.print_exc()
            compile_time_s = round(time.perf_counter() - t0, 1)
            save_results(
                {
                    "experiment": "encoder",
                    "status": "trace_failed",
                    "compile_time_s": compile_time_s,
                    "error": str(e)[:500],
                    "max_seq_len": MAX_SEQ_LEN,
                },
                RESULTS_PATH,
            )
            return 1
        compile_time_s = round(time.perf_counter() - t0, 1)
        torch.jit.save(traced_enc, str(TRACED_ENCODER))
        size_mb = TRACED_ENCODER.stat().st_size / 1e6
        logger.info(
            f"[encoder] compiled in {compile_time_s}s, saved {TRACED_ENCODER.name} ({size_mb:.1f} MB)"
        )
        del te, wrapper
        gc.collect()

    if os.environ.get("ZIMAGE_SKIP_BENCH", "0") == "1":
        save_results(
            {
                "experiment": "encoder",
                "status": "ok",
                "compile_time_s": compile_time_s,
                "max_seq_len": MAX_SEQ_LEN,
                "note": "bench skipped by ZIMAGE_SKIP_BENCH=1",
            },
            RESULTS_PATH,
        )
        return 0

    logger.info("[encoder] running traced encoder on Neuron (first call = DMA)…")
    inputs = tok(
        [formatted],
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = traced_enc(inputs.input_ids, inputs.attention_mask.long())
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        enc_out = traced_enc(inputs.input_ids, inputs.attention_mask.long())
        t_second = time.perf_counter() - t0
    logger.info(
        f"[encoder] first={t_first * 1000:.1f}ms  second={t_second * 1000:.1f}ms  "
        f"shape={list(enc_out.shape)}"
    )

    if not TRACED_TRANSFORMER.exists() or not TRACED_VAE_512.exists():
        save_results(
            {
                "experiment": "encoder",
                "status": "missing_artifacts",
                "t_encoder_warm_ms": round(t_second * 1000, 1),
                "t_encoder_first_ms": round(t_first * 1000, 1),
            },
            RESULTS_PATH,
        )
        return 1

    logger.info("[encoder] loading traced transformer + VAE alongside encoder…")
    traced_xfm = torch.jit.load(str(TRACED_TRANSFORMER)).eval()
    traced_vae = torch.jit.load(str(TRACED_VAE_512)).eval()

    with open(model_path / "vae/config.json") as fh:
        vcfg = json.load(fh)
    vae_scaling = vcfg.get("scaling_factor", 0.18215)
    vae_shift = vcfg.get("shift_factor", 0.0) or 0.0

    with open(model_path / "scheduler/scheduler_config.json") as fh:
        scfg = json.load(fh)

    def run_once(seed: int = SEED):
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        inputs = tok(
            [formatted],
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeds = traced_enc(inputs.input_ids, inputs.attention_mask.long())
        mask = inputs.attention_mask[0].bool()
        emb = embeds[0][mask]
        cap = pad_caption(emb, max_seq_len=512, device=None)
        t_enc = time.perf_counter() - t0

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=scfg.get("num_train_timesteps", 1000),
            shift=scfg.get("shift", 3.0),
            use_dynamic_shifting=scfg.get("use_dynamic_shifting", False),
        )
        image_seq_len = (LATENT_H // 2) * (LATENT_W // 2)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", BASE_IMAGE_SEQ_LEN),
            scheduler.config.get("max_image_seq_len", MAX_IMAGE_SEQ_LEN),
            scheduler.config.get("base_shift", BASE_SHIFT),
            scheduler.config.get("max_shift", MAX_SHIFT),
        )
        scheduler.sigma_min = 0.0
        timesteps, _ = retrieve_timesteps(scheduler, STEPS, torch.device("cpu"), sigmas=None, mu=mu)
        scheduler._step_index = 0

        gen_cpu = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(
            (1, IN_CH, LATENT_H, LATENT_W), generator=gen_cpu, device="cpu", dtype=torch.float32
        )

        timesteps_list = timesteps.cpu().tolist()
        sigmas_list = scheduler.sigmas.cpu().tolist()

        t0 = time.perf_counter()
        for i, t_val in enumerate(timesteps_list[:STEPS]):
            if t_val == 0 and i == len(timesteps_list) - 1:
                continue
            t_tensor = torch.tensor([t_val], dtype=torch.bfloat16)
            t_norm = ((1000 - t_tensor) / 1000).to(torch.bfloat16)
            lat_in = latents.to(torch.bfloat16)[0].unsqueeze(1)
            with torch.no_grad():
                model_out = traced_xfm(lat_in, t_norm, cap)
            noise_pred = -model_out.float()
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]
            latents = latents.to(torch.float32) + (sigma_next - sigma) * noise_pred
        t_denoise = time.perf_counter() - t0

        t0 = time.perf_counter()
        lat_in = (latents.to(torch.float32) / vae_scaling) + vae_shift
        with torch.no_grad():
            img = traced_vae(lat_in)
            if hasattr(img, "sample"):
                img = img.sample
        t_vae = time.perf_counter() - t0

        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        img = (img * 255).round().astype("uint8")
        pil = Image.fromarray(img[0])

        return pil, {
            "n_tokens": int(n_tok),
            "t_encode_s": round(t_enc, 3),
            "t_denoise_s": round(t_denoise, 3),
            "t_vae_s": round(t_vae, 3),
            "t_total_s": round(time.perf_counter() - t_total, 3),
        }

    logger.info("\n[[encoder] warmup…")
    _, warm = run_once()
    logger.info(f"[encoder]   warmup: {warm}")

    logger.info("\n[[encoder] 3 fresh runs…")
    runs = []
    pil_last = None
    for i in range(3):
        pil_last, times = run_once()
        runs.append(times)
    logger.info(f"[encoder]   run {i + 1}/3: {times}")

    OUTPUT_IMG = OUTPUTS_DIR / "encoder_sample_512.png"
    pil_last.save(str(OUTPUT_IMG))

    totals = [r["t_total_s"] for r in runs]
    result = {
        "experiment": "encoder",
        "description": (
            f"Traced Qwen3 encoder @ max_seq_len={MAX_SEQ_LEN} "
            "with inline_weights_to_neff=False + traced xfm "
            "+ traced VAE @ 512×512"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "max_seq_len": MAX_SEQ_LEN,
        "compile_time_s": compile_time_s,
        "encoder_neff_mb": round(TRACED_ENCODER.stat().st_size / 1e6, 1),
        "t_encoder_first_call_ms": round(t_first * 1000, 1),
        "t_encoder_second_call_ms": round(t_second * 1000, 1),
        "runs": runs,
        "fresh_mean_s": round(float(np.mean(totals)), 3),
        "fresh_min_s": round(float(np.min(totals)), 3),
        "fresh_median_s": round(float(np.median(totals)), 3),
        "vs_baseline_5_58": round(float(np.mean(totals)) - 5.577, 3),
        "image_stats": image_stats(OUTPUT_IMG),
        "output_path": str(OUTPUT_IMG),
    }
    if GOLDEN_512.exists():
        result["vs_golden_512"] = compare_to_golden(OUTPUT_IMG, GOLDEN_512)
    save_results(result, RESULTS_PATH)
    logger.info(
        f"\n[[encoder] OK: fresh mean {result['fresh_mean_s']}s "
        f"(vs baseline 5.58s: {result['vs_baseline_5_58']:+.3f}s)"
    )
    return 0


def main() -> int:
    return _run_full_trace()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TimeoutError as e:
        logger.error(f"encoder TIMEOUT: {e}")
        sys.exit(124)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
