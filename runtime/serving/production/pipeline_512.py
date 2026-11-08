from __future__ import annotations

import gc
import io
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from lib.paths import (
    COMPILED_MODELS_DIR,
    DEFAULT_ARTIFACTS_DIR,
    NEFF_TRANSFORMER_512,
    NEFF_VAE_512,
)
from serving.production.base_pipeline import BasePaths, BasePipeline, InferenceTiming

if TYPE_CHECKING:
    import PIL.Image
    import torch


@dataclass(frozen=True)
class InferenceParams:
    prompt: str = ""
    num_inference_steps: int = 8
    guidance_scale: float = 0.0
    seed: int = 42
    max_sequence_length: int = 512
    upscale_to_1024: bool = False


@dataclass
class ZImagePaths(BasePaths):
    project_root: Path = field(
        default_factory=lambda: Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    )
    traced_transformer: Path = field(init=False)
    traced_vae: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "traced_transformer",
            self.project_root / COMPILED_MODELS_DIR / NEFF_TRANSFORMER_512,
        )
        object.__setattr__(
            self, "traced_vae", self.project_root / COMPILED_MODELS_DIR / NEFF_VAE_512
        )

    def validate(self, *, require_encoder: bool = True) -> None:
        missing = []
        for name in ("model_weights", "traced_transformer", "traced_vae"):
            p = getattr(self, name)
            if not p.exists():
                missing.append(f"{name} → {p}")
        if require_encoder and not self.traced_encoder.exists():
            missing.append(f"traced_encoder (optional) → {self.traced_encoder}")
        if missing:
            raise FileNotFoundError(
                "ZImagePipeline required artifacts missing:\n  " + "\n  ".join(missing)
            )

    @property
    def has_traced_encoder(self) -> bool:
        return self.traced_encoder.exists()


TRACED_ENCODER_MAX_SEQ_LEN = 256


class ZImagePipeline(BasePipeline):
    def __init__(
        self,
        paths: ZImagePaths | None = None,
        cpu_threads: int = 16,
        verbose: bool = True,
        encoder_mode: str = "inline",
    ):
        if paths is None:
            paths = ZImagePaths()
        paths.validate(require_encoder=False)

        super().__init__(paths, verbose)

        valid_modes = {"inline", "subprocess", "cpu"}
        if encoder_mode not in valid_modes:
            raise ValueError(
                f"encoder_mode must be one of {sorted(valid_modes)}, got {encoder_mode!r}"
            )
        self._encoder_mode_requested = encoder_mode

        self._setup_neuron_env(cpu_threads, encoder_mode)

        import torch as _torch

        _torch.set_num_threads(cpu_threads)
        self._torch = _torch
        self._load_all()

    def _setup_neuron_env(self, cpu_threads: int, encoder_mode: str) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("PJRT_DEVICE", "NEURON")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if encoder_mode == "subprocess":
            os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
        else:
            os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0,1")
        default_cache = str(self._paths.project_root / COMPILED_MODELS_DIR / "cache_512")
        os.environ.setdefault("NEURON_COMPILE_CACHE_URL", default_cache)

    def _load_all(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        paths: ZImagePaths = self._paths  # type: ignore[assignment]

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(paths.model_weights / "tokenizer"),
            trust_remote_code=True,  # nosec B615 - local path, no remote download
        )

        self._encoder_path = "cpu_fp32"
        self._traced_encoder = None
        self._cpu_encoder = None

        mode = self._encoder_mode_requested

        t0 = time.perf_counter()
        self._log(
            f"loading traced transformer NEFF "
            f"({paths.traced_transformer.stat().st_size / 1e9:.2f} GB) …"
        )
        self._traced_xfm = torch.jit.load(str(paths.traced_transformer)).eval()
        self._log(f"  transformer loaded [{time.perf_counter() - t0:.1f}s]")

        t0 = time.perf_counter()
        self._log(f"loading traced VAE NEFF ({paths.traced_vae.stat().st_size / 1e6:.0f} MB) …")
        self._traced_vae = torch.jit.load(str(paths.traced_vae)).eval()
        self._log(f"  VAE loaded [{time.perf_counter() - t0:.1f}s]")

        gc.collect()

        if mode == "subprocess" and paths.encoder_server.exists() and paths.has_traced_encoder:
            t0 = time.perf_counter()
            self._log(f"spawning encoder subprocess on Core 1 (script={paths.encoder_server})…")
            if self._init_encoder_subprocess():
                self._log(f"  encoder subprocess ready [{time.perf_counter() - t0:.1f}s]")
            else:
                mode = "inline"

        if mode == "inline" and paths.has_traced_encoder:
            t0 = time.perf_counter()
            try:
                self._log(
                    f"loading traced Qwen3 encoder NEFF "
                    f"({paths.traced_encoder.stat().st_size / 1e9:.2f} GB) …"
                )
                self._traced_encoder = torch.jit.load(str(paths.traced_encoder)).eval()
                self._encoder_path = "neuron"
                self._log(f"  encoder loaded [{time.perf_counter() - t0:.1f}s]")
            except Exception as e:
                self._log(f"  traced encoder load failed ({e}); falling back to CPU FP32")
                self._traced_encoder = None

        if self._traced_encoder is None and self._encoder_client is None:
            t0 = time.perf_counter()
            self._log("loading Qwen3 text encoder (CPU, bf16 to fit ml.inf2.xlarge RAM) …")
            self._cpu_encoder = (
                AutoModel.from_pretrained(
                    str(paths.model_weights / "text_encoder"),
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,  # nosec B615 - local path, no remote download
                )
                .to("cpu")
                .eval()
            )
            self._log(f"  CPU encoder ready [{time.perf_counter() - t0:.1f}s]")
            self._encoder_path = "cpu_fp32"

        with open(paths.model_weights / "vae" / "config.json") as fh:
            vcfg = json.load(fh)
        self._vae_scaling = vcfg.get("scaling_factor", 0.18215)
        self._vae_shift = vcfg.get("shift_factor", 0.0) or 0.0

        with open(paths.model_weights / "scheduler" / "scheduler_config.json") as fh:
            self._scheduler_cfg = json.load(fh)

        gc.collect()

        if self._traced_encoder is not None:
            self._log("warming up traced encoder (first DMA ~7 s)…")
            t0 = time.perf_counter()
            with torch.no_grad():
                dummy_ids = torch.zeros((1, TRACED_ENCODER_MAX_SEQ_LEN), dtype=torch.long)
                dummy_mask = torch.zeros((1, TRACED_ENCODER_MAX_SEQ_LEN), dtype=torch.long)
                _ = self._traced_encoder(dummy_ids, dummy_mask)
            self._log(f"  encoder warm [{time.perf_counter() - t0:.1f}s]")

        self._log(
            f"ready.  encoder path = {self._encoder_path}  "
            f"(requested mode = {self._encoder_mode_requested})"
        )

    @classmethod
    def from_default_artifacts(cls, **kwargs) -> ZImagePipeline:
        return cls(paths=ZImagePaths(), **kwargs)

    def _encode_neuron(self, formatted: str) -> tuple[torch.Tensor, int]:
        import torch

        inputs = self._tokenizer(
            [formatted],
            padding="max_length",
            max_length=TRACED_ENCODER_MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        n_tok = int((inputs.attention_mask[0] == 1).sum().item())
        with torch.no_grad():
            embeds = self._traced_encoder(inputs.input_ids, inputs.attention_mask.long())
        mask = inputs.attention_mask[0].bool()
        emb = embeds[0][mask]
        cap_dim = emb.shape[-1]
        out = torch.zeros(512, cap_dim, dtype=emb.dtype)
        valid = min(emb.shape[0], 512)
        out[:valid] = emb[:valid]
        return out, n_tok

    def _encode_subprocess(self, prompt: str, seed: int = 0) -> tuple[torch.Tensor, int]:
        import torch

        result = self._encoder_client.encode(prompt, seed=seed)
        embeds = torch.load(
            io.BytesIO(result["embeds_pt_bytes"]),
            map_location="cpu",
            weights_only=True,
        )
        out = embeds[0].contiguous()
        return out, int(result["n_tokens"])

    def _encode_cpu_fp32(self, formatted: str) -> tuple[torch.Tensor, int]:
        import torch

        inputs = self._tokenizer(
            [formatted], padding=False, truncation=True, max_length=512, return_tensors="pt"
        )
        n_tok = inputs.input_ids.shape[1]
        with torch.no_grad():
            embeds = (
                self._cpu_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask.bool(),
                    output_hidden_states=True,
                    use_cache=False,
                )
                .hidden_states[-2]
                .bfloat16()
            )
        cap_dim = embeds.shape[-1]
        out = torch.zeros(512, cap_dim, dtype=embeds.dtype)
        valid = min(embeds.shape[1], 512)
        out[:valid] = embeds[0, :valid]
        return out, n_tok

    def _encode(
        self, prompt: str, max_sequence_length: int = 512, seed: int = 0
    ) -> tuple[torch.Tensor, int, str]:
        if self._encoder_client is not None:
            try:
                cap, n = self._encode_subprocess(prompt, seed=seed)
                return cap, n, "subprocess"
            except Exception as e:
                self._log(f"subprocess encode failed ({e}); restarting…")
                if self._restart_encoder_subprocess():
                    cap, n = self._encode_subprocess(prompt, seed=seed)
                    return cap, n, "subprocess"
                else:
                    self._encoder_client = None

        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        if self._traced_encoder is not None:
            cap, n = self._encode_neuron(formatted)
            return cap, n, "neuron"

        if self._cpu_encoder is None:
            import torch
            from transformers import AutoModel

            self._log("lazy-loading CPU FP32 encoder (no NEFF available) …")
            self._cpu_encoder = (
                AutoModel.from_pretrained(
                    str(self._paths.model_weights / "text_encoder"),
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,  # nosec B615 - local path, no remote download
                )
                .to("cpu")
                .eval()
            )
        cap, n = self._encode_cpu_fp32(formatted)
        return cap, n, "cpu_fp32"

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 8,
        max_sequence_length: int = 512,
        upscale_to_1024: bool = False,
    ) -> tuple[PIL.Image.Image, InferenceTiming]:
        import torch
        from PIL import Image

        timing = InferenceTiming(resolution=512)
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        cap, n_tok, path = self._encode(prompt, max_sequence_length, seed=seed)
        timing.t_encode_s = round(time.perf_counter() - t0, 3)
        timing.n_tokens = int(n_tok)
        timing.encoder_mode = path

        LATENT_HW = 64
        IN_CH = 16
        _, timesteps_list, sigmas_list = self._build_scheduler(num_inference_steps, LATENT_HW)

        gen_cpu = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(
            (1, IN_CH, LATENT_HW, LATENT_HW), generator=gen_cpu, device="cpu", dtype=torch.float32
        )

        t0 = time.perf_counter()
        for i, t_val in enumerate(timesteps_list[:num_inference_steps]):
            if t_val == 0 and i == len(timesteps_list) - 1:
                continue
            t_tensor = torch.tensor([t_val], dtype=torch.bfloat16)
            t_norm = ((1000 - t_tensor) / 1000).to(torch.bfloat16)
            lat_in = latents.to(torch.bfloat16)[0].unsqueeze(1)
            with torch.no_grad():
                model_out = self._traced_xfm(lat_in, t_norm, cap)
            noise_pred = -model_out.float()
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]
            latents = latents.to(torch.float32) + (sigma_next - sigma) * noise_pred
        timing.t_denoise_s = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        lat_in = (latents.to(torch.float32) / self._vae_scaling) + self._vae_shift
        with torch.no_grad():
            img = self._traced_vae(lat_in)
            if hasattr(img, "sample"):
                img = img.sample
        timing.t_vae_s = round(time.perf_counter() - t0, 3)

        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        img = (img * 255).round().astype("uint8")
        pil = Image.fromarray(img[0])

        if upscale_to_1024:
            t0 = time.perf_counter()
            pil = pil.resize((1024, 1024), Image.LANCZOS)
            timing.t_vae_s += round(time.perf_counter() - t0, 3)

        timing.t_total_s = round(time.perf_counter() - t_total, 3)
        return pil, timing

    def generate_streaming(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 8,
        max_sequence_length: int = 512,
        upscale_to_1024: bool = False,
        denoise_viz: bool = True,
    ) -> Iterator[tuple[str, int, PIL.Image.Image, InferenceTiming | None]]:
        import torch
        from PIL import Image

        timing = InferenceTiming(resolution=512)
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        cap, n_tok, path = self._encode(prompt, max_sequence_length, seed=seed)
        timing.t_encode_s = round(time.perf_counter() - t0, 3)
        timing.n_tokens = int(n_tok)
        timing.encoder_mode = path
        logger.info("streaming encode={:.3f}s path={}", timing.t_encode_s, path)

        LATENT_HW = 64
        IN_CH = 16
        _, timesteps_list, sigmas_list = self._build_scheduler(num_inference_steps, LATENT_HW)

        gen_cpu = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(
            (1, IN_CH, LATENT_HW, LATENT_HW), generator=gen_cpu, device="cpu", dtype=torch.float32
        )

        t0 = time.perf_counter()
        for i, t_val in enumerate(timesteps_list[:num_inference_steps]):
            if t_val == 0 and i == len(timesteps_list) - 1:
                continue
            t_step = time.perf_counter()
            t_tensor = torch.tensor([t_val], dtype=torch.bfloat16)
            t_norm = ((1000 - t_tensor) / 1000).to(torch.bfloat16)
            lat_in = latents.to(torch.bfloat16)[0].unsqueeze(1)
            with torch.no_grad():
                model_out = self._traced_xfm(lat_in, t_norm, cap)
            noise_pred = -model_out.float()
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]
            latents = latents.to(torch.float32) + (sigma_next - sigma) * noise_pred
            t_dit = time.perf_counter() - t_step

            if denoise_viz and i < num_inference_steps - 1:
                try:
                    t_vae_start = time.perf_counter()
                    lat_in_vae = (latents.to(torch.float32) / self._vae_scaling) + self._vae_shift
                    with torch.no_grad():
                        inter = self._traced_vae(lat_in_vae)
                        if hasattr(inter, "sample"):
                            inter = inter.sample
                    inter = (inter / 2 + 0.5).clamp(0, 1)
                    inter = inter.cpu().permute(0, 2, 3, 1).float().numpy()
                    inter = (inter * 255).round().astype("uint8")
                    t_vae = time.perf_counter() - t_vae_start
                    logger.info("step {} dit={:.3f}s vae={:.3f}s", i, t_dit, t_vae)
                    yield ("intermediate", i, Image.fromarray(inter[0]), None)
                except Exception:
                    logger.info("step {} dit={:.3f}s vae=FAIL", i, t_dit)
            else:
                logger.info("step {} dit={:.3f}s", i, t_dit)

        timing.t_denoise_s = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        lat_in = (latents.to(torch.float32) / self._vae_scaling) + self._vae_shift
        with torch.no_grad():
            img = self._traced_vae(lat_in)
            if hasattr(img, "sample"):
                img = img.sample
        timing.t_vae_s = round(time.perf_counter() - t0, 3)
        logger.info("final vae={:.3f}s denoise_total={:.3f}s", timing.t_vae_s, timing.t_denoise_s)

        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        img = (img * 255).round().astype("uint8")
        pil = Image.fromarray(img[0])

        if upscale_to_1024:
            t0 = time.perf_counter()
            pil = pil.resize((1024, 1024), Image.LANCZOS)
            timing.t_vae_s += round(time.perf_counter() - t0, 3)

        timing.t_total_s = round(time.perf_counter() - t_total, 3)
        logger.info("streaming total={:.3f}s", timing.t_total_s)
        yield ("final", num_inference_steps, pil, timing)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo on Inferentia2 — one-shot generator."
    )
    parser.add_argument("prompt")
    parser.add_argument(
        "-o", "--output", default="out.png", help="output PNG path (default: out.png)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--upscale", action="store_true", help="LANCZOS upscale 512 -> 1024 before saving"
    )
    parser.add_argument(
        "--encoder-mode",
        choices=("inline", "subprocess", "cpu"),
        default="inline",
        help="inline: encoder in main proc (legacy); "
        "subprocess: encoder on Core 1 via subprocess "
        "(required for ml.inf2.xlarge); "
        "cpu: CPU-FP32 fallback only.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _CODE_ROOT = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
    sys.path.insert(0, str(_CODE_ROOT / "src"))
    sys.path.insert(0, str(_CODE_ROOT))

    with ZImagePipeline.from_default_artifacts(
        verbose=not args.quiet,
        encoder_mode=args.encoder_mode,
    ) as pipe:
        img, timing = pipe.generate(
            args.prompt,
            seed=args.seed,
            num_inference_steps=args.steps,
            max_sequence_length=args.max_seq_len,
            upscale_to_1024=args.upscale,
        )
        img.save(args.output)
        logger.info(
            f"saved {args.output}  ({img.size[0]}x{img.size[1]})  "
            f"total={timing.t_total_s}s  "
            f"encode={timing.t_encode_s}s [{timing.encoder_mode}]  "
            f"denoise={timing.t_denoise_s}s  "
            f"vae={timing.t_vae_s}s  "
            f"tokens={timing.n_tokens}"
        )
