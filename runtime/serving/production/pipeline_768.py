from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from lib.neuron_cache import resolve_compile_cache
from lib.paths import (
    CACHE_768_PREFIX,
    COMPILED_MODELS_DIR,
    DEFAULT_ARTIFACTS_DIR,
    NEFF_VAE_768,
)
from serving.production.base_pipeline import BasePaths, BasePipeline, InferenceTiming

if TYPE_CHECKING:
    import PIL.Image
    import torch


@dataclass
class ZImagePaths768(BasePaths):
    project_root: Path = field(
        default_factory=lambda: Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    )
    traced_vae: Path = field(init=False)
    compile_cache: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "traced_vae", self.project_root / COMPILED_MODELS_DIR / NEFF_VAE_768
        )
        cache_version = os.environ.get("NEURON_CACHE_VERSION", "")
        cache_subdir = f"{CACHE_768_PREFIX}/{cache_version}" if cache_version else CACHE_768_PREFIX
        object.__setattr__(
            self, "compile_cache", self.project_root / COMPILED_MODELS_DIR / cache_subdir
        )

    def validate(self) -> None:
        missing = []
        for name in ("model_weights", "traced_vae"):
            p = getattr(self, name)
            if not p.exists():
                missing.append(f"{name} -> {p}")
        if missing:
            raise FileNotFoundError(
                "ZImagePipeline768 required artifacts missing:\n  " + "\n  ".join(missing)
            )
        self.compile_cache.mkdir(parents=True, exist_ok=True)


class ZImagePipeline768(BasePipeline):
    def __init__(
        self, paths: ZImagePaths768 | None = None, cpu_threads: int = 16, verbose: bool = True
    ):
        if paths is None:
            paths = ZImagePaths768()
        paths.validate()

        super().__init__(paths, verbose)

        self._setup_neuron_env(cpu_threads)

        import torch as _torch

        _torch.set_num_threads(cpu_threads)
        self._torch = _torch

        if os.environ.get("ZIMAGE_SKIP_ENCODER_SUBPROCESS") == "1":
            self._log("encoder subprocess skipped (ZIMAGE_SKIP_ENCODER_SUBPROCESS=1)")
        else:
            try:
                self._log("starting encoder subprocess on Core 1...")
                self._init_encoder_subprocess()
                os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
            except Exception as e:
                self._log(f"subprocess encoder failed ({e}); falling back to CPU FP32")

        self._load_main_pipeline()

    def _setup_neuron_env(self, cpu_threads: int) -> None:
        paths: ZImagePaths768 = self._paths  # type: ignore[assignment]
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("PJRT_DEVICE", "NEURON")
        os.environ.setdefault("NEURON_NUM_WORKERS", "4")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
        os.environ["ZIMAGE_MARK_STEP_EVERY_N_LAYERS"] = "10"
        os.environ.setdefault("USE_TRACED_VAE", "1")
        os.environ["NEURON_VAE_PATH"] = str(paths.traced_vae)
        cache_subdir = str(
            paths.compile_cache.relative_to(paths.project_root / COMPILED_MODELS_DIR)
        )
        os.environ["NEURON_COMPILE_CACHE_URL"] = str(
            resolve_compile_cache(paths.compile_cache, cache_subdir)
        )
        os.environ["NEURON_CC_FLAGS"] = (
            "--model-type transformer --enable-fast-loading-neuron-binaries --auto-cast matmult"
        )

    def _load_main_pipeline(self) -> None:
        _code_root = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
        sys.path.insert(0, str(_code_root / "src"))
        sys.path.insert(0, str(_code_root))
        import torch

        from neuron.runtime import apply_patches, get_xla_device, load_all_on_neuron
        from utils import ensure_model_weights

        apply_patches(attention_backend="native")

        t0 = time.perf_counter()
        self._log("loading transformer + VAE + CPU encoder (fallback) ...")
        xla_device = get_xla_device()
        model_path = ensure_model_weights(str(self._paths.model_weights), verify=False)
        self._components = load_all_on_neuron(model_path, xla_device, torch.bfloat16, verbose=False)
        self._xla_device = xla_device
        from neuron import runtime as neuron_utils

        self._neuron_utils = neuron_utils
        self._log(f"  loaded in {time.perf_counter() - t0:.1f}s")
        self._warmed_up = False

    def warmup(self, prompt: str | None = None, seed: int = 42) -> float:
        from neuron.runtime import generate_on_neuron

        self._neuron_utils._PROMPT_EMBED_CACHE.clear()
        self._log("warmup call (compiles / cache-hits sub-graphs)...")
        t0 = time.perf_counter()
        warmup_prompt = prompt or "warmup"

        _ = generate_on_neuron(
            **self._components,
            prompt=warmup_prompt,
            height=768,
            width=768,
            num_inference_steps=8,
            guidance_scale=0.0,
            seed=seed,
            max_sequence_length=512,
            verbose=False,
        )
        t_after_gen = time.perf_counter()
        self._log(f"  warmup pass 1 (generate) done in {t_after_gen - t0:.1f}s")

        try:
            for _frame in self.generate_streaming(
                warmup_prompt,
                seed=seed,
                num_inference_steps=8,
                denoise_viz=True,
            ):
                pass
            self._log(
                f"  warmup pass 2 (streaming) done in {time.perf_counter() - t_after_gen:.1f}s"
            )
        except Exception as exc:
            self._log(
                f"  warmup pass 2 (streaming) FAILED: {exc!r} — "
                f"streaming-path PJRT cache may be incomplete"
            )

        dt = time.perf_counter() - t0
        self._log(f"  warmup TOTAL {dt:.1f}s")
        self._warmed_up = True
        return dt

    def _encode_via_subprocess(self, prompt: str, seed: int) -> tuple[torch.Tensor, int]:
        return self._encode_subprocess_to_xla(prompt, seed, self._xla_device)

    @classmethod
    def from_default_artifacts(cls, **kwargs) -> ZImagePipeline768:
        return cls(paths=ZImagePaths768(), **kwargs)

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 8,
        max_sequence_length: int = 512,
    ) -> tuple[PIL.Image.Image, InferenceTiming]:
        from neuron.runtime import generate_on_neuron

        timing = InferenceTiming(resolution=768)
        t_total = time.perf_counter()

        self._neuron_utils._PROMPT_EMBED_CACHE.clear()

        if self._encoder_client is not None:
            t0 = time.perf_counter()
            try:
                cap, n_tokens = self._encode_via_subprocess(prompt, seed)
                timing.t_encode_s = round(time.perf_counter() - t0, 3)
                timing.n_tokens = int(n_tokens)
                timing.encoder_mode = "subprocess"
                cache_key = ((prompt,), 512)
                self._neuron_utils._PROMPT_EMBED_CACHE[cache_key] = [cap]
            except Exception as e:
                self._log(f"subprocess encode failed ({e}); falling back")
                timing.encoder_mode = "cpu_fp32"
        else:
            timing.encoder_mode = "cpu_fp32"

        t_gen0 = time.perf_counter()
        imgs = generate_on_neuron(
            **self._components,
            prompt=prompt,
            height=768,
            width=768,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            seed=seed,
            max_sequence_length=max_sequence_length,
            verbose=False,
        )
        t_gen = time.perf_counter() - t_gen0

        timing.t_denoise_s = round(t_gen, 3)
        timing.t_total_s = round(time.perf_counter() - t_total, 3)
        return imgs[0], timing

    def generate_streaming(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 8,
        max_sequence_length: int = 512,
        denoise_viz: bool = True,
        **_kwargs,
    ) -> Iterator[tuple[str, int, PIL.Image.Image, InferenceTiming | None]]:
        import sys as _sys

        import torch
        from PIL import Image
        from zimage.pipeline import calculate_shift, retrieve_timesteps

        from neuron.runtime import _mark_step, pad_caption

        _code_root = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
        _sys.path.insert(0, str(_code_root / "src"))
        from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT

        timing = InferenceTiming(resolution=768)
        t_total = time.perf_counter()
        self._neuron_utils._PROMPT_EMBED_CACHE.clear()

        transformer = self._components["transformer"]
        vae = self._components["vae"]
        text_encoder = self._components["text_encoder"]
        tokenizer = self._components["tokenizer"]
        scheduler = self._components["scheduler"]
        xla_device = self._xla_device
        dtype = torch.bfloat16

        # ---- Text encoding: subprocess (fast path) or inline CPU FP32 fallback ---
        prompt_embeds = None
        if self._encoder_client is not None:
            t_enc = time.perf_counter()
            try:
                cap, n_tokens = self._encode_via_subprocess(prompt, seed)
                timing.t_encode_s = round(time.perf_counter() - t_enc, 3)
                timing.n_tokens = int(n_tokens)
                timing.encoder_mode = "subprocess"
                prompt_embeds = [cap]
            except Exception as exc:
                self._log(f"subprocess encode failed ({exc}); falling back to CPU FP32")
                prompt_embeds = None

        if prompt_embeds is None:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            probe = tokenizer(
                [formatted],
                padding=False,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )
            timing.n_tokens = int(probe.input_ids.shape[1])
            t_enc = time.perf_counter()
            inputs = tokenizer(
                [formatted],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                embeds = (
                    text_encoder(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask.bool(),
                        output_hidden_states=True,
                    )
                    .hidden_states[-2]
                    .bfloat16()
                )
            masks = inputs.attention_mask.bool()
            emb = embeds[0][masks[0]]
            cap = pad_caption(emb, max_seq_len=512, device=xla_device)
            prompt_embeds = [cap]
            timing.t_encode_s = round(time.perf_counter() - t_enc, 3)
            timing.encoder_mode = "cpu_fp32"

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_scale = vae_scale_factor * 2
        height = width = 768
        height_latent = 2 * (height // vae_scale)
        width_latent = 2 * (width // vae_scale)
        image_seq_len = (height_latent // 2) * (width_latent // 2)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", BASE_IMAGE_SEQ_LEN),
            scheduler.config.get("max_image_seq_len", MAX_IMAGE_SEQ_LEN),
            scheduler.config.get("base_shift", BASE_SHIFT),
            scheduler.config.get("max_shift", MAX_SHIFT),
        )
        scheduler.sigma_min = 0.0
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            xla_device,
            sigmas=None,
            mu=mu,
        )

        shape = (1, transformer.in_channels, height_latent, width_latent)
        gen_cpu = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(shape, generator=gen_cpu, device="cpu", dtype=torch.float32).to(
            xla_device
        )
        _mark_step()
        scheduler._step_index = 0

        use_traced = os.environ.get("USE_TRACED_VAE", "1") == "1"
        traced_vae = None
        if use_traced:
            traced_vae_path = os.environ.get("NEURON_VAE_PATH", str(self._paths.traced_vae))
            if self._neuron_utils._TRACED_VAE is None:
                self._neuron_utils._TRACED_VAE = torch.jit.load(traced_vae_path)
            traced_vae = self._neuron_utils._TRACED_VAE
        shift = getattr(vae.config, "shift_factor", 0.0) or 0.0

        def _decode_latents_to_pil(lats_xla) -> PIL.Image.Image:
            _mark_step()
            lat_cpu = lats_xla.to(torch.float32).cpu()
            lat_in = lat_cpu / vae.config.scaling_factor + shift
            with torch.no_grad():
                if traced_vae is not None:
                    img = traced_vae(lat_in)
                else:
                    img = vae.decode(lat_in, return_dict=False)[0]
            img = (img / 2 + 0.5).clamp(0, 1)
            arr = img.cpu().permute(0, 2, 3, 1).float().numpy()
            arr = (arr * 255).round().astype("uint8")
            return Image.fromarray(arr[0])

        t0_denoise = time.perf_counter()
        for i, t in enumerate(timesteps.cpu().tolist()):
            if t == 0 and i == len(timesteps) - 1:
                continue
            t_tensor = torch.tensor([t], device=xla_device)
            timestep = t_tensor.expand(1)
            t_norm = (1000 - timestep) / 1000
            lat_in = latents.to(dtype).unsqueeze(2)
            lat_list = list(lat_in.unbind(dim=0))
            with torch.no_grad():
                model_out, _ = transformer(lat_list, t_norm, prompt_embeds)
            if os.environ.get("SKIP_INTERMEDIATE_MARK_STEP") != "1":
                _mark_step()
            noise_pred = torch.stack([o.float() for o in model_out], dim=0)
            noise_pred = -noise_pred.squeeze(2)

            if not hasattr(scheduler, "_sigmas_py"):
                scheduler._sigmas_py = [float(x) for x in scheduler.sigmas.detach().cpu().tolist()]
            sigma = scheduler._sigmas_py[scheduler._step_index]
            sigma_next = scheduler._sigmas_py[scheduler._step_index + 1]
            dt_val = sigma_next - sigma
            latents = latents.to(torch.float32) + dt_val * noise_pred
            scheduler._step_index += 1
            _mark_step()

            if denoise_viz and i < num_inference_steps - 1:
                try:
                    inter_pil = _decode_latents_to_pil(latents)
                    yield ("intermediate", i, inter_pil, None)
                except Exception as exc:
                    self._log(f"intermediate decode failed (step {i}): {exc}")
        timing.t_denoise_s = round(time.perf_counter() - t0_denoise, 3)

        t0_vae = time.perf_counter()
        final_pil = _decode_latents_to_pil(latents)
        timing.t_vae_s = round(time.perf_counter() - t0_vae, 3)
        timing.t_total_s = round(time.perf_counter() - t_total, 3)
        yield ("final", num_inference_steps, final_pil, timing)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Z-Image-Turbo 768x768 generator on Inferentia2.")
    parser.add_argument("prompt")
    parser.add_argument("-o", "--output", default="out_768.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with ZImagePipeline768.from_default_artifacts(verbose=not args.quiet) as pipe:
        pipe.warmup()
        img, timing = pipe.generate(args.prompt, seed=args.seed, num_inference_steps=args.steps)
        img.save(args.output)
        logger.info(
            f"saved {args.output}  ({img.size[0]}x{img.size[1]})  "
            f"total={timing.t_total_s}s  "
            f"encoder={timing.encoder_mode}  "
            f"tokens={timing.n_tokens}"
        )
