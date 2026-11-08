from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from lib.paths import DEFAULT_ARTIFACTS_DIR, WEIGHTS_DIR

if TYPE_CHECKING:
    import PIL.Image


@dataclass
class InferenceTimingGpu:
    t_total_s: float = 0.0
    t_load_s: float = 0.0
    height: int = 0
    width: int = 0
    n_steps: int = 0
    backend: str = "cuda"


@dataclass
class ZImagePathsGpu:
    project_root: Path = Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    model_weights: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "model_weights", self.project_root / WEIGHTS_DIR)

    def validate(self) -> None:
        if not self.model_weights.exists():
            raise FileNotFoundError(
                f"ZImagePipelineGpu required artifacts missing: {self.model_weights}"
            )


class ZImagePipelineGpu:
    def __init__(
        self,
        paths: ZImagePathsGpu | None = None,
        allowed_resolutions: tuple[int, ...] = (512, 768, 1024),
        verbose: bool = True,
    ):
        if paths is None:
            paths = ZImagePathsGpu()
        paths.validate()
        self._paths = paths
        self._verbose = verbose
        self._allowed = set(allowed_resolutions)

        import torch
        from diffusers import ZImagePipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; ZImagePipelineGpu requires a GPU host")

        t0 = time.perf_counter()
        self._log(f"loading Z-Image-Turbo weights from {paths.model_weights} …")
        self._pipe = ZImagePipeline.from_pretrained(
            str(paths.model_weights),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self._pipe.to("cuda")
        self._load_s = time.perf_counter() - t0
        self._log(f"loaded in {self._load_s:.1f}s")
        self._warmed_up = False

    @classmethod
    def from_default_artifacts(cls, **kwargs) -> ZImagePipelineGpu:
        return cls(paths=ZImagePathsGpu(), **kwargs)

    def _log(self, msg: str) -> None:
        if self._verbose:
            logger.info(msg)

    def warmup(self, prompt: str = "warmup", seed: int = 0) -> float:
        t0 = time.perf_counter()
        self._log("warmup generation …")
        target_res = max(self._allowed)
        self.generate(
            prompt,
            seed=seed,
            num_inference_steps=2,
            height=target_res,
            width=target_res,
        )
        elapsed = time.perf_counter() - t0
        self._warmed_up = True
        self._log(f"warmup complete in {elapsed:.1f}s")
        return elapsed

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 8,
        height: int | None = None,
        width: int | None = None,
    ) -> tuple[PIL.Image.Image, InferenceTimingGpu]:
        import torch

        target_h = height or max(self._allowed)
        target_w = width or target_h

        steps_arg = num_inference_steps

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._pipe(
                prompt=prompt,
                height=target_h,
                width=target_w,
                num_inference_steps=steps_arg,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(seed),
            )
        image = out.images[0]
        del out
        gc.collect()
        torch.cuda.empty_cache()
        timing = InferenceTimingGpu(
            t_total_s=round(time.perf_counter() - t0, 2),
            t_load_s=round(self._load_s, 1),
            height=target_h,
            width=target_w,
            n_steps=steps_arg,
        )
        return image, timing

    def generate_streaming(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 8,
        height: int | None = None,
        width: int | None = None,
        **kwargs,
    ):
        import queue
        import threading

        import numpy as np
        import torch
        from PIL import Image as PILImage

        target_h = height or max(self._allowed)
        target_w = width or target_h
        steps_arg = num_inference_steps

        frame_queue: queue.Queue = queue.Queue()
        _SENTINEL = None

        def _run_pipeline():
            step_count = [0]

            def _step_callback(p, step_index, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                scaled = latents / p.vae.config.scaling_factor
                with torch.no_grad():
                    decoded = (
                        p.vae.decode(scaled.to(dtype=p.vae.dtype), return_dict=False)[0]
                        .float()
                        .cpu()
                    )
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                arr = decoded.permute(0, 2, 3, 1).numpy()
                arr = (arr[0] * 255).round().astype(np.uint8)
                frame_queue.put(("intermediate", step_count[0], PILImage.fromarray(arr)))
                step_count[0] += 1
                return callback_kwargs

            with torch.inference_mode():
                out = self._pipe(
                    prompt=prompt,
                    height=target_h,
                    width=target_w,
                    num_inference_steps=steps_arg,
                    guidance_scale=0.0,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    callback_on_step_end=_step_callback,
                )
            frame_queue.put(("final", steps_arg, out.images[0]))
            frame_queue.put(_SENTINEL)

        t0 = time.perf_counter()
        thread = threading.Thread(target=_run_pipeline, daemon=True)
        thread.start()

        while True:
            item = frame_queue.get()
            if item is _SENTINEL:
                break
            frame_type, step, img = item
            elapsed = round(time.perf_counter() - t0, 2)
            yield (frame_type, step, img, {"elapsed_s": elapsed})

        thread.join()
        gc.collect()
        torch.cuda.empty_cache()
