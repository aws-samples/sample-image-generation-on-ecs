from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from lib.paths import (
    COMPILED_MODELS_DIR,
    EFS_ARTIFACTS_PATH,
    NEFF_ENCODER_256,
    WEIGHTS_DIR,
)

if TYPE_CHECKING:
    import PIL.Image


@dataclass
class BasePaths:
    project_root: Path = field(
        default_factory=lambda: Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", EFS_ARTIFACTS_PATH))
    )
    model_weights: Path = field(init=False)
    traced_encoder: Path = field(init=False)
    encoder_server: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_weights", self.project_root / WEIGHTS_DIR)
        object.__setattr__(
            self,
            "traced_encoder",
            self.project_root / COMPILED_MODELS_DIR / NEFF_ENCODER_256,
        )
        object.__setattr__(
            self,
            "encoder_server",
            Path(__file__).resolve().parent.parent / "encoder_server.py",
        )

    def validate(self) -> None:
        missing = []
        for name in ("model_weights", "traced_encoder"):
            p = getattr(self, name)
            if not p.exists():
                missing.append(f"{name} → {p}")
        if missing:
            raise FileNotFoundError("Required artifacts missing:\n  " + "\n  ".join(missing))


@dataclass
class InferenceTiming:
    t_encode_s: float = 0.0
    t_denoise_s: float = 0.0
    t_vae_s: float = 0.0
    t_total_s: float = 0.0
    n_tokens: int = 0
    encoder_mode: str = ""
    resolution: int = 0
    backend: str = "neuron"


class BasePipeline(ABC):
    def __init__(self, paths: BasePaths, verbose: bool = True) -> None:
        self._paths = paths
        self._verbose = verbose
        self._encoder_client = None
        self._encoder_path = "none"

    def _log(self, msg: str) -> None:
        if self._verbose:
            logger.info(msg)

    def _init_encoder_subprocess(self) -> bool:
        from serving.encoder_client import EncoderClient

        try:
            self._encoder_client = EncoderClient(self._paths.encoder_server)
            self._encoder_path = "subprocess"
            self._log("encoder subprocess ready")
            return True
        except Exception as e:
            self._log(f"subprocess encoder failed ({e}); falling back to inline")
            self._encoder_client = None
            return False

    def _restart_encoder_subprocess(self) -> bool:
        from serving.encoder_client import EncoderClient

        try:
            self._encoder_client = EncoderClient(self._paths.encoder_server)
            self._encoder_path = "subprocess"
            self._log("subprocess restarted successfully")
            return True
        except Exception as e:
            self._log(f"subprocess restart failed ({e})")
            self._encoder_client = None
            return False

    def close(self) -> None:
        if self._encoder_client is not None:
            with contextlib.suppress(Exception):
                self._encoder_client.close()
            self._encoder_client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _encode_subprocess_to_xla(
        self, prompt: str, seed: int, xla_device: object
    ) -> tuple[object, int]:
        import io

        import torch

        assert self._encoder_client is not None, "subprocess encoder not started"
        result = self._encoder_client.encode(prompt, seed=seed)
        embeds = torch.load(
            io.BytesIO(result["embeds_pt_bytes"]),
            map_location="cpu",
            weights_only=True,
        )
        n_tokens = int(result["n_tokens"])
        from neuron.runtime import pad_caption

        valid = embeds[0, :n_tokens]
        cap = pad_caption(valid, max_seq_len=512, device=xla_device)
        return cap, n_tokens

    def _build_scheduler(self, num_inference_steps: int, latent_hw: int):
        import sys as _sys

        import torch

        _code_root = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
        _sys.path.insert(0, str(_code_root / "src"))
        _sys.path.insert(0, str(_code_root))
        from zimage.pipeline import calculate_shift, retrieve_timesteps
        from zimage.scheduler import FlowMatchEulerDiscreteScheduler

        from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT

        scfg = self._scheduler_cfg
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=scfg.get("num_train_timesteps", 1000),
            shift=scfg.get("shift", 3.0),
            use_dynamic_shifting=scfg.get("use_dynamic_shifting", False),
        )
        image_seq_len = (latent_hw // 2) ** 2
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", BASE_IMAGE_SEQ_LEN),
            scheduler.config.get("max_image_seq_len", MAX_IMAGE_SEQ_LEN),
            scheduler.config.get("base_shift", BASE_SHIFT),
            scheduler.config.get("max_shift", MAX_SHIFT),
        )
        scheduler.sigma_min = 0.0
        timesteps, _ = retrieve_timesteps(
            scheduler, num_inference_steps, torch.device("cpu"), sigmas=None, mu=mu
        )
        scheduler._step_index = 0
        return scheduler, timesteps.cpu().tolist(), scheduler.sigmas.cpu().tolist()

    @abstractmethod
    def generate(
        self, prompt: str, seed: int = 42, num_inference_steps: int = 8, **kwargs
    ) -> tuple[PIL.Image.Image, InferenceTiming]: ...

    @abstractmethod
    def generate_streaming(
        self, prompt: str, seed: int = 42, num_inference_steps: int = 8, **kwargs
    ) -> Iterator[tuple[str, int, PIL.Image.Image, InferenceTiming | None]]: ...
