from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    from lib.paths import EFS_ARTIFACTS_PATH
except ModuleNotFoundError:
    EFS_ARTIFACTS_PATH = "/mnt/efs/artifacts"


class NeuronEnvConfig(BaseModel):
    pjrt_device: str = "NEURON"
    neuron_rt_visible_cores: str = "0,1"
    neuron_rt_exec_timeout: int = 120
    neuron_cc_flags: str = ""
    neuron_compile_cache_url: str = ""
    neuron_cache_version: str = ""
    neuron_num_workers: int = 4
    neuron_fuse_softmax: bool = True
    omp_num_threads: int = 16
    mkl_num_threads: int = 16
    tokenizers_parallelism: str = "false"
    use_traced_vae: bool = True
    vae_on_cpu: bool = False
    mark_step_every_n_layers: int = 0


class RuntimeConfig(BaseModel):
    resolution: int = Field(default=512, ge=512, le=1024)
    backend: Literal["neuron", "gpu"] = "neuron"
    artifacts_dir: Path = Path(EFS_ARTIFACTS_PATH)
    encoder_mode: Literal["subprocess", "inline", "cpu"] = "subprocess"
    server_port: int = Field(default=8080, ge=1, le=65535)
    allowed_resolutions: list[int] = Field(default_factory=lambda: [512])
    eager_load: bool = True
    neuron: NeuronEnvConfig = Field(default_factory=NeuronEnvConfig)

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        import os

        resolution = int(os.environ.get("ZIMAGE_DEFAULT_RESOLUTION", "512"))
        allowed_str = os.environ.get("ZIMAGE_ALLOWED_RESOLUTIONS", str(resolution))
        allowed = [int(x) for x in allowed_str.split(",") if x.strip()]

        neuron = NeuronEnvConfig(
            pjrt_device=os.environ.get("PJRT_DEVICE", "NEURON"),
            neuron_rt_visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES", "0,1"),
            neuron_rt_exec_timeout=int(os.environ.get("NEURON_RT_EXEC_TIMEOUT", "120")),
            neuron_cc_flags=os.environ.get("NEURON_CC_FLAGS", ""),
            neuron_compile_cache_url=os.environ.get("NEURON_COMPILE_CACHE_URL", ""),
            neuron_cache_version=os.environ.get("NEURON_CACHE_VERSION", ""),
            neuron_num_workers=int(os.environ.get("NEURON_NUM_WORKERS", "4")),
            neuron_fuse_softmax=os.environ.get("NEURON_FUSE_SOFTMAX", "1") == "1",
            omp_num_threads=int(os.environ.get("OMP_NUM_THREADS", "16")),
            mkl_num_threads=int(os.environ.get("MKL_NUM_THREADS", "16")),
            tokenizers_parallelism=os.environ.get("TOKENIZERS_PARALLELISM", "false"),
            use_traced_vae=os.environ.get("USE_TRACED_VAE", "1") == "1",
            vae_on_cpu=os.environ.get("VAE_ON_CPU", "0") == "1",
            mark_step_every_n_layers=int(os.environ.get("ZIMAGE_MARK_STEP_EVERY_N_LAYERS", "0")),
        )

        backend = os.environ.get("ZIMAGE_BACKEND", "neuron").lower()
        encoder_mode = os.environ.get("ZIMAGE_ENCODER_MODE", "subprocess")

        return cls(
            resolution=resolution,
            backend=backend,  # type: ignore[arg-type]  # validated by Pydantic
            artifacts_dir=Path(os.environ.get("ZIMAGE_ARTIFACTS_DIR", EFS_ARTIFACTS_PATH)),
            encoder_mode=encoder_mode,  # type: ignore[arg-type]  # validated by Pydantic
            server_port=int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080")),
            allowed_resolutions=allowed,
            eager_load=os.environ.get("ZIMAGE_EAGER_LOAD", "true").lower() == "true",
            neuron=neuron,
        )

    def apply_to_environment(self) -> None:
        import os

        os.environ.setdefault("PJRT_DEVICE", self.neuron.pjrt_device)
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", self.neuron.neuron_rt_visible_cores)
        os.environ.setdefault("OMP_NUM_THREADS", str(self.neuron.omp_num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.neuron.mkl_num_threads))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", self.neuron.tokenizers_parallelism)
        if self.neuron.neuron_compile_cache_url:
            os.environ.setdefault("NEURON_COMPILE_CACHE_URL", self.neuron.neuron_compile_cache_url)
        if self.neuron.neuron_cc_flags:
            os.environ.setdefault("NEURON_CC_FLAGS", self.neuron.neuron_cc_flags)
