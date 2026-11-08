from __future__ import annotations

import os

import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(
            (
                "ZIMAGE_",
                "NEURON_",
                "PJRT_",
                "OMP_",
                "MKL_",
                "TOKENIZERS_",
                "USE_",
                "VAE_",
                "SAGEMAKER_",
            )
        ):
            monkeypatch.delenv(key, raising=False)


def test_default_config():
    from lib.config import RuntimeConfig

    cfg = RuntimeConfig()
    assert cfg.resolution == 512
    assert cfg.backend == "neuron"
    assert cfg.server_port == 8080


def test_from_environment(monkeypatch):
    monkeypatch.setenv("ZIMAGE_DEFAULT_RESOLUTION", "768")
    monkeypatch.setenv("ZIMAGE_BACKEND", "gpu")
    monkeypatch.setenv("SAGEMAKER_BIND_TO_PORT", "9090")
    monkeypatch.setenv("NEURON_RT_EXEC_TIMEOUT", "86400")

    from lib.config import RuntimeConfig

    cfg = RuntimeConfig.from_environment()
    assert cfg.resolution == 768
    assert cfg.backend == "gpu"
    assert cfg.server_port == 9090
    assert cfg.neuron.neuron_rt_exec_timeout == 86400


def test_invalid_resolution_rejected():
    from lib.config import RuntimeConfig

    with pytest.raises(ValidationError):
        RuntimeConfig(resolution=256)


def test_apply_to_environment(monkeypatch):
    from lib.config import NeuronEnvConfig, RuntimeConfig

    cfg = RuntimeConfig(neuron=NeuronEnvConfig(omp_num_threads=32))
    cfg.apply_to_environment()
    assert os.environ.get("OMP_NUM_THREADS") == "32"


def test_allowed_resolutions_parsing(monkeypatch):
    monkeypatch.setenv("ZIMAGE_ALLOWED_RESOLUTIONS", "512,768,1024")

    from lib.config import RuntimeConfig

    cfg = RuntimeConfig.from_environment()
    assert cfg.allowed_resolutions == [512, 768, 1024]
