from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_MOCKED_MODULES = [
    "torch",
    "torch.jit",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",
    "torch_neuronx",
    "torch_xla",
    "torch_xla.core",
    "torch_xla.core.xla_model",
    "transformers",
    "transformers.models",
    "transformers.models.auto",
    "transformers.models.auto.auto_factory",
    "transformers.models.auto.configuration_auto",
    "transformers.AutoModel",
    "transformers.AutoTokenizer",
    "transformers.AutoConfig",
    "PIL",
    "PIL.Image",
    "neuronxcc",
    "zimage",
    "zimage.scheduler",
    "zimage.pipeline",
    "config",
]


@pytest.fixture(scope="session", autouse=True)
def _mock_ml_deps():
    originals = {}
    for mod in _MOCKED_MODULES:
        originals[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock()

    runtime_path = str(Path(__file__).resolve().parents[2] / "runtime")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)

    yield

    for mod in _MOCKED_MODULES:
        if originals[mod] is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = originals[mod]
