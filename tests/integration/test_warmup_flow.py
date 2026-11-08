from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_RUNTIME_PATH = str(Path(__file__).resolve().parents[2] / "runtime")
if _RUNTIME_PATH not in sys.path:
    sys.path.insert(0, _RUNTIME_PATH)

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

for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


@pytest.fixture
def artifacts_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


class TestVerifyEfsArtifacts:
    def test_logs_present_weights(self, artifacts_dir):
        from serving.warmup_cache import _verify_efs_artifacts

        weights_dir = artifacts_dir / "ckpts" / "Z-Image-Turbo"
        weights_dir.mkdir(parents=True)
        (weights_dir / "model_index.json").write_text("{}")
        (weights_dir / "weight_0.safetensors").write_bytes(b"\x00" * 100)

        _verify_efs_artifacts(artifacts_dir, 768)

    def test_logs_missing_weights(self, artifacts_dir):
        from serving.warmup_cache import _verify_efs_artifacts

        _verify_efs_artifacts(artifacts_dir, 768)

    def test_checks_correct_neffs_per_resolution(self, artifacts_dir):
        from serving.warmup_cache import _verify_efs_artifacts

        compiled_dir = artifacts_dir / "compiled_models"
        compiled_dir.mkdir(parents=True)
        (compiled_dir / "encoder_256.pt").write_bytes(b"\x00" * 100)
        (compiled_dir / "vae_768.pt").write_bytes(b"\x00" * 100)

        _verify_efs_artifacts(artifacts_dir, 768)


class TestMainArgumentParsing:
    def test_resolution_is_required(self, monkeypatch):
        from serving.warmup_cache import main

        monkeypatch.setattr(sys, "argv", ["warmup_cache.py"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2

    def test_resolution_only_accepted(self, monkeypatch, tmp_path):
        from serving.warmup_cache import main

        monkeypatch.setattr(sys, "argv", ["warmup_cache.py", "--resolution", "768"])
        monkeypatch.setenv("ZIMAGE_ARTIFACTS_DIR", str(tmp_path / "art"))

        mock_pipe = MagicMock()
        mock_module = MagicMock()
        mock_module.ZImagePipeline768.from_default_artifacts.return_value = mock_pipe

        with patch.dict("sys.modules", {"serving.production.pipeline_768": mock_module}):
            result = main()

        assert result == 0
        mock_pipe.warmup.assert_called_once()

    def test_invalid_resolution_rejected(self, monkeypatch):
        from serving.warmup_cache import main

        monkeypatch.setattr(
            sys,
            "argv",
            ["warmup_cache.py", "--resolution", "512"],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2

    def test_valid_resolution_768_accepted(self, monkeypatch, tmp_path):
        from serving.warmup_cache import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "warmup_cache.py",
                "--resolution",
                "768",
            ],
        )
        monkeypatch.setenv("ZIMAGE_ARTIFACTS_DIR", str(tmp_path / "art"))

        mock_pipe = MagicMock()
        mock_module = MagicMock()
        mock_module.ZImagePipeline768.from_default_artifacts.return_value = mock_pipe

        with patch.dict("sys.modules", {"serving.production.pipeline_768": mock_module}):
            result = main()

        assert result == 0
        mock_pipe.warmup.assert_called_once()

    def test_valid_resolution_1024_accepted(self, monkeypatch, tmp_path):
        from serving.warmup_cache import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "warmup_cache.py",
                "--resolution",
                "1024",
            ],
        )
        monkeypatch.setenv("ZIMAGE_ARTIFACTS_DIR", str(tmp_path / "art"))

        mock_pipe = MagicMock()
        mock_module = MagicMock()
        mock_module.ZImagePipeline1024.from_default_artifacts.return_value = mock_pipe

        with patch.dict("sys.modules", {"serving.production.pipeline_1024": mock_module}):
            result = main()

        assert result == 0
        mock_pipe.warmup.assert_called_once()

    def test_warmup_exception_returns_1(self, monkeypatch, tmp_path):
        from serving.warmup_cache import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "warmup_cache.py",
                "--resolution",
                "768",
            ],
        )
        monkeypatch.setenv("ZIMAGE_ARTIFACTS_DIR", str(tmp_path / "art"))

        mock_pipe = MagicMock()
        mock_pipe.warmup.side_effect = RuntimeError("import failed")
        mock_module = MagicMock()
        mock_module.ZImagePipeline768.from_default_artifacts.return_value = mock_pipe

        with patch.dict("sys.modules", {"serving.production.pipeline_768": mock_module}):
            result = main()

        assert result == 1
