from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def ZImagePaths():
    from serving.production.pipeline_512 import ZImagePaths

    return ZImagePaths


@pytest.fixture
def ZImagePaths768():
    from serving.production.pipeline_768 import ZImagePaths768

    return ZImagePaths768


@pytest.fixture
def ZImagePaths1024():
    from serving.production.pipeline_1024 import ZImagePaths1024

    return ZImagePaths1024


class TestZImagePaths512DefaultRoot:
    def test_default_project_root(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.project_root == Path("/opt/ml/zimage_artifacts")


class TestZImagePaths512CustomRoot:
    def test_custom_root_propagates(self, tmp_path: Path, ZImagePaths):
        paths = ZImagePaths(project_root=tmp_path)
        assert paths.project_root == tmp_path
        assert paths.model_weights == tmp_path / "ckpts" / "Z-Image-Turbo"
        assert paths.traced_transformer == tmp_path / "compiled_models" / "transformer_512.pt"
        assert paths.traced_vae == tmp_path / "compiled_models" / "vae_512.pt"
        assert paths.traced_encoder == tmp_path / "compiled_models" / "encoder_256.pt"


class TestZImagePaths512PostInit:
    def test_model_weights_path(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.model_weights == Path("/opt/ml/zimage_artifacts/ckpts/Z-Image-Turbo")

    def test_traced_transformer_path(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.traced_transformer == Path(
            "/opt/ml/zimage_artifacts/compiled_models/transformer_512.pt"
        )

    def test_traced_vae_path(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.traced_vae == Path("/opt/ml/zimage_artifacts/compiled_models/vae_512.pt")

    def test_traced_encoder_path(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.traced_encoder == Path(
            "/opt/ml/zimage_artifacts/compiled_models/encoder_256.pt"
        )


class TestZImagePaths512ArtifactFilenames:
    def test_transformer_filename(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.traced_transformer.name == "transformer_512.pt"

    def test_vae_filename(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.traced_vae.name == "vae_512.pt"

    def test_encoder_filename(self, ZImagePaths):
        paths = ZImagePaths()
        assert paths.traced_encoder.name == "encoder_256.pt"


class TestZImagePaths512ValidateRaises:
    def test_validate_missing_all(self, tmp_path: Path, ZImagePaths):
        paths = ZImagePaths(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate(require_encoder=False)
        msg = str(exc_info.value)
        assert "ZImagePipeline required artifacts missing:" in msg
        assert "model_weights" in msg
        assert "traced_transformer" in msg
        assert "traced_vae" in msg

    def test_validate_missing_encoder_when_required(self, tmp_path: Path, ZImagePaths):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "transformer_512.pt").write_bytes(b"fake")
        (tmp_path / "compiled_models" / "vae_512.pt").write_bytes(b"fake")

        paths = ZImagePaths(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate(require_encoder=True)
        msg = str(exc_info.value)
        assert "traced_encoder (optional)" in msg

    def test_validate_missing_transformer(self, tmp_path: Path, ZImagePaths):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "vae_512.pt").write_bytes(b"fake")

        paths = ZImagePaths(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate(require_encoder=False)
        msg = str(exc_info.value)
        assert "traced_transformer" in msg


class TestZImagePaths512ValidatePasses:
    def test_validate_passes_without_encoder(self, tmp_path: Path, ZImagePaths):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "transformer_512.pt").write_bytes(b"fake")
        (tmp_path / "compiled_models" / "vae_512.pt").write_bytes(b"fake")

        paths = ZImagePaths(project_root=tmp_path)
        paths.validate(require_encoder=False)

    def test_validate_passes_with_encoder(self, tmp_path: Path, ZImagePaths):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "transformer_512.pt").write_bytes(b"fake")
        (tmp_path / "compiled_models" / "vae_512.pt").write_bytes(b"fake")
        (tmp_path / "compiled_models" / "encoder_256.pt").write_bytes(b"fake")

        paths = ZImagePaths(project_root=tmp_path)
        paths.validate(require_encoder=True)


class TestZImagePaths512HasTracedEncoder:
    def test_returns_false_when_missing(self, tmp_path: Path, ZImagePaths):
        paths = ZImagePaths(project_root=tmp_path)
        assert paths.has_traced_encoder is False

    def test_returns_true_when_present(self, tmp_path: Path, ZImagePaths):
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "encoder_256.pt").write_bytes(b"fake")

        paths = ZImagePaths(project_root=tmp_path)
        assert paths.has_traced_encoder is True


class TestZImagePaths768DefaultRoot:
    def test_default_project_root(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.project_root == Path("/opt/ml/zimage_artifacts")


class TestZImagePaths768CustomRoot:
    def test_custom_root_propagates(self, tmp_path: Path, ZImagePaths768):
        paths = ZImagePaths768(project_root=tmp_path)
        assert paths.project_root == tmp_path
        assert paths.model_weights == tmp_path / "ckpts" / "Z-Image-Turbo"
        assert paths.traced_vae == tmp_path / "compiled_models" / "vae_768.pt"
        assert paths.traced_encoder == tmp_path / "compiled_models" / "encoder_256.pt"
        assert paths.compile_cache == tmp_path / "compiled_models" / "cache_768"
        assert paths.encoder_server.name == "encoder_server.py"


class TestZImagePaths768PostInit:
    def test_model_weights_path(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.model_weights == Path("/opt/ml/zimage_artifacts/ckpts/Z-Image-Turbo")

    def test_traced_vae_path(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.traced_vae == Path("/opt/ml/zimage_artifacts/compiled_models/vae_768.pt")

    def test_traced_encoder_path(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.traced_encoder == Path(
            "/opt/ml/zimage_artifacts/compiled_models/encoder_256.pt"
        )

    def test_compile_cache_path(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.compile_cache == Path("/opt/ml/zimage_artifacts/compiled_models/cache_768")

    def test_encoder_server_path(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.encoder_server.name == "encoder_server.py"


class TestZImagePaths768ArtifactFilenames:
    def test_vae_filename(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.traced_vae.name == "vae_768.pt"

    def test_encoder_filename(self, ZImagePaths768):
        paths = ZImagePaths768()
        assert paths.traced_encoder.name == "encoder_256.pt"


class TestZImagePaths768ValidateRaises:
    def test_validate_missing_all(self, tmp_path: Path, ZImagePaths768):
        paths = ZImagePaths768(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate()
        msg = str(exc_info.value)
        assert "ZImagePipeline768 required artifacts missing:" in msg
        assert "model_weights" in msg
        assert "traced_vae" in msg

    def test_validate_missing_vae_only(self, tmp_path: Path, ZImagePaths768):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models" / "cache_768").mkdir(parents=True)

        paths = ZImagePaths768(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate()
        msg = str(exc_info.value)
        assert "traced_vae" in msg
        assert "model_weights" not in msg


class TestZImagePaths768ValidatePasses:
    def test_validate_passes(self, tmp_path: Path, ZImagePaths768):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "vae_768.pt").write_bytes(b"fake")
        (tmp_path / "compiled_models" / "cache_768").mkdir(parents=True)

        paths = ZImagePaths768(project_root=tmp_path)
        paths.validate()


class TestZImagePaths1024DefaultRoot:
    def test_default_project_root(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.project_root == Path("/opt/ml/zimage_artifacts")


class TestZImagePaths1024CustomRoot:
    def test_custom_root_propagates(self, tmp_path: Path, ZImagePaths1024):
        paths = ZImagePaths1024(project_root=tmp_path)
        assert paths.project_root == tmp_path
        assert paths.model_weights == tmp_path / "ckpts" / "Z-Image-Turbo"
        assert paths.traced_vae == tmp_path / "compiled_models" / "vae_1024.pt"
        assert paths.traced_encoder == tmp_path / "compiled_models" / "encoder_256.pt"
        assert paths.compile_cache == tmp_path / "compiled_models" / "cache_1024"
        assert paths.encoder_server.name == "encoder_server.py"


class TestZImagePaths1024PostInit:
    def test_model_weights_path(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.model_weights == Path("/opt/ml/zimage_artifacts/ckpts/Z-Image-Turbo")

    def test_traced_vae_path(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.traced_vae == Path("/opt/ml/zimage_artifacts/compiled_models/vae_1024.pt")

    def test_traced_encoder_path(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.traced_encoder == Path(
            "/opt/ml/zimage_artifacts/compiled_models/encoder_256.pt"
        )

    def test_compile_cache_path(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.compile_cache == Path("/opt/ml/zimage_artifacts/compiled_models/cache_1024")

    def test_encoder_server_path(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.encoder_server.name == "encoder_server.py"


class TestZImagePaths1024ArtifactFilenames:
    def test_vae_filename(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.traced_vae.name == "vae_1024.pt"

    def test_encoder_filename(self, ZImagePaths1024):
        paths = ZImagePaths1024()
        assert paths.traced_encoder.name == "encoder_256.pt"


class TestZImagePaths1024ValidateRaises:
    def test_validate_missing_all(self, tmp_path: Path, ZImagePaths1024):
        paths = ZImagePaths1024(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate()
        msg = str(exc_info.value)
        assert "ZImagePipeline1024 required artifacts missing:" in msg
        assert "model_weights" in msg
        assert "traced_vae" in msg

    def test_validate_missing_vae_only(self, tmp_path: Path, ZImagePaths1024):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)

        paths = ZImagePaths1024(project_root=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            paths.validate()
        msg = str(exc_info.value)
        assert "traced_vae" in msg
        assert "model_weights" not in msg


class TestZImagePaths1024ValidatePasses:
    def test_validate_passes_and_creates_compile_cache(self, tmp_path: Path, ZImagePaths1024):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "vae_1024.pt").write_bytes(b"fake")

        paths = ZImagePaths1024(project_root=tmp_path)
        paths.validate()
        assert paths.compile_cache.exists()
        assert paths.compile_cache.is_dir()

    def test_validate_creates_compile_cache_with_parents(self, tmp_path: Path, ZImagePaths1024):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "vae_1024.pt").write_bytes(b"fake")
        cache_path = tmp_path / "compiled_models" / "cache_1024"
        assert not cache_path.exists()

        paths = ZImagePaths1024(project_root=tmp_path)
        paths.validate()

        assert cache_path.exists()
        assert cache_path.is_dir()

    def test_validate_does_not_check_neuronxcc_subdir(self, tmp_path: Path, ZImagePaths1024):
        (tmp_path / "ckpts" / "Z-Image-Turbo").mkdir(parents=True)
        (tmp_path / "compiled_models").mkdir(parents=True)
        (tmp_path / "compiled_models" / "vae_1024.pt").write_bytes(b"fake")

        paths = ZImagePaths1024(project_root=tmp_path)
        paths.validate()
        assert list(paths.compile_cache.iterdir()) == []
