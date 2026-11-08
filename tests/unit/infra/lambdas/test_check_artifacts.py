from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_ZIP_MAGIC = b"PK\x03\x04"
_EOCD_MAGIC = b"PK\x05\x06"


@pytest.fixture
def check_artifacts(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    lambda_dir = repo_root / "infra" / "lambdas" / "check_artifacts"
    monkeypatch.syspath_prepend(str(lambda_dir))
    if "index" in sys.modules:
        del sys.modules["index"]
    mod = importlib.import_module("index")
    monkeypatch.setattr(mod, "EFS_MOUNT", tmp_path)
    return mod


def _create_valid_neff(path: Path, size: int = 1_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_ZIP_MAGIC)
        middle_size = size - 4 - 64
        if middle_size > 0:
            f.write(b"\x00" * middle_size)
        f.write(b"\x00" * 30 + _EOCD_MAGIC + b"\x00" * 30)


def _create_neff_files(base_dir: Path, names: list[str], prefix: str = "compile") -> None:
    for name in names:
        _create_valid_neff(base_dir / prefix / name)


def _create_cache_neffs(base_dir: Path, prefix: str, count: int) -> None:
    cache_dir = base_dir / "compiled_models" / prefix
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (cache_dir / f"neuronxcc/MODULE_{i}").mkdir(parents=True, exist_ok=True)
        (cache_dir / f"neuronxcc/MODULE_{i}/model.neff").write_bytes(b"\x00" * 100)


ALL_NEFFS = [
    "transformer_512.pt",
    "encoder_256.pt",
    "vae_512.pt",
    "vae_768.pt",
    "vae_1024.pt",
]


def test_check_only_returns_all_cached_when_neffs_and_pjrt_complete(check_artifacts, tmp_path):
    _create_neff_files(tmp_path, ALL_NEFFS)
    _create_cache_neffs(tmp_path, "cache_768", 7)
    _create_cache_neffs(tmp_path, "cache_1024", 7)

    out = check_artifacts.handler(
        {
            "artifacts_bucket": "test-bucket",
            "check_only": True,
            "require_pjrt_cache": True,
        },
        context=None,
    )

    assert out["all_cached"] is True
    assert out["pjrt_cache_ok"] is True
    assert out["missing"] == 0


def test_check_only_returns_not_cached_when_pjrt_empty(check_artifacts, tmp_path):
    _create_neff_files(tmp_path, ALL_NEFFS)

    out = check_artifacts.handler(
        {
            "artifacts_bucket": "test-bucket",
            "check_only": True,
            "require_pjrt_cache": True,
        },
        context=None,
    )

    assert out["all_cached"] is False
    assert out["pjrt_cache_ok"] is False
    assert out["missing"] == 0


def test_check_only_returns_not_cached_when_pjrt_partial(check_artifacts, tmp_path):
    _create_neff_files(tmp_path, ALL_NEFFS)
    _create_cache_neffs(tmp_path, "cache_768", 3)
    _create_cache_neffs(tmp_path, "cache_1024", 3)

    out = check_artifacts.handler(
        {
            "artifacts_bucket": "test-bucket",
            "check_only": True,
            "require_pjrt_cache": True,
        },
        context=None,
    )

    assert out["all_cached"] is False
    assert out["pjrt_cache_ok"] is False


def test_check_only_skips_pjrt_when_not_required(check_artifacts, tmp_path):
    _create_neff_files(tmp_path, ALL_NEFFS)

    out = check_artifacts.handler(
        {"artifacts_bucket": "test-bucket", "check_only": True},
        context=None,
    )

    assert out["all_cached"] is True


def test_check_only_missing_neff_is_not_cached(check_artifacts, tmp_path):
    _create_neff_files(tmp_path, ["vae_512.pt", "vae_768.pt", "vae_1024.pt"])
    _create_cache_neffs(tmp_path, "cache_768", 7)
    _create_cache_neffs(tmp_path, "cache_1024", 7)

    out = check_artifacts.handler(
        {
            "artifacts_bucket": "test-bucket",
            "check_only": True,
            "require_pjrt_cache": True,
        },
        context=None,
    )

    assert out["all_cached"] is False
    assert out["missing"] == 2


def test_verify_mode_skips_zip_magic_for_non_pt(check_artifacts, tmp_path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "model_index.json").write_text('{"_class_name": "FluxPipeline"}')

    out = check_artifacts.handler(
        {
            "artifacts_bucket": "test-bucket",
            "check_only": False,
            "key_prefix": "weights",
            "expected_files": ["model_index.json"],
            "min_bytes": 10,
        },
        context=None,
    )

    assert out["failures"] == []
    assert len(out["verified"]) == 1
    assert out["verified"][0]["magic_ok"] is True


def test_verify_mode_raises_on_missing_files(check_artifacts, tmp_path):
    with pytest.raises(RuntimeError, match="NEFF verification failed"):
        check_artifacts.handler(
            {"artifacts_bucket": "test-bucket"},
            context=None,
        )


def test_file_too_small_fails(check_artifacts, tmp_path):
    compile_dir = tmp_path / "compile"
    compile_dir.mkdir(parents=True)
    (compile_dir / "transformer_512.pt").write_bytes(b"\x00" * 100)

    with pytest.raises(RuntimeError, match="NEFF verification failed") as exc_info:
        check_artifacts.handler(
            {
                "artifacts_bucket": "test-bucket",
                "expected_files": ["transformer_512.pt"],
                "min_bytes": 1024,
            },
            context=None,
        )

    assert "size" in str(exc_info.value)
    assert "< min_bytes" in str(exc_info.value)


def test_wrong_magic_bytes_fails(check_artifacts, tmp_path):
    compile_dir = tmp_path / "compile"
    compile_dir.mkdir(parents=True)
    with open(compile_dir / "transformer_512.pt", "wb") as f:
        f.write(b"\x00\x00\x00\x00")
        f.write(b"\x00" * 5_000_000)

    with pytest.raises(RuntimeError, match="NEFF verification failed") as exc_info:
        check_artifacts.handler(
            {
                "artifacts_bucket": "test-bucket",
                "expected_files": ["transformer_512.pt"],
            },
            context=None,
        )

    assert "first 4 bytes" in str(exc_info.value)


def test_cache_version_prefix(check_artifacts, tmp_path):
    _create_neff_files(tmp_path, ALL_NEFFS)
    _create_cache_neffs(tmp_path, "cache_768/v2.25", 7)
    _create_cache_neffs(tmp_path, "cache_1024/v2.25", 7)

    out = check_artifacts.handler(
        {
            "artifacts_bucket": "test-bucket",
            "check_only": True,
            "require_pjrt_cache": True,
            "cache_version": "v2.25",
        },
        context=None,
    )

    assert out["all_cached"] is True
    assert out["pjrt_cache_ok"] is True
