from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

LAMBDAS_DIR = Path(__file__).resolve().parents[2] / "infra" / "lambdas"
VERIFY_NEFFS_DIR = str(LAMBDAS_DIR / "check_artifacts")
CACHE_PURGE_DIR = str(LAMBDAS_DIR / "cache_purge")


ZIP_MAGIC = b"PK\x03\x04"
EOCD_MAGIC = b"PK\x05\x06"

DEFAULT_FILES = [
    "transformer_512.pt",
    "encoder_256.pt",
    "vae_512.pt",
    "vae_768.pt",
    "vae_1024.pt",
]


def _make_client_error(code: str = "404", message: str = "Not Found") -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "HeadObject",
    )


def _make_body(data: bytes) -> MagicMock:
    body = MagicMock()
    body.read.return_value = data
    return body


def _valid_tail_bytes() -> bytes:
    padding = b"\x00" * 30
    return padding + EOCD_MAGIC + (b"\x00" * 30)


def _make_valid_pt_file(path: Path, size: int = 5_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(ZIP_MAGIC)
        f.write(b"\x00" * (size - 4 - 64))
        f.write(_valid_tail_bytes())


def _load_check_artifacts():
    if "index" in sys.modules:
        del sys.modules["index"]
    original_path = sys.path.copy()
    sys.path.insert(0, VERIFY_NEFFS_DIR)
    try:
        import index

        return index
    finally:
        sys.path[:] = original_path


def _load_cache_purge():
    if "index" in sys.modules:
        del sys.modules["index"]
    original_path = sys.path.copy()
    sys.path.insert(0, CACHE_PURGE_DIR)
    try:
        import index

        return index
    finally:
        sys.path[:] = original_path


class TestVerifyNeffsHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.module = _load_check_artifacts()
        self.efs_root = tmp_path
        self.module.EFS_MOUNT = tmp_path
        yield
        if "index" in sys.modules:
            del sys.modules["index"]

    def test_all_files_present_and_valid(self):
        compile_dir = self.efs_root / "compile"
        for fname in DEFAULT_FILES:
            _make_valid_pt_file(compile_dir / fname)

        event = {"artifacts_bucket": "test-bucket"}
        result = self.module.handler(event, None)

        assert result["failures"] == []
        assert len(result["verified"]) == 5
        for entry in result["verified"]:
            assert entry["magic_ok"] is True
            assert entry["eocd_ok"] is True
            assert entry["size_bytes"] == 5_000_000
            assert "key" in entry

    def test_all_files_missing_raises(self):
        event = {"artifacts_bucket": "test-bucket"}
        with pytest.raises(RuntimeError, match="NEFF verification failed"):
            self.module.handler(event, None)

    def test_check_only_all_present(self):
        compile_dir = self.efs_root / "compile"
        for fname in DEFAULT_FILES:
            _make_valid_pt_file(compile_dir / fname)

        event = {"artifacts_bucket": "test-bucket", "check_only": True}
        result = self.module.handler(event, None)

        assert result["all_cached"] is True
        assert result["verified"] == 5
        assert result["missing"] == 0

    def test_check_only_some_missing(self):
        compile_dir = self.efs_root / "compile"
        for fname in DEFAULT_FILES[2:]:
            _make_valid_pt_file(compile_dir / fname)

        event = {"artifacts_bucket": "test-bucket", "check_only": True}
        result = self.module.handler(event, None)

        assert result["all_cached"] is False
        assert result["verified"] == 3
        assert result["missing"] == 2

    def test_file_present_but_too_small(self):
        compile_dir = self.efs_root / "compile"
        compile_dir.mkdir(parents=True, exist_ok=True)
        (compile_dir / "transformer_512.pt").write_bytes(b"\x00" * 100)

        event = {
            "artifacts_bucket": "test-bucket",
            "expected_files": ["transformer_512.pt"],
            "min_bytes": 1024,
        }
        with pytest.raises(RuntimeError, match="NEFF verification failed") as exc_info:
            self.module.handler(event, None)

        error_msg = str(exc_info.value)
        assert "size" in error_msg
        assert "< min_bytes" in error_msg

    def test_file_present_wrong_magic_bytes(self):
        compile_dir = self.efs_root / "compile"
        compile_dir.mkdir(parents=True, exist_ok=True)
        with open(compile_dir / "transformer_512.pt", "wb") as f:
            f.write(b"\x00\x00\x00\x00")
            f.write(b"\x00" * 5_000_000)

        event = {
            "artifacts_bucket": "test-bucket",
            "expected_files": ["transformer_512.pt"],
        }
        with pytest.raises(RuntimeError, match="NEFF verification failed") as exc_info:
            self.module.handler(event, None)

        error_msg = str(exc_info.value)
        assert "first 4 bytes" in error_msg


class TestCachePurgeHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EFS_MOUNT_PATH", str(tmp_path))
        self.efs_root = tmp_path
        self.module = _load_cache_purge()
        self.module.EFS_MOUNT_PATH = str(tmp_path)
        yield
        if "index" in sys.modules:
            del sys.modules["index"]

    def test_no_failed_entries(self):
        cache_dir = self.efs_root / "cache_1024" / "neuronxcc-abc" / "MODULE_0"
        cache_dir.mkdir(parents=True)
        (cache_dir / "compile_cache_log.json").write_text(json.dumps({"compile_status": "success"}))
        (cache_dir / "model.neff").write_bytes(b"neff-data")

        event = {"cache_prefixes": ["cache_1024/"]}
        result = self.module.handler(event, None)

        assert result["purged_count"] == 0
        assert result["kept_count"] == 1

    def test_one_failed_entry_purged(self):
        failed_dir = self.efs_root / "cache_1024" / "neuronxcc-abc" / "MODULE_0"
        failed_dir.mkdir(parents=True)
        (failed_dir / "compile_cache_log.json").write_text(json.dumps({"compile_status": "failed"}))
        (failed_dir / "model.neff").write_bytes(b"corrupt")

        good_dir = self.efs_root / "cache_1024" / "neuronxcc-def" / "MODULE_0"
        good_dir.mkdir(parents=True)
        (good_dir / "compile_cache_log.json").write_text(json.dumps({"compile_status": "success"}))
        (good_dir / "model.neff").write_bytes(b"good-neff")

        event = {"cache_prefixes": ["cache_1024/"]}
        result = self.module.handler(event, None)

        assert result["purged_count"] == 2
        assert not failed_dir.exists()
        assert good_dir.exists()

    def test_empty_prefix(self):
        event = {"cache_prefixes": ["cache_nonexistent/"]}
        result = self.module.handler(event, None)

        assert result["purged_count"] == 0
        assert result["kept_count"] == 0
