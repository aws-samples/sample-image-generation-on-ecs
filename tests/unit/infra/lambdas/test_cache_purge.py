from __future__ import annotations

import json
import sys
from pathlib import Path

CACHE_PURGE_DIR = str(Path(__file__).resolve().parents[4] / "infra" / "lambdas" / "cache_purge")


def _load_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("EFS_MOUNT_PATH", str(tmp_path))
    if "index" in sys.modules:
        del sys.modules["index"]
    original_path = sys.path.copy()
    sys.path.insert(0, CACHE_PURGE_DIR)
    try:
        import index

        index.EFS_MOUNT_PATH = str(tmp_path)
        return index
    finally:
        sys.path[:] = original_path


class TestCachePurgeHandler:
    def test_purges_failed_entries(self, tmp_path, monkeypatch):
        module = _load_handler(tmp_path, monkeypatch)

        failed_dir = tmp_path / "cache_1024" / "neuron-abc" / "MODULE_0"
        failed_dir.mkdir(parents=True)
        (failed_dir / "compile_cache_log.json").write_text(json.dumps({"compile_status": "failed"}))
        (failed_dir / "model.neff").write_bytes(b"corrupt")

        good_dir = tmp_path / "cache_1024" / "neuron-def" / "MODULE_0"
        good_dir.mkdir(parents=True)
        (good_dir / "compile_cache_log.json").write_text(json.dumps({"compile_status": "success"}))
        (good_dir / "model.neff").write_bytes(b"good")

        result = module.handler({"cache_prefixes": ["cache_1024/"]}, None)
        assert result["purged_count"] == 2
        assert not failed_dir.exists()
        assert good_dir.exists()

    def test_no_failed_entries(self, tmp_path, monkeypatch):
        module = _load_handler(tmp_path, monkeypatch)

        good_dir = tmp_path / "cache_768" / "neuron-ok" / "MODULE_0"
        good_dir.mkdir(parents=True)
        (good_dir / "compile_cache_log.json").write_text(json.dumps({"compile_status": "success"}))
        (good_dir / "model.neff").write_bytes(b"good")

        result = module.handler({"cache_prefixes": ["cache_768/"]}, None)
        assert result["purged_count"] == 0

    def test_uses_default_prefixes(self, tmp_path, monkeypatch):
        module = _load_handler(tmp_path, monkeypatch)

        result = module.handler({}, None)
        assert result["purged_count"] == 0
        assert result["kept_count"] == 0
