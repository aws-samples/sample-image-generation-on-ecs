from __future__ import annotations

from pathlib import Path

import pytest
from runtime.lib import neuron_cache
from runtime.lib.neuron_cache import resolve_compile_cache


def test_writable_efs_cache_returned_in_place(tmp_path: Path) -> None:
    efs_cache = tmp_path / "compiled_models" / "cache_768" / "v1"
    efs_cache.mkdir(parents=True)
    (efs_cache / "model.neff").write_bytes(b"neff")

    result = resolve_compile_cache(efs_cache, "cache_768/v1")

    assert result == efs_cache


def test_readonly_cache_mirrored_to_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    efs_cache = tmp_path / "efs" / "cache_768" / "v1"
    efs_cache.mkdir(parents=True)
    (efs_cache / "model.neff").write_bytes(b"neff-bytes")
    (efs_cache / "sub").mkdir()
    (efs_cache / "sub" / "part.neff").write_bytes(b"part")

    local_base = tmp_path / "scratch"
    monkeypatch.setattr(neuron_cache, "LOCAL_CACHE_BASE", local_base)
    monkeypatch.setattr(neuron_cache, "_is_writable", lambda p: p != efs_cache)

    result = resolve_compile_cache(efs_cache, "cache_768/v1")

    assert result == local_base / "cache_768/v1"
    assert (result / "model.neff").read_bytes() == b"neff-bytes"
    assert (result / "sub" / "part.neff").read_bytes() == b"part"


def test_missing_source_returns_empty_writable_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    efs_cache = tmp_path / "efs" / "cache_1024" / "v1"
    local_base = tmp_path / "scratch"
    monkeypatch.setattr(neuron_cache, "LOCAL_CACHE_BASE", local_base)
    monkeypatch.setattr(neuron_cache, "_is_writable", lambda p: p != efs_cache)

    result = resolve_compile_cache(efs_cache, "cache_1024/v1")

    assert result == local_base / "cache_1024/v1"
    assert result.is_dir()
    assert not any(result.iterdir())
