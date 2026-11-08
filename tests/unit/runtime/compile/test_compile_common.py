from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_tmpdir = tempfile.mkdtemp(prefix="zimg_test_")
os.environ["ZIMAGE_ROOT"] = _tmpdir

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "runtime" / "compile"))

import _common
from _common import ROOT, CompileConfig


class TestROOT:
    def test_root_is_path(self):
        assert isinstance(ROOT, Path)

    def test_root_exists_as_constant(self):
        assert hasattr(_common, "ROOT")


class TestCompileConfigConstruction:
    def test_default_construction_required_fields(self):
        cfg = CompileConfig(resolution=512)
        assert cfg.resolution == 512
        assert cfg.max_sequence_length == 512
        assert cfg.vae_mode == "cpu_fp32"
        assert cfg.attention_backend == "native"

    def test_construction_with_all_required_fields(self):
        cfg = CompileConfig(resolution=1024, max_sequence_length=256)
        assert cfg.resolution == 1024
        assert cfg.max_sequence_length == 256

    def test_custom_attention_backend(self):
        cfg = CompileConfig(resolution=512, attention_backend="nki")
        assert cfg.attention_backend == "nki"

    def test_custom_vae_mode(self):
        cfg = CompileConfig(resolution=768, vae_mode="neuron_lazy")
        assert cfg.vae_mode == "neuron_lazy"

    def test_extra_cc_flags_default_empty(self):
        cfg = CompileConfig(resolution=512)
        assert cfg.extra_cc_flags == ""


class TestCompileConfigHeightWidth:
    def test_height_returns_resolution_512(self):
        cfg = CompileConfig(resolution=512)
        assert cfg.height == 512

    def test_width_returns_resolution_512(self):
        cfg = CompileConfig(resolution=512)
        assert cfg.width == 512

    def test_height_returns_resolution_768(self):
        cfg = CompileConfig(resolution=768)
        assert cfg.height == 768

    def test_width_returns_resolution_1024(self):
        cfg = CompileConfig(resolution=1024)
        assert cfg.width == 1024


class TestCompileConfigCacheTag:
    def test_cache_tag_512_512(self):
        cfg = CompileConfig(resolution=512, max_sequence_length=512)
        assert cfg.cache_tag == "512_512"

    def test_cache_tag_768_256(self):
        cfg = CompileConfig(resolution=768, max_sequence_length=256)
        assert cfg.cache_tag == "768_256"

    def test_cache_tag_1024_128(self):
        cfg = CompileConfig(resolution=1024, max_sequence_length=128)
        assert cfg.cache_tag == "1024_128"

    def test_cache_tag_1024_512(self):
        cfg = CompileConfig(resolution=1024, max_sequence_length=512)
        assert cfg.cache_tag == "1024_512"


class TestCompileConfigCachePath:
    def test_cache_path_is_path_object(self):
        cfg = CompileConfig(resolution=512)
        assert isinstance(cfg.cache_path, Path)

    def test_cache_path_contains_cache_tag(self):
        cfg = CompileConfig(resolution=768, max_sequence_length=256)
        assert cfg.cache_path == ROOT / "compiled_models" / "cache_768_256"

    def test_cache_path_under_root(self):
        cfg = CompileConfig(resolution=1024, max_sequence_length=512)
        assert cfg.cache_path.parts[-2] == "compiled_models"
        assert cfg.cache_path.name == "cache_1024_512"


class TestCompileConfigImageTokens:
    def test_image_tokens_512(self):
        cfg = CompileConfig(resolution=512)
        assert cfg.image_tokens == 1024

    def test_image_tokens_768(self):
        cfg = CompileConfig(resolution=768)
        assert cfg.image_tokens == 2304

    def test_image_tokens_1024(self):
        cfg = CompileConfig(resolution=1024)
        assert cfg.image_tokens == 4096


class TestCompileConfigTotalTokens:
    def test_total_tokens_512_res_512_seq(self):
        cfg = CompileConfig(resolution=512, max_sequence_length=512)
        assert cfg.total_tokens == 1024 + 512

    def test_total_tokens_1024_res_128_seq(self):
        cfg = CompileConfig(resolution=1024, max_sequence_length=128)
        assert cfg.total_tokens == 4096 + 128

    def test_total_tokens_equals_image_plus_seq(self):
        cfg = CompileConfig(resolution=768, max_sequence_length=256)
        assert cfg.total_tokens == cfg.image_tokens + cfg.max_sequence_length
