from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestModelConfig:
    def test_adaln_embed_dim(self) -> None:
        from neuron.config.model import ADALN_EMBED_DIM

        assert ADALN_EMBED_DIM == 256

    def test_seq_multi_of(self) -> None:
        from neuron.config.model import SEQ_MULTI_OF

        assert SEQ_MULTI_OF == 32

    def test_rope_constants(self) -> None:
        from neuron.config.model import ROPE_AXES_DIMS, ROPE_AXES_LENS, ROPE_THETA

        assert ROPE_THETA == 256.0
        assert len(ROPE_AXES_DIMS) == 3
        assert len(ROPE_AXES_LENS) == 3
        assert sum(ROPE_AXES_DIMS) == 128

    def test_frequency_embedding_constants(self) -> None:
        from neuron.config.model import FREQUENCY_EMBEDDING_SIZE, MAX_PERIOD

        assert FREQUENCY_EMBEDDING_SIZE == 256
        assert MAX_PERIOD == 10000

    def test_image_seq_len_constants(self) -> None:
        from neuron.config.model import (
            BASE_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_IMAGE_SEQ_LEN,
            MAX_SHIFT,
        )

        assert BASE_IMAGE_SEQ_LEN == 256
        assert MAX_IMAGE_SEQ_LEN == 4096
        assert BASE_SHIFT < MAX_SHIFT

    def test_vae_defaults(self) -> None:
        from neuron.config.model import (
            DEFAULT_VAE_IN_CHANNELS,
            DEFAULT_VAE_LATENT_CHANNELS,
            DEFAULT_VAE_NORM_NUM_GROUPS,
            DEFAULT_VAE_OUT_CHANNELS,
            DEFAULT_VAE_SCALE_FACTOR,
            DEFAULT_VAE_SCALING_FACTOR,
        )

        assert DEFAULT_VAE_SCALE_FACTOR == 8
        assert DEFAULT_VAE_IN_CHANNELS == 3
        assert DEFAULT_VAE_OUT_CHANNELS == 3
        assert DEFAULT_VAE_LATENT_CHANNELS == 4
        assert DEFAULT_VAE_NORM_NUM_GROUPS == 32
        assert pytest.approx(0.18215) == DEFAULT_VAE_SCALING_FACTOR

    def test_transformer_defaults(self) -> None:
        from neuron.config.model import (
            DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
            DEFAULT_TRANSFORMER_DIM,
            DEFAULT_TRANSFORMER_IN_CHANNELS,
            DEFAULT_TRANSFORMER_N_HEADS,
            DEFAULT_TRANSFORMER_N_KV_HEADS,
            DEFAULT_TRANSFORMER_N_LAYERS,
            DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
            DEFAULT_TRANSFORMER_NORM_EPS,
            DEFAULT_TRANSFORMER_PATCH_SIZE,
            DEFAULT_TRANSFORMER_QK_NORM,
        )

        assert DEFAULT_TRANSFORMER_PATCH_SIZE == (2,)
        assert DEFAULT_TRANSFORMER_IN_CHANNELS == 16
        assert DEFAULT_TRANSFORMER_DIM == 3840
        assert DEFAULT_TRANSFORMER_N_LAYERS == 30
        assert DEFAULT_TRANSFORMER_N_REFINER_LAYERS == 2
        assert DEFAULT_TRANSFORMER_N_HEADS == DEFAULT_TRANSFORMER_N_KV_HEADS
        assert pytest.approx(1e-5) == DEFAULT_TRANSFORMER_NORM_EPS
        assert DEFAULT_TRANSFORMER_QK_NORM is True
        assert DEFAULT_TRANSFORMER_CAP_FEAT_DIM == 2560

    def test_scheduler_defaults(self) -> None:
        from neuron.config.model import (
            DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS,
            DEFAULT_SCHEDULER_SHIFT,
            DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING,
        )

        assert DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS == 1000
        assert DEFAULT_SCHEDULER_SHIFT == 3.0
        assert DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING is False

    def test_load_defaults(self) -> None:
        from neuron.config.model import DEFAULT_LOAD_DEVICE, DEFAULT_LOAD_DTYPE_STR

        assert DEFAULT_LOAD_DEVICE == "cuda"
        assert DEFAULT_LOAD_DTYPE_STR == "bfloat16"

    def test_bytes_per_gb(self) -> None:
        from neuron.config.model import BYTES_PER_GB

        assert BYTES_PER_GB == 2**30


class TestInferenceConfig:
    def test_default_dimensions(self) -> None:
        from neuron.config.inference import DEFAULT_HEIGHT, DEFAULT_WIDTH

        assert DEFAULT_HEIGHT == 1024
        assert DEFAULT_WIDTH == 1024

    def test_default_inference_steps(self) -> None:
        from neuron.config.inference import DEFAULT_INFERENCE_STEPS

        assert DEFAULT_INFERENCE_STEPS == 8

    def test_default_guidance_scale(self) -> None:
        from neuron.config.inference import DEFAULT_GUIDANCE_SCALE

        assert DEFAULT_GUIDANCE_SCALE == 0.0

    def test_default_cfg_truncation(self) -> None:
        from neuron.config.inference import DEFAULT_CFG_TRUNCATION

        assert DEFAULT_CFG_TRUNCATION == 1.0

    def test_default_max_sequence_length(self) -> None:
        from neuron.config.inference import DEFAULT_MAX_SEQUENCE_LENGTH

        assert DEFAULT_MAX_SEQUENCE_LENGTH == 512


@pytest.fixture
def helpers_module():
    import sys

    config_mock = MagicMock()
    config_mock.BYTES_PER_GB = 2**30
    sys.modules["config"] = config_mock

    from neuron.utils import helpers

    helpers.BYTES_PER_GB = 2**30
    return helpers


class TestFormatBytes:
    def test_zero_bytes(self, helpers_module) -> None:
        assert helpers_module.format_bytes(0) == "0.00 GB"

    def test_one_gb(self, helpers_module) -> None:
        assert helpers_module.format_bytes(2**30) == "1.00 GB"

    def test_fractional_gb(self, helpers_module) -> None:
        half_gb = 2**30 / 2
        assert helpers_module.format_bytes(half_gb) == "0.50 GB"

    def test_large_value(self, helpers_module) -> None:
        ten_gb = 10 * 2**30
        assert helpers_module.format_bytes(ten_gb) == "10.00 GB"


class TestPrintMemoryStats:
    def test_logs_warning_when_cuda_unavailable(self, helpers_module) -> None:
        import torch

        torch.cuda.is_available.return_value = False

        helpers_module.print_memory_stats("test_stage")

        torch.cuda.is_available.assert_called()

    def test_calls_cuda_stats_when_available(self, helpers_module) -> None:
        import torch

        torch.cuda.is_available.return_value = True
        torch.cuda.max_memory_allocated.return_value = 2**30
        torch.cuda.max_memory_reserved.return_value = 2**31
        torch.cuda.memory_allocated.return_value = 2**29
        torch.cuda.memory_reserved.return_value = 2**30

        helpers_module.print_memory_stats("inference")

        torch.cuda.synchronize.assert_called()
        torch.cuda.max_memory_allocated.assert_called()
        torch.cuda.max_memory_reserved.assert_called()


class TestComputeFileMd5:
    def test_known_content(self, tmp_path, helpers_module) -> None:
        import hashlib

        test_file = tmp_path / "test.bin"
        content = b"hello world"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = helpers_module.compute_file_md5(test_file)
        assert result == expected

    def test_empty_file(self, tmp_path, helpers_module) -> None:
        import hashlib

        test_file = tmp_path / "empty.bin"
        test_file.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        result = helpers_module.compute_file_md5(test_file)
        assert result == expected

    def test_large_file_chunked(self, tmp_path, helpers_module) -> None:
        import hashlib

        test_file = tmp_path / "large.bin"
        content = b"x" * 20000
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = helpers_module.compute_file_md5(test_file, chunk_size=4096)
        assert result == expected

    def test_custom_chunk_size(self, tmp_path, helpers_module) -> None:
        import hashlib

        test_file = tmp_path / "chunk.bin"
        content = b"abcdef" * 1000
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = helpers_module.compute_file_md5(test_file, chunk_size=128)
        assert result == expected


class TestLoadManifest:
    def test_nonexistent_file_returns_empty(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "missing.txt"
        result = helpers_module.load_manifest(manifest_file)
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "empty.txt"
        manifest_file.write_text("")
        result = helpers_module.load_manifest(manifest_file)
        assert result == {}

    def test_comments_and_blank_lines_skipped(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "manifest.txt"
        manifest_file.write_text("# comment\n\n# another comment\n")
        result = helpers_module.load_manifest(manifest_file)
        assert result == {}

    def test_file_path_only(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "manifest.txt"
        manifest_file.write_text("model/weights.bin\nvae/decoder.pt\n")
        result = helpers_module.load_manifest(manifest_file)
        assert result == {"model/weights.bin": None, "vae/decoder.pt": None}

    def test_md5_first_then_path(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "manifest.txt"
        md5 = "a" * 32
        manifest_file.write_text(f"{md5} model/weights.bin\n")
        result = helpers_module.load_manifest(manifest_file)
        assert result == {"model/weights.bin": md5}

    def test_path_first_then_md5(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "manifest.txt"
        md5 = "abc123"
        manifest_file.write_text(f"model/weights.bin {md5}\n")
        result = helpers_module.load_manifest(manifest_file)
        assert result == {"model/weights.bin": md5}

    def test_invalid_line_skipped(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "manifest.txt"
        manifest_file.write_text("too many parts here extra\nvalid_file.bin\n")
        result = helpers_module.load_manifest(manifest_file)
        assert result == {"valid_file.bin": None}

    def test_mixed_format(self, tmp_path, helpers_module) -> None:
        manifest_file = tmp_path / "manifest.txt"
        md5 = "a" * 32
        content = (
            f"# Manifest\n{md5} model/weights.bin\nvae/decoder.pt\n\nconfig.json non_hex_hash\n"
        )
        manifest_file.write_text(content)
        result = helpers_module.load_manifest(manifest_file)
        assert result == {
            "model/weights.bin": md5,
            "vae/decoder.pt": None,
            "config.json": "non_hex_hash",
        }


class TestVerifyFileIntegrity:
    def test_all_files_present_no_checksums(self, tmp_path, helpers_module) -> None:
        (tmp_path / "a.bin").write_bytes(b"data")
        (tmp_path / "b.bin").write_bytes(b"data")
        manifest = {"a.bin": None, "b.bin": None}

        all_valid, missing, corrupted = helpers_module.verify_file_integrity(tmp_path, manifest)
        assert all_valid is True
        assert missing == []
        assert corrupted == []

    def test_missing_file_reported(self, tmp_path, helpers_module) -> None:
        (tmp_path / "a.bin").write_bytes(b"data")
        manifest = {"a.bin": None, "missing.bin": None}

        all_valid, missing, corrupted = helpers_module.verify_file_integrity(tmp_path, manifest)
        assert all_valid is False
        assert "missing.bin" in missing
        assert corrupted == []

    def test_checksum_match(self, tmp_path, helpers_module) -> None:
        import hashlib

        content = b"hello"
        (tmp_path / "file.bin").write_bytes(content)
        md5 = hashlib.sha256(content).hexdigest()
        manifest = {"file.bin": md5}

        all_valid, missing, corrupted = helpers_module.verify_file_integrity(
            tmp_path, manifest, verify_checksums=True
        )
        assert all_valid is True
        assert missing == []
        assert corrupted == []

    def test_checksum_mismatch(self, tmp_path, helpers_module) -> None:
        (tmp_path / "file.bin").write_bytes(b"actual content")
        manifest = {"file.bin": "0000000000000000000000000000dead"}

        all_valid, missing, corrupted = helpers_module.verify_file_integrity(
            tmp_path, manifest, verify_checksums=True
        )
        assert all_valid is False
        assert missing == []
        assert "file.bin" in corrupted

    def test_skip_checksum_when_disabled(self, tmp_path, helpers_module) -> None:
        (tmp_path / "file.bin").write_bytes(b"actual content")
        manifest = {"file.bin": "0000000000000000000000000000dead"}

        all_valid, missing, corrupted = helpers_module.verify_file_integrity(
            tmp_path, manifest, verify_checksums=False
        )
        assert all_valid is True
        assert corrupted == []

    def test_skip_checksum_when_none(self, tmp_path, helpers_module) -> None:
        (tmp_path / "file.bin").write_bytes(b"content")
        manifest = {"file.bin": None}

        all_valid, missing, corrupted = helpers_module.verify_file_integrity(
            tmp_path, manifest, verify_checksums=True
        )
        assert all_valid is True
        assert corrupted == []

    def test_empty_manifest(self, tmp_path, helpers_module) -> None:
        all_valid, missing, corrupted = helpers_module.verify_file_integrity(
            tmp_path, {}, verify_checksums=True
        )
        assert all_valid is True
        assert missing == []
        assert corrupted == []

    def test_unreadable_file_treated_as_corrupted(self, tmp_path, helpers_module) -> None:
        (tmp_path / "file.bin").write_bytes(b"data")
        manifest = {"file.bin": "a" * 32}

        with patch.object(
            helpers_module,
            "compute_file_md5",
            side_effect=PermissionError("cannot read"),
        ):
            all_valid, missing, corrupted = helpers_module.verify_file_integrity(
                tmp_path, manifest, verify_checksums=True
            )
        assert all_valid is False
        assert "file.bin" in corrupted


class TestEnsureModelWeights:
    @pytest.fixture(autouse=True)
    def _mock_huggingface(self):
        import sys

        mock_hf = MagicMock()
        sys.modules["huggingface_hub"] = mock_hf
        yield mock_hf
        sys.modules.pop("huggingface_hub", None)

    def test_returns_path_when_all_files_present(self, tmp_path, helpers_module) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"data")

        with (
            patch.object(
                helpers_module,
                "load_manifest",
                return_value={"weights.bin": None},
            ),
            patch.object(
                helpers_module,
                "verify_file_integrity",
                return_value=(True, [], []),
            ),
        ):
            result = helpers_module.ensure_model_weights(
                str(model_dir), manifest_name="test_manifest.txt"
            )
        assert result == model_dir

    def test_downloads_when_files_missing(
        self, tmp_path, helpers_module, _mock_huggingface
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_snapshot = MagicMock()
        _mock_huggingface.snapshot_download = mock_snapshot

        with (
            patch.object(
                helpers_module,
                "load_manifest",
                return_value={"weights.bin": None},
            ),
            patch.object(
                helpers_module,
                "verify_file_integrity",
                side_effect=[
                    (False, ["weights.bin"], []),
                    (True, [], []),
                ],
            ),
        ):
            result = helpers_module.ensure_model_weights(
                str(model_dir),
                repo_id="org/model",
                manifest_name="manifest.txt",
            )

        mock_snapshot.assert_called_once()
        assert result == model_dir

    def test_raises_when_download_fails(self, tmp_path, helpers_module, _mock_huggingface) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        _mock_huggingface.snapshot_download = MagicMock(side_effect=RuntimeError("network error"))

        with (
            patch.object(
                helpers_module,
                "load_manifest",
                return_value={"weights.bin": None},
            ),
            patch.object(
                helpers_module,
                "verify_file_integrity",
                return_value=(False, ["weights.bin"], []),
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to download model weights"):
                helpers_module.ensure_model_weights(
                    str(model_dir),
                    repo_id="org/model",
                    manifest_name="manifest.txt",
                )

    def test_raises_when_still_invalid_after_download(
        self, tmp_path, helpers_module, _mock_huggingface
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        _mock_huggingface.snapshot_download = MagicMock()

        with (
            patch.object(
                helpers_module,
                "load_manifest",
                return_value={"weights.bin": None},
            ),
            patch.object(
                helpers_module,
                "verify_file_integrity",
                side_effect=[
                    (False, ["weights.bin"], []),
                    (False, ["weights.bin"], []),
                ],
            ),
        ):
            with pytest.raises(FileNotFoundError, match="Still missing"):
                helpers_module.ensure_model_weights(
                    str(model_dir),
                    repo_id="org/model",
                    manifest_name="manifest.txt",
                )

    def test_returns_existing_dir_when_no_manifest(self, tmp_path, helpers_module) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch.object(helpers_module, "load_manifest", return_value={}):
            result = helpers_module.ensure_model_weights(
                str(model_dir), manifest_name="missing.txt"
            )
        assert result == model_dir

    def test_downloads_when_no_manifest_and_dir_missing(
        self, tmp_path, helpers_module, _mock_huggingface
    ) -> None:
        model_dir = tmp_path / "nonexistent_model"
        _mock_huggingface.snapshot_download = MagicMock(side_effect=RuntimeError("network error"))

        with patch.object(helpers_module, "load_manifest", return_value={}):
            with pytest.raises(RuntimeError):
                helpers_module.ensure_model_weights(
                    str(model_dir),
                    repo_id="org/model",
                    manifest_name="missing.txt",
                )
