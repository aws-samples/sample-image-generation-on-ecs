from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def warmup_module():
    from runtime.serving import warmup_cache

    return warmup_cache


class TestVerifyEfsArtifacts:
    def test_logs_present_weights(self, tmp_path, warmup_module):
        artifacts = tmp_path / "artifacts"
        weights = artifacts / "ckpts" / "Z-Image-Turbo"
        weights.mkdir(parents=True)
        (weights / "model_index.json").write_text("{}")
        (weights / "model.safetensors").write_bytes(b"\x00" * 100)

        warmup_module._verify_efs_artifacts(artifacts, 768)

    def test_handles_missing_weights(self, tmp_path, warmup_module):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        warmup_module._verify_efs_artifacts(artifacts, 768)

    def test_checks_768_neffs(self, tmp_path, warmup_module):
        artifacts = tmp_path / "artifacts"
        compiled = artifacts / "compiled_models"
        compiled.mkdir(parents=True)
        (compiled / "encoder_256.pt").write_bytes(b"\x00" * 100)
        (compiled / "vae_768.pt").write_bytes(b"\x00" * 100)

        warmup_module._verify_efs_artifacts(artifacts, 768)

    def test_checks_1024_neffs(self, tmp_path, warmup_module):
        artifacts = tmp_path / "artifacts"
        compiled = artifacts / "compiled_models"
        compiled.mkdir(parents=True)
        (compiled / "encoder_256.pt").write_bytes(b"\x00" * 100)
        (compiled / "vae_1024.pt").write_bytes(b"\x00" * 100)

        warmup_module._verify_efs_artifacts(artifacts, 1024)

    def test_checks_512_neffs(self, tmp_path, warmup_module):
        artifacts = tmp_path / "artifacts"
        compiled = artifacts / "compiled_models"
        compiled.mkdir(parents=True)
        (compiled / "encoder_256.pt").write_bytes(b"\x00" * 100)
        (compiled / "transformer_512.pt").write_bytes(b"\x00" * 100)
        (compiled / "vae_512.pt").write_bytes(b"\x00" * 100)

        warmup_module._verify_efs_artifacts(artifacts, 512)


class TestBuildEndpointEnv:
    def test_768_env_matches_endpoint(self, warmup_module):
        env = warmup_module._build_endpoint_env(768)

        assert env["PJRT_DEVICE"] == "NEURON"
        assert env["NEURON_RT_VISIBLE_CORES"] == "0,1"
        assert env["NEURON_FUSE_SOFTMAX"] == "1"
        assert env["NEURON_RT_EXEC_TIMEOUT"] == "120"
        assert env["USE_TRACED_VAE"] == "1"
        assert env["VAE_ON_CPU"] == "0"
        assert env["ZIMAGE_ENCODER_MODE"] == "subprocess"
        assert env["ZIMAGE_DEFAULT_RESOLUTION"] == "768"

    def test_1024_env_has_vae_on_cpu(self, warmup_module):
        env = warmup_module._build_endpoint_env(1024)

        assert env["VAE_ON_CPU"] == "1"
        assert env["USE_TRACED_VAE"] == "1"
        assert env["ZIMAGE_DEFAULT_RESOLUTION"] == "1024"


class TestRunNeuronParallelCompile:
    @patch("runtime.serving.warmup_cache.subprocess.run")
    def test_collect_success(self, mock_run, warmup_module):
        mock_run.return_value = MagicMock(returncode=0)
        env = warmup_module._build_endpoint_env(768)

        result = warmup_module._run_neuron_parallel_compile("collect", 768, env)

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "neuron_parallel_compile"
        assert "--command" in cmd
        assert "collect" in cmd
        assert "--resolution" in cmd
        assert "768" in cmd

    @patch("runtime.serving.warmup_cache.subprocess.run")
    def test_compile_success(self, mock_run, warmup_module):
        mock_run.return_value = MagicMock(returncode=0)
        env = warmup_module._build_endpoint_env(1024)

        result = warmup_module._run_neuron_parallel_compile("compile", 1024, env)

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert "compile" in cmd
        assert "1024" in cmd

    @patch("runtime.serving.warmup_cache.subprocess.run")
    def test_nonzero_exit_returns_false(self, mock_run, warmup_module):
        mock_run.return_value = MagicMock(returncode=1)
        env = warmup_module._build_endpoint_env(768)

        result = warmup_module._run_neuron_parallel_compile("collect", 768, env)

        assert result is False

    @patch("runtime.serving.warmup_cache._run_fallback_compile", return_value=True)
    @patch("runtime.serving.warmup_cache.subprocess.run", side_effect=FileNotFoundError)
    def test_falls_back_when_binary_missing(self, mock_run, mock_fallback, warmup_module):
        env = warmup_module._build_endpoint_env(768)

        result = warmup_module._run_neuron_parallel_compile("collect", 768, env)

        assert result is True
        mock_fallback.assert_called_once_with("collect", 768, env, 7200)

    @patch("runtime.serving.warmup_cache.subprocess.run")
    def test_timeout_returns_false(self, mock_run, warmup_module):
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=100)
        env = warmup_module._build_endpoint_env(768)

        result = warmup_module._run_neuron_parallel_compile("collect", 768, env, timeout=100)

        assert result is False


class TestRunFallbackCompile:
    @patch("runtime.serving.warmup_cache.subprocess.run")
    def test_collect_sets_extract_graphs_env(self, mock_run, warmup_module):
        mock_run.return_value = MagicMock(returncode=0)
        env = warmup_module._build_endpoint_env(768)

        result = warmup_module._run_fallback_compile("collect", 768, env)

        assert result is True
        call_env = mock_run.call_args[1]["env"]
        assert call_env["NEURON_EXTRACT_GRAPHS_ONLY"] == "1"

    @patch("runtime.serving.warmup_cache.subprocess.run")
    def test_compile_sets_parallel_compile_env(self, mock_run, warmup_module):
        mock_run.return_value = MagicMock(returncode=0)
        env = warmup_module._build_endpoint_env(768)

        result = warmup_module._run_fallback_compile("compile", 768, env)

        assert result is True
        call_env = mock_run.call_args[1]["env"]
        assert call_env["NEURON_PARALLEL_COMPILE"] == "1"
