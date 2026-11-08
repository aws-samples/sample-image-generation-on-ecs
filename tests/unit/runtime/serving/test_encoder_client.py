from __future__ import annotations

import json
import struct
import subprocess
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def encoder_module():
    from serving import encoder_client

    return encoder_client


@pytest.fixture
def EncoderClientCls(encoder_module):
    return encoder_module.EncoderClient


class TestResolvePython:
    def test_returns_first_existing_candidate(self, EncoderClientCls):
        with (
            patch.object(Path, "exists", side_effect=[True]),
            patch("serving.encoder_client.shutil.which") as mock_which,
        ):
            result = EncoderClientCls._resolve_python()

        assert result == "/opt/conda/bin/python3"
        mock_which.assert_not_called()

    def test_falls_to_which_if_conda_missing(self, EncoderClientCls):
        with (
            patch.object(Path, "exists", return_value=False),
            patch("serving.encoder_client.shutil.which", return_value="/usr/bin/python3"),
        ):
            result = EncoderClientCls._resolve_python()

        assert result == "/usr/bin/python3"

    def test_falls_through_to_shutil_which(self, EncoderClientCls):
        with (
            patch.object(Path, "exists", side_effect=[False, False]),
            patch(
                "serving.encoder_client.shutil.which", return_value="/usr/bin/python3"
            ) as mock_which,
        ):
            result = EncoderClientCls._resolve_python()

        assert result == "/usr/bin/python3"
        mock_which.assert_called_once_with("python3")

    def test_raises_if_no_candidate_found(self, EncoderClientCls):
        with (
            patch.object(Path, "exists", return_value=False),
            patch("serving.encoder_client.shutil.which", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="could not find a python interpreter"):
                EncoderClientCls._resolve_python()


class TestInit:
    def test_raises_if_child_does_not_emit_ready_byte(self, EncoderClientCls):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"X")

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc),
        ):
            with pytest.raises(RuntimeError, match="didn't emit R"):
                EncoderClientCls(Path("/fake/encoder_server.py"))

    def test_raises_if_child_stdout_eof(self, EncoderClientCls):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"")

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc),
        ):
            with pytest.raises(RuntimeError, match="encoder EOF"):
                EncoderClientCls(Path("/fake/encoder_server.py"))

    def test_success_on_ready_byte(self, EncoderClientCls):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"R")

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            client = EncoderClientCls(Path("/fake/encoder_server.py"))

        assert client.proc is mock_proc
        mock_popen.assert_called_once()


def _make_framed_response(json_data: dict, payload: bytes = b"") -> bytes:
    json_bytes = json.dumps(json_data).encode()
    return (
        struct.pack(">I", len(json_bytes)) + json_bytes + struct.pack(">I", len(payload)) + payload
    )


class TestEncode:
    def _make_client(self, EncoderClientCls, response_bytes: bytes):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"R" + response_bytes)
        mock_proc.stdin = BytesIO()

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc),
        ):
            client = EncoderClientCls(Path("/fake/encoder_server.py"))

        mock_proc.stdin = MagicMock()
        return client

    def test_encode_returns_result_dict(self, EncoderClientCls):
        tensor_payload = b"\x00" * 128
        response = _make_framed_response({"n_tokens": 64}, tensor_payload)
        client = self._make_client(EncoderClientCls, response)

        result = client.encode("a cat", seed=42)

        assert result["n_tokens"] == 64
        assert result["embeds_pt_bytes"] == tensor_payload

    def test_encode_sends_correct_json_frame(self, EncoderClientCls):
        response = _make_framed_response({"n_tokens": 5}, b"data")
        client = self._make_client(EncoderClientCls, response)

        client.encode("hello world", seed=7)

        stdin = client.proc.stdin
        calls = stdin.write.call_args_list
        assert len(calls) >= 3

        json_len_bytes = calls[0][0][0]
        struct.unpack(">I", json_len_bytes)
        json_bytes = calls[1][0][0]
        parsed = json.loads(json_bytes)
        assert parsed["op"] == "encode"
        assert parsed["prompt"] == "hello world"
        assert parsed["seed"] == 7

        stdin.flush.assert_called_once()

    def test_encode_raises_on_error_key(self, EncoderClientCls):
        response = _make_framed_response({"error": "OOM in encoder"})
        client = self._make_client(EncoderClientCls, response)

        with pytest.raises(RuntimeError, match="encoder error: OOM in encoder"):
            client.encode("test prompt", seed=0)


class TestClose:
    def _make_initialized_client(self, EncoderClientCls):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"R")

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc),
        ):
            client = EncoderClientCls(Path("/fake/encoder_server.py"))

        return client

    def test_close_is_idempotent(self, EncoderClientCls):
        client = self._make_initialized_client(EncoderClientCls)
        client.proc.wait.return_value = 0

        client.close()
        client.close()

        assert client.proc is None

    def test_close_calls_stdin_close_and_wait(self, EncoderClientCls):
        client = self._make_initialized_client(EncoderClientCls)
        client.proc.wait.return_value = 0

        proc = client.proc
        client.close()

        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    def test_close_terminates_on_first_timeout(self, EncoderClientCls):
        client = self._make_initialized_client(EncoderClientCls)
        proc = client.proc
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="enc", timeout=10),
            0,
        ]

        client.close()

        proc.terminate.assert_called_once()
        assert proc.kill.call_count == 0
        assert client.proc is None

    def test_close_kills_on_second_timeout(self, EncoderClientCls):
        client = self._make_initialized_client(EncoderClientCls)
        proc = client.proc
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="enc", timeout=10),
            subprocess.TimeoutExpired(cmd="enc", timeout=5),
            0,
        ]

        client.close()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert client.proc is None

    def test_close_handles_stdin_close_exception(self, EncoderClientCls):
        client = self._make_initialized_client(EncoderClientCls)
        proc = client.proc
        proc.stdin.close.side_effect = BrokenPipeError("already closed")
        proc.wait.return_value = 0

        client.close()
        assert client.proc is None


class TestContextManager:
    def test_context_manager_calls_close_on_exit(self, EncoderClientCls):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"R")
        mock_proc.wait.return_value = 0

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc),
        ):
            with EncoderClientCls(Path("/fake/encoder_server.py")) as client:
                assert client.proc is mock_proc

        assert client.proc is None

    def test_context_manager_calls_close_on_exception(self, EncoderClientCls):
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"R")
        mock_proc.wait.return_value = 0

        with (
            patch.object(EncoderClientCls, "_resolve_python", return_value="/usr/bin/python3"),
            patch("serving.encoder_client.subprocess.Popen", return_value=mock_proc),
        ):
            with pytest.raises(ValueError, match="test error"):
                with EncoderClientCls(Path("/fake/encoder_server.py")) as client:
                    raise ValueError("test error")

        assert client.proc is None
