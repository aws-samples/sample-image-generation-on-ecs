from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def encoder_server_module():
    from serving import encoder_server

    return encoder_server


class TestReadFrame:
    def test_reads_valid_frame(self, encoder_server_module):
        json_data = {"op": "encode", "prompt": "hello", "seed": 42}
        payload = b"binary_data"
        json_bytes = json.dumps(json_data).encode()
        frame = (
            struct.pack(">I", len(json_bytes))
            + json_bytes
            + struct.pack(">I", len(payload))
            + payload
        )
        stream = io.BytesIO(frame)

        result = encoder_server_module._read_frame(stream)

        assert result is not None
        parsed_json, parsed_payload = result
        assert parsed_json == json_data
        assert parsed_payload == payload

    def test_returns_none_on_empty_stream(self, encoder_server_module):
        stream = io.BytesIO(b"")
        assert encoder_server_module._read_frame(stream) is None

    def test_returns_none_on_short_header(self, encoder_server_module):
        stream = io.BytesIO(b"\x00\x01")
        assert encoder_server_module._read_frame(stream) is None

    def test_reads_empty_payload(self, encoder_server_module):
        json_data = {"op": "encode", "prompt": "test", "seed": 0}
        json_bytes = json.dumps(json_data).encode()
        frame = struct.pack(">I", len(json_bytes)) + json_bytes + struct.pack(">I", 0)
        stream = io.BytesIO(frame)

        result = encoder_server_module._read_frame(stream)

        assert result is not None
        parsed_json, parsed_payload = result
        assert parsed_json == json_data
        assert parsed_payload == b""


class TestWriteFrame:
    def test_writes_valid_frame(self, encoder_server_module):
        stream = io.BytesIO()
        json_data = {"n_tokens": 64}
        payload = b"tensor_bytes"

        encoder_server_module._write_frame(stream, json_data, payload)

        stream.seek(0)
        json_len = struct.unpack(">I", stream.read(4))[0]
        json_bytes = stream.read(json_len)
        assert json.loads(json_bytes) == json_data
        payload_len = struct.unpack(">I", stream.read(4))[0]
        assert stream.read(payload_len) == payload

    def test_writes_empty_payload(self, encoder_server_module):
        stream = io.BytesIO()
        encoder_server_module._write_frame(stream, {"ok": True})

        stream.seek(0)
        json_len = struct.unpack(">I", stream.read(4))[0]
        stream.read(json_len)
        payload_len = struct.unpack(">I", stream.read(4))[0]
        assert payload_len == 0

    def test_flushes_after_write(self, encoder_server_module):
        stream = MagicMock()
        encoder_server_module._write_frame(stream, {"ok": True}, b"data")
        stream.flush.assert_called_once()

    def test_frame_roundtrip(self, encoder_server_module):
        json_data = {"n_tokens": 128}
        payload = b"\x00" * 64
        stream = io.BytesIO()

        encoder_server_module._write_frame(stream, json_data, payload)

        stream.seek(0)
        result = encoder_server_module._read_frame(stream)
        assert result is not None
        assert result[0] == json_data
        assert result[1] == payload

    def test_writes_correct_byte_order(self, encoder_server_module):
        stream = io.BytesIO()
        json_data = {"x": 1}
        payload = b"A" * 256

        encoder_server_module._write_frame(stream, json_data, payload)

        stream.seek(0)
        json_len = struct.unpack(">I", stream.read(4))[0]
        stream.read(json_len)
        raw = stream.read(4)
        assert raw == b"\x00\x00\x01\x00"


class TestMain:
    @pytest.fixture
    def mock_torch(self):
        torch_mock = sys.modules["torch"]
        torch_mock.reset_mock()

        tensor_mock = MagicMock()
        tensor_mock.shape = (1, 128, 2560)
        tensor_mock.__getitem__ = MagicMock(return_value=tensor_mock)
        tensor_mock.cpu.return_value = tensor_mock
        tensor_mock.contiguous.return_value = tensor_mock
        torch_mock.zeros.return_value = tensor_mock
        torch_mock.long = "torch.long"
        torch_mock.no_grad.return_value.__enter__ = MagicMock()
        torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        model_mock = MagicMock()
        model_mock.eval.return_value = model_mock
        enc_output = MagicMock()
        enc_output.shape = (1, 128, 2560)
        model_mock.return_value = enc_output
        torch_mock.jit.load.return_value = model_mock

        def fake_save(tensor, buf):
            buf.write(b"\x80fake_tensor_data")

        torch_mock.save.side_effect = fake_save

        return torch_mock

    @pytest.fixture
    def mock_tokenizer(self):
        transformers_mock = sys.modules["transformers"]
        transformers_mock.reset_mock()

        tok_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.return_value = tok_mock
        tok_mock.apply_chat_template.return_value = "<think>\n</think>\na cat"

        tok_output = MagicMock()
        input_ids_mock = MagicMock()
        tok_output.input_ids = input_ids_mock

        attn_mask_mock = MagicMock()
        attn_mask_mock.long.return_value = attn_mask_mock
        mask_row = MagicMock()
        mask_row.__eq__ = MagicMock(
            return_value=MagicMock(
                sum=MagicMock(return_value=MagicMock(item=MagicMock(return_value=64)))
            )
        )
        mask_row.bool.return_value = MagicMock(
            tolist=MagicMock(return_value=[True] * 64 + [False] * 64)
        )
        attn_mask_mock.__getitem__ = MagicMock(return_value=mask_row)
        tok_output.attention_mask = attn_mask_mock

        tok_mock.return_value = tok_output

        return tok_mock, transformers_mock

    def _make_request_frame(self, prompt: str, seed: int) -> bytes:
        json_data = {"op": "encode", "prompt": prompt, "seed": seed}
        json_bytes = json.dumps(json_data).encode()
        return struct.pack(">I", len(json_bytes)) + json_bytes + struct.pack(">I", 0)

    def test_main_exits_on_missing_model_file(self, encoder_server_module):
        with patch.object(Path, "exists", return_value=False):
            result = encoder_server_module.main()
        assert result == 1

    def test_main_exits_cleanly_on_empty_stdin(
        self, encoder_server_module, mock_torch, mock_tokenizer
    ):
        tok_mock, transformers_mock = mock_tokenizer
        data_stream = io.BytesIO()
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(b"")

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", return_value=MagicMock(st_size=5_000_000_000)),
            patch.object(encoder_server_module, "_data_stream", data_stream),
            patch("sys.stdin", mock_stdin),
        ):
            result = encoder_server_module.main()

        assert result == 0

    def test_main_sends_ready_byte(self, encoder_server_module, mock_torch, mock_tokenizer):
        tok_mock, transformers_mock = mock_tokenizer
        data_stream = io.BytesIO()
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(b"")

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", return_value=MagicMock(st_size=5_000_000_000)),
            patch.object(encoder_server_module, "_data_stream", data_stream),
            patch("sys.stdin", mock_stdin),
        ):
            encoder_server_module.main()

        data_stream.seek(0)
        assert data_stream.read(1) == b"R"

    def test_main_processes_valid_request(self, encoder_server_module, mock_torch, mock_tokenizer):
        tok_mock, transformers_mock = mock_tokenizer
        request = self._make_request_frame("a cat", 42)
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(request)
        data_stream = io.BytesIO()

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", return_value=MagicMock(st_size=5_000_000_000)),
            patch.object(encoder_server_module, "_data_stream", data_stream),
            patch("sys.stdin", mock_stdin),
        ):
            result = encoder_server_module.main()

        assert result == 0

        data_stream.seek(1)
        resp = encoder_server_module._read_frame(data_stream)
        assert resp is not None
        resp_json, resp_payload = resp
        assert "n_tokens" in resp_json
        assert resp_json["n_tokens"] == 64
        assert len(resp_payload) > 0

    def test_main_handles_bad_json_frame(self, encoder_server_module, mock_torch, mock_tokenizer):
        tok_mock, transformers_mock = mock_tokenizer

        bad_json = b"not valid json {"
        frame = struct.pack(">I", len(bad_json)) + bad_json + struct.pack(">I", 0)
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(frame)
        data_stream = io.BytesIO()

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", return_value=MagicMock(st_size=5_000_000_000)),
            patch.object(encoder_server_module, "_data_stream", data_stream),
            patch("sys.stdin", mock_stdin),
        ):
            result = encoder_server_module.main()

        assert result == 0

        data_stream.seek(1)
        resp = encoder_server_module._read_frame(data_stream)
        assert resp is not None
        assert "error" in resp[0]

    def test_main_handles_encoding_exception(
        self, encoder_server_module, mock_torch, mock_tokenizer
    ):
        tok_mock, transformers_mock = mock_tokenizer
        tok_mock.apply_chat_template.side_effect = RuntimeError("tokenizer exploded")

        request = self._make_request_frame("trigger error", 1)
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(request)
        data_stream = io.BytesIO()

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", return_value=MagicMock(st_size=5_000_000_000)),
            patch.object(encoder_server_module, "_data_stream", data_stream),
            patch("sys.stdin", mock_stdin),
        ):
            result = encoder_server_module.main()

        assert result == 0

        data_stream.seek(1)
        resp = encoder_server_module._read_frame(data_stream)
        assert resp is not None
        assert "error" in resp[0]
        assert "tokenizer exploded" in resp[0]["error"]

    def test_main_processes_multiple_requests(
        self, encoder_server_module, mock_torch, mock_tokenizer
    ):
        tok_mock, transformers_mock = mock_tokenizer

        req1 = self._make_request_frame("first prompt", 1)
        req2 = self._make_request_frame("second prompt", 2)
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(req1 + req2)
        data_stream = io.BytesIO()

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", return_value=MagicMock(st_size=5_000_000_000)),
            patch.object(encoder_server_module, "_data_stream", data_stream),
            patch("sys.stdin", mock_stdin),
        ):
            result = encoder_server_module.main()

        assert result == 0

        data_stream.seek(1)
        resp1 = encoder_server_module._read_frame(data_stream)
        resp2 = encoder_server_module._read_frame(data_stream)

        assert resp1 is not None
        assert resp2 is not None
        assert "n_tokens" in resp1[0]
        assert "n_tokens" in resp2[0]


class TestModuleConstants:
    def test_max_seq_len_encoder(self, encoder_server_module):
        assert encoder_server_module.MAX_SEQ_LEN_ENCODER == 256

    def test_out_seq_len(self, encoder_server_module):
        assert encoder_server_module.OUT_SEQ_LEN == 512

    def test_traced_encoder_path(self, encoder_server_module):
        expected = Path("/opt/ml/zimage_artifacts/compiled_models/encoder_256.pt")
        assert expected == encoder_server_module.TRACED_ENCODER_PT

    def test_artifacts_dir_path(self, encoder_server_module):
        expected = Path("/opt/ml/zimage_artifacts")
        assert expected == encoder_server_module.ARTIFACTS_DIR
