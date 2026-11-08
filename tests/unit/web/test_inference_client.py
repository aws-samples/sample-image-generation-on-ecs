from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _set_alb_url(monkeypatch):
    monkeypatch.setattr("inference_client.ALB_URL", "http://test-alb:80")


@pytest.fixture
def mock_http():
    with patch("inference_client._http") as m:
        yield m


class TestInvokeJson:
    def test_success(self, mock_http):
        from inference_client import invoke_json

        mock_http.request.return_value = MagicMock(
            status=200, data=json.dumps({"image_b64": "abc123"}).encode()
        )
        result = invoke_json(512, {"prompt": "test", "seed": 42})
        assert result == {"image_b64": "abc123"}
        mock_http.request.assert_called_once()
        args = mock_http.request.call_args
        assert args[0] == ("POST", "http://test-alb:80/invocations")
        assert args[1]["headers"]["X-Target-Resolution"] == "512"

    def test_non_200_raises(self, mock_http):
        from inference_client import invoke_json

        mock_http.request.return_value = MagicMock(status=503, data=b"loading")
        with pytest.raises(RuntimeError, match="503"):
            invoke_json(768, {"prompt": "test"})


class TestInvokeStreaming:
    def test_yields_chunks(self, mock_http):
        from inference_client import invoke_streaming

        mock_resp = MagicMock(status=200)
        mock_resp.stream.return_value = iter([b"chunk1", b"chunk2"])
        mock_http.request.return_value = mock_resp

        chunks = list(invoke_streaming(1024, {"prompt": "test"}))
        assert chunks == [b"chunk1", b"chunk2"]
        mock_resp.release_conn.assert_called_once()

    def test_non_200_raises(self, mock_http):
        from inference_client import invoke_streaming

        mock_http.request.return_value = MagicMock(status=500)
        with pytest.raises(RuntimeError, match="500"):
            list(invoke_streaming(512, {"prompt": "test"}))


class TestHealthCheck:
    def test_healthy(self, mock_http):
        from inference_client import health_check

        mock_http.request.return_value = MagicMock(status=200)
        result = health_check(512)
        assert result["status"] == "healthy"
        assert result["resolution"] == 512

    def test_unhealthy(self, mock_http):
        from inference_client import health_check

        mock_http.request.return_value = MagicMock(status=503)
        result = health_check(512)
        assert result["status"] == "unhealthy"
        assert result["code"] == 503

    def test_connection_error(self, mock_http):
        from inference_client import health_check

        mock_http.request.side_effect = Exception("Connection refused")
        result = health_check(512)
        assert result["status"] == "error"
        assert "Connection refused" in result["error"]
