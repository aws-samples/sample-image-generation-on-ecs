from __future__ import annotations

import importlib
import json
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ws_handler(monkeypatch):
    monkeypatch.setenv("ZIMAGE_GALLERY_BUCKET", "test-bucket")
    monkeypatch.setenv("ZIMAGE_GALLERY_TABLE", "test-table")
    monkeypatch.setenv("ZIMAGE_REGION", "us-east-1")
    monkeypatch.setenv("ZIMAGE_PREFIX", "dev-zimg")
    monkeypatch.setenv("ZIMAGE_INFERENCE_ALB_URL", "http://test-alb.internal")

    repo_root = Path(__file__).resolve().parents[3]
    backend_dir = repo_root / "web" / "backend"
    monkeypatch.syspath_prepend(str(backend_dir))

    if "websocket_handler" in sys.modules:
        del sys.modules["websocket_handler"]

    with patch("boto3.client"), patch("boto3.resource"), patch("urllib3.PoolManager"):
        mod = importlib.import_module("websocket_handler")

    return mod


def test_resolve_endpoint_gpu(ws_handler):
    result = ws_handler._resolve_endpoint(768, "gpu")
    assert result == "ecs:gpu"


def test_resolve_endpoint_all_resolutions(ws_handler):
    assert ws_handler._resolve_endpoint(512, "gpu") == "ecs:gpu"
    assert ws_handler._resolve_endpoint(1024, "gpu") == "ecs:gpu"


def test_streaming_invocation_sends_correct_headers(ws_handler, monkeypatch):
    mock_http = MagicMock()
    monkeypatch.setattr(ws_handler, "_http", mock_http)

    frame_payload = b"PNG_DATA"
    frame_header = struct.pack(">III", 2, 8, len(frame_payload))
    chunk = frame_header + frame_payload

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.stream.return_value = iter([chunk])
    mock_http.request.return_value = mock_response

    mock_apigw = MagicMock()
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://presigned-url"
    monkeypatch.setattr(ws_handler, "s3_client", mock_s3)
    monkeypatch.setattr(ws_handler, "dynamodb_resource", MagicMock())

    event = {
        "requestContext": {
            "routeKey": "$default",
            "connectionId": "test-conn",
            "domainName": "test.execute-api.us-east-1.amazonaws.com",
            "stage": "v1",
        },
        "body": json.dumps(
            {
                "prompt": "test prompt",
                "width": 768,
                "height": 768,
                "denoise_viz": True,
            }
        ),
    }

    with patch.object(ws_handler, "_get_apigw_client", return_value=mock_apigw):
        ws_handler.handler(event, None)

    mock_http.request.assert_called_once()
    call_kwargs = mock_http.request.call_args
    assert call_kwargs[0][0] == "POST"
    assert "http://test-alb.internal/invocations" in call_kwargs[0][1]
    headers = call_kwargs[1]["headers"]
    assert headers["X-Target-Resolution"] == "gpu"
    assert headers["Accept"] == "application/octet-stream"


def test_json_invocation_path(ws_handler, monkeypatch):
    mock_http = MagicMock()
    monkeypatch.setattr(ws_handler, "_http", mock_http)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.data = json.dumps(
        {
            "image_b64": "dGVzdA==",
            "elapsed_s": 5.0,
            "resolution": 512,
        }
    ).encode()
    mock_http.request.return_value = mock_response

    mock_apigw = MagicMock()
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://presigned-url"
    monkeypatch.setattr(ws_handler, "s3_client", mock_s3)
    monkeypatch.setattr(ws_handler, "dynamodb_resource", MagicMock())

    event = {
        "requestContext": {
            "routeKey": "$default",
            "connectionId": "test-conn",
            "domainName": "test.execute-api.us-east-1.amazonaws.com",
            "stage": "v1",
        },
        "body": json.dumps(
            {
                "prompt": "test prompt",
                "width": 512,
                "height": 512,
                "denoise_viz": False,
            }
        ),
    }

    with patch.object(ws_handler, "_get_apigw_client", return_value=mock_apigw):
        ws_handler.handler(event, None)

    call_kwargs = mock_http.request.call_args
    headers = call_kwargs[1]["headers"]
    assert headers["X-Target-Resolution"] == "gpu"
    assert "Accept" not in headers or headers.get("Accept") != "application/octet-stream"
