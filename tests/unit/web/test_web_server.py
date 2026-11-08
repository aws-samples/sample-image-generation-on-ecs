from __future__ import annotations

import struct
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from server import app
from streaming import FrameType


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestHealthCheck:
    def test_returns_health_status(self, client):
        with patch("inference_client.health_check", return_value={"status": "healthy"}):
            response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

    def test_degraded_when_endpoint_unhealthy(self, client):
        with patch("inference_client.health_check", return_value={"status": "error"}):
            response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


class TestGetConfig:
    def test_returns_endpoint_config(self, client):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "default_backend" in data
        assert "gpu_enabled" in data
        assert "inferentia_enabled" in data
        assert "gpu_max_pixels" in data
        assert "gpu_min_dimension" in data
        assert "gpu_dimension_step" in data
        assert "region" in data
        assert "ws_heartbeat_interval" in data

    def test_gpu_pixel_budget_in_config(self, client):
        response = client.get("/api/config")
        data = response.json()
        assert data["gpu_max_pixels"] == 1048576
        assert data["gpu_dimension_step"] == 8


class TestGenerateResolutionValidation:
    def _auth_headers(self):
        return {"Authorization": "Bearer test-token"}

    @pytest.fixture(autouse=True)
    def _patch_auth(self):
        with patch("server.require_role", return_value=lambda: None):
            with patch("server.Depends", return_value=None):
                yield

    def test_gpu_valid_square(self, client):
        with patch("inference_client.invoke_json", return_value={"image_b64": "abc"}):
            resp = client.post(
                "/api/generate",
                json={"prompt": "test", "width": 512, "height": 512, "backend": "gpu"},
            )
        assert resp.status_code == 200

    def test_gpu_valid_landscape(self, client):
        with patch("inference_client.invoke_json", return_value={"image_b64": "abc"}):
            resp = client.post(
                "/api/generate",
                json={"prompt": "test", "width": 1280, "height": 720, "backend": "gpu"},
            )
        assert resp.status_code == 200

    def test_gpu_exceeds_pixel_budget(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "width": 1920, "height": 1080, "backend": "gpu"},
        )
        assert resp.status_code == 422
        assert "exceeds" in resp.json()["detail"]

    def test_gpu_not_divisible_by_step(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "width": 513, "height": 512, "backend": "gpu"},
        )
        assert resp.status_code == 422
        assert "divisible" in resp.json()["detail"]

    def test_gpu_below_min_dimension(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "width": 128, "height": 128, "backend": "gpu"},
        )
        assert resp.status_code == 422

    def test_neuron_valid_square(self, client):
        with patch("inference_client.invoke_json", return_value={"image_b64": "abc"}):
            resp = client.post(
                "/api/generate",
                json={"prompt": "test", "width": 768, "height": 768, "backend": "neuron"},
            )
        assert resp.status_code == 200

    def test_neuron_rejects_non_square(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "width": 1280, "height": 720, "backend": "neuron"},
        )
        assert resp.status_code == 422
        assert "Neuron" in resp.json()["detail"]

    def test_neuron_rejects_invalid_resolution(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "width": 640, "height": 640, "backend": "neuron"},
        )
        assert resp.status_code == 422
        assert "Neuron" in resp.json()["detail"]

    def test_resolution_backwards_compat(self, client):
        with patch("inference_client.invoke_json", return_value={"image_b64": "abc"}):
            resp = client.post(
                "/api/generate",
                json={"prompt": "test", "resolution": 768, "backend": "gpu"},
            )
        assert resp.status_code == 200


class TestListGallery:
    def test_returns_mock_gallery_in_local_mode(self, client):
        response = client.get("/api/gallery")
        assert response.status_code == 200
        data = response.json()
        assert "images" in data
        assert "count" in data
        assert len(data["images"]) > 0
        img = data["images"][0]
        assert "id" in img
        assert "prompt" in img
        assert "url" in img


class TestGetImage:
    def test_returns_404_for_nonexistent_image(self, client):
        with patch("gallery_routes.gallery_manager") as mock_gm:
            mock_gm.get_image = AsyncMock(return_value=None)
            response = client.get("/api/gallery/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_returns_image_when_found(self, client):
        image_data = {
            "id": "img-123",
            "prompt": "a sunset",
            "resolution": 512,
            "url": "https://example.com/img.png",
        }
        with patch("gallery_routes.gallery_manager") as mock_gm:
            mock_gm.get_image = AsyncMock(return_value=image_data)
            response = client.get("/api/gallery/img-123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "img-123"
        assert data["prompt"] == "a sunset"


class TestDeleteImage:
    def test_returns_404_for_nonexistent_image(self, client):
        with patch("gallery_routes.gallery_manager") as mock_gm:
            mock_gm.delete_image = AsyncMock(return_value=False)
            response = client.delete("/api/gallery/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_soft_delete_success(self, client):
        with patch("gallery_routes.gallery_manager") as mock_gm:
            mock_gm.delete_image = AsyncMock(return_value=True)
            response = client.delete("/api/gallery/img-456")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["image_id"] == "img-456"
        assert data["hard_deleted"] is False

    def test_hard_delete_with_query_param(self, client):
        with patch("gallery_routes.gallery_manager") as mock_gm:
            mock_gm.delete_image = AsyncMock(return_value=True)
            response = client.delete("/api/gallery/img-789?hard=true")

        assert response.status_code == 200
        data = response.json()
        assert data["hard_deleted"] is True


class TestWebSocketGenerate:
    def test_connects_and_receives_frames_local_mode(self, client):
        with client.websocket_connect("/ws/generate") as ws:
            ws.send_json({"prompt": "a cat in space", "resolution": 512, "steps": 8})
            msg = ws.receive_json()
            assert msg["type"] == "request_id"
            assert "request_id" in msg
            assert msg["backend"] == "mock"
            frame_bytes = ws.receive_bytes()
            ft, step, plen = struct.unpack(">III", frame_bytes[:12])
            assert ft in (FrameType.INTERMEDIATE, FrameType.FINAL)
            assert len(frame_bytes) == 12 + plen

    def test_receives_multiple_frames_with_denoise_viz(self, client):
        with client.websocket_connect("/ws/generate") as ws:
            ws.send_json({"prompt": "test", "resolution": 512, "steps": 8, "denoise_viz": True})
            msg = ws.receive_json()
            assert msg["type"] == "request_id"
            frames = []
            try:
                while True:
                    data = ws.receive_bytes()
                    frames.append(data)
            except Exception:
                pass
            assert len(frames) >= 2
            last_ft = struct.unpack(">I", frames[-1][:4])[0]
            assert last_ft == FrameType.COMPLETE

    def test_missing_prompt_sends_error_and_closes(self, client):
        with client.websocket_connect("/ws/generate") as ws:
            ws.send_json({"prompt": "", "resolution": 512})
            frame_bytes = ws.receive_bytes()
            ft = struct.unpack(">I", frame_bytes[:4])[0]
            assert ft == FrameType.ERROR
            _, _, plen = struct.unpack(">III", frame_bytes[:12])
            error_msg = frame_bytes[12 : 12 + plen].decode("utf-8")
            assert "required" in error_msg.lower()
