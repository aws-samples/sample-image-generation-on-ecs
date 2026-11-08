from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import config

    snapshot = {
        "gallery_bucket": getattr(config.settings, "gallery_bucket", ""),
        "cognito_user_pool_id": config.settings.cognito_user_pool_id,
        "cognito_app_client_id": config.settings.cognito_app_client_id,
        "cognito_domain": config.settings.cognito_domain,
    }
    config.settings.gallery_bucket = "test-bucket"
    config.settings.cognito_user_pool_id = "us-east-1_secret"
    config.settings.cognito_app_client_id = "secret-client-id"
    config.settings.cognito_domain = "test.auth.us-east-1.amazoncognito.com"

    from server import app

    yield TestClient(app)

    for k, v in snapshot.items():
        setattr(config.settings, k, v)


class TestConfigEndpoint:
    def test_does_not_expose_user_pool_id(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "cognito_user_pool_id" not in body, (
            "User Pool ID must not be in /api/config — frontend reads it from the meta tag"
        )

    def test_does_not_expose_app_client_id(self, client):
        resp = client.get("/api/config")
        body = resp.json()
        assert "cognito_app_client_id" not in body, (
            "App Client ID must not be in /api/config — frontend reads it from the meta tag"
        )

    def test_returns_non_secret_runtime_config(self, client):
        resp = client.get("/api/config")
        body = resp.json()
        assert "default_backend" in body
        assert "gpu_enabled" in body
        assert "gpu_max_pixels" in body
        assert "websocket_url" in body
        assert "region" in body
