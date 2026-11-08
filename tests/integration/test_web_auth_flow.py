from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_WEB_BACKEND = str(Path(__file__).resolve().parents[2] / "web" / "backend")
if _WEB_BACKEND not in sys.path:
    sys.path.insert(0, _WEB_BACKEND)


class _FakeSettings:
    cognito_user_pool_id = "us-east-1_testPool"
    cognito_app_client_id = "test-client-id-123"
    cognito_domain = "test-app.auth.us-east-1.amazoncognito.com"


@pytest.fixture
def client():
    import auth_routes

    with patch.object(auth_routes, "settings", _FakeSettings()):
        app = FastAPI()
        app.include_router(auth_routes.router)
        yield TestClient(app)


class TestAuthRoutes:
    def test_callback_route_removed(self, client):
        resp = client.get("/auth/callback", follow_redirects=False)
        assert resp.status_code == 404

    def test_refresh_route_removed(self, client):
        resp = client.post("/auth/refresh", json={"refresh_token": "x"})
        assert resp.status_code == 404


class TestAuthLogout:
    def test_redirects_to_cognito_logout(self, client):
        resp = client.get(
            "/auth/logout",
            headers={"x-forwarded-proto": "https", "host": "d123.cloudfront.net"},
            follow_redirects=False,
        )
        assert resp.status_code in (302, 307)
        location = resp.headers["location"]
        assert "/logout" in location
        assert "client_id=test-client-id-123" in location
        assert "logout_uri=" in location
        assert "test-app.auth.us-east-1.amazoncognito.com" in location
