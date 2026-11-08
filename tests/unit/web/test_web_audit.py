from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


class TestAudit:
    def test_audit_emits_structured_log(self, capsys):
        from loguru import logger

        from audit import audit

        logger.remove()
        sink_id = logger.add(sys.stderr, format="{message} {extra}")
        try:
            audit(
                actor_sub="user-1",
                actor_email="admin@test.com",
                action="delete_endpoint",
                resource="dev-zimg-endpoint-512",
            )
            err = capsys.readouterr().err
            assert "delete_endpoint" in err
            assert "dev-zimg-endpoint-512" in err
        finally:
            logger.remove(sink_id)

    def test_audit_includes_extra_kwargs(self, capsys):
        from loguru import logger

        from audit import audit

        logger.remove()
        sink_id = logger.add(sys.stderr, format="{message} {extra}")
        try:
            audit(
                actor_sub="u",
                actor_email="u@e.com",
                action="scale_endpoint",
                resource="ep-1",
                min_capacity=1,
                max_capacity=4,
            )
            err = capsys.readouterr().err
            assert "min_capacity" in err
            assert "max_capacity" in err
        finally:
            logger.remove(sink_id)


class TestDeleteEndpointConfirm:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import admin
        import config

        config.settings.local = True
        from auth import TokenClaims

        async def _stub_admin():
            return TokenClaims(sub="u", email="admin@t.com", groups=["admins"])

        app = FastAPI()
        app.dependency_overrides[admin._require_admin] = _stub_admin
        app.include_router(admin.router)
        return TestClient(app)

    def test_delete_without_confirm_rejects(self, client):
        resp = client.delete("/admin/api/endpoints/test-zimg-inference-512")
        assert resp.status_code == 400
        assert "confirm" in resp.json()["detail"].lower()

    def test_delete_with_wrong_confirm_rejects(self, client):
        resp = client.delete("/admin/api/endpoints/test-zimg-inference-512?confirm=wrong-name")
        assert resp.status_code == 400

    def test_delete_with_matching_confirm_proceeds(self, client):
        with patch("admin.boto3"):
            resp = client.delete(
                "/admin/api/endpoints/test-zimg-inference-512?confirm=test-zimg-inference-512"
            )
        assert resp.status_code == 200
