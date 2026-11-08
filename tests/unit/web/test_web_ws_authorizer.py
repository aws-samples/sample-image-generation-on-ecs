from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from jose import jwt

_TEST_SECRET = "test-secret-key-for-ws-tickets-64chars-long-enough-for-hmac256xx"


@pytest.fixture
def env_vars(monkeypatch):
    monkeypatch.setenv(
        "ZIMAGE_WS_TICKET_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:123:secret:test"
    )
    monkeypatch.setenv("ZIMAGE_REGION", "us-east-1")


def _make_event(
    ticket: str | None, method_arn: str = "arn:aws:execute-api:us-east-1:123:abc/v1/$connect"
):
    qs = {"ticket": ticket} if ticket else {}
    return {
        "type": "REQUEST",
        "methodArn": method_arn,
        "headers": {},
        "queryStringParameters": qs,
        "requestContext": {"connectionId": "abc-123", "routeKey": "$connect"},
    }


def _make_ticket(claims: dict) -> str:
    return jwt.encode(claims, _TEST_SECRET, algorithm="HS256")


class TestAuthorizerHandler:
    def test_allows_valid_ticket(self, env_vars):
        from ws_authorizer import handler

        now = int(time.time())
        ticket = _make_ticket(
            {
                "sub": "user-1",
                "email": "u@test.com",
                "groups": ["users"],
                "iat": now,
                "exp": now + 30,
                "purpose": "ws-connect",
            }
        )
        event = _make_event(ticket)
        with patch("ws_authorizer._get_secret", return_value=_TEST_SECRET):
            result = handler(event, None)

        assert result["principalId"] == "user-1"
        assert result["policyDocument"]["Statement"][0]["Effect"] == "Allow"

    def test_denies_missing_ticket(self, env_vars):
        from ws_authorizer import handler

        event = _make_event(ticket=None)
        result = handler(event, None)
        assert result["policyDocument"]["Statement"][0]["Effect"] == "Deny"

    def test_denies_invalid_ticket(self, env_vars):
        from ws_authorizer import handler

        event = _make_event(ticket="not-a-jwt")
        with patch("ws_authorizer._get_secret", return_value=_TEST_SECRET):
            result = handler(event, None)
        assert result["policyDocument"]["Statement"][0]["Effect"] == "Deny"

    def test_denies_expired_ticket(self, env_vars):
        from ws_authorizer import handler

        now = int(time.time())
        ticket = _make_ticket(
            {
                "sub": "user-1",
                "email": "u@test.com",
                "groups": ["users"],
                "iat": now - 60,
                "exp": now - 30,
                "purpose": "ws-connect",
            }
        )
        event = _make_event(ticket)
        with patch("ws_authorizer._get_secret", return_value=_TEST_SECRET):
            result = handler(event, None)
        assert result["policyDocument"]["Statement"][0]["Effect"] == "Deny"

    def test_denies_wrong_purpose(self, env_vars):
        from ws_authorizer import handler

        now = int(time.time())
        ticket = _make_ticket(
            {
                "sub": "user-1",
                "email": "u@test.com",
                "groups": ["users"],
                "iat": now,
                "exp": now + 30,
                "purpose": "wrong-purpose",
            }
        )
        event = _make_event(ticket)
        with patch("ws_authorizer._get_secret", return_value=_TEST_SECRET):
            result = handler(event, None)
        assert result["policyDocument"]["Statement"][0]["Effect"] == "Deny"

    def test_extracts_groups_into_context(self, env_vars):
        from ws_authorizer import handler

        now = int(time.time())
        ticket = _make_ticket(
            {
                "sub": "admin-1",
                "email": "admin@test.com",
                "groups": ["admins", "users"],
                "iat": now,
                "exp": now + 30,
                "purpose": "ws-connect",
            }
        )
        event = _make_event(ticket)
        with patch("ws_authorizer._get_secret", return_value=_TEST_SECRET):
            result = handler(event, None)

        ctx = result.get("context", {})
        assert ctx.get("sub") == "admin-1"
        assert ctx.get("email") == "admin@test.com"
        assert "admins" in ctx.get("groups", "")
        assert "users" in ctx.get("groups", "")

    def test_policy_resource_matches_method_arn(self, env_vars):
        from ws_authorizer import handler

        now = int(time.time())
        custom_arn = "arn:aws:execute-api:us-east-1:999:xyz/v1/$connect"
        ticket = _make_ticket(
            {
                "sub": "user-1",
                "email": "u@test.com",
                "groups": ["users"],
                "iat": now,
                "exp": now + 30,
                "purpose": "ws-connect",
            }
        )
        event = _make_event(ticket, method_arn=custom_arn)
        with patch("ws_authorizer._get_secret", return_value=_TEST_SECRET):
            result = handler(event, None)

        resource = result["policyDocument"]["Statement"][0]["Resource"]
        assert resource == custom_arn
