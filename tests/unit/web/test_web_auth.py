from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


def _run(coro):
    return asyncio.run(coro)


def _fake_request(aws_event: dict | None = None):

    class _Req:
        def __init__(self):
            self.scope = {"aws.event": aws_event} if aws_event is not None else {}

    return _Req()


class TestRoleEnum:
    def test_user_value(self):
        from auth import Role

        assert Role.USER == "users"
        assert Role.USER.value == "users"

    def test_admin_value(self):
        from auth import Role

        assert Role.ADMIN == "admins"
        assert Role.ADMIN.value == "admins"


class TestTokenClaims:
    def test_construction_with_groups(self):
        from auth import TokenClaims

        claims = TokenClaims(sub="user-123", email="test@example.com", groups=["admins", "users"])
        assert claims.sub == "user-123"
        assert claims.email == "test@example.com"
        assert claims.groups == ["admins", "users"]

    def test_construction_without_groups_defaults_empty(self):
        from auth import TokenClaims

        claims = TokenClaims(sub="user-456", email="other@example.com")
        assert claims.sub == "user-456"
        assert claims.email == "other@example.com"
        assert claims.groups == []


class TestGetCurrentUser:
    @patch("auth.settings")
    def test_raises_401_when_no_authorization(self, mock_settings):
        from fastapi import HTTPException

        from auth import get_current_user

        mock_settings.local = False
        with pytest.raises(HTTPException) as exc_info:
            _run(get_current_user(_fake_request(), authorization=None))
        assert exc_info.value.status_code == 401
        assert "Not authenticated" in exc_info.value.detail

    @patch("auth._decode_token")
    @patch("auth.settings")
    def test_raises_401_for_invalid_token(self, mock_settings, mock_decode):
        from fastapi import HTTPException

        from auth import get_current_user

        mock_settings.local = False
        mock_decode.side_effect = HTTPException(status_code=401, detail="Token validation failed")
        with pytest.raises(HTTPException) as exc_info:
            _run(get_current_user(_fake_request(), authorization="Bearer bad-token"))
        assert exc_info.value.status_code == 401
        mock_decode.assert_called_once_with("bad-token")

    @patch("auth._decode_token")
    @patch("auth.settings")
    def test_returns_claims_for_valid_token(self, mock_settings, mock_decode):
        from auth import TokenClaims, get_current_user

        mock_settings.local = False
        expected = TokenClaims(sub="u1", email="u1@test.com", groups=["users"])
        mock_decode.return_value = expected
        result = _run(get_current_user(_fake_request(), authorization="Bearer valid-token"))
        assert result == expected
        mock_decode.assert_called_once_with("valid-token")


class TestRequireRole:
    @patch("auth.settings")
    def test_admin_role_passes_when_user_has_admins_group(self, mock_settings):
        from auth import Role, TokenClaims, require_role

        mock_settings.local = False
        checker = require_role(Role.ADMIN)
        user = TokenClaims(sub="a1", email="admin@test.com", groups=["admins", "users"])
        result = _run(checker(user=user))
        assert result == user

    @patch("auth.settings")
    def test_admin_role_raises_403_when_only_users_group(self, mock_settings):
        from fastapi import HTTPException

        from auth import Role, TokenClaims, require_role

        mock_settings.local = False
        checker = require_role(Role.ADMIN)
        user = TokenClaims(sub="u1", email="user@test.com", groups=["users"])
        with pytest.raises(HTTPException) as exc_info:
            _run(checker(user=user))
        assert exc_info.value.status_code == 403
        assert "admins" in exc_info.value.detail

    @patch("auth.settings")
    def test_user_role_passes_for_authenticated_user(self, mock_settings):
        from auth import Role, TokenClaims, require_role

        mock_settings.local = False
        checker = require_role(Role.USER)
        user = TokenClaims(sub="u1", email="user@test.com", groups=["users"])
        result = _run(checker(user=user))
        assert result == user


class TestClaimsFromRequestScope:
    @patch("auth.settings")
    def test_extracts_claims_from_aws_event(self, mock_settings):
        from auth import get_current_user

        mock_settings.local = False
        mock_settings.is_lambda = True
        aws_event = {
            "requestContext": {
                "authorizer": {
                    "jwt": {
                        "claims": {
                            "sub": "u-from-scope",
                            "email": "scope@test.com",
                            "cognito:groups": ["users", "admins"],
                        }
                    }
                }
            }
        }
        result = _run(get_current_user(_fake_request(aws_event=aws_event), authorization=None))
        assert result.sub == "u-from-scope"
        assert result.email == "scope@test.com"
        assert "admins" in result.groups

    @patch("auth.settings")
    def test_handles_string_groups_serialized_by_api_gw(self, mock_settings):
        from auth import get_current_user

        mock_settings.local = False
        mock_settings.is_lambda = True
        aws_event = {
            "requestContext": {
                "authorizer": {
                    "jwt": {
                        "claims": {
                            "sub": "u",
                            "email": "u@test.com",
                            "cognito:groups": "users,admins",
                        }
                    }
                }
            }
        }
        result = _run(get_current_user(_fake_request(aws_event=aws_event), authorization=None))
        assert "users" in result.groups
        assert "admins" in result.groups


class TestLocalBypassSafety:
    @patch("auth.settings")
    def test_local_bypass_active_when_local_and_not_lambda(self, mock_settings):
        from auth import _LOCAL_ADMIN, get_current_user

        mock_settings.local = True
        mock_settings.is_lambda = False
        result = _run(get_current_user(_fake_request(), authorization=None))
        assert result == _LOCAL_ADMIN

    @patch("auth.settings")
    def test_local_bypass_suppressed_in_lambda(self, mock_settings):
        from fastapi import HTTPException

        from auth import get_current_user

        mock_settings.local = True
        mock_settings.is_lambda = True
        with pytest.raises(HTTPException) as exc_info:
            _run(get_current_user(_fake_request(), authorization=None))
        assert exc_info.value.status_code == 401

    @patch("auth.settings")
    def test_require_role_bypass_suppressed_in_lambda(self, mock_settings):
        from fastapi import HTTPException

        from auth import Role, TokenClaims, require_role

        mock_settings.local = True
        mock_settings.is_lambda = True
        checker = require_role(Role.ADMIN)
        non_admin = TokenClaims(sub="u1", email="user@test.com", groups=["users"])
        with pytest.raises(HTTPException) as exc_info:
            _run(checker(user=non_admin))
        assert exc_info.value.status_code == 403


class TestWsAuthenticate:
    def test_raises_401_when_no_token(self):
        from fastapi import HTTPException

        from auth import ws_authenticate

        with pytest.raises(HTTPException) as exc_info:
            _run(ws_authenticate(token=None))
        assert exc_info.value.status_code == 401
        assert "Token required" in exc_info.value.detail

    @patch("auth._decode_token")
    def test_returns_claims_for_valid_token(self, mock_decode):
        from auth import TokenClaims, ws_authenticate

        expected = TokenClaims(sub="ws-user", email="ws@test.com", groups=["users"])
        mock_decode.return_value = expected
        result = _run(ws_authenticate(token="ws-valid-token"))
        assert result == expected
        mock_decode.assert_called_once_with("ws-valid-token")
