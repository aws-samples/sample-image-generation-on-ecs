from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from jose import jwk, jwt


@pytest.fixture(autouse=True)
def _clear_jwks_cache():
    from jwt_verify import _load_jwks_cached

    _load_jwks_cached.cache_clear()
    yield
    _load_jwks_cached.cache_clear()


@pytest.fixture
def rsa_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_jwk = jwk.construct(public_pem.decode(), algorithm="RS256").to_dict()
    pub_jwk["kid"] = "test-key-1"
    pub_jwk["use"] = "sig"
    pub_jwk["alg"] = "RS256"
    return {
        "private_pem": private_pem.decode(),
        "jwks": {"keys": [pub_jwk]},
        "kid": "test-key-1",
    }


@pytest.fixture
def issuer():
    return "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_testpool"


@pytest.fixture
def audience():
    return "test-client-id"


@pytest.fixture
def jwks_url(issuer):
    return f"{issuer}/.well-known/jwks.json"


def _make_token(rsa_keypair, claims):
    return jwt.encode(
        claims,
        rsa_keypair["private_pem"],
        algorithm="RS256",
        headers={"kid": rsa_keypair["kid"]},
    )


def _mock_jwks_response(rsa_keypair):
    resp = MagicMock()
    resp.read.return_value = json.dumps(rsa_keypair["jwks"]).encode()
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: None
    return resp


class TestVerifyToken:
    def test_valid_token_returns_claims(self, rsa_keypair, issuer, audience, jwks_url):
        from jwt_verify import verify_token

        now = int(time.time())
        token = _make_token(
            rsa_keypair,
            {
                "sub": "user-123",
                "email": "u@test.com",
                "iss": issuer,
                "aud": audience,
                "exp": now + 3600,
                "iat": now,
                "token_use": "id",
                "cognito:groups": ["users", "admins"],
            },
        )
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            claims = verify_token(token, jwks_url=jwks_url, audience=audience, issuer=issuer)
        assert claims["sub"] == "user-123"
        assert claims["email"] == "u@test.com"
        assert claims["cognito:groups"] == ["users", "admins"]

    def test_rejects_expired_token(self, rsa_keypair, issuer, audience, jwks_url):
        from jose.exceptions import ExpiredSignatureError

        from jwt_verify import verify_token

        now = int(time.time())
        token = _make_token(
            rsa_keypair,
            {
                "sub": "user-123",
                "iss": issuer,
                "aud": audience,
                "exp": now - 60,
                "iat": now - 3660,
                "token_use": "id",
            },
        )
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            with pytest.raises(ExpiredSignatureError):
                verify_token(token, jwks_url=jwks_url, audience=audience, issuer=issuer)

    def test_rejects_wrong_issuer(self, rsa_keypair, issuer, audience, jwks_url):
        from jose.exceptions import JWTClaimsError

        from jwt_verify import verify_token

        now = int(time.time())
        token = _make_token(
            rsa_keypair,
            {
                "sub": "user-123",
                "iss": "https://evil.example.com/wrong-issuer",
                "aud": audience,
                "exp": now + 3600,
                "iat": now,
                "token_use": "id",
            },
        )
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            with pytest.raises(JWTClaimsError):
                verify_token(token, jwks_url=jwks_url, audience=audience, issuer=issuer)

    def test_rejects_wrong_audience(self, rsa_keypair, issuer, audience, jwks_url):
        from jose.exceptions import JWTClaimsError

        from jwt_verify import verify_token

        now = int(time.time())
        token = _make_token(
            rsa_keypair,
            {
                "sub": "user-123",
                "iss": issuer,
                "aud": "different-client-id",
                "exp": now + 3600,
                "iat": now,
                "token_use": "id",
            },
        )
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            with pytest.raises(JWTClaimsError):
                verify_token(token, jwks_url=jwks_url, audience=audience, issuer=issuer)

    def test_rejects_missing_kid(self, rsa_keypair, issuer, audience, jwks_url):
        from jwt_verify import JWTVerifyError, verify_token

        bad_token = jwt.encode(
            {"sub": "x", "iss": issuer, "aud": audience, "exp": int(time.time()) + 3600},
            rsa_keypair["private_pem"],
            algorithm="RS256",
        )
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            with pytest.raises(JWTVerifyError):
                verify_token(bad_token, jwks_url=jwks_url, audience=audience, issuer=issuer)

    def test_rejects_unknown_kid(self, rsa_keypair, issuer, audience, jwks_url):
        from jwt_verify import JWTVerifyError, verify_token

        bad_token = jwt.encode(
            {"sub": "x", "iss": issuer, "aud": audience, "exp": int(time.time()) + 3600},
            rsa_keypair["private_pem"],
            algorithm="RS256",
            headers={"kid": "unknown-kid-999"},
        )
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            with pytest.raises(JWTVerifyError):
                verify_token(bad_token, jwks_url=jwks_url, audience=audience, issuer=issuer)

    def test_rejects_invalid_signature(self, rsa_keypair, issuer, audience, jwks_url):
        from jose import JWTError

        from jwt_verify import verify_token

        now = int(time.time())
        token = _make_token(
            rsa_keypair,
            {
                "sub": "user-123",
                "iss": issuer,
                "aud": audience,
                "exp": now + 3600,
                "iat": now,
            },
        )
        parts = token.split(".")
        tampered = ".".join([parts[0], parts[1], "AAAAAAAA" + parts[2][8:]])
        with patch("urllib.request.urlopen", return_value=_mock_jwks_response(rsa_keypair)):
            with pytest.raises(JWTError):
                verify_token(tampered, jwks_url=jwks_url, audience=audience, issuer=issuer)


class TestJwksCache:
    def test_jwks_fetched_once_for_same_url(self, rsa_keypair, issuer, audience, jwks_url):
        from jwt_verify import _load_jwks_cached, verify_token

        _load_jwks_cached.cache_clear()

        now = int(time.time())
        token = _make_token(
            rsa_keypair,
            {
                "sub": "u",
                "iss": issuer,
                "aud": audience,
                "exp": now + 3600,
                "iat": now,
            },
        )
        urlopen_mock = MagicMock(return_value=_mock_jwks_response(rsa_keypair))
        with patch("urllib.request.urlopen", urlopen_mock):
            verify_token(token, jwks_url=jwks_url, audience=audience, issuer=issuer)
            urlopen_mock.return_value = _mock_jwks_response(rsa_keypair)
            verify_token(token, jwks_url=jwks_url, audience=audience, issuer=issuer)

        assert urlopen_mock.call_count == 1
