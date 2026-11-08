from __future__ import annotations

import json
import urllib.request
from functools import lru_cache

from jose import jwt


class JWTVerifyError(Exception):
    pass


@lru_cache(maxsize=4)
def _load_jwks_cached(jwks_url: str) -> dict:
    if not jwks_url.startswith("https://"):
        raise ValueError("JWKS URL must use HTTPS")
    # nosec B310: HTTPS scheme enforced above.
    with urllib.request.urlopen(jwks_url, timeout=5) as resp:  # nosec B310
        return json.loads(resp.read())


def verify_token(token: str, *, jwks_url: str, audience: str, issuer: str) -> dict:
    try:
        header = jwt.get_unverified_header(token)
    except Exception as e:
        raise JWTVerifyError(f"Failed to read token header: {e}") from e

    kid = header.get("kid")
    if not kid:
        raise JWTVerifyError("Token header missing 'kid'")

    jwks = _load_jwks_cached(jwks_url)
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if not key:
        raise JWTVerifyError(f"No JWKS key matches kid={kid}")

    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=audience,
        issuer=issuer,
        options={
            "verify_at_hash": False,
        },
    )
