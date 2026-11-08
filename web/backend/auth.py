from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel

from config import settings


class Role(StrEnum):
    USER = "users"
    ADMIN = "admins"


class TokenClaims(BaseModel):
    sub: str
    email: str
    groups: list[str] = []


_LOCAL_ADMIN = TokenClaims(sub="local-dev", email="admin@localhost", groups=["users", "admins"])


def _is_local_bypass_safe() -> bool:
    return settings.local and not settings.is_lambda


def _decode_token(token: str) -> TokenClaims:
    from jwt_verify import JWTVerifyError, verify_token

    if not settings.cognito_user_pool_id or not settings.cognito_app_client_id:
        raise HTTPException(status_code=401, detail="Auth not configured")

    region = settings.region
    pool_id = settings.cognito_user_pool_id
    jwks_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json"
    issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"

    try:
        claims = verify_token(
            token,
            jwks_url=jwks_url,
            audience=settings.cognito_app_client_id,
            issuer=issuer,
        )
    except (JWTVerifyError, Exception):
        raise HTTPException(status_code=401, detail="Token verification failed") from None

    if claims.get("token_use") not in ("id", "access"):
        raise HTTPException(status_code=401, detail="Invalid token type")

    groups = claims.get("cognito:groups", [])
    return TokenClaims(
        sub=claims.get("sub", ""),
        email=claims.get("email", ""),
        groups=groups if isinstance(groups, list) else [],
    )


def _claims_from_request_scope(request: Request) -> TokenClaims | None:
    aws_event = request.scope.get("aws.event")
    if not isinstance(aws_event, dict):
        return None
    rc = aws_event.get("requestContext") or {}
    authz = rc.get("authorizer") or {}
    jwt_claims = (authz.get("jwt") or {}).get("claims") or {}
    if not jwt_claims:
        return None
    groups = jwt_claims.get("cognito:groups", [])
    if isinstance(groups, str):
        import re

        groups = [g for g in re.split(r"[,\s]+", groups.strip("[]")) if g]
    return TokenClaims(
        sub=jwt_claims.get("sub", ""),
        email=jwt_claims.get("email", ""),
        groups=groups,
    )


async def get_current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> TokenClaims:
    if _is_local_bypass_safe():
        return _LOCAL_ADMIN

    scope_claims = _claims_from_request_scope(request)
    if scope_claims is not None:
        return scope_claims

    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.removeprefix("Bearer ").strip()
    return _decode_token(token)


def require_role(role: Role):

    async def _check(user: Annotated[TokenClaims, Depends(get_current_user)]) -> TokenClaims:
        if _is_local_bypass_safe():
            return _LOCAL_ADMIN
        if role.value not in user.groups:
            raise HTTPException(
                status_code=403,
                detail=f"Requires {role.value} role",
            )
        return user

    return _check


async def ws_authenticate(token: str | None = Query(default=None)) -> TokenClaims:
    if not token:
        raise HTTPException(status_code=401, detail="Token required for WebSocket")
    return _decode_token(token)
