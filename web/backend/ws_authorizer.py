from __future__ import annotations

import os
from functools import lru_cache

from loguru import logger

_REGION = os.environ["ZIMAGE_REGION"]


@lru_cache(maxsize=1)
def _get_secret() -> str:
    import boto3

    secret_arn = os.environ.get("ZIMAGE_WS_TICKET_SECRET_ARN", "")
    if not secret_arn:
        raise RuntimeError("ZIMAGE_WS_TICKET_SECRET_ARN not set")
    client = boto3.client("secretsmanager")
    return client.get_secret_value(SecretId=secret_arn)["SecretString"]


def _policy(principal_id: str, effect: str, resource: str, context: dict | None = None) -> dict:
    doc = {
        "principalId": principal_id,
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [{"Action": "execute-api:Invoke", "Effect": effect, "Resource": resource}],
        },
    }
    if context:
        doc["context"] = context
    return doc


def handler(event: dict, _context) -> dict:
    method_arn = event.get("methodArn", "*")
    qs = event.get("queryStringParameters") or {}
    ticket = qs.get("ticket")

    if not ticket:
        logger.info("WS authorizer: no ticket, denying")
        return _policy("anonymous", "Deny", method_arn)

    try:
        from jose import jwt

        secret = _get_secret()
        claims = jwt.decode(ticket, secret, algorithms=["HS256"])
        if claims.get("purpose") != "ws-connect":
            raise ValueError("invalid ticket purpose")
    except Exception as e:
        logger.warning(f"WS authorizer: ticket rejected: {type(e).__name__}: {e}")
        return _policy("anonymous", "Deny", method_arn)

    sub = claims.get("sub", "unknown")
    groups = claims.get("groups", [])
    context = {
        "sub": sub,
        "email": claims.get("email", ""),
        "groups": ",".join(groups) if isinstance(groups, list) else str(groups),
    }
    logger.info(f"WS authorizer: allowing sub={sub}")
    return _policy(sub, "Allow", method_arn, context=context)
