from __future__ import annotations

import os
from functools import lru_cache

import boto3


@lru_cache(maxsize=1)
def get_ws_ticket_secret() -> str:
    secret_arn = os.environ.get("ZIMAGE_WS_TICKET_SECRET_ARN", "")
    if not secret_arn:
        raise RuntimeError("ZIMAGE_WS_TICKET_SECRET_ARN not set")
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_arn)
    return response["SecretString"]
