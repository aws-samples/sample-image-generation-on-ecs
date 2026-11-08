from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError


@pytest.fixture
def resolver(monkeypatch):
    repo_root = Path(__file__).resolve().parents[4]
    lambda_dir = repo_root / "infra" / "lambdas"
    monkeypatch.syspath_prepend(str(lambda_dir))
    if "ecr_digest_resolver" in sys.modules:
        del sys.modules["ecr_digest_resolver"]
    return importlib.import_module("ecr_digest_resolver")


def _create_event(repo: str = "dev-zimg-serving-768", tag: str = "latest") -> dict:
    return {
        "RequestType": "Create",
        "RequestId": "test-request-id",
        "ResponseURL": "https://example.invalid/cfn",
        "StackId": "arn:aws:cloudformation:us-east-1:111111111111:stack/test/uuid",
        "LogicalResourceId": "TestCR",
        "ResourceProperties": {
            "ServiceToken": "arn:aws:lambda:...",
            "RepositoryName": repo,
            "ImageTag": tag,
        },
    }


def test_returns_digest_uri_on_success(resolver, monkeypatch):
    fake_ecr = MagicMock()
    fake_ecr.batch_get_image.return_value = {
        "images": [
            {
                "imageId": {
                    "imageTag": "latest",
                    "imageDigest": "sha256:abc123",
                }
            }
        ],
        "failures": [],
    }
    fake_sts = MagicMock()
    fake_sts.get_caller_identity.return_value = {"Account": "111111111111"}

    def fake_client(name):
        return {"ecr": fake_ecr, "sts": fake_sts}[name]

    monkeypatch.setattr(resolver.boto3, "client", fake_client)
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    out = resolver.handler(_create_event(), context=None)

    assert out["Data"]["Digest"] == "sha256:abc123"
    assert out["Data"]["ImageUri"] == (
        "111111111111.dkr.ecr.us-east-1.amazonaws.com/dev-zimg-serving-768@sha256:abc123"
    )
    assert "@sha256:abc123" in out["PhysicalResourceId"]


def test_propagates_access_denied(resolver, monkeypatch):
    fake_ecr = MagicMock()
    fake_ecr.batch_get_image.side_effect = ClientError(
        error_response={
            "Error": {
                "Code": "AccessDeniedException",
                "Message": ("User ... is not authorized to perform: ecr:BatchGetImage"),
            }
        },
        operation_name="BatchGetImage",
    )
    monkeypatch.setattr(resolver.boto3, "client", lambda name: fake_ecr)

    with pytest.raises(ClientError) as excinfo:
        resolver.handler(_create_event(), context=None)

    assert "AccessDeniedException" in str(excinfo.value)


def test_propagates_image_not_found(resolver, monkeypatch):
    fake_ecr = MagicMock()
    fake_ecr.batch_get_image.return_value = {
        "images": [],
        "failures": [{"failureCode": "ImageNotFound", "failureReason": "not found"}],
    }
    monkeypatch.setattr(resolver.boto3, "client", lambda name: fake_ecr)

    with pytest.raises(RuntimeError, match="ECR image not found"):
        resolver.handler(_create_event(), context=None)


def test_delete_request_is_noop(resolver, monkeypatch):
    fake_ecr = MagicMock()
    monkeypatch.setattr(resolver.boto3, "client", lambda name: fake_ecr)

    event = _create_event()
    event["RequestType"] = "Delete"
    event["PhysicalResourceId"] = "previous-uri"

    out = resolver.handler(event, context=None)

    assert out["PhysicalResourceId"] == "previous-uri"
    assert out["Data"] == {}
    fake_ecr.batch_get_image.assert_not_called()
