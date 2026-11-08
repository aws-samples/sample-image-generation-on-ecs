from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    has_aws = bool(os.environ.get("AWS_ACCOUNT_ID") or os.environ.get("CDK_DEFAULT_ACCOUNT"))
    for item in items:
        if "aws" in item.keywords and not has_aws:
            item.add_marker(pytest.mark.skip(reason="needs AWS creds/env"))


@pytest.fixture(scope="session", autouse=True)
def _set_cdk_env():
    os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
    os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")
    return
