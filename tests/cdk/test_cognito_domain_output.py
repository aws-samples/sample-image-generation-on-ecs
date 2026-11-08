from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")

WEB_TEMPLATE = Path(__file__).resolve().parents[2] / "cdk.out" / "dev-zimg-web.template.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("CDK_TESTS_SKIP") == "1",
    reason="CDK synth tests require writable JSII cache (set CDK_TESTS_SKIP=1 to skip)",
)


def _domain_logical_id(resources: dict) -> str:
    ids = [k for k, v in resources.items() if v["Type"] == "AWS::Cognito::UserPoolDomain"]
    assert len(ids) == 1, ids
    return ids[0]


def _assert_fqdn_join(value: object, domain_id: str) -> None:
    assert isinstance(value, dict), value
    assert "Fn::Join" in value, value
    sep, parts = value["Fn::Join"]
    assert sep == ""
    assert parts[0] == {"Ref": domain_id}
    assert len(parts) == 2, parts
    assert isinstance(parts[1], str), parts
    assert parts[1].startswith(".auth.")
    assert parts[1].endswith(".amazoncognito.com")


def test_cognito_auth_domain_fqdn_references_user_pool_domain() -> None:
    import aws_cdk as cdk
    from infra.stacks.constructs.web.cognito import CognitoAuth

    app = cdk.App()
    stack = cdk.Stack(app, "s", env=cdk.Environment(account="123456789012", region="us-east-1"))
    auth = CognitoAuth(
        stack, "Auth", prefix="dev-zimg", admin_email="a@example.com", domain_prefix="x-auth"
    )
    cdk.CfnOutput(stack, "CognitoDomain", value=auth.domain_fqdn)
    template = cdk.assertions.Template.from_stack(stack).to_json()

    domain_id = _domain_logical_id(template["Resources"])
    value = template["Outputs"]["CognitoDomain"]["Value"]
    _assert_fqdn_join(value, domain_id)
    assert value["Fn::Join"][1][1] == ".auth.us-east-1.amazoncognito.com"


@pytest.mark.skipif(not WEB_TEMPLATE.exists(), reason="run `make synth` first")
def test_synth_web_output_and_lambda_env_use_domain_resource() -> None:
    template = json.loads(WEB_TEMPLATE.read_text())
    resources = template["Resources"]
    domain_id = _domain_logical_id(resources)

    output_value = template["Outputs"]["CognitoDomain"]["Value"]
    _assert_fqdn_join(output_value, domain_id)

    env_values = [
        r["Properties"]["Environment"]["Variables"]["ZIMAGE_COGNITO_DOMAIN"]
        for r in resources.values()
        if r["Type"] == "AWS::Lambda::Function"
        and "ZIMAGE_COGNITO_DOMAIN"
        in r.get("Properties", {}).get("Environment", {}).get("Variables", {})
    ]
    assert len(env_values) == 1, env_values
    assert env_values[0] == output_value
