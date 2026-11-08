from __future__ import annotations

import os

os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ZIMAGE_ENV", "dev")

try:
    _CDK_AVAILABLE = True
except Exception:
    _CDK_AVAILABLE = False

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CDK_TESTS_SKIP") == "1",
    reason="CDK synth tests require writable JSII cache (set CDK_TESTS_SKIP=1 to skip)",
)


def _synth_web_stack():
    import aws_cdk as cdk
    from aws_cdk import assertions
    from infra.config.settings import DeploySettings
    from infra.stacks.cicd import CicdStack
    from infra.stacks.inference_infra import InferenceInfraStack
    from infra.stacks.platform import PlatformStack
    from infra.stacks.web import WebStack

    settings = DeploySettings(env_override="dev")
    app = cdk.App()
    env = settings.cdk_env()

    platform = PlatformStack(app, "p", settings=settings, env=env)
    cicd = CicdStack(app, "c", settings=settings, platform=platform, env=env)
    infra = InferenceInfraStack(app, "i", settings=settings, platform=platform, env=env)
    web = WebStack(
        app,
        "w",
        settings=settings,
        platform=platform,
        infra=infra,
        findings_topic=cicd.findings_topic,
        env=env,
    )
    return assertions.Template.from_stack(web)


def test_websocket_connect_route_has_authorizer() -> None:
    template = _synth_web_stack()
    template.has_resource_properties(
        "AWS::ApiGatewayV2::Route",
        {
            "RouteKey": "$connect",
            "AuthorizationType": "CUSTOM",
        },
    )


def test_websocket_authorizer_lambda_exists() -> None:
    template = _synth_web_stack()
    template.has_resource_properties(
        "AWS::ApiGatewayV2::Authorizer",
        {
            "AuthorizerType": "REQUEST",
            "IdentitySource": ["route.request.querystring.ticket"],
        },
    )
