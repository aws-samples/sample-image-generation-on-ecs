from __future__ import annotations

import os

os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ZIMAGE_ENV", "dev")


def _synth_compile_sfn_template():
    import aws_cdk as cdk
    from aws_cdk import assertions
    from infra.config.settings import DeploySettings
    from infra.stacks.cicd import CicdStack
    from infra.stacks.compile_sfn import CompileStateMachineStack
    from infra.stacks.platform import PlatformStack

    settings = DeploySettings(env_override="dev")
    app = cdk.App()
    env = settings.cdk_env()
    platform = PlatformStack(app, "p", settings=settings, env=env)
    cicd = CicdStack(app, "c", settings=settings, platform=platform, env=env)
    sfn_stack = CompileStateMachineStack(
        app, "s", settings=settings, platform=platform, cicd=cicd, env=env
    )
    return assertions.Template.from_stack(sfn_stack)


def _flatten_statements(template):
    for resource_type in ("AWS::IAM::ManagedPolicy", "AWS::IAM::Policy"):
        for body in template.find_resources(resource_type).values():
            doc = body.get("Properties", {}).get("PolicyDocument", {})
            yield from doc.get("Statement", [])


def _resource_strings(stmt) -> list[str]:
    import json

    res = stmt.get("Resource", [])
    if not isinstance(res, list):
        res = [res]
    out = []
    for r in res:
        if isinstance(r, str):
            out.append(r)
        else:
            out.append(json.dumps(r))
    return out


def test_cache_purge_has_efs_access():
    from aws_cdk.assertions import Match

    template = _synth_compile_sfn_template()
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "FunctionName": Match.string_like_regexp(".*cache-purge.*"),
            "FileSystemConfigs": Match.any_value(),
        },
    )
