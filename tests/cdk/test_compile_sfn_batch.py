from __future__ import annotations

import os

os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ZIMAGE_ENV", "dev")


def _compute_envs(*, inferentia_enabled: bool, gpu_enabled: bool) -> list[dict]:
    import aws_cdk as cdk
    from aws_cdk import assertions
    from infra.config.settings import DeploySettings
    from infra.stacks.cicd import CicdStack
    from infra.stacks.compile_sfn import CompileStateMachineStack
    from infra.stacks.platform import PlatformStack

    settings = DeploySettings(env_override="dev")
    settings.inferentia_endpoint.enabled = inferentia_enabled
    settings.gpu_endpoint.enabled = gpu_enabled
    app = cdk.App()
    env = settings.cdk_env()
    platform = PlatformStack(app, "p", settings=settings, env=env)
    cicd = CicdStack(app, "c", settings=settings, platform=platform, env=env)
    sfn_stack = CompileStateMachineStack(
        app, "s", settings=settings, platform=platform, cicd=cicd, env=env
    )
    template = assertions.Template.from_stack(sfn_stack)
    ces = template.find_resources("AWS::Batch::ComputeEnvironment")
    return [body["Properties"]["ComputeResources"] for body in ces.values()]


def test_gpu_only_creates_no_inf2_compute_env():
    resources = _compute_envs(inferentia_enabled=False, gpu_enabled=True)
    assert len(resources) == 1
    for res in resources:
        assert not any("inf2" in t for t in res["InstanceTypes"])
        assert res.get("MinvCpus", 0) == 0


def test_inferentia_enabled_creates_inf2_warmup_compute_env():
    resources = _compute_envs(inferentia_enabled=True, gpu_enabled=False)
    assert len(resources) == 2
    assert ["inf2.8xlarge"] in [res["InstanceTypes"] for res in resources]
