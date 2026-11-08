from __future__ import annotations

import os

os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ZIMAGE_ENV", "dev")


def test_all_stacks_synthesize() -> None:
    import aws_cdk as cdk
    from infra.config.settings import DeploySettings
    from infra.stacks.cicd import CicdStack
    from infra.stacks.compile_sfn import CompileStateMachineStack
    from infra.stacks.inference_ecs import InferenceEcsStack
    from infra.stacks.inference_infra import InferenceInfraStack
    from infra.stacks.platform import PlatformStack
    from infra.stacks.web import WebStack

    settings = DeploySettings(env_override="dev")
    app = cdk.App()
    env = settings.cdk_env()

    platform = PlatformStack(app, "test-platform", settings=settings, env=env)
    cicd = CicdStack(app, "test-cicd", settings=settings, platform=platform, env=env)
    infra = InferenceInfraStack(
        app, "test-inference-infra", settings=settings, platform=platform, env=env
    )
    inference_ecs = InferenceEcsStack(
        app,
        "test-inference-ecs",
        settings=settings,
        platform=platform,
        cicd=cicd,
        env=env,
    )
    CompileStateMachineStack(
        app,
        "test-compile-sfn",
        settings=settings,
        platform=platform,
        cicd=cicd,
        env=env,
    )
    expected_stacks = 6

    WebStack(
        app,
        "test-web",
        settings=settings,
        platform=platform,
        infra=infra,
        findings_topic=cicd.findings_topic,
        inference_alb_url=f"http://{inference_ecs.ecs_cluster.alb_dns}",
        env=env,
    )

    assembly = app.synth()
    assert len(assembly.stacks) == expected_stacks
