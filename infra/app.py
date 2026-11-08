#!/usr/bin/env python3

from __future__ import annotations

import warnings

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from infra.config.settings import DeploySettings
from infra.nag.suppressions import apply_nag_suppressions
from infra.stacks.cicd import CicdStack
from infra.stacks.compile_sfn import CompileStateMachineStack
from infra.stacks.inference_ecs import InferenceEcsStack
from infra.stacks.inference_infra import InferenceInfraStack
from infra.stacks.platform import PlatformStack
from infra.stacks.web import WebStack
from infra.utils.naming import ResourceNamer
from infra.utils.tagging import build_tags


def main() -> None:
    warnings.filterwarnings("ignore", message="Typeguard cannot check")

    settings = DeploySettings()
    namer = ResourceNamer(settings.environment)
    app = cdk.App()
    env = settings.cdk_env()

    platform = PlatformStack(app, namer.stack_name("platform"), settings=settings, env=env)
    cicd = CicdStack(app, namer.stack_name("cicd"), settings=settings, platform=platform, env=env)
    infra = InferenceInfraStack(
        app,
        namer.stack_name("inference-infra"),
        settings=settings,
        platform=platform,
        env=env,
    )
    inference_ecs = InferenceEcsStack(
        app,
        namer.stack_name("inference-ecs"),
        settings=settings,
        platform=platform,
        cicd=cicd,
        env=env,
    )
    compile_sfn = CompileStateMachineStack(
        app,
        namer.stack_name("compile-sfn"),
        settings=settings,
        platform=platform,
        cicd=cicd,
        env=env,
    )

    web = WebStack(
        app,
        namer.stack_name("web"),
        settings=settings,
        platform=platform,
        infra=infra,
        findings_topic=cicd.findings_topic,
        inference_alb_url=f"http://{inference_ecs.ecs_cluster.alb_dns}",
        inference_alb_sg=inference_ecs.ecs_cluster.alb_security_group,
        env=env,
    )

    tags = build_tags(settings.environment, settings.tags)
    all_stacks = [
        platform,
        cicd,
        infra,
        compile_sfn,
        inference_ecs,
        web,
    ]
    for stack in all_stacks:
        for key, value in tags.items():
            cdk.Tags.of(stack).add(key, value)

    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    apply_nag_suppressions(
        platform,
        cicd,
        infra,
        compile_sfn=compile_sfn,
    )

    app.synth()


if __name__ == "__main__":
    main()
