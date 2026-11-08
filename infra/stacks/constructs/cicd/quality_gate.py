from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

from aws_cdk import Duration
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

SCRIPTS_DIR = Path(__file__).parent / "scripts"


@dataclass(frozen=True)
class QualityGateTargets:
    lint: str = "lint"
    test: str = "test"
    security: str = "security"
    synth: str | None = "synth"
    install_extras: list[str] = field(default_factory=list)


def _build_project(
    scope: Construct,
    construct_id: str,
    *,
    project_name: str,
    script_file: str,
    kms_key: kms.IKey,
    timeout_minutes: int = 20,
    compute_type: codebuild.ComputeType = codebuild.ComputeType.MEDIUM,
    env_vars: dict[str, str] | None = None,
) -> codebuild.PipelineProject:
    log_group = logs.LogGroup(
        scope,
        f"{construct_id}Logs",
        retention=logs.RetentionDays.ONE_MONTH,
        encryption_key=kms_key,
    )

    script = (SCRIPTS_DIR / script_file).read_text()
    script_b64 = base64.b64encode(script.encode()).decode()

    build_spec: dict = {
        "version": "0.2",
        "phases": {
            "install": {"runtime-versions": {"python": "3.13"}},
            "build": {
                "commands": [
                    f"echo {script_b64} | base64 -d > /tmp/run.sh",
                    "chmod +x /tmp/run.sh",
                    "/tmp/run.sh",  # nosec B108
                ]
            },
        },
    }
    if env_vars:
        build_spec["env"] = {"variables": env_vars}

    project = codebuild.PipelineProject(
        scope,
        construct_id,
        project_name=project_name,
        build_spec=codebuild.BuildSpec.from_object(build_spec),
        environment=codebuild.BuildEnvironment(
            build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2023_5,
            privileged=False,
            compute_type=compute_type,
        ),
        logging=codebuild.LoggingOptions(
            cloud_watch=codebuild.CloudWatchLoggingOptions(enabled=True, log_group=log_group),
        ),
        timeout=Duration.minutes(timeout_minutes),
        encryption_key=kms_key,
    )

    NagSuppressions.add_resource_suppressions(
        project,
        [
            {
                "id": "AwsSolutions-IAM5",
                "reason": "CodeBuild default policy includes CW Logs wildcards.",
            }
        ],
        apply_to_children=True,
    )
    return project


class QualityGate(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        kms_key: kms.IKey,
        name_prefix: str = "quality-gate",
        targets: QualityGateTargets | None = None,
    ) -> None:
        super().__init__(scope, construct_id)
        targets = targets or QualityGateTargets()

        self.lint_project = _build_project(
            self,
            "Lint",
            project_name=f"{name_prefix}-lint",
            script_file="lint.sh",
            kms_key=kms_key,
            timeout_minutes=10,
            compute_type=codebuild.ComputeType.SMALL,
            env_vars={"MAKE_TARGET": targets.lint},
        )

        self.test_project = _build_project(
            self,
            "Tests",
            project_name=f"{name_prefix}-tests",
            script_file="test.sh",
            kms_key=kms_key,
            timeout_minutes=15,
            env_vars={"MAKE_TARGET": targets.test},
        )

        self.security_project = _build_project(
            self,
            "SecurityScan",
            project_name=f"{name_prefix}-security",
            script_file="security_scan.sh",
            kms_key=kms_key,
            timeout_minutes=20,
            env_vars={"MAKE_TARGET": targets.security},
        )

        self.synth_project: codebuild.PipelineProject | None = None
        if targets.synth:
            self.synth_project = _build_project(
                self,
                "Synth",
                project_name=f"{name_prefix}-synth",
                script_file="synth.sh",
                kms_key=kms_key,
                timeout_minutes=10,
                compute_type=codebuild.ComputeType.MEDIUM,
                env_vars={"MAKE_TARGET": targets.synth},
            )
