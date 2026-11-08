from __future__ import annotations

import base64
from pathlib import Path

from aws_cdk import Duration
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

SCRIPTS_DIR = Path(__file__).parent / "scripts"


class CompileModelsProject(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        kms_key: kms.IKey,
        prefix: str,
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=kms_key,
        )

        script = (SCRIPTS_DIR / "compile_models.sh").read_text()
        script_b64 = base64.b64encode(script.encode()).decode()

        self.project = codebuild.PipelineProject(
            self,
            "Project",
            project_name=f"{prefix}-compile-models",
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "env": {"variables": {"ZIMG_PREFIX": prefix}},
                    "phases": {
                        "install": {"runtime-versions": {"python": "3.11"}},
                        "build": {
                            "commands": [
                                f"echo {script_b64} | base64 -d > /tmp/compile-models.sh",
                                "chmod +x /tmp/compile-models.sh",
                                "/tmp/compile-models.sh",  # nosec B108
                            ]
                        },
                    },
                }
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2023_5,
                privileged=False,
                compute_type=codebuild.ComputeType.SMALL,
            ),
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(enabled=True, log_group=log_group),
            ),
            timeout=Duration.minutes(195),
            encryption_key=kms_key,
        )

        self.sfn_policy = iam.ManagedPolicy(
            self,
            "SfnPolicy",
            managed_policy_name=f"{prefix}-compile-models-sfn",
            statements=[
                iam.PolicyStatement(
                    actions=["states:StartExecution", "states:DescribeExecution"],
                    resources=[f"arn:aws:states:*:*:stateMachine:{prefix}-compile-sfn"],
                ),
                iam.PolicyStatement(
                    actions=["states:DescribeExecution"],
                    resources=[f"arn:aws:states:*:*:execution:{prefix}-compile-sfn:*"],
                ),
                iam.PolicyStatement(
                    actions=["s3:HeadObject", "s3:GetObject"],
                    resources=[
                        f"arn:aws:s3:::{prefix}-artifacts-*",
                        f"arn:aws:s3:::{prefix}-artifacts-*/compile/*",
                    ],
                ),
            ],
        )
        self.project.role.add_managed_policy(self.sfn_policy)

        NagSuppressions.add_resource_suppressions(
            self.project,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SFN execution ARN includes dynamic name suffix. "
                    "S3 bucket name includes account/region.",
                },
            ],
            apply_to_children=True,
        )
