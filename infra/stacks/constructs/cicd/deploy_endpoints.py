from __future__ import annotations

import base64
from pathlib import Path

from aws_cdk import Aws, Duration
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.utils.constants import CDK_BOOTSTRAP_QUALIFIER

SCRIPTS_DIR = Path(__file__).parent / "scripts"


class DeployEndpointsProject(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        kms_key: kms.IKey,
        account: str,
        endpoint_stack_short: str,
        prefix: str = "dev-zimg",
        environment: str = "dev",
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=kms_key,
        )

        script = (SCRIPTS_DIR / "deploy_endpoints.sh").read_text()
        script_b64 = base64.b64encode(script.encode()).decode()

        self.project = codebuild.PipelineProject(
            self,
            "Project",
            project_name=f"{prefix}-deploy-{endpoint_stack_short}",
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "env": {
                        "variables": {
                            "ZIMG_PREFIX": prefix,
                            "ZIMG_ENDPOINT_STACK": endpoint_stack_short,
                            "AWS_ACCOUNT_ID": account,
                            "DEPLOY_REGION": Aws.REGION,
                            "ZIMAGE_ENV": environment,
                        }
                    },
                    "phases": {
                        "install": {"runtime-versions": {"python": "3.13", "nodejs": "20"}},
                        "build": {
                            "commands": [
                                f"echo {script_b64} | base64 -d > /tmp/deploy-endpoint.sh",
                                "chmod +x /tmp/deploy-endpoint.sh",
                                "/tmp/deploy-endpoint.sh",  # nosec B108
                            ]
                        },
                    },
                }
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2023_5,
                privileged=False,
                compute_type=codebuild.ComputeType.MEDIUM,
            ),
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(enabled=True, log_group=log_group),
            ),
            timeout=Duration.minutes(120),
            encryption_key=kms_key,
        )

        self.cdk_deploy_policy = iam.ManagedPolicy(
            self,
            "CdkDeployPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["sts:AssumeRole"],
                    resources=[
                        f"arn:aws:iam::{account}:role/cdk-{CDK_BOOTSTRAP_QUALIFIER}-deploy-role-{account}-{Aws.REGION}",
                        f"arn:aws:iam::{account}:role/cdk-{CDK_BOOTSTRAP_QUALIFIER}-file-publishing-role-{account}-{Aws.REGION}",
                        f"arn:aws:iam::{account}:role/cdk-{CDK_BOOTSTRAP_QUALIFIER}-image-publishing-role-{account}-{Aws.REGION}",
                        f"arn:aws:iam::{account}:role/cdk-{CDK_BOOTSTRAP_QUALIFIER}-lookup-role-{account}-{Aws.REGION}",
                        f"arn:aws:iam::{account}:role/cdk-{CDK_BOOTSTRAP_QUALIFIER}-cfn-exec-role-{account}-{Aws.REGION}",
                    ],
                ),
                iam.PolicyStatement(
                    actions=[
                        "cloudformation:DescribeStacks",
                        "cloudformation:ListStacks",
                        "cloudformation:GetTemplate",
                        "cloudformation:DescribeStackEvents",
                    ],
                    resources=[f"arn:aws:cloudformation:{Aws.REGION}:{account}:stack/*/*"],
                ),
                iam.PolicyStatement(
                    actions=["ssm:GetParameter"],
                    resources=[f"arn:aws:ssm:{Aws.REGION}:{account}:parameter/cdk-bootstrap/*"],
                ),
                iam.PolicyStatement(
                    actions=["ecr:DescribeImages"],
                    resources=[f"arn:aws:ecr:{Aws.REGION}:{account}:repository/{prefix}-*"],
                ),
            ],
        )
        self.project.role.add_managed_policy(self.cdk_deploy_policy)

        NagSuppressions.add_resource_suppressions(
            self.project,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CDK deploy needs sts:AssumeRole for all 5 bootstrap roles. "
                    "CFN describe/SSM read are scoped to stack/parameter ARNs.",
                },
            ],
            apply_to_children=True,
        )
