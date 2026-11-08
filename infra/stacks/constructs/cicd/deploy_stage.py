from __future__ import annotations

from aws_cdk import Aws, Duration
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from constructs import Construct

from infra.utils.constants import CDK_BOOTSTRAP_QUALIFIER


class DeployProject(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        kms_key: kms.IKey,
        account: str,
        prefix: str = "dev-zimg",
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=kms_key,
        )

        deploy_and_resolve = (
            'resolve_tag() { aws ecr describe-images --repository-name "$1" '
            "--query 'imageDetails | sort_by(@, &imagePushedAt) | "
            "[-1].imageTags[?@ != `latest`] | [0]' --output text 2>/dev/null || echo latest; }; "
            f'TAG_512=$(resolve_tag "{prefix}-serving-512"); '
            f'TAG_COMPILER=$(resolve_tag "{prefix}-compiler"); '
            'echo "[deploy] image tags: 512=$TAG_512 compiler=$TAG_COMPILER"; '
            f'cdk deploy "{prefix}-inference-infra" "{prefix}-compile-sfn"'
            " --require-approval never --exclusively --concurrency 1"
            ' --context "image_tag_512=$TAG_512"'
            ' --context "image_tag_compiler=$TAG_COMPILER"'
        )

        deploy_cmds = [
            "set -euo pipefail",
            "npm install -g aws-cdk@2.1130.0",
            "pip install --quiet uv==0.7.12",
            "uv sync --frozen --quiet",
            "source .venv/bin/activate",
            "cd $CODEBUILD_SRC_DIR",
            f"export CDK_DEFAULT_ACCOUNT={account}",
            f"export CDK_DEFAULT_REGION={Aws.REGION}",
            deploy_and_resolve,
        ]

        self.project = codebuild.PipelineProject(
            self,
            "Project",
            project_name=f"{prefix}-deploy-inference",
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "install": {"runtime-versions": {"python": "3.13", "nodejs": "20"}},
                        "build": {"commands": deploy_cmds},
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
            timeout=Duration.minutes(90),
            encryption_key=kms_key,
        )

        self.cdk_deploy_policy = iam.ManagedPolicy(
            self,
            "CdkDeployPolicy",
            managed_policy_name=f"{prefix}-deploy-cdk-assume-roles",
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
                iam.PolicyStatement(
                    actions=["iam:CreateServiceLinkedRole"],
                    resources=[
                        f"arn:aws:iam::{account}:role/aws-service-role/batch.amazonaws.com/*",
                    ],
                ),
            ],
        )
        self.project.role.add_managed_policy(self.cdk_deploy_policy)
