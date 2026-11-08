from __future__ import annotations

from aws_cdk import Duration
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.utils.constants import AWS_DLC_ACCOUNT_ID
from infra.utils.enums import BuildTarget


class ImageBuildProject(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        target_name: str,
        repo: ecr.IRepository,
        kms_key: kms.IKey,
        base_image: str | None = None,
        resolution: int | None = None,
        is_inference: bool = False,
        prefix: str = "dev-zimg",
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=kms_key,
        )

        build_cmds = self._build_commands(
            target_name=target_name,
            base_image=base_image,
            resolution=resolution,
            is_inference=is_inference,
        )

        env_vars = {
            "ECR_REGISTRY": f"{repo.env.account}.dkr.ecr.{repo.env.region}.amazonaws.com",
            "ECR_REPO": repo.repository_name,
        }

        self.project = codebuild.PipelineProject(
            self,
            "Project",
            project_name=f"{prefix}-{target_name}-build",
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "env": {"variables": env_vars},
                    "phases": {"build": {"commands": build_cmds}},
                }
            ),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2023_5,
                privileged=True,
                compute_type=codebuild.ComputeType.LARGE,
            ),
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(enabled=True, log_group=log_group),
            ),
            timeout=Duration.minutes(60),
            encryption_key=kms_key,
        )

        self.ecr_push_policy = iam.ManagedPolicy(
            self,
            "EcrPushPolicy",
            managed_policy_name=f"{prefix}-{target_name}-build-ecr-push",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:CompleteLayerUpload",
                        "ecr:InitiateLayerUpload",
                        "ecr:PutImage",
                        "ecr:UploadLayerPart",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    resources=[repo.repository_arn],
                ),
                iam.PolicyStatement(
                    sid="PullDlcBaseImages",
                    actions=[
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    resources=[
                        f"arn:aws:ecr:*:{AWS_DLC_ACCOUNT_ID}:repository/pytorch-inference-neuronx",
                        f"arn:aws:ecr:*:{AWS_DLC_ACCOUNT_ID}:repository/pytorch-inference",
                    ],
                ),
                iam.PolicyStatement(
                    actions=["ecr:GetAuthorizationToken"],
                    resources=["*"],
                ),
            ],
        )
        self.project.role.add_managed_policy(self.ecr_push_policy)

        self.s3_read_policy: iam.ManagedPolicy | None = None

        NagSuppressions.add_resource_suppressions(
            self.project,
            [
                {
                    "id": "AwsSolutions-CB3",
                    "reason": "docker-in-docker requires privileged mode.",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CW Logs wildcards from CodeBuild default.",
                },
            ],
            apply_to_children=True,
        )

    @staticmethod
    def _build_commands(
        *,
        target_name: str,
        base_image: str | None,
        resolution: int | None,
        is_inference: bool,
    ) -> list[str]:
        try:
            target = BuildTarget(target_name)
            dockerfile = target.dockerfile
            context_dir = target.context_dir
        except ValueError:
            dockerfile = "runtime/Dockerfile.inference"
            context_dir = "runtime/"

        cmds = [
            "set -euo pipefail",
            'aws ecr get-login-password --region "$AWS_REGION" | '
            'docker login --username AWS --password-stdin "$ECR_REGISTRY"',
            'aws ecr get-login-password --region "$AWS_REGION" | '
            f'docker login --username AWS --password-stdin {AWS_DLC_ACCOUNT_ID}.dkr.ecr."$AWS_REGION".amazonaws.com',
            "cd $CODEBUILD_SRC_DIR",
            "export SHORT_SHA=${CODEBUILD_BUILD_ID##*:}; export SHORT_SHA=${SHORT_SHA:0:12}",
        ]

        build_args = []
        if base_image:
            build_args.append(f'--build-arg BASE_IMAGE="{base_image}"')
            cmds.append(f'docker pull "{base_image}"')
        if resolution is not None:
            build_args.append(f"--build-arg RESOLUTION={resolution}")

        cmds.append(
            f"docker build -f {dockerfile} {' '.join(build_args)} "
            f"-t $ECR_REGISTRY/$ECR_REPO:$SHORT_SHA {context_dir}"
        )
        cmds += [
            "docker push $ECR_REGISTRY/$ECR_REPO:$SHORT_SHA",
        ]
        return cmds
