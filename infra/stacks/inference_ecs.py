from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_ecr as ecr
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.config.settings import DeploySettings
from infra.stacks.cicd import CicdStack
from infra.stacks.constructs.compute.inferentia_endpoint import InferentiaCluster
from infra.stacks.platform import PlatformStack
from infra.utils.enums import BuildTarget


class InferenceEcsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: DeploySettings,
        platform: PlatformStack,
        cicd: CicdStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = settings.prefix

        ecr_repos = {
            512: cicd.inference_repo.repository,
            768: cicd.inference_repo_768.repository,
            1024: cicd.inference_repo_1024.repository,
        }

        gpu_repo = (
            ecr.Repository.from_repository_name(
                self, "InferenceGpuRepoRef", f"{prefix}-{BuildTarget.SERVING_GPU}"
            )
            if settings.gpu_endpoint.enabled
            else None
        )

        self.ecs_cluster = InferentiaCluster(
            self,
            "EcsInference",
            vpc=platform.vpc,
            private_subnets=platform.private_subnets,
            config=settings.inferentia_endpoint,
            ecr_repos=ecr_repos,
            gpu_repo=gpu_repo,
            gpu_config=settings.gpu_endpoint if settings.gpu_endpoint.enabled else None,
            efs_filesystem=platform.efs.filesystem,
            efs_ro_access_point=platform.efs.ro_access_point,
            efs_security_group=platform.efs.security_group,
            access_logs_bucket=platform.elb_access_logs_bucket,
            prefix=prefix,
        )

        CfnOutput(
            self,
            "InferenceAlbDns",
            value=self.ecs_cluster.alb_dns,
            description="Internal ALB DNS for ECS inference",
        )
        CfnOutput(
            self,
            "InferenceAlbUrl",
            value=f"http://{self.ecs_cluster.alb_dns}",
        )

        NagSuppressions.add_stack_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ECS instance roles require wildcard for ecs:Submit*, "
                    "ecr:GetAuthorizationToken, and S3 object-level access.",
                    "appliesTo": [
                        "Resource::*",
                        "Action::ecs:Submit*",
                        "Resource::arn:aws:ecs:<AWS::Region>:<AWS::AccountId>:cluster/*",
                        "Resource::arn:aws:logs:<AWS::Region>:<AWS::AccountId>:log-group:/ecs/*",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AWS-managed policies for ECS instance role, SSM, and CDK drain hook.",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                        "service-role/AmazonEC2ContainerServiceforEC2Role",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore",
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                        "service-role/AWSLambdaBasicExecutionRole",
                    ],
                },
                {
                    "id": "AwsSolutions-EC23",
                    "reason": "Security group egress is required for S3/ECR/NAT.",
                },
                {
                    "id": "AwsSolutions-AS3",
                    "reason": "ASG scaling notifications not needed for "
                    "single-task-per-instance inference fleet.",
                },
                {
                    "id": "AwsSolutions-ECS2",
                    "reason": "Neuron env vars are non-sensitive compile flags; "
                    "Secrets Manager adds unnecessary latency at cold start.",
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "CDK-managed drain hook Lambda runtime is auto-selected.",
                },
                {
                    "id": "AwsSolutions-SNS3",
                    "reason": "CDK-managed ASG drain hook SNS topic is internal.",
                },
            ],
        )
