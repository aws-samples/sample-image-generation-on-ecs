from __future__ import annotations

from pathlib import Path

from aws_cdk import Aws, Duration, RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

WEB_BACKEND_DIR = Path(__file__).resolve().parents[4] / "web" / "backend"


class ApiLambda(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        region: str,
        gallery_bucket_name: str,
        gallery_table_name: str,
        cognito_user_pool_id: str,
        cognito_app_client_id: str,
        cognito_domain: str,
        websocket_url: str = "",
        inference_alb_url: str = "",
        deps_layer: lambda_.ILayerVersion,
        vpc: ec2.IVpc | None = None,
        vpc_subnets: list[ec2.ISubnet] | None = None,
        kms_key_arn: str = "",
        gpu_enabled: bool = True,
        inferentia_enabled: bool = True,
        gpu_max_pixels: int = 1048576,
        gpu_min_dimension: int = 256,
        gpu_dimension_step: int = 8,
        ws_ticket_secret_arn: str = "",
    ) -> None:
        super().__init__(scope, construct_id)

        from infra.stacks.constructs.web.lambda_policies import lambda_execution_policies

        log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/aws/lambda/{prefix}-web-api",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.role = iam.Role(
            self,
            "Role",
            role_name=f"{prefix}-web-api-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=lambda_execution_policies(self, log_group, vpc),
        )

        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "elasticloadbalancing:DescribeTargetGroups",
                    "elasticloadbalancing:DescribeTargetHealth",
                ],
                resources=["*"],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecs:DescribeServices",
                    "ecs:UpdateService",
                    "ecs:ListTasks",
                ],
                resources=[
                    f"arn:aws:ecs:{Aws.REGION}:{Aws.ACCOUNT_ID}:service/{prefix}-inference-cluster/*",
                    f"arn:aws:ecs:{Aws.REGION}:{Aws.ACCOUNT_ID}:cluster/{prefix}-*",
                ],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
                resources=[
                    f"arn:aws:s3:::{gallery_bucket_name}",
                    f"arn:aws:s3:::{gallery_bucket_name}/*",
                ],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:DescribeAlarms", "cloudwatch:GetMetricData"],
                resources=["*"],
            )
        )
        if kms_key_arn:
            self.role.add_to_policy(
                iam.PolicyStatement(
                    actions=["kms:Decrypt", "kms:GenerateDataKey", "kms:DescribeKey"],
                    resources=[kms_key_arn],
                )
            )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "dynamodb:DeleteItem",
                    "dynamodb:UpdateItem",
                ],
                resources=[
                    f"arn:aws:dynamodb:{region}:*:table/{gallery_table_name}",
                    f"arn:aws:dynamodb:{region}:*:table/{gallery_table_name}/index/*",
                ],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "states:DescribeExecution",
                    "states:ListExecutions",
                    "states:StartExecution",
                    "states:StopExecution",
                ],
                resources=[f"arn:aws:states:{region}:*:stateMachine:{prefix}-compile-sfn"],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["states:DescribeExecution", "states:StopExecution"],
                resources=[f"arn:aws:states:{region}:*:execution:{prefix}-compile-sfn:*"],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["codepipeline:GetPipelineState"],
                resources=[f"arn:aws:codepipeline:{region}:*:{prefix}-inference-pipeline"],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "application-autoscaling:RegisterScalableTarget",
                    "application-autoscaling:PutScalingPolicy",
                    "application-autoscaling:DescribeScalableTargets",
                ],
                resources=["*"],
            )
        )

        fn_kwargs: dict = {
            "function_name": f"{prefix}-web-api",
            "runtime": lambda_.Runtime.PYTHON_3_13,
            "handler": "lambda_handler.handler",
            "code": lambda_.Code.from_asset(str(WEB_BACKEND_DIR)),
            "layers": [deps_layer],
            "role": self.role,
            "memory_size": 1024,
            "timeout": Duration.seconds(30),
            "log_group": log_group,
            "tracing": lambda_.Tracing.ACTIVE,
            "reserved_concurrent_executions": 50,
            "environment": {
                "ZIMAGE_GALLERY_BUCKET": gallery_bucket_name,
                "ZIMAGE_GALLERY_TABLE": gallery_table_name,
                "ZIMAGE_REGION": region,
                "ZIMAGE_PREFIX": prefix,
                "ZIMAGE_ENDPOINT_512": f"{prefix}-endpoint-512",
                "ZIMAGE_ENDPOINT_768": f"{prefix}-endpoint-768",
                "ZIMAGE_ENDPOINT_1024": f"{prefix}-endpoint-1024",
                "ZIMAGE_ENDPOINT_GPU": f"{prefix}-endpoint-gpu",
                "ZIMAGE_COGNITO_USER_POOL_ID": cognito_user_pool_id,
                "ZIMAGE_COGNITO_APP_CLIENT_ID": cognito_app_client_id,
                "ZIMAGE_COGNITO_DOMAIN": cognito_domain,
                "ZIMAGE_WEBSOCKET_URL": websocket_url,
                "ZIMAGE_INFERENCE_ALB_URL": inference_alb_url,
                "ZIMAGE_GPU_ENABLED": str(gpu_enabled).lower(),
                "ZIMAGE_INFERENTIA_ENABLED": str(inferentia_enabled).lower(),
                "ZIMAGE_DEFAULT_BACKEND": "gpu" if gpu_enabled else "neuron",
                "ZIMAGE_GPU_MAX_PIXELS": str(gpu_max_pixels),
                "ZIMAGE_GPU_MIN_DIMENSION": str(gpu_min_dimension),
                "ZIMAGE_GPU_DIMENSION_STEP": str(gpu_dimension_step),
                "ZIMAGE_WS_TICKET_SECRET_ARN": ws_ticket_secret_arn,
            },
        }
        if vpc and vpc_subnets:
            fn_kwargs["vpc"] = vpc
            fn_kwargs["vpc_subnets"] = ec2.SubnetSelection(subnets=vpc_subnets)

        self.function = lambda_.Function(self, "Function", **fn_kwargs)

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Wildcards required: CloudWatch DescribeAlarms/GetMetricData, "
                    "application-autoscaling DescribeScalableTargets, ELBv2 Describe*, "
                    "and ec2 ENI actions do not support resource-level permissions.",
                },
            ],
            apply_to_children=True,
        )
