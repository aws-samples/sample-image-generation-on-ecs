from __future__ import annotations

from pathlib import Path

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

WEB_BACKEND_DIR = Path(__file__).resolve().parents[4] / "web" / "backend"


class WebSocketLambda(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        region: str,
        gallery_bucket_name: str,
        gallery_table_name: str,
        deps_layer: lambda_.ILayerVersion,
        vpc: ec2.IVpc | None = None,
        vpc_subnets: list[ec2.ISubnet] | None = None,
        inference_alb_url: str = "",
        kms_key_arn: str = "",
        gpu_enabled: bool = True,
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/aws/lambda/{prefix}-web-ws",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        from infra.stacks.constructs.web.lambda_policies import lambda_execution_policies

        self.role = iam.Role(
            self,
            "Role",
            role_name=f"{prefix}-web-ws-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=lambda_execution_policies(self, log_group, vpc),
        )

        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[
                    f"arn:aws:s3:::{gallery_bucket_name}",
                    f"arn:aws:s3:::{gallery_bucket_name}/*",
                ],
            )
        )

        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["execute-api:ManageConnections"],
                resources=["arn:aws:execute-api:*:*:*/@connections/*"],
            )
        )

        if kms_key_arn:
            self.role.add_to_policy(
                iam.PolicyStatement(
                    actions=["kms:GenerateDataKey", "kms:Decrypt", "kms:DescribeKey"],
                    resources=[kms_key_arn],
                )
            )

        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                ],
                resources=[
                    f"arn:aws:dynamodb:{region}:*:table/{gallery_table_name}",
                    f"arn:aws:dynamodb:{region}:*:table/{gallery_table_name}/index/*",
                ],
            )
        )

        fn_kwargs: dict = {
            "function_name": f"{prefix}-web-ws",
            "runtime": lambda_.Runtime.PYTHON_3_13,
            "handler": "websocket_handler.handler",
            "code": lambda_.Code.from_asset(str(WEB_BACKEND_DIR)),
            "layers": [deps_layer],
            "role": self.role,
            "memory_size": 2048,
            "timeout": Duration.seconds(120),
            "log_group": log_group,
            "tracing": lambda_.Tracing.ACTIVE,
            "reserved_concurrent_executions": 100,
            "environment": {
                "ZIMAGE_GALLERY_BUCKET": gallery_bucket_name,
                "ZIMAGE_GALLERY_TABLE": gallery_table_name,
                "ZIMAGE_REGION": region,
                "ZIMAGE_PREFIX": prefix,
                "ZIMAGE_INFERENCE_ALB_URL": inference_alb_url,
                "ZIMAGE_DEFAULT_BACKEND": "gpu" if gpu_enabled else "neuron",
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
                    "reason": "Wildcards required: ec2 ENI actions and execute-api "
                    "ManageConnections do not support resource-level permissions.",
                },
            ],
            apply_to_children=True,
        )
