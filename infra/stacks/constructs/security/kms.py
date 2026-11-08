from __future__ import annotations

from aws_cdk import Aws, Duration, RemovalPolicy
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from constructs import Construct


class KmsKey(Construct):
    def __init__(self, scope: Construct, construct_id: str, *, prefix: str = "") -> None:
        super().__init__(scope, construct_id)

        alias_name = f"alias/{prefix}-z-image" if prefix else "alias/z-image-inferentia"

        self.key = kms.Key(
            self,
            "Key",
            alias=alias_name,
            description="Z-Image encryption key",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,
            pending_window=Duration.days(7),
        )

        self.key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudWatchLogs",
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal(f"logs.{Aws.REGION}.amazonaws.com")],
                actions=[
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:DescribeKey",
                ],
                resources=["*"],
            )
        )

        self.key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudTrail",
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("cloudtrail.amazonaws.com")],
                actions=[
                    "kms:GenerateDataKey*",
                    "kms:DescribeKey",
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                ],
                resources=["*"],
            )
        )
