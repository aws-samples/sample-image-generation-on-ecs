from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from constructs import Construct

from infra.config.settings import DeploySettings
from infra.stacks.constructs.storage.s3 import SecureBucket
from infra.stacks.platform import PlatformStack


class InferenceInfraStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: DeploySettings,
        platform: PlatformStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = settings.prefix
        acct, region = settings.account, settings.region

        self.gallery_bucket = SecureBucket(
            self,
            "Gallery",
            encryption_key=platform.key_data,
            bucket_name=f"{prefix}-gallery-{acct}-{region}",
            log_bucket=platform.access_logs_bucket,
            lifecycle_rules=[],
        ).bucket

        self.gallery_table = dynamodb.Table(
            self,
            "GalleryTable",
            table_name=f"{prefix}-gallery",
            partition_key=dynamodb.Attribute(name="user_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
        )
        self.gallery_table.add_global_secondary_index(
            index_name="id-index",
            partition_key=dynamodb.Attribute(name="id", type=dynamodb.AttributeType.STRING),
        )

        CfnOutput(self, "GalleryBucket", value=self.gallery_bucket.bucket_name)
        CfnOutput(self, "GalleryTableName", value=self.gallery_table.table_name)
        CfnOutput(self, "Region", value=region)
