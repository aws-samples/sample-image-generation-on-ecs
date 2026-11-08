from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.config.settings import DeploySettings
from infra.stacks.constructs.network.vpc import Vpc
from infra.stacks.constructs.security.kms import KmsKey
from infra.stacks.constructs.storage.efs import SecureEfs
from infra.stacks.constructs.storage.s3 import SecureBucket


class PlatformStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: DeploySettings,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        kms_construct = KmsKey(self, "Kms", prefix=settings.prefix)
        self.key = kms_construct.key
        self.key_logs = self.key
        self.key_artifacts = self.key
        self.key_data = self.key

        self.access_logs_bucket = SecureBucket(
            self,
            "AccessLogs",
            encryption_key=self.key,
            bucket_name=f"{settings.prefix}-access-logs-{settings.account}-{settings.region}",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-logs",
                    enabled=True,
                    expiration=Duration.days(365),
                    noncurrent_version_expiration=Duration.days(90),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
            ],
            versioned=True,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
        ).bucket

        self.elb_access_logs_bucket = s3.Bucket(
            self,
            "ElbAccessLogs",
            bucket_name=f"{settings.prefix}-elb-access-logs-{settings.account}-{settings.region}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-logs",
                    enabled=True,
                    expiration=Duration.days(365),
                    noncurrent_version_expiration=Duration.days(90),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
            ],
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        NagSuppressions.add_resource_suppressions(
            self.elb_access_logs_bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "This IS the access-log target bucket; self-logging would loop.",
                }
            ],
        )

        network = Vpc(
            self,
            "Network",
            nat_count=settings.network.nat_count,
            account=settings.account,
            az_count=settings.network.az_count,
        )
        self.vpc = network.vpc
        self.private_subnets = network.private_subnets
        self.public_subnets = network.public_subnets
        self.endpoint_sg = network.endpoint_sg

        self.efs = SecureEfs(
            self,
            "Efs",
            vpc=self.vpc,
            private_subnets=self.private_subnets,
            kms_key=self.key_artifacts,
            prefix=settings.prefix,
        )

        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        CfnOutput(self, "AccessLogsBucketName", value=self.access_logs_bucket.bucket_name)
        CfnOutput(self, "KmsKeyArn", value=self.key.key_arn)
        CfnOutput(self, "EfsFileSystemId", value=self.efs.filesystem.file_system_id)
        CfnOutput(self, "EfsRoAccessPointId", value=self.efs.ro_access_point.access_point_id)
        CfnOutput(self, "EfsRwAccessPointId", value=self.efs.rw_access_point.access_point_id)
