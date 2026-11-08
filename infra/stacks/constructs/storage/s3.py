from __future__ import annotations

from aws_cdk import RemovalPolicy
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from constructs import Construct


class SecureBucket(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        encryption_key: kms.IKey,
        bucket_name: str | None = None,
        log_bucket: s3.IBucket | None = None,
        lifecycle_rules: list[s3.LifecycleRule] | None = None,
        versioned: bool = True,
        removal_policy: RemovalPolicy = RemovalPolicy.DESTROY,
        object_ownership: s3.ObjectOwnership = s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
        event_bridge_enabled: bool = False,
    ) -> None:
        super().__init__(scope, construct_id)

        self.bucket = s3.Bucket(
            self,
            "Bucket",
            bucket_name=bucket_name,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=encryption_key,
            bucket_key_enabled=True,
            versioned=versioned,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            public_read_access=False,
            server_access_logs_bucket=log_bucket,
            server_access_logs_prefix=f"{construct_id}/" if log_bucket else None,
            lifecycle_rules=lifecycle_rules or [],
            removal_policy=removal_policy,
            auto_delete_objects=(removal_policy == RemovalPolicy.DESTROY),
            object_ownership=object_ownership,
            event_bridge_enabled=event_bridge_enabled,
        )
