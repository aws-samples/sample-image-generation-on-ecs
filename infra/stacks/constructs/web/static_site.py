from __future__ import annotations

from aws_cdk import RemovalPolicy
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct


class StaticSite(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        account: str,
        region: str,
    ) -> None:
        super().__init__(scope, construct_id)

        bucket_name = f"{prefix}-web-static-{account}-{region}"

        self.bucket = s3.Bucket(
            self,
            "Bucket",
            bucket_name=bucket_name,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        NagSuppressions.add_resource_suppressions(
            self.bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "Static site bucket does not need access logging. "
                    "CloudFront WAF provides request visibility.",
                },
                {
                    "id": "AwsSolutions-S2",
                    "reason": "Static site assets are immutable deployments. "
                    "Versioning adds unnecessary cost for replaced builds.",
                },
            ],
            apply_to_children=True,
        )

    def grant_cloudfront_oac_read(self, distribution_arn: str) -> None:
        self.bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudFrontOAC",
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("cloudfront.amazonaws.com")],
                actions=["s3:GetObject"],
                resources=[f"{self.bucket.bucket_arn}/*"],
                conditions={"StringEquals": {"AWS:SourceArn": distribution_arn}},
            )
        )
