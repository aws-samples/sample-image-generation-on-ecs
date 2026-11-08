from __future__ import annotations

import subprocess
from pathlib import Path

import jsii
from aws_cdk import BundlingOptions, CfnOutput, Duration, ILocalBundling, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.config.settings import DeploySettings
from infra.stacks.constructs.web import (
    AdminErrorRateAlarm,
    ApiLambda,
    CognitoAuth,
    CognitoCallbackUpdater,
    EnhancePromptLambda,
    HttpApiGateway,
    StaticSite,
    WebCdnV2,
    WebSocketApiGateway,
    WebSocketLambda,
    WsAuthorizerLambda,
)
from infra.stacks.inference_infra import InferenceInfraStack
from infra.stacks.platform import PlatformStack

WEB_BACKEND_DIR = Path(__file__).resolve().parents[2] / "web" / "backend"


@jsii.implements(ILocalBundling)
class _PipLocalBundler:
    def try_bundle(self, output_dir: str, options) -> bool:
        target = f"{output_dir}/python"
        subprocess.check_call(
            ["pip", "install", "-r", str(WEB_BACKEND_DIR / "requirements.txt"), "-t", target, "-q"]
        )
        return True


class WebStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: DeploySettings,
        platform: PlatformStack,
        infra: InferenceInfraStack,
        findings_topic: sns.ITopic,
        inference_alb_url: str = "",
        inference_alb_sg: ec2.ISecurityGroup | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = settings.prefix
        region = settings.region

        self.ws_ticket_secret = secretsmanager.Secret(
            self,
            "WsTicketSecret",
            secret_name=f"{prefix}-ws-ticket-secret",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=64,
                exclude_punctuation=True,
            ),
        )
        NagSuppressions.add_resource_suppressions(
            self.ws_ticket_secret,
            [
                {
                    "id": "AwsSolutions-SMG4",
                    "reason": "HMAC signing key is static by design; rotation would "
                    "invalidate in-flight tickets and requires coordinated Lambda redeploy.",
                }
            ],
        )

        self.auth = CognitoAuth(
            self,
            "Auth",
            prefix=prefix,
            admin_email=settings.security.findings_email,
            domain_prefix=settings.cognito_domain_prefix,
        )

        self.static_site = StaticSite(
            self,
            "StaticSite",
            prefix=prefix,
            account=self.account,
            region=region,
        )

        deps_layer = lambda_.LayerVersion(
            self,
            "DepsLayer",
            layer_version_name=f"{prefix}-web-deps",
            code=lambda_.Code.from_asset(
                str(WEB_BACKEND_DIR),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_13.bundling_image,
                    local=_PipLocalBundler(),
                    command=["bash", "-c", "echo 'Docker fallback not used'"],
                ),
            ),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_13],
            description="Z-Image web backend dependencies",
        )

        self.api_lambda = ApiLambda(
            self,
            "ApiLambda",
            prefix=prefix,
            region=region,
            gallery_bucket_name=infra.gallery_bucket.bucket_name,
            gallery_table_name=infra.gallery_table.table_name,
            cognito_user_pool_id=self.auth.user_pool.user_pool_id,
            cognito_app_client_id=self.auth.app_client.user_pool_client_id,
            cognito_domain=self.auth.domain_fqdn,
            inference_alb_url=inference_alb_url,
            deps_layer=deps_layer,
            vpc=platform.vpc,
            vpc_subnets=platform.private_subnets,
            kms_key_arn=platform.key.key_arn,
            gpu_enabled=settings.gpu_endpoint.enabled,
            inferentia_enabled=settings.inferentia_endpoint.enabled,
            gpu_max_pixels=settings.gpu_endpoint.max_pixels,
            gpu_min_dimension=settings.gpu_endpoint.min_dimension,
            gpu_dimension_step=settings.gpu_endpoint.dimension_step,
            ws_ticket_secret_arn=self.ws_ticket_secret.secret_arn,
        )

        self.ws_lambda = WebSocketLambda(
            self,
            "WsLambda",
            prefix=prefix,
            region=region,
            gallery_bucket_name=infra.gallery_bucket.bucket_name,
            gallery_table_name=infra.gallery_table.table_name,
            deps_layer=deps_layer,
            vpc=platform.vpc,
            vpc_subnets=platform.private_subnets,
            inference_alb_url=inference_alb_url,
            kms_key_arn=platform.key.key_arn,
            gpu_enabled=settings.gpu_endpoint.enabled,
        )

        if inference_alb_sg:
            for i, fn in enumerate([self.ws_lambda.function, self.api_lambda.function]):
                ec2.CfnSecurityGroupIngress(
                    self,
                    f"AlbFromLambda{i}",
                    group_id=inference_alb_sg.security_group_id,
                    source_security_group_id=fn.connections.security_groups[0].security_group_id,
                    ip_protocol="tcp",
                    from_port=80,
                    to_port=80,
                    description="Lambda to inference ALB",
                )

        self.ws_authorizer = WsAuthorizerLambda(
            self,
            "WsAuthorizer",
            prefix=prefix,
            region=region,
            cognito_user_pool_id=self.auth.user_pool.user_pool_id,
            cognito_app_client_id=self.auth.app_client.user_pool_client_id,
            deps_layer=deps_layer,
            ws_ticket_secret_arn=self.ws_ticket_secret.secret_arn,
        )

        self.ws_ticket_secret.grant_read(self.api_lambda.function)
        self.ws_ticket_secret.grant_read(self.ws_authorizer.function)

        self.enhance_lambda = EnhancePromptLambda(
            self,
            "EnhanceLambda",
            prefix=prefix,
            region=region,
        )

        self.http_api = HttpApiGateway(
            self,
            "HttpApi",
            prefix=prefix,
            handler=self.api_lambda.function,
            user_pool=self.auth.user_pool,
            user_pool_client_id=self.auth.app_client.user_pool_client_id,
            region=region,
            enhance_handler=self.enhance_lambda.function,
        )

        self.ws_api = WebSocketApiGateway(
            self,
            "WsApi",
            prefix=prefix,
            handler=self.ws_lambda.function,
            authorizer_handler=self.ws_authorizer.function,
        )

        self.ws_lambda.function.add_environment("WEBSOCKET_API_ENDPOINT", self.ws_api.callback_url)

        self.api_lambda.function.add_environment("ZIMAGE_WEBSOCKET_URL", self.ws_api.url)

        self.cdn = WebCdnV2(
            self,
            "Cdn",
            prefix=prefix,
            static_bucket=self.static_site.bucket,
            api_url=self.http_api.url,
            access_logs_bucket=platform.elb_access_logs_bucket,
        )

        self.api_lambda.function.add_environment("ZIMAGE_CDN_URL", self.cdn.url)

        self.static_site.grant_cloudfront_oac_read(
            f"arn:aws:cloudfront::{self.account}:distribution/"
            f"{self.cdn.distribution.distribution_id}"
        )

        CognitoCallbackUpdater(
            self,
            "UpdateCallbackUrls",
            user_pool=self.auth.user_pool,
            app_client=self.auth.app_client,
            cdn_domain=self.cdn.distribution.distribution_domain_name,
        )

        warmup_rule = events.Rule(
            self, "LambdaWarmupRule", schedule=events.Schedule.rate(Duration.minutes(5))
        )
        for fn in [self.api_lambda.function, self.ws_lambda.function, self.enhance_lambda.function]:
            warmup_rule.add_target(
                targets.LambdaFunction(
                    fn, event=events.RuleTargetInput.from_object({"warmup": True})
                )
            )

        self.admin_alarm = AdminErrorRateAlarm(
            self,
            "AdminAlarm",
            prefix=prefix,
            api_lambda=self.api_lambda.function,
            findings_topic=findings_topic,
        )

        CfnOutput(self, "WebUrl", value=self.cdn.url)
        CfnOutput(self, "WebSocketUrl", value=self.ws_api.url)
        CfnOutput(self, "UserPoolId", value=self.auth.user_pool.user_pool_id)
        CfnOutput(self, "AppClientId", value=self.auth.app_client.user_pool_client_id)
        CfnOutput(self, "CognitoDomain", value=self.auth.domain_fqdn)
        CfnOutput(self, "StaticBucketName", value=self.static_site.bucket.bucket_name)

        NagSuppressions.add_stack_suppressions(
            self,
            [
                {"id": "AwsSolutions-L1", "reason": "CDK-internal custom resource Lambda."},
                {
                    "id": "AwsSolutions-IAM4",
                    "appliesTo": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                    ],
                    "reason": "CDK-internal custom resource Lambdas use AWS managed policy.",
                },
            ],
        )
