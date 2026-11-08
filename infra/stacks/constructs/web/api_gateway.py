from __future__ import annotations

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct


class HttpApiGateway(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        handler: lambda_.IFunction,
        user_pool: cognito.IUserPool,
        user_pool_client_id: str,
        region: str,
        enhance_handler: lambda_.IFunction | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool.user_pool_id}"
        jwt_authorizer = authorizers.HttpJwtAuthorizer(
            "CognitoAuth",
            jwt_issuer=issuer,
            jwt_audience=[user_pool_client_id],
        )

        lambda_integration = integrations.HttpLambdaIntegration(
            "ApiIntegration",
            handler=handler,
        )

        self.api = apigwv2.HttpApi(
            self,
            "HttpApi",
            api_name=f"{prefix}-web-api",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=["*"],
                allow_methods=[
                    apigwv2.CorsHttpMethod.GET,
                    apigwv2.CorsHttpMethod.POST,
                    apigwv2.CorsHttpMethod.PUT,
                    apigwv2.CorsHttpMethod.DELETE,
                    apigwv2.CorsHttpMethod.OPTIONS,
                ],
                allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
                max_age=Duration.hours(1),
            ),
        )

        self.api.add_routes(
            path="/api/config",
            methods=[apigwv2.HttpMethod.GET],
            integration=lambda_integration,
        )

        self.api.add_routes(
            path="/api/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=lambda_integration,
            authorizer=jwt_authorizer,
        )

        self.api.add_routes(
            path="/admin/api/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=lambda_integration,
            authorizer=jwt_authorizer,
        )

        if enhance_handler:
            enhance_integration = integrations.HttpLambdaIntegration(
                "EnhanceIntegration",
                handler=enhance_handler,
            )
            self.api.add_routes(
                path="/api/enhance",
                methods=[apigwv2.HttpMethod.POST],
                integration=enhance_integration,
                authorizer=jwt_authorizer,
            )

        self.api.add_routes(
            path="/auth/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=lambda_integration,
        )

        self._access_log_group = logs.LogGroup(
            self,
            "AccessLogs",
            log_group_name=f"/aws/apigateway/{prefix}-web-api",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        default_stage = self.api.default_stage
        if default_stage is not None:
            cfn_stage = default_stage.node.default_child
            cfn_stage.add_property_override(
                "DefaultRouteSettings",
                {"ThrottlingRateLimit": 20, "ThrottlingBurstLimit": 50},
            )
            cfn_stage.add_property_override(
                "AccessLogSettings",
                {
                    "DestinationArn": self._access_log_group.log_group_arn,
                    "Format": '{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","method":"$context.httpMethod","path":"$context.path","status":"$context.status","latency":"$context.responseLatency"}',
                },
            )

        self.url = self.api.url or ""

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-APIG4",
                    "reason": "Auth routes (/auth/*) intentionally have no "
                    "authorizer — needed for OAuth callback/logout flow.",
                },
                {
                    "id": "AwsSolutions-APIG1",
                    "reason": "Access logging configured via L1 override on the default stage.",
                },
            ],
            apply_to_children=True,
        )
