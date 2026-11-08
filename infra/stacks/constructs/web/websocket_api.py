from __future__ import annotations

from aws_cdk import RemovalPolicy
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_authorizers as ws_authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct


class WebSocketApiGateway(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        handler: lambda_.IFunction,
        authorizer_handler: lambda_.IFunction,
    ) -> None:
        super().__init__(scope, construct_id)

        ws_integration = integrations.WebSocketLambdaIntegration(
            "WsIntegration",
            handler=handler,
        )

        connect_authorizer = ws_authorizers.WebSocketLambdaAuthorizer(
            "ConnectAuthorizer",
            handler=authorizer_handler,
            identity_source=["route.request.querystring.ticket"],
        )

        self.api = apigwv2.WebSocketApi(
            self,
            "WebSocketApi",
            api_name=f"{prefix}-web-ws",
            connect_route_options=apigwv2.WebSocketRouteOptions(
                integration=ws_integration,
                authorizer=connect_authorizer,
            ),
            default_route_options=apigwv2.WebSocketRouteOptions(
                integration=ws_integration,
            ),
            disconnect_route_options=apigwv2.WebSocketRouteOptions(
                integration=ws_integration,
            ),
        )

        self._access_log_group = logs.LogGroup(
            self,
            "AccessLogs",
            log_group_name=f"/aws/apigateway/{prefix}-web-ws/v1",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.stage = apigwv2.WebSocketStage(
            self,
            "Stage",
            web_socket_api=self.api,
            stage_name="v1",
            auto_deploy=True,
        )

        cfn_stage = self.stage.node.default_child
        cfn_stage.add_property_override(
            "AccessLogSettings",
            {
                "DestinationArn": self._access_log_group.log_group_arn,
                "Format": '{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","routeKey":"$context.routeKey","status":"$context.status","connectionId":"$context.connectionId"}',
            },
        )

        handler.add_permission(
            "WsDefaultRouteInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{self.api.env.region}:{self.api.env.account}:{self.api.api_id}/*/$default",
        )
        handler.add_permission(
            "WsDisconnectRouteInvoke",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=f"arn:aws:execute-api:{self.api.env.region}:{self.api.env.account}:{self.api.api_id}/*/$disconnect",
        )

        self.callback_url = (
            f"https://{self.api.api_id}.execute-api."
            f"{self.api.env.region}.amazonaws.com/{self.stage.stage_name}"
        )

        self.url = (
            f"wss://{self.api.api_id}.execute-api."
            f"{self.api.env.region}.amazonaws.com/{self.stage.stage_name}"
        )

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-APIG4",
                    "reason": "$default and $disconnect routes inherit the $connect "
                    "authorization decision — only authenticated clients can establish "
                    "a connection in the first place. WebSocket routes are tied to a "
                    "connectionId, so once $connect is denied, no $default/$disconnect "
                    "messages are possible. The $connect route is authorized via "
                    "ConnectAuthorizer (Lambda authorizer validating HMAC ticket).",
                },
                {
                    "id": "AwsSolutions-APIG1",
                    "reason": "Access logging configured via L1 override on the v1 stage.",
                },
            ],
            apply_to_children=True,
        )
