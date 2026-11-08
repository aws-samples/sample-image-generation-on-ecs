from __future__ import annotations

from pathlib import Path

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

WEB_BACKEND_DIR = Path(__file__).resolve().parents[4] / "web" / "backend"


class WsAuthorizerLambda(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        region: str,
        cognito_user_pool_id: str,
        cognito_app_client_id: str,
        deps_layer: lambda_.ILayerVersion,
        ws_ticket_secret_arn: str = "",
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/aws/lambda/{prefix}-web-ws-authorizer",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        from infra.stacks.constructs.web.lambda_policies import lambda_execution_policies

        self.role = iam.Role(
            self,
            "Role",
            role_name=f"{prefix}-web-ws-authorizer-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=lambda_execution_policies(self, log_group),
        )

        self.function = lambda_.Function(
            self,
            "Function",
            function_name=f"{prefix}-web-ws-authorizer",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="ws_authorizer.handler",
            code=lambda_.Code.from_asset(str(WEB_BACKEND_DIR)),
            layers=[deps_layer],
            role=self.role,
            memory_size=256,
            timeout=Duration.seconds(10),
            log_group=log_group,
            environment={
                "ZIMAGE_REGION": region,
                "ZIMAGE_COGNITO_USER_POOL_ID": cognito_user_pool_id,
                "ZIMAGE_COGNITO_APP_CLIENT_ID": cognito_app_client_id,
                "ZIMAGE_WS_TICKET_SECRET_ARN": ws_ticket_secret_arn,
            },
        )

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Log group ARN uses :* suffix per AWS convention.",
                }
            ],
            apply_to_children=True,
        )
