from __future__ import annotations

from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from aws_cdk import custom_resources as cr
from constructs import Construct


class CognitoCallbackUpdater(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        user_pool: cognito.IUserPool,
        app_client: cognito.IUserPoolClient,
        cdn_domain: str,
    ) -> None:
        super().__init__(scope, construct_id)

        params = {
            "UserPoolId": user_pool.user_pool_id,
            "ClientId": app_client.user_pool_client_id,
            "CallbackURLs": [
                f"https://{cdn_domain}/auth/callback",
                "http://localhost:5173/auth/callback",
            ],
            "LogoutURLs": [f"https://{cdn_domain}/", "http://localhost:5173/"],
            "AllowedOAuthFlows": ["code"],
            "AllowedOAuthScopes": ["openid", "email", "profile"],
            "AllowedOAuthFlowsUserPoolClient": True,
            "SupportedIdentityProviders": ["COGNITO"],
            "ExplicitAuthFlows": [
                "ALLOW_USER_SRP_AUTH",
                "ALLOW_USER_PASSWORD_AUTH",
                "ALLOW_REFRESH_TOKEN_AUTH",
            ],
        }
        sdk_call = cr.AwsSdkCall(
            service="CognitoIdentityServiceProvider",
            action="updateUserPoolClient",
            parameters=params,
            physical_resource_id=cr.PhysicalResourceId.of("update-callback-urls"),
        )
        cr.AwsCustomResource(
            self,
            "Resource",
            on_create=sdk_call,
            on_update=sdk_call,
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["cognito-idp:UpdateUserPoolClient"],
                        resources=[user_pool.user_pool_arn],
                    )
                ]
            ),
        )
