from __future__ import annotations

from aws_cdk import Duration, Fn, RemovalPolicy, Stack
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from aws_cdk import custom_resources as cr
from constructs import Construct


class CognitoAuth(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        admin_email: str,
        domain_prefix: str,
    ) -> None:
        super().__init__(scope, construct_id)

        self.user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=f"{prefix}-users",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            feature_plan=cognito.FeaturePlan.PLUS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.users_group = cognito.CfnUserPoolGroup(
            self,
            "UsersGroup",
            user_pool_id=self.user_pool.user_pool_id,
            group_name="users",
            description="Regular users — can generate images and view gallery",
        )

        self.admins_group = cognito.CfnUserPoolGroup(
            self,
            "AdminsGroup",
            user_pool_id=self.user_pool.user_pool_id,
            group_name="admins",
            description="Administrators — full infrastructure control",
        )

        self.app_client = self.user_pool.add_client(
            "WebClient",
            user_pool_client_name=f"{prefix}-web",
            generate_secret=False,
            auth_flows=cognito.AuthFlow(user_srp=True, user_password=True),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
            ),
            supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
            id_token_validity=Duration.hours(1),
            access_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
            prevent_user_existence_errors=True,
        )

        self.domain = self.user_pool.add_domain(
            "Domain",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix),
        )
        self.domain_fqdn = Fn.join(
            "",
            [self.domain.domain_name, f".auth.{Stack.of(self).region}.amazoncognito.com"],
        )

        admin_user_cr = cr.AwsCustomResource(
            self,
            "AdminUser",
            on_create=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="adminCreateUser",
                parameters={
                    "UserPoolId": self.user_pool.user_pool_id,
                    "Username": admin_email,
                    "UserAttributes": [
                        {"Name": "email", "Value": admin_email},
                        {"Name": "email_verified", "Value": "true"},
                    ],
                    "DesiredDeliveryMediums": ["EMAIL"],
                },
                physical_resource_id=cr.PhysicalResourceId.of(f"admin-{admin_email}"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["cognito-idp:AdminCreateUser"],
                        resources=[self.user_pool.user_pool_arn],
                    )
                ]
            ),
        )

        group_membership = cr.AwsCustomResource(
            self,
            "AdminGroupMembership",
            on_create=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="adminAddUserToGroup",
                parameters={
                    "UserPoolId": self.user_pool.user_pool_id,
                    "Username": admin_email,
                    "GroupName": "admins",
                },
                physical_resource_id=cr.PhysicalResourceId.of(f"admin-group-{admin_email}"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["cognito-idp:AdminAddUserToGroup"],
                        resources=[self.user_pool.user_pool_arn],
                    )
                ]
            ),
        )
        users_membership = cr.AwsCustomResource(
            self,
            "AdminUsersGroupMembership",
            on_create=cr.AwsSdkCall(
                service="CognitoIdentityServiceProvider",
                action="adminAddUserToGroup",
                parameters={
                    "UserPoolId": self.user_pool.user_pool_id,
                    "Username": admin_email,
                    "GroupName": "users",
                },
                physical_resource_id=cr.PhysicalResourceId.of(f"users-group-{admin_email}"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["cognito-idp:AdminAddUserToGroup"],
                        resources=[self.user_pool.user_pool_arn],
                    )
                ]
            ),
        )

        admin_user_cr.node.add_dependency(self.admins_group)
        admin_user_cr.node.add_dependency(self.users_group)
        group_membership.node.add_dependency(admin_user_cr)
        users_membership.node.add_dependency(admin_user_cr)
