from __future__ import annotations

from aws_cdk import Duration
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct


class VpcLambda(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        function_name: str,
        handler: str,
        code_path: str,
        vpc: ec2.IVpc,
        subnets: list[ec2.ISubnet],
        security_groups: list[ec2.ISecurityGroup],
        timeout: Duration | None = None,
        memory_size: int = 256,
        environment: dict[str, str] | None = None,
        s3_read_arns: list[str] | None = None,
        s3_write_arns: list[str] | None = None,
        kms_key: kms.IKey | None = None,
        filesystem: lambda_.FileSystem | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        self.log_group = logs.LogGroup(
            self,
            "LogGroup",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        fn_kwargs: dict = {
            "function_name": function_name,
            "runtime": lambda_.Runtime.PYTHON_3_13,
            "code": lambda_.Code.from_asset(code_path),
            "handler": handler,
            "timeout": timeout,
            "memory_size": memory_size,
            "environment": environment or {},
            "log_group": self.log_group,
            "vpc": vpc,
            "vpc_subnets": ec2.SubnetSelection(subnets=subnets),
            "security_groups": security_groups,
        }
        if filesystem:
            fn_kwargs["filesystem"] = filesystem
        self.function = lambda_.Function(self, "Fn", **fn_kwargs)

        if s3_read_arns:
            self.s3_read_policy = iam.ManagedPolicy(
                self,
                "S3ReadPolicy",
                managed_policy_name=f"{function_name}-s3-read",
                statements=[
                    iam.PolicyStatement(
                        actions=["s3:GetObject", "s3:HeadObject", "s3:ListBucket"],
                        resources=s3_read_arns,
                    ),
                ],
            )
            self.function.role.add_managed_policy(self.s3_read_policy)

        if s3_write_arns:
            self.s3_write_policy = iam.ManagedPolicy(
                self,
                "S3WritePolicy",
                managed_policy_name=f"{function_name}-s3-write",
                statements=[
                    iam.PolicyStatement(
                        actions=["s3:PutObject", "s3:DeleteObject"],
                        resources=s3_write_arns,
                    ),
                ],
            )
            self.function.role.add_managed_policy(self.s3_write_policy)

        if kms_key:
            kms_key.grant_decrypt(self.function)
