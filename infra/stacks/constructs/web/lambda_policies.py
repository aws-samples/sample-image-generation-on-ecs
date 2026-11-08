from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from constructs import Construct


def lambda_execution_policies(
    scope: Construct,
    log_group: logs.ILogGroup,
    vpc: ec2.IVpc | None = None,
) -> list[iam.IManagedPolicy]:
    log_group_arn = log_group.log_group_arn
    policies: list[iam.IManagedPolicy] = [
        iam.ManagedPolicy(
            scope,
            "LogsPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                    resources=[log_group_arn, f"{log_group_arn}:*"],
                )
            ],
        )
    ]

    if vpc:
        policies.append(
            iam.ManagedPolicy(
                scope,
                "VpcPolicy",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "ec2:CreateNetworkInterface",
                            "ec2:DescribeNetworkInterfaces",
                            "ec2:DeleteNetworkInterface",
                        ],
                        resources=["*"],
                    )
                ],
            )
        )

    return policies
