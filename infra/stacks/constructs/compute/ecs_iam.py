from __future__ import annotations

from aws_cdk import Aws
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from constructs import Construct


def create_ecs_instance_role(scope: Construct, res_str: str) -> iam.Role:
    return iam.Role(
        scope,
        f"InstanceRole{res_str}",
        assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
        managed_policies=[
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonEC2ContainerServiceforEC2Role"
            ),
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
        ],
    )


def create_ecs_execution_role(scope: Construct, ecr_repos: dict[int, ecr.IRepository]) -> iam.Role:
    role = iam.Role(
        scope,
        "ExecRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    role.add_to_policy(iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"]))
    for repo in ecr_repos.values():
        repo.grant_pull(role)
    role.add_to_policy(
        iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/ecs/*"],
        )
    )
    return role


def create_ecs_task_role(
    scope: Construct,
    *,
    efs_filesystem_arn: str = "",
) -> iam.Role:
    role = iam.Role(
        scope,
        "TaskRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    )
    if efs_filesystem_arn:
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["elasticfilesystem:ClientMount"],
                resources=[efs_filesystem_arn],
            )
        )
    return role
