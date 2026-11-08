from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import Aws
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from constructs import Construct


@dataclass
class BatchRoles:
    instance_role: iam.Role
    instance_profile: iam.CfnInstanceProfile
    job_execution_role: iam.Role
    job_role: iam.Role
    batch_instance_policy: iam.ManagedPolicy
    job_execution_policy: iam.ManagedPolicy
    batch_job_policy: iam.ManagedPolicy


def create_batch_roles(
    scope: Construct,
    *,
    inference_repo: ecr.IRepository,
    efs_filesystem_arn: str = "",
    efs_rw_access_point_arn: str = "",
) -> BatchRoles:
    batch_instance_policy = iam.ManagedPolicy(
        scope,
        "BatchInstancePolicy",
        statements=[
            iam.PolicyStatement(
                actions=[
                    "ecs:DeregisterContainerInstance",
                    "ecs:RegisterContainerInstance",
                    "ecs:UpdateContainerInstancesState",
                    "ecs:Submit*",
                ],
                resources=[f"arn:aws:ecs:{Aws.REGION}:{Aws.ACCOUNT_ID}:cluster/*"],
            ),
            iam.PolicyStatement(
                actions=[
                    "ecs:DiscoverPollEndpoint",
                    "ecs:Poll",
                    "ecs:StartTelemetrySession",
                    "ecs:CreateCluster",
                ],
                resources=["*"],
            ),
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/aws/batch/*"],
            ),
            iam.PolicyStatement(
                actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                resources=[f"arn:aws:ecr:{Aws.REGION}:{Aws.ACCOUNT_ID}:repository/*"],
            ),
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            ),
            iam.PolicyStatement(
                actions=[
                    "ssm:GetParameter",
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                ],
                resources=[f"arn:aws:ssm:{Aws.REGION}:{Aws.ACCOUNT_ID}:parameter/*"],
            ),
        ],
    )
    instance_role = iam.Role(
        scope,
        "InstanceRole",
        assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
        description="Role for Batch EC2 instances executing warmup jobs",
        managed_policies=[batch_instance_policy],
    )
    instance_profile = iam.CfnInstanceProfile(
        scope,
        "InstanceProfile",
        roles=[instance_role.role_name],
    )
    instance_profile.node.add_dependency(instance_role)

    job_execution_policy = iam.ManagedPolicy(
        scope,
        "JobExecPolicy",
        statements=[
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=[inference_repo.repository_arn],
            ),
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            ),
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/aws/batch/*"],
            ),
        ],
    )
    job_execution_role = iam.Role(
        scope,
        "JobExecutionRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        description="ECS execution role for warmup Batch jobs",
        managed_policies=[job_execution_policy],
    )

    efs_resources = [efs_filesystem_arn] if efs_filesystem_arn else ["*"]
    efs_conditions = (
        {"StringEquals": {"elasticfilesystem:AccessPointArn": efs_rw_access_point_arn}}
        if efs_rw_access_point_arn
        else {}
    )
    batch_job_policy = iam.ManagedPolicy(
        scope,
        "BatchJobPolicy",
        statements=[
            iam.PolicyStatement(
                actions=[
                    "elasticfilesystem:ClientMount",
                    "elasticfilesystem:ClientWrite",
                ],
                resources=efs_resources,
                conditions=efs_conditions,
            ),
        ],
    )
    job_role = iam.Role(
        scope,
        "JobRole",
        assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        description="Runtime role for warmup_cache.py inside the container",
        managed_policies=[batch_job_policy],
    )

    return BatchRoles(
        instance_role=instance_role,
        instance_profile=instance_profile,
        job_execution_role=job_execution_role,
        job_role=job_role,
        batch_instance_policy=batch_instance_policy,
        job_execution_policy=job_execution_policy,
        batch_job_policy=batch_job_policy,
    )
