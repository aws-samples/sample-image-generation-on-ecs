from __future__ import annotations

from aws_cdk import Aws
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from constructs import Construct


def create_sm_processing_role(
    scope: Construct,
    *,
    stack_name: str,
    key_artifacts: kms.IKey,
    environment: str,
) -> tuple[iam.ManagedPolicy, iam.Role]:
    policy = iam.ManagedPolicy(
        scope,
        "SmProcessingPolicy",
        managed_policy_name=f"{stack_name}-sm-processing",
        statements=[
            iam.PolicyStatement(
                actions=["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey"],
                resources=[key_artifacts.key_arn],
            ),
            iam.PolicyStatement(
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=[f"arn:aws:ecr:{Aws.REGION}:{Aws.ACCOUNT_ID}:repository/{environment}-*"],
            ),
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            ),
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:AddTags",
                    "sagemaker:ListTags",
                ],
                resources=[f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:processing-job/zi-*"],
            ),
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[
                    f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/aws/sagemaker/ProcessingJobs",
                    f"arn:aws:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:log-group:/aws/sagemaker/ProcessingJobs:log-stream:*",
                ],
            ),
            iam.PolicyStatement(
                sid="VpcNetworking",
                actions=[
                    "ec2:CreateNetworkInterface",
                    "ec2:CreateNetworkInterfacePermission",
                    "ec2:DeleteNetworkInterface",
                    "ec2:DeleteNetworkInterfacePermission",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DescribeVpcs",
                    "ec2:DescribeDhcpOptions",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups",
                ],
                resources=["*"],
            ),
        ],
    )

    role = iam.Role(
        scope,
        "SmProcessingRole",
        assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        description="Execution role for compile-sfn Processing Jobs",
        managed_policies=[policy],
    )

    return policy, role


def create_sfn_execution_policy(
    scope: Construct,
    *,
    stack_name: str,
    sm_role_arn: str,
    region: str,
    account: str,
) -> iam.ManagedPolicy:
    return iam.ManagedPolicy(
        scope,
        "SfnExecutionPolicy",
        managed_policy_name=f"{stack_name}-sfn-execution",
        statements=[
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[sm_role_arn],
                conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            ),
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:ListTags",
                    "sagemaker:AddTags",
                ],
                resources=[f"arn:aws:sagemaker:{region}:{account}:processing-job/zi-*"],
            ),
            iam.PolicyStatement(
                actions=["events:PutTargets", "events:PutRule", "events:DescribeRule"],
                resources=[
                    f"arn:aws:events:{region}:{account}:rule/"
                    "StepFunctionsGetEventsForSageMakerProcessingJobsRule",
                ],
            ),
        ],
    )
