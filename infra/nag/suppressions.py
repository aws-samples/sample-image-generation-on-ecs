from __future__ import annotations

import contextlib

from aws_cdk import Stack
from cdk_nag import NagSuppressions

CDK_CUSTOM_RESOURCE_PREFIX = "AWS679f53fac"

CDK_FRAMEWORK_SUPPRESSIONS = [
    {
        "id": "AwsSolutions-IAM4",
        "reason": "CDK-internal framework Lambda.",
        "appliesTo": [
            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        ],
    },
    {
        "id": "AwsSolutions-IAM5",
        "reason": "CDK-internal framework Lambda scoped permissions.",
    },
    {
        "id": "AwsSolutions-L1",
        "reason": "CDK-internal framework Lambda runtime.",
    },
]


def apply_nag_suppressions(
    platform: Stack,
    cicd: Stack,
    infra: Stack,
    compile_sfn: Stack | None = None,
) -> None:
    _suppress_platform_framework(platform)

    for stack in (cicd, infra):
        _suppress_framework_lambdas(stack)

    _suppress_framework_lambdas(cicd, pattern="BucketNotificationsHandler")
    _suppress_framework_lambdas(infra, pattern="CDKBucketDeployment")

    if compile_sfn is not None:
        _suppress_framework_lambdas(compile_sfn)
        _suppress_log_retention_lambdas(compile_sfn)


def _suppress_platform_framework(platform: Stack) -> None:
    _safe_suppress(
        platform,
        f"/{platform.stack_name}/AccessLogs/Bucket/Resource",
        [
            {
                "id": "AwsSolutions-S1",
                "reason": "Access-logs bucket cannot log to itself.",
            }
        ],
    )


def _suppress_framework_lambdas(stack: Stack, pattern: str | None = None) -> None:
    match_prefix = pattern or CDK_CUSTOM_RESOURCE_PREFIX

    for child in stack.node.find_all():
        node = getattr(child, "node", None)
        if not node:
            continue
        if match_prefix in node.id:
            with contextlib.suppress(Exception):
                NagSuppressions.add_resource_suppressions(
                    child,
                    CDK_FRAMEWORK_SUPPRESSIONS,
                    apply_to_children=True,
                )


def _suppress_log_retention_lambdas(stack: Stack) -> None:
    for child in stack.node.find_all():
        node = getattr(child, "node", None)
        if not node:
            continue
        if node.id.startswith("LogRetention"):
            with contextlib.suppress(Exception):
                NagSuppressions.add_resource_suppressions(
                    child,
                    CDK_FRAMEWORK_SUPPRESSIONS,
                    apply_to_children=True,
                )


def _safe_suppress(stack: Stack, path: str, suppressions: list[dict]) -> None:
    with contextlib.suppress(Exception):
        NagSuppressions.add_resource_suppressions_by_path(stack, path, suppressions)
