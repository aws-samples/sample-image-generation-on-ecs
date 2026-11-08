from __future__ import annotations

from aws_cdk import Duration
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_efs as efs
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from constructs import Construct
from runtime.lib.paths import EFS_MOUNT_PATH

from infra.stacks.constructs.compute.lambda_ import VpcLambda


def create_compile_lambdas(
    scope: Construct,
    *,
    prefix: str,
    vpc: ec2.IVpc,
    private_subnets: list[ec2.ISubnet],
    endpoint_sg: ec2.ISecurityGroup,
    region: str,
    kms_key: kms.IKey | None = None,
    efs_filesystem: efs.IFileSystem | None = None,
    efs_ro_access_point: efs.IAccessPoint | None = None,
    efs_rw_access_point: efs.IAccessPoint | None = None,
) -> dict[str, VpcLambda]:

    purge_fs_kwargs = {}
    if efs_filesystem and efs_rw_access_point:
        purge_fs_kwargs["filesystem"] = lambda_.FileSystem.from_efs_access_point(
            efs_rw_access_point, EFS_MOUNT_PATH
        )

    cache_purge = VpcLambda(
        scope,
        "CachePurgeLambda",
        function_name=f"{prefix}-cache-purge",
        handler="index.handler",
        code_path="infra/lambdas/cache_purge",
        timeout=Duration.minutes(5),
        memory_size=512,
        vpc=vpc,
        subnets=private_subnets,
        security_groups=[endpoint_sg],
        environment={
            "REGION": region,
            "EFS_MOUNT_PATH": EFS_MOUNT_PATH,
        },
        kms_key=kms_key,
        **purge_fs_kwargs,
    )

    fs_kwargs = {}
    if efs_filesystem and efs_ro_access_point:
        fs_kwargs["filesystem"] = lambda_.FileSystem.from_efs_access_point(
            efs_ro_access_point, EFS_MOUNT_PATH
        )

    check_artifacts = VpcLambda(
        scope,
        "CheckArtifactsLambda",
        function_name=f"{prefix}-check-artifacts",
        handler="index.handler",
        code_path="infra/lambdas/check_artifacts",
        timeout=Duration.seconds(60),
        memory_size=512,
        vpc=vpc,
        subnets=private_subnets,
        security_groups=[endpoint_sg],
        environment={
            "REGION": region,
            "EFS_MOUNT_PATH": EFS_MOUNT_PATH,
        },
        kms_key=kms_key,
        **fs_kwargs,
    )

    return {
        "cache_purge": cache_purge,
        "check_artifacts": check_artifacts,
    }
