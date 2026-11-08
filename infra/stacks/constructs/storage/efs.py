from __future__ import annotations

from aws_cdk import RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_efs as efs
from aws_cdk import aws_kms as kms
from constructs import Construct


class SecureEfs(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        private_subnets: list[ec2.ISubnet],
        kms_key: kms.IKey,
        prefix: str,
    ) -> None:
        super().__init__(scope, construct_id)

        self.security_group = ec2.SecurityGroup(
            self,
            "EfsSG",
            vpc=vpc,
            description=f"{prefix} EFS - NFS access from VPC",
            allow_all_outbound=True,
        )

        self.filesystem = efs.FileSystem(
            self,
            "Fs",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=private_subnets),
            security_group=self.security_group,
            encrypted=True,
            kms_key=kms_key,
            performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
            throughput_mode=efs.ThroughputMode.ELASTIC,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_14_DAYS,
            removal_policy=RemovalPolicy.DESTROY,
            file_system_name=f"{prefix}-artifacts",
        )

        self.ro_access_point = efs.AccessPoint(
            self,
            "RoAccessPoint",
            file_system=self.filesystem,
            path="/artifacts",
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="0755"),
        )

        self.rw_access_point = efs.AccessPoint(
            self,
            "RwAccessPoint",
            file_system=self.filesystem,
            path="/artifacts",
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="0755"),
        )
