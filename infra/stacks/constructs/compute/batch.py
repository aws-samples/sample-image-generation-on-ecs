from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_efs as efs
from constructs import Construct
from runtime.lib.paths import EFS_ARTIFACTS_PATH

from infra.config.neuron_env import NEURON_GRAPH_ENV, neuron_env_for_resolution
from infra.stacks.constructs.compute.base_batch import BaseBatchConstruct, BatchConfig


@dataclass(frozen=True)
class BatchJobQueue:
    job_queue_arn: str
    job_queue_name: str


@dataclass(frozen=True)
class BatchJobDefinition:
    job_definition_arn: str
    job_definition_name: str


_WARMUP_CMD = [
    "python3",
    "/opt/ml/code/serving/warmup_cache.py",
    "--resolution",
    "Ref::resolution",
]


class WarmupBatch(BaseBatchConstruct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        subnets: list[ec2.ISubnet],
        inference_repo: ecr.IRepository,
        prefix: str,
        image_tag: str = "latest",
        efs_filesystem: efs.IFileSystem | None = None,
        efs_access_point: efs.IAccessPoint | None = None,
        efs_security_group: ec2.ISecurityGroup | None = None,
    ) -> None:
        artifacts_dir = EFS_ARTIFACTS_PATH if efs_filesystem else "/opt/ml/code/artifacts"

        config = BatchConfig(
            sg_id="WarmupBatchSG",
            sg_description="Z-Image warmup Batch EC2 hosts - egress only",
            launch_template_id="WarmupLaunchTemplate",
            launch_template_name=f"{prefix}-warmup-lt",
            ce_id="WarmupCE",
            ce_name=f"{prefix}-warmup-ce",
            job_queue_id="WarmupJobQueue",
            job_queue_name=f"{prefix}-warmup-jq",
            job_def_id="WarmupJobDef",
            job_def_name=f"{prefix}-warmup-jd",
            instance_types=["inf2.8xlarge"],
            ami_ssm_path="/aws/service/ecs/optimized-ami/amazon-linux-2023/neuron/recommended/image_id",
            min_vcpus=0,
            max_vcpus=128,
            container_command=_WARMUP_CMD,
            vcpus=32,
            memory=122880,
            environment={
                **NEURON_GRAPH_ENV,
                "NEURON_RT_VISIBLE_CORES": "0,1",
                "NEURON_CACHE_VERSION": neuron_env_for_resolution(768)["NEURON_CACHE_VERSION"],
                "PYTHONPATH": "/opt/ml/code",
                "ZIMAGE_ARTIFACTS_DIR": artifacts_dir,
            },
            retry_attempts=3,
            timeout_seconds=10800,
            neuron_devices=True,
        )

        super().__init__(
            scope,
            construct_id,
            vpc=vpc,
            subnets=subnets,
            ecr_repo=inference_repo,
            config=config,
            image_tag=image_tag,
            efs_filesystem=efs_filesystem,
            efs_access_point=efs_access_point,
            efs_security_group=efs_security_group,
        )
