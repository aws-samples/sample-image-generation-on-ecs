from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_efs as efs
from constructs import Construct
from runtime.lib.paths import EFS_ARTIFACTS_PATH

from infra.stacks.constructs.compute.base_batch import BaseBatchConstruct, BatchConfig

_TRACE_CMD = [
    "/opt/ml/code/compile/compile.py",
]


class CpuBatch(BaseBatchConstruct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        subnets: list[ec2.ISubnet],
        efs_filesystem: efs.IFileSystem,
        efs_rw_access_point: efs.IAccessPoint,
        compile_cpu_repo: ecr.IRepository,
        prefix: str,
        image_tag: str = "latest",
        efs_security_group: ec2.ISecurityGroup | None = None,
    ) -> None:
        config = BatchConfig(
            sg_id="CpuBatchSG",
            sg_description="Z-Image CPU Batch EC2 hosts - egress only",
            launch_template_id="CpuLaunchTemplate",
            launch_template_name=f"{prefix}-cpu-trace-lt",
            ce_id="CpuTraceCE",
            ce_name=f"{prefix}-cpu-trace-ce",
            job_queue_id="CpuTraceJobQueue",
            job_queue_name=f"{prefix}-cpu-trace-jq",
            job_def_id="CpuTraceJobDef",
            job_def_name=f"{prefix}-cpu-trace-jd",
            instance_types=["c6i.8xlarge"],
            ami_ssm_path="/aws/service/ecs/optimized-ami/amazon-linux-2023/recommended/image_id",
            min_vcpus=0,
            max_vcpus=64,
            container_command=_TRACE_CMD,
            vcpus=32,
            memory=60000,
            environment={
                "PJRT_DEVICE": "CPU",
                "ZIMAGE_ARTIFACTS_DIR": EFS_ARTIFACTS_PATH,
                "PYTHONPATH": "/opt/ml/code",
                "OMP_NUM_THREADS": "32",
                "NEURON_CC_FLAGS": "--optlevel=1 --retry_failed_compilation",
                "NEURON_RT_VISIBLE_CORES": "",
                "NEURON_RT_NUM_CORES": "0",
                "NEURON_INTERNAL_MOCK_TPU": "1",
                "TPU_NUM_DEVICES": "1",
                "ZIMAGE_ALLOW_CPU_COMPILE": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": "inf2",
            },
            retry_attempts=2,
            timeout_seconds=7200,
            neuron_devices=False,
            privileged=False,
        )

        super().__init__(
            scope,
            construct_id,
            vpc=vpc,
            subnets=subnets,
            ecr_repo=compile_cpu_repo,
            config=config,
            image_tag=image_tag,
            efs_filesystem=efs_filesystem,
            efs_access_point=efs_rw_access_point,
            efs_security_group=efs_security_group,
        )
