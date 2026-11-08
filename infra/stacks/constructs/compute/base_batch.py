from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import Aws
from aws_cdk import aws_batch as batch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_efs as efs
from aws_cdk import aws_ssm as ssm
from constructs import Construct
from runtime.lib.paths import EFS_ARTIFACTS_PATH

from infra.stacks.constructs.compute.batch_iam import create_batch_roles


@dataclass(frozen=True)
class BatchConfig:
    sg_id: str
    sg_description: str
    launch_template_id: str
    launch_template_name: str
    ce_id: str
    ce_name: str
    job_queue_id: str
    job_queue_name: str
    job_def_id: str
    job_def_name: str

    instance_types: list[str]
    ami_ssm_path: str
    min_vcpus: int
    max_vcpus: int

    container_command: list[str]
    vcpus: int
    memory: int
    environment: dict[str, str]
    retry_attempts: int
    timeout_seconds: int

    neuron_devices: bool = False

    privileged: bool | None = None


class BaseBatchConstruct(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        subnets: list[ec2.ISubnet],
        ecr_repo: ecr.IRepository,
        config: BatchConfig,
        image_tag: str = "latest",
        efs_filesystem: efs.IFileSystem | None = None,
        efs_access_point: efs.IAccessPoint | None = None,
        efs_security_group: ec2.ISecurityGroup | None = None,
    ) -> None:
        super().__init__(scope, construct_id)
        self._image_tag = image_tag

        self.security_group = ec2.SecurityGroup(
            self,
            config.sg_id,
            vpc=vpc,
            description=config.sg_description,
            allow_all_outbound=True,
        )

        if efs_security_group:
            ec2.CfnSecurityGroupIngress(
                self,
                "EfsFromBatch",
                group_id=efs_security_group.security_group_id,
                source_security_group_id=self.security_group.security_group_id,
                ip_protocol="tcp",
                from_port=2049,
                to_port=2049,
                description="Batch hosts to EFS",
            )

        roles = create_batch_roles(
            self,
            inference_repo=ecr_repo,
            efs_filesystem_arn=efs_filesystem.file_system_arn if efs_filesystem else "",
            efs_rw_access_point_arn=efs_access_point.access_point_arn if efs_access_point else "",
        )

        ami_id = ssm.StringParameter.value_for_string_parameter(self, config.ami_ssm_path)
        launch_template = ec2.CfnLaunchTemplate(
            self,
            config.launch_template_id,
            launch_template_name=config.launch_template_name,
            launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                image_id=ami_id,
                metadata_options=ec2.CfnLaunchTemplate.MetadataOptionsProperty(
                    http_tokens="required",
                    http_put_response_hop_limit=2,
                ),
                block_device_mappings=[
                    ec2.CfnLaunchTemplate.BlockDeviceMappingProperty(
                        device_name="/dev/xvda",
                        ebs=ec2.CfnLaunchTemplate.EbsProperty(
                            volume_size=200,
                            volume_type="gp3",
                            delete_on_termination=True,
                            encrypted=True,
                        ),
                    ),
                ],
            ),
        )

        ce = batch.CfnComputeEnvironment(
            self,
            config.ce_id,
            compute_environment_name=config.ce_name,
            type="MANAGED",
            state="ENABLED",
            compute_resources=batch.CfnComputeEnvironment.ComputeResourcesProperty(
                type="EC2",
                allocation_strategy="BEST_FIT_PROGRESSIVE",
                minv_cpus=config.min_vcpus,
                maxv_cpus=config.max_vcpus,
                instance_types=config.instance_types,
                subnets=[s.subnet_id for s in subnets],
                security_group_ids=[self.security_group.security_group_id],
                instance_role=roles.instance_profile.attr_arn,
                update_to_latest_image_version=True,
                launch_template=batch.CfnComputeEnvironment.LaunchTemplateSpecificationProperty(
                    launch_template_id=launch_template.ref,
                    version="$Latest",
                ),
            ),
            update_policy=batch.CfnComputeEnvironment.UpdatePolicyProperty(
                terminate_jobs_on_update=False,
            ),
            replace_compute_environment=False,
        )
        ce.node.add_dependency(roles.instance_profile)
        ce.node.add_dependency(roles.instance_role)
        ce.node.add_dependency(launch_template)

        job_queue = batch.CfnJobQueue(
            self,
            config.job_queue_id,
            job_queue_name=config.job_queue_name,
            priority=1,
            state="ENABLED",
            compute_environment_order=[
                batch.CfnJobQueue.ComputeEnvironmentOrderProperty(
                    compute_environment=ce.ref,
                    order=1,
                ),
            ],
        )
        job_queue.node.add_dependency(ce)

        image_uri = f"{ecr_repo.repository_uri}:{image_tag}"

        volumes: list = []
        mount_points: list = []
        efs_volume_name = "efs-artifacts"

        if efs_filesystem is not None and efs_access_point is not None:
            volumes.append(
                batch.CfnJobDefinition.VolumesProperty(
                    name=efs_volume_name,
                    efs_volume_configuration=batch.CfnJobDefinition.EfsVolumeConfigurationProperty(
                        file_system_id=efs_filesystem.file_system_id,
                        transit_encryption="ENABLED",
                        authorization_config=batch.CfnJobDefinition.AuthorizationConfigProperty(
                            access_point_id=efs_access_point.access_point_id,
                            iam="ENABLED",
                        ),
                    ),
                )
            )
            mount_points.append(
                batch.CfnJobDefinition.MountPointsProperty(
                    container_path=EFS_ARTIFACTS_PATH,
                    source_volume=efs_volume_name,
                    read_only=False,
                )
            )

        container_props: dict = {
            "image": image_uri,
            "vcpus": config.vcpus,
            "memory": config.memory,
            "execution_role_arn": roles.job_execution_role.role_arn,
            "job_role_arn": roles.job_role.role_arn,
            "command": config.container_command,
            "environment": [
                batch.CfnJobDefinition.EnvironmentProperty(name=k, value=v)
                for k, v in config.environment.items()
            ],
        }

        if config.neuron_devices:
            container_props["linux_parameters"] = batch.CfnJobDefinition.LinuxParametersProperty(
                devices=[
                    batch.CfnJobDefinition.DeviceProperty(
                        host_path="/dev/neuron0",
                        container_path="/dev/neuron0",
                        permissions=["read", "write", "mknod"],
                    )
                ],
            )

        if config.privileged is not None:
            container_props["privileged"] = config.privileged

        if volumes:
            container_props["volumes"] = volumes
            container_props["mount_points"] = mount_points

        job_def = batch.CfnJobDefinition(
            self,
            config.job_def_id,
            job_definition_name=config.job_def_name,
            type="container",
            platform_capabilities=["EC2"],
            retry_strategy=batch.CfnJobDefinition.RetryStrategyProperty(
                attempts=config.retry_attempts,
            ),
            timeout=batch.CfnJobDefinition.TimeoutProperty(
                attempt_duration_seconds=config.timeout_seconds,
            ),
            container_properties=batch.CfnJobDefinition.ContainerPropertiesProperty(
                **container_props,
            ),
        )
        job_def.node.add_dependency(roles.job_execution_role)
        job_def.node.add_dependency(roles.job_role)

        from infra.stacks.constructs.compute.batch import (
            BatchJobDefinition,
            BatchJobQueue,
        )

        self.job_queue = BatchJobQueue(
            job_queue_arn=f"arn:aws:batch:{Aws.REGION}:{Aws.ACCOUNT_ID}:job-queue/{job_queue.job_queue_name}",
            job_queue_name=job_queue.job_queue_name,
        )
        self.job_def = BatchJobDefinition(
            job_definition_arn=job_def.ref,
            job_definition_name=job_def.job_definition_name,
        )

        self.instance_role = roles.instance_role
        self.job_execution_role = roles.job_execution_role
        self.job_role = roles.job_role
        self.compute_environment = ce
        self.batch_instance_policy = roles.batch_instance_policy
        self.job_execution_policy = roles.job_execution_policy
        self.batch_job_policy = roles.batch_job_policy
