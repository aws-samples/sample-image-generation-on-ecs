from __future__ import annotations

import jsii
from aws_cdk import Aspects, Duration, IAspect, RemovalPolicy
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_efs as efs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct, IConstruct, Node
from runtime.lib.paths import EFS_ARTIFACTS_PATH

from infra.config.neuron_env import neuron_env_for_resolution
from infra.stacks.constructs.compute.ecs_iam import (
    create_ecs_execution_role,
    create_ecs_instance_role,
    create_ecs_task_role,
)
from infra.utils.contracts import GpuEndpointConfig, InferentiaEndpointConfig


class InferentiaCluster(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        private_subnets: list[ec2.ISubnet],
        ecr_repos: dict[int, ecr.IRepository],
        gpu_repo: ecr.IRepository | None = None,
        gpu_config: GpuEndpointConfig | None = None,
        efs_filesystem: efs.IFileSystem | None = None,
        efs_ro_access_point: efs.IAccessPoint | None = None,
        efs_security_group: ec2.ISecurityGroup | None = None,
        access_logs_bucket: s3.IBucket | None = None,
        prefix: str,
        config: InferentiaEndpointConfig | None = None,
    ) -> None:
        super().__init__(scope, construct_id)
        cfg = config or InferentiaEndpointConfig()
        self._gpu_cfg = gpu_config or GpuEndpointConfig()
        self._prefix = prefix
        self._vpc = vpc
        self._subnets = private_subnets
        self._cfg = cfg
        self._efs_filesystem = efs_filesystem
        self._efs_ro_access_point = efs_ro_access_point
        self._efs_sg = efs_security_group

        self.cluster = ecs.Cluster(
            self,
            "Cluster",
            cluster_name=f"{prefix}-inference-cluster",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENHANCED,
        )

        self._task_execution_role = create_ecs_execution_role(self, ecr_repos)
        self._task_role = create_ecs_task_role(
            self,
            efs_filesystem_arn=efs_filesystem.file_system_arn if efs_filesystem else "",
        )

        self._alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            vpc=vpc,
            description="ECS inference ALB (restricted to authorized callers)",
            allow_all_outbound=True,
        )
        self.alb_security_group = self._alb_sg

        self._task_sg = ec2.SecurityGroup(
            self,
            "TaskSg",
            vpc=vpc,
            description="ECS inference tasks (inf2)",
            allow_all_outbound=True,
        )
        self._task_sg.add_ingress_rule(
            self._alb_sg, ec2.Port.tcp(cfg.container_port), "ALB to container"
        )
        if self._efs_sg:
            ec2.CfnSecurityGroupIngress(
                self,
                "EfsFromTasks",
                group_id=self._efs_sg.security_group_id,
                source_security_group_id=self._task_sg.security_group_id,
                ip_protocol="tcp",
                from_port=2049,
                to_port=2049,
                description="ECS inference tasks to EFS",
            )

        self.alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            load_balancer_name=f"{prefix}-inference-alb",
            vpc=vpc,
            internet_facing=False,
            vpc_subnets=ec2.SubnetSelection(subnets=private_subnets),
            security_group=self._alb_sg,
            idle_timeout=Duration.seconds(cfg.alb_idle_timeout_s),
        )

        if access_logs_bucket:
            self.alb.log_access_logs(access_logs_bucket, prefix="alb")

        self._listener = self.alb.add_listener(
            "Listener",
            port=80,
            open=False,
            default_action=elbv2.ListenerAction.fixed_response(
                status_code=400, message_body="Missing or unsupported X-Target-Resolution header"
            ),
        )

        self.services: dict[int | str, ecs.Ec2Service] = {}
        if cfg.enabled:
            for resolution, repo in ecr_repos.items():
                self._create_resolution(resolution, repo)

        if gpu_repo and self._gpu_cfg.enabled:
            self._create_gpu_service(gpu_repo)

        Aspects.of(self).add(_CapacityProviderDependency(list(self.services.values())))

        self.alb_dns = self.alb.load_balancer_dns_name

    def _create_resolution(self, resolution: int, repo: ecr.IRepository) -> None:
        res_str = str(resolution)
        instance_role = create_ecs_instance_role(self, res_str)
        lt = ec2.LaunchTemplate(
            self,
            f"Lt{res_str}",
            launch_template_name=f"{self._prefix}-ecs-{res_str}-lt",
            instance_type=ec2.InstanceType(self._cfg.instance_type),
            machine_image=ec2.MachineImage.from_ssm_parameter(
                "/aws/service/ecs/optimized-ami/amazon-linux-2023/neuron/recommended/image_id",
            ),
            role=instance_role,
            security_group=self._task_sg,
            user_data=ec2.UserData.for_linux(),
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        200, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True
                    ),
                )
            ],
            require_imdsv2=True,
            http_put_response_hop_limit=2,
        )
        asg = autoscaling.AutoScalingGroup(
            self,
            f"Asg{res_str}",
            auto_scaling_group_name=f"{self._prefix}-ecs-{res_str}-asg",
            vpc=self._vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=self._subnets),
            launch_template=lt,
            min_capacity=self._cfg.min_capacity,
            max_capacity=self._cfg.max_capacity,
            new_instances_protected_from_scale_in=True,
        )
        cp = ecs.AsgCapacityProvider(
            self,
            f"Cp{res_str}",
            capacity_provider_name=f"{self._prefix}-{res_str}-cp",
            auto_scaling_group=asg,
            enable_managed_scaling=True,
            target_capacity_percent=100,
            enable_managed_termination_protection=True,
        )
        self.cluster.add_asg_capacity_provider(cp)

        task_def = ecs.Ec2TaskDefinition(
            self,
            f"TaskDef{res_str}",
            family=f"{self._prefix}-inference-{res_str}",
            network_mode=ecs.NetworkMode.AWS_VPC,
            task_role=self._task_role,
            execution_role=self._task_execution_role,
        )

        if self._efs_filesystem and self._efs_ro_access_point:
            task_def.add_volume(
                name="efs-artifacts",
                efs_volume_configuration=ecs.EfsVolumeConfiguration(
                    file_system_id=self._efs_filesystem.file_system_id,
                    transit_encryption="ENABLED",
                    authorization_config=ecs.AuthorizationConfig(
                        access_point_id=self._efs_ro_access_point.access_point_id,
                        iam="ENABLED",
                    ),
                ),
            )

        env_vars = neuron_env_for_resolution(resolution)
        env_vars.update(
            {
                "NEURON_RT_VISIBLE_CORES": "0,1",
                "ZIMAGE_ARTIFACTS_DIR": EFS_ARTIFACTS_PATH,
                "ZIMAGE_EFS_DIR": EFS_ARTIFACTS_PATH,
                "ZIMAGE_ROOT": "/opt/ml/code",
                "ZIMAGE_ALLOWED_RESOLUTIONS": str(resolution),
                "ZIMAGE_EAGER_LOAD": "true",
                "SAGEMAKER_BIND_TO_PORT": "8080",
            }
        )

        log_group = logs.LogGroup(
            self,
            f"Logs{res_str}",
            log_group_name=f"/ecs/{self._prefix}-inference-{res_str}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        image_tag = Node.of(self).try_get_context(f"image_tag_{res_str}") or "latest"

        linux_params = ecs.LinuxParameters(self, f"LP{res_str}")
        linux_params.add_devices(
            ecs.Device(
                host_path="/dev/neuron0",
                container_path="/dev/neuron0",
                permissions=[
                    ecs.DevicePermission.READ,
                    ecs.DevicePermission.WRITE,
                    ecs.DevicePermission.MKNOD,
                ],
            )
        )
        linux_params.add_tmpfs(
            ecs.Tmpfs(
                container_path="/tmp", size=1024, mount_options=[ecs.TmpfsMountOption.NOSUID]
            ),
        )

        container = task_def.add_container(
            "inference",
            image=ecs.ContainerImage.from_ecr_repository(repo, tag=image_tag),
            cpu=32768,
            memory_limit_mib=122880,
            linux_parameters=linux_params,
            environment=env_vars,
            readonly_root_filesystem=True,
            port_mappings=[
                ecs.PortMapping(container_port=self._cfg.container_port, protocol=ecs.Protocol.TCP)
            ],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="inference", log_group=log_group),
        )

        if self._efs_filesystem and self._efs_ro_access_point:
            container.add_mount_points(
                ecs.MountPoint(
                    container_path=EFS_ARTIFACTS_PATH,
                    source_volume="efs-artifacts",
                    read_only=True,
                )
            )

        service = ecs.Ec2Service(
            self,
            f"Svc{res_str}",
            service_name=f"{self._prefix}-inference-{res_str}",
            cluster=self.cluster,
            task_definition=task_def,
            desired_count=self._cfg.min_capacity,
            min_healthy_percent=0,
            max_healthy_percent=100,
            vpc_subnets=ec2.SubnetSelection(subnets=self._subnets),
            security_groups=[self._task_sg],
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(capacity_provider=cp.capacity_provider_name, weight=1)
            ],
            health_check_grace_period=Duration.seconds(self._cfg.health_check_grace_s),
        )
        self.services[resolution] = service

        tg = elbv2.ApplicationTargetGroup(
            self,
            f"Tg{res_str}",
            target_group_name=f"{self._prefix}-inf-{res_str}-tg",
            vpc=self._vpc,
            port=self._cfg.container_port,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            load_balancing_algorithm_type=elbv2.TargetGroupLoadBalancingAlgorithmType.LEAST_OUTSTANDING_REQUESTS,
            health_check=elbv2.HealthCheck(
                path="/ping",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                healthy_threshold_count=2,
                unhealthy_threshold_count=10,
            ),
            deregistration_delay=Duration.seconds(120),
        )
        service.attach_to_application_target_group(tg)

        elbv2.ApplicationListenerRule(
            self,
            f"Rule{res_str}",
            listener=self._listener,
            priority=resolution,
            conditions=[elbv2.ListenerCondition.http_header("X-Target-Resolution", [res_str])],
            target_groups=[tg],
        )

        scaling = service.auto_scale_task_count(
            min_capacity=self._cfg.min_capacity,
            max_capacity=self._cfg.max_capacity,
        )
        scaling.scale_on_request_count(
            f"Scale{res_str}",
            requests_per_target=3,
            target_group=tg,
            scale_out_cooldown=Duration.seconds(0),
            scale_in_cooldown=Duration.seconds(300),
        )

    def _create_gpu_service(self, repo: ecr.IRepository) -> None:
        gpu_cfg = self._gpu_cfg
        instance_role = create_ecs_instance_role(self, "gpu")
        lt = ec2.LaunchTemplate(
            self,
            "LtGpu",
            launch_template_name=f"{self._prefix}-ecs-gpu-lt",
            instance_type=ec2.InstanceType(gpu_cfg.instance_type),
            machine_image=ecs.EcsOptimizedImage.amazon_linux2023(
                hardware_type=ecs.AmiHardwareType.GPU
            ),
            role=instance_role,
            security_group=self._task_sg,
            user_data=ec2.UserData.for_linux(),
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        gpu_cfg.root_volume_gb,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        encrypted=True,
                    ),
                )
            ],
            require_imdsv2=True,
            http_put_response_hop_limit=2,
        )
        asg = autoscaling.AutoScalingGroup(
            self,
            "AsgGpu",
            auto_scaling_group_name=f"{self._prefix}-ecs-gpu-asg",
            vpc=self._vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=self._subnets),
            launch_template=lt,
            min_capacity=gpu_cfg.min_capacity,
            max_capacity=gpu_cfg.max_capacity,
            new_instances_protected_from_scale_in=True,
        )
        cp = ecs.AsgCapacityProvider(
            self,
            "CpGpu",
            capacity_provider_name=f"{self._prefix}-gpu-cp",
            auto_scaling_group=asg,
            enable_managed_scaling=True,
            target_capacity_percent=100,
            enable_managed_termination_protection=True,
        )
        self.cluster.add_asg_capacity_provider(cp)

        task_def = ecs.Ec2TaskDefinition(
            self,
            "TaskDefGpu",
            family=f"{self._prefix}-inference-gpu",
            network_mode=ecs.NetworkMode.AWS_VPC,
            task_role=self._task_role,
            execution_role=self._task_execution_role,
        )

        if self._efs_filesystem and self._efs_ro_access_point:
            task_def.add_volume(
                name="efs-artifacts",
                efs_volume_configuration=ecs.EfsVolumeConfiguration(
                    file_system_id=self._efs_filesystem.file_system_id,
                    transit_encryption="ENABLED",
                    authorization_config=ecs.AuthorizationConfig(
                        access_point_id=self._efs_ro_access_point.access_point_id,
                        iam="ENABLED",
                    ),
                ),
            )

        env_vars = {
            "ZIMAGE_BACKEND": "gpu",
            "ZIMAGE_GPU_MAX_PIXELS": str(gpu_cfg.max_pixels),
            "ZIMAGE_GPU_MIN_DIMENSION": str(gpu_cfg.min_dimension),
            "ZIMAGE_GPU_DIMENSION_STEP": str(gpu_cfg.dimension_step),
            "ZIMAGE_ARTIFACTS_DIR": EFS_ARTIFACTS_PATH,
            "ZIMAGE_EFS_DIR": EFS_ARTIFACTS_PATH,
            "ZIMAGE_ROOT": "/opt/ml/code",
            "ZIMAGE_EAGER_LOAD": "true",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "SAGEMAKER_BIND_TO_PORT": str(self._cfg.container_port),
        }
        log_group = logs.LogGroup(
            self,
            "LogsGpu",
            log_group_name=f"/ecs/{self._prefix}-inference-gpu",
            retention=logs.RetentionDays.ONE_MONTH,
        )
        gpu_image_tag = Node.of(self).try_get_context("image_tag_gpu") or "latest"
        gpu_linux_params = ecs.LinuxParameters(self, "LPGpu")
        gpu_linux_params.add_tmpfs(
            ecs.Tmpfs(
                container_path="/tmp", size=1024, mount_options=[ecs.TmpfsMountOption.NOSUID]
            ),
        )
        gpu_container = task_def.add_container(
            "inference",
            image=ecs.ContainerImage.from_ecr_repository(repo, tag=gpu_image_tag),
            gpu_count=1,
            memory_limit_mib=gpu_cfg.memory_mib,
            cpu=gpu_cfg.cpu,
            environment=env_vars,
            linux_parameters=gpu_linux_params,
            readonly_root_filesystem=True,
            port_mappings=[
                ecs.PortMapping(container_port=self._cfg.container_port, protocol=ecs.Protocol.TCP)
            ],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="inference", log_group=log_group),
        )

        if self._efs_filesystem and self._efs_ro_access_point:
            gpu_container.add_mount_points(
                ecs.MountPoint(
                    container_path=EFS_ARTIFACTS_PATH,
                    source_volume="efs-artifacts",
                    read_only=True,
                )
            )

        service = ecs.Ec2Service(
            self,
            "SvcGpu",
            service_name=f"{self._prefix}-inference-gpu",
            cluster=self.cluster,
            task_definition=task_def,
            desired_count=gpu_cfg.min_capacity,
            min_healthy_percent=0,
            max_healthy_percent=100,
            vpc_subnets=ec2.SubnetSelection(subnets=self._subnets),
            security_groups=[self._task_sg],
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(capacity_provider=cp.capacity_provider_name, weight=1)
            ],
            health_check_grace_period=Duration.seconds(gpu_cfg.startup_timeout_s),
        )
        self.services["gpu"] = service

        tg = elbv2.ApplicationTargetGroup(
            self,
            "TgGpu",
            target_group_name=f"{self._prefix}-inf-gpu-tg",
            vpc=self._vpc,
            port=self._cfg.container_port,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            load_balancing_algorithm_type=elbv2.TargetGroupLoadBalancingAlgorithmType.LEAST_OUTSTANDING_REQUESTS,
            health_check=elbv2.HealthCheck(
                path="/ping",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                healthy_threshold_count=2,
                unhealthy_threshold_count=10,
            ),
            deregistration_delay=Duration.seconds(120),
        )
        service.attach_to_application_target_group(tg)

        elbv2.ApplicationListenerRule(
            self,
            "RuleGpu",
            listener=self._listener,
            priority=10,
            conditions=[elbv2.ListenerCondition.http_header("X-Target-Resolution", ["gpu"])],
            target_groups=[tg],
        )

        scaling = service.auto_scale_task_count(
            min_capacity=gpu_cfg.min_capacity,
            max_capacity=gpu_cfg.max_capacity,
        )
        scaling.scale_on_request_count(
            "GpuRequestScaling",
            requests_per_target=1,
            target_group=tg,
            scale_out_cooldown=Duration.seconds(0),
            scale_in_cooldown=Duration.seconds(300),
        )


@jsii.implements(IAspect)
class _CapacityProviderDependency:
    def __init__(self, services: list[ecs.Ec2Service]) -> None:
        self._services = services

    def visit(self, node: IConstruct) -> None:
        if (
            getattr(node, "cfn_resource_type", None)
            != "AWS::ECS::ClusterCapacityProviderAssociations"
        ):
            return
        for service in self._services:
            service.node.add_dependency(node)
