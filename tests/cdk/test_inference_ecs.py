from __future__ import annotations

import os

import aws_cdk as cdk
import aws_cdk.assertions as assertions

os.environ.setdefault("CDK_DEFAULT_ACCOUNT", "123456789012")
os.environ.setdefault("CDK_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ZIMAGE_ENV", "dev")


def _synth_ecs_stack(*, inferentia_enabled=True, gpu_enabled=False):
    from infra.config.settings import DeploySettings
    from infra.stacks.cicd import CicdStack
    from infra.stacks.inference_ecs import InferenceEcsStack
    from infra.stacks.platform import PlatformStack

    settings = DeploySettings(env_override="dev")
    settings.inferentia_endpoint.enabled = inferentia_enabled
    settings.gpu_endpoint.enabled = gpu_enabled
    app = cdk.App()
    env = settings.cdk_env()

    platform = PlatformStack(app, "test-platform", settings=settings, env=env)
    cicd = CicdStack(app, "test-cicd", settings=settings, platform=platform, env=env)
    stack = InferenceEcsStack(
        app,
        "test-inference-ecs",
        settings=settings,
        platform=platform,
        cicd=cicd,
        env=env,
    )
    return assertions.Template.from_stack(stack)


def test_ecs_cluster_created():
    template = _synth_ecs_stack()
    template.resource_count_is("AWS::ECS::Cluster", 1)


def test_three_asg_created_gpu_disabled():
    template = _synth_ecs_stack()
    template.resource_count_is("AWS::AutoScaling::AutoScalingGroup", 3)


def test_three_ecs_services_gpu_disabled():
    template = _synth_ecs_stack()
    template.resource_count_is("AWS::ECS::Service", 3)


def test_three_task_definitions_gpu_disabled():
    template = _synth_ecs_stack()
    template.resource_count_is("AWS::ECS::TaskDefinition", 3)


def test_alb_is_internal():
    template = _synth_ecs_stack()
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        {"Scheme": "internal"},
    )


def test_alb_idle_timeout_300():
    template = _synth_ecs_stack()
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        {
            "LoadBalancerAttributes": assertions.Match.array_with(
                [
                    assertions.Match.object_like(
                        {"Key": "idle_timeout.timeout_seconds", "Value": "300"}
                    )
                ]
            )
        },
    )


def test_target_groups_health_check_ping():
    template = _synth_ecs_stack()
    tgs = template.find_resources("AWS::ElasticLoadBalancingV2::TargetGroup")
    for _id, tg in tgs.items():
        props = tg["Properties"]
        assert props["HealthCheckPath"] == "/ping"
        assert props["Port"] == 8080


def test_neuron_task_defs_device_mapping():
    template = _synth_ecs_stack()
    task_defs = template.find_resources("AWS::ECS::TaskDefinition")
    device_mapped_count = 0
    for _id, td in task_defs.items():
        containers = td["Properties"]["ContainerDefinitions"]
        for c in containers:
            assert not c.get("Privileged"), f"Container in {_id} should not be privileged"
            linux_params = c.get("LinuxParameters", {})
            devices = linux_params.get("Devices", [])
            for dev in devices:
                if dev.get("HostPath") == "/dev/neuron0":
                    device_mapped_count += 1
    assert device_mapped_count == 3


def test_listener_rules_use_header_routing():
    template = _synth_ecs_stack()
    rules = template.find_resources("AWS::ElasticLoadBalancingV2::ListenerRule")
    resolutions_found = set()
    for _id, rule in rules.items():
        conditions = rule["Properties"]["Conditions"]
        for cond in conditions:
            if cond.get("Field") == "http-header":
                values = cond["HttpHeaderConfig"]["Values"]
                resolutions_found.update(values)
    assert resolutions_found == {"512", "768", "1024"}


def test_gpu_only_no_neuron_asg():
    template = _synth_ecs_stack(inferentia_enabled=False, gpu_enabled=True)
    asgs = template.find_resources("AWS::AutoScaling::AutoScalingGroup")
    for _id, asg in asgs.items():
        launch_config = asg.get("Properties", {})
        instance_type = launch_config.get("InstanceType", "")
        assert "inf2" not in str(instance_type), f"Found inf2 ASG in GPU-only mode: {_id}"


def test_gpu_only_has_gpu_service():
    template = _synth_ecs_stack(inferentia_enabled=False, gpu_enabled=True)
    template.resource_count_is("AWS::ECS::Service", 1)


def test_gpu_only_has_alb():
    template = _synth_ecs_stack(inferentia_enabled=False, gpu_enabled=True)
    template.resource_count_is("AWS::ElasticLoadBalancingV2::LoadBalancer", 1)
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        {"Scheme": "internal"},
    )


def test_gpu_only_routes_on_gpu_header():
    template = _synth_ecs_stack(inferentia_enabled=False, gpu_enabled=True)
    rules = template.find_resources("AWS::ElasticLoadBalancingV2::ListenerRule")
    resolutions_found = set()
    for _id, rule in rules.items():
        conditions = rule["Properties"]["Conditions"]
        for cond in conditions:
            if cond.get("Field") == "http-header":
                values = cond["HttpHeaderConfig"]["Values"]
                resolutions_found.update(values)
    assert "gpu" in resolutions_found
