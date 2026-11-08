from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_logs as logs
from aws_cdk import aws_stepfunctions as sfn
from constructs import Construct

from infra.config.settings import DeploySettings
from infra.stacks.cicd import CicdStack
from infra.stacks.compile_sfn.definition import build_state_machine_definition
from infra.stacks.constructs.compute.batch import WarmupBatch
from infra.stacks.constructs.compute.cpu_batch import CpuBatch
from infra.stacks.platform import PlatformStack


class CompileStateMachineStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: DeploySettings,
        platform: PlatformStack,
        cicd: CicdStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = settings.prefix
        acct = self.account
        region = self.region

        warmup_image_tag = self.node.try_get_context("image_tag_512") or "latest"
        compile_image_tag = self.node.try_get_context("image_tag_compiler") or "latest"

        self.warmup_batch: WarmupBatch | None = None
        if settings.inferentia_endpoint.enabled:
            self.warmup_batch = WarmupBatch(
                self,
                "WarmupBatch",
                vpc=platform.vpc,
                subnets=platform.private_subnets,
                inference_repo=cicd.inference_repo.repository,
                prefix=prefix,
                image_tag=warmup_image_tag,
                efs_filesystem=platform.efs.filesystem,
                efs_access_point=platform.efs.rw_access_point,
                efs_security_group=platform.efs.security_group,
            )

        self.cpu_batch = CpuBatch(
            self,
            "CpuBatch",
            vpc=platform.vpc,
            subnets=platform.private_subnets,
            efs_filesystem=platform.efs.filesystem,
            efs_rw_access_point=platform.efs.rw_access_point,
            compile_cpu_repo=cicd.compile_cpu_repo.repository,
            prefix=prefix,
            image_tag=compile_image_tag,
            efs_security_group=platform.efs.security_group,
        )

        from infra.stacks.compile_sfn.lambdas import create_compile_lambdas

        _lambdas = create_compile_lambdas(
            self,
            prefix=prefix,
            vpc=platform.vpc,
            private_subnets=platform.private_subnets,
            endpoint_sg=platform.endpoint_sg,
            region=region,
            kms_key=platform.key_artifacts,
            efs_filesystem=platform.efs.filesystem,
            efs_ro_access_point=platform.efs.ro_access_point,
            efs_rw_access_point=platform.efs.rw_access_point,
        )
        self.cache_purge = _lambdas["cache_purge"]
        self.check_artifacts = _lambdas["check_artifacts"]

        from infra.stacks.compile_sfn.iam import create_sm_processing_role

        self.sm_processing_policy, self.sm_role = create_sm_processing_role(
            self,
            stack_name=self.stack_name,
            key_artifacts=platform.key_artifacts,
            environment=str(settings.environment),
        )

        definition = build_state_machine_definition(
            self,
            job_definition_arn=(
                self.warmup_batch.job_def.job_definition_arn if self.warmup_batch else ""
            ),
            job_queue_arn=self.warmup_batch.job_queue.job_queue_arn if self.warmup_batch else "",
            cpu_job_definition_arn=self.cpu_batch.job_def.job_definition_arn,
            cpu_job_queue_arn=self.cpu_batch.job_queue.job_queue_arn,
            prefix=prefix,
            check_artifacts_function=self.check_artifacts.function,
            cache_purge_function=self.cache_purge.function,
            inferentia_enabled=settings.inferentia_endpoint.enabled,
        )

        sm_log_group = logs.LogGroup(
            self,
            "StateMachineLogs",
            log_group_name=f"/aws/vendedlogs/states/{prefix}-compile-sfn",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.state_machine = sfn.StateMachine(
            self,
            "StateMachine",
            state_machine_name=f"{prefix}-compile-sfn",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.hours(6),
            logs=sfn.LogOptions(
                destination=sm_log_group,
                level=sfn.LogLevel.ALL,
                include_execution_data=True,
            ),
            tracing_enabled=True,
        )
        from infra.stacks.compile_sfn.iam import create_sfn_execution_policy

        self.sfn_execution_policy = create_sfn_execution_policy(
            self,
            stack_name=self.stack_name,
            sm_role_arn=self.sm_role.role_arn,
            region=region,
            account=acct,
        )

        self.state_machine.role.add_managed_policy(self.sfn_execution_policy)

        CfnOutput(self, "StateMachineArn", value=self.state_machine.state_machine_arn)
        if self.warmup_batch:
            CfnOutput(self, "WarmupJobQueueArn", value=self.warmup_batch.job_queue.job_queue_arn)
            CfnOutput(
                self,
                "WarmupJobDefinitionArn",
                value=self.warmup_batch.job_def.job_definition_arn,
            )
        CfnOutput(self, "CpuTraceJobQueueArn", value=self.cpu_batch.job_queue.job_queue_arn)
        CfnOutput(
            self,
            "CpuTraceJobDefinitionArn",
            value=self.cpu_batch.job_def.job_definition_arn,
        )

        from cdk_nag import NagSuppressions

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "S3 prefix wildcards and log-stream wildcards are architecturally required."
                    ),
                },
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CDK-internal framework Lambdas use AWSLambdaBasicExecutionRole.",
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "Lambda runtime pinned to Python 3.13.",
                },
            ],
            apply_to_children=True,
        )
