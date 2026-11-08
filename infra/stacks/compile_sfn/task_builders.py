from __future__ import annotations

from aws_cdk import Duration
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct
from runtime.lib.paths import EFS_ARTIFACTS_PATH

from infra.utils.constants import HF_MODEL_REVISION, HF_REPO_ID


def build_download_weights_task(
    scope: Construct,
    *,
    cpu_job_definition_arn: str,
    cpu_job_queue_arn: str,
    prefix: str,
) -> sfn.IChainable:
    t = tasks.BatchSubmitJob(
        scope,
        "DownloadWeights",
        job_definition_arn=cpu_job_definition_arn,
        job_name=f"{prefix}-download-weights",
        job_queue_arn=cpu_job_queue_arn,
        container_overrides=tasks.BatchContainerOverrides(
            command=[
                "/opt/ml/code/compile/download_weights.py",
            ],
            environment={
                "HF_REPO_ID": HF_REPO_ID,
                "HF_MODEL_REVISION": HF_MODEL_REVISION,
                "ZIMAGE_ARTIFACTS_DIR": EFS_ARTIFACTS_PATH,
            },
        ),
        result_path="$.downloadWeights",
        task_timeout=sfn.Timeout.duration(Duration.hours(2)),
    )
    t.add_retry(
        errors=["States.TaskFailed", "Batch.BatchException", "Batch.ServerException"],
        interval=Duration.seconds(60),
        max_attempts=2,
        backoff_rate=2.0,
    )
    return t


def build_neuron_trace_task(
    scope: Construct,
    *,
    spec: dict,
    job_definition_arn: str,
    job_queue_arn: str,
    prefix: str,
) -> sfn.IChainable:
    step = spec["step"]
    t = tasks.BatchSubmitJob(
        scope,
        step,
        job_definition_arn=job_definition_arn,
        job_name=f"{prefix}-{step.lower()[:20]}",
        job_queue_arn=job_queue_arn,
        container_overrides=tasks.BatchContainerOverrides(
            command=["python3", "/opt/ml/code/compile/compile.py"],
            environment={
                "TRACE_SCRIPT": spec["script"],
                "EXPECTED_OUTPUT_FILE": spec["output"],
                "COMPILE_OUTPUT_MODE": "batch",
                "ZIMAGE_ROOT": "/opt/ml/code",
                "ZIMAGE_ARTIFACTS_DIR": EFS_ARTIFACTS_PATH,
                "ZIMAGE_OUTPUT_DIR": "/opt/ml/code/compiled_models",
                "ZIMAGE_TRANSFORMER_PASSTHROUGH": "0",
                "ZIMAGE_ENCODER_PASSTHROUGH": "0",
                "ZIMAGE_ALLOW_CPU_COMPILE": "0",
            },
        ),
        result_path=f"$.traces.{step}",
        task_timeout=sfn.Timeout.duration(Duration.hours(2)),
    )
    t.add_retry(
        errors=["States.TaskFailed", "Batch.BatchException", "Batch.ServerException"],
        interval=Duration.seconds(60),
        max_attempts=2,
        backoff_rate=2.0,
    )
    return t


def build_vae_trace_task_batch(
    scope: Construct,
    *,
    spec: dict,
    job_definition_arn: str,
    job_queue_arn: str,
    prefix: str,
) -> sfn.IChainable:
    step = spec["step"]
    t = tasks.BatchSubmitJob(
        scope,
        step,
        job_definition_arn=job_definition_arn,
        job_name=f"{prefix}-{step.lower()[:20]}",
        job_queue_arn=job_queue_arn,
        container_overrides=tasks.BatchContainerOverrides(
            command=[
                "/opt/ml/code/compile/compile.py",
            ],
            environment={
                "TRACE_SCRIPT": spec["script"],
                "EXPECTED_OUTPUT_FILE": spec["output"],
                "COMPILE_OUTPUT_MODE": "batch",
                "ZIMAGE_ROOT": "/opt/ml/code",
                "ZIMAGE_ARTIFACTS_DIR": EFS_ARTIFACTS_PATH,
            },
        ),
        result_path=f"$.traces.{step}",
        task_timeout=sfn.Timeout.duration(Duration.hours(2)),
    )
    t.add_retry(
        errors=["States.TaskFailed", "Batch.BatchException", "Batch.ServerException"],
        interval=Duration.seconds(60),
        max_attempts=2,
        backoff_rate=2.0,
    )
    return t


def build_warmup_task_batch(
    scope: Construct,
    *,
    resolution: int,
    job_definition_arn: str,
    job_queue_arn: str,
    prefix: str,
) -> sfn.IChainable:
    t = tasks.BatchSubmitJob(
        scope,
        f"WarmupCacheBatch{resolution}",
        job_definition_arn=job_definition_arn,
        job_name=f"{prefix}-warmup-{resolution}",
        job_queue_arn=job_queue_arn,
        payload=sfn.TaskInput.from_object(
            {
                "resolution": str(resolution),
            }
        ),
        result_path=f"$.warmup.res{resolution}",
        task_timeout=sfn.Timeout.duration(Duration.hours(3)),
    )
    t.add_retry(
        errors=[
            "Batch.BatchException",
            "Batch.ClientException",
            "Batch.ServerException",
            "States.TaskFailed",
        ],
        interval=Duration.seconds(60),
        max_attempts=2,
        backoff_rate=2.0,
    )
    return t
