from __future__ import annotations

from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct
from runtime.lib.cache_version import compute_cache_version
from runtime.lib.paths import CACHE_768_PREFIX, CACHE_1024_PREFIX, COMPILED_MODELS_DIR

from infra.stacks.compile_sfn.task_builders import (
    build_download_weights_task,
    build_neuron_trace_task,
    build_vae_trace_task_batch,
)
from infra.stacks.constructs.compute.compile_tasks import NEURON_TRACE_SPECS, TRACE_SPECS


def build_state_machine_definition(
    scope: Construct,
    *,
    job_definition_arn: str,
    job_queue_arn: str,
    cpu_job_definition_arn: str,
    cpu_job_queue_arn: str,
    prefix: str,
    check_artifacts_function,
    cache_purge_function,
    inferentia_enabled: bool = True,
) -> sfn.IChainable:
    cache_version = compute_cache_version()

    download_weights = build_download_weights_task(
        scope,
        cpu_job_definition_arn=cpu_job_definition_arn,
        cpu_job_queue_arn=cpu_job_queue_arn,
        prefix=prefix,
    )

    if not inferentia_enabled:
        check_weights = tasks.LambdaInvoke(
            scope,
            "CheckWeights",
            lambda_function=check_artifacts_function,
            payload=sfn.TaskInput.from_object(
                {
                    "check_only": True,
                    "key_prefix": "weights",
                    "expected_files": ["model_index.json"],
                    "min_bytes": 100,
                }
            ),
            result_path="$.weightsCheck",
            retry_on_service_exceptions=True,
        )
        done = sfn.Pass(scope, "WeightsReady")
        weights_choice = sfn.Choice(scope, "WeightsExist?")
        weights_choice.when(
            sfn.Condition.boolean_equals("$.weightsCheck.Payload.all_cached", True),
            done,
        )
        weights_choice.otherwise(download_weights.next(done))
        return check_weights.next(weights_choice)

    compile_parallel = sfn.Parallel(scope, "CompileTraces", result_path="$.traces")
    for spec in TRACE_SPECS:
        compile_parallel.branch(
            build_vae_trace_task_batch(
                scope,
                spec=spec,
                job_definition_arn=cpu_job_definition_arn,
                job_queue_arn=cpu_job_queue_arn,
                prefix=prefix,
            )
        )
    for spec in NEURON_TRACE_SPECS:
        compile_parallel.branch(
            build_neuron_trace_task(
                scope,
                spec=spec,
                job_definition_arn=job_definition_arn,
                job_queue_arn=job_queue_arn,
                prefix=prefix,
            )
        )

    from infra.stacks.compile_sfn.task_builders import build_warmup_task_batch

    warmup_parallel = sfn.Parallel(scope, "WarmupCache", result_path="$.warmup")
    for res in (768, 1024):
        warmup_parallel.branch(
            build_warmup_task_batch(
                scope,
                resolution=res,
                job_definition_arn=job_definition_arn,
                job_queue_arn=job_queue_arn,
                prefix=prefix,
            )
        )

    verify = tasks.LambdaInvoke(
        scope,
        "VerifyNEFFs",
        lambda_function=check_artifacts_function,
        payload=sfn.TaskInput.from_object(
            {
                "key_prefix": COMPILED_MODELS_DIR,
                "require_pjrt_cache": True,
                "cache_version": cache_version,
            }
        ),
        result_path="$.verify",
        retry_on_service_exceptions=True,
    )

    purge = tasks.LambdaInvoke(
        scope,
        "PurgeStaleCache",
        lambda_function=cache_purge_function,
        payload=sfn.TaskInput.from_object(
            {
                "cache_prefixes": [
                    f"{CACHE_1024_PREFIX}/{cache_version}/",
                    f"{CACHE_768_PREFIX}/{cache_version}/",
                ],
            }
        ),
        result_path="$.purge",
    )

    check_cache = tasks.LambdaInvoke(
        scope,
        "CheckCache",
        lambda_function=check_artifacts_function,
        payload=sfn.TaskInput.from_object(
            {
                "key_prefix": COMPILED_MODELS_DIR,
                "check_only": True,
                "require_pjrt_cache": True,
                "cache_version": cache_version,
            }
        ),
        result_path="$.cacheCheck",
        retry_on_service_exceptions=True,
    )

    check_weights = tasks.LambdaInvoke(
        scope,
        "CheckWeights",
        lambda_function=check_artifacts_function,
        payload=sfn.TaskInput.from_object(
            {
                "check_only": True,
                "key_prefix": "weights",
                "expected_files": ["model_index.json"],
                "min_bytes": 100,
            }
        ),
        result_path="$.weightsCheck",
        retry_on_service_exceptions=True,
    )

    check_traces = tasks.LambdaInvoke(
        scope,
        "CheckTraces",
        lambda_function=check_artifacts_function,
        payload=sfn.TaskInput.from_object(
            {
                "key_prefix": COMPILED_MODELS_DIR,
                "check_only": True,
            }
        ),
        result_path="$.traceCheck",
        retry_on_service_exceptions=True,
    )

    all_cached = sfn.Pass(scope, "AllCached")

    warmup_parallel.next(verify).next(purge)
    tail = warmup_parallel

    traces_choice = sfn.Choice(scope, "TracesExist?")
    traces_choice.when(
        sfn.Condition.boolean_equals("$.traceCheck.Payload.all_cached", True),
        tail,
    )
    traces_choice.otherwise(compile_parallel.next(tail))

    post_weights = check_traces.next(traces_choice)

    weights_choice = sfn.Choice(scope, "WeightsExist?")
    weights_choice.when(
        sfn.Condition.boolean_equals("$.weightsCheck.Payload.all_cached", True),
        post_weights,
    )
    weights_choice.otherwise(download_weights.next(post_weights))

    cache_choice = sfn.Choice(scope, "CacheHit?")
    cache_choice.when(
        sfn.Condition.boolean_equals("$.cacheCheck.Payload.all_cached", True),
        all_cached,
    )
    cache_choice.otherwise(check_weights.next(weights_choice))

    return check_cache.next(cache_choice)
