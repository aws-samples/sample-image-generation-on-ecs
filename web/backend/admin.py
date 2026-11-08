from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Annotated

import boto3
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from audit import audit
from auth import Role, TokenClaims, require_role
from config import settings

router = APIRouter(prefix="/admin/api", tags=["admin"])

_require_admin = require_role(Role.ADMIN)


class EndpointStatus(BaseModel):
    name: str
    status: str
    instance_type: str
    current_instance_count: int
    desired_instance_count: int


class EndpointMetrics(BaseModel):
    invocations: list[float]
    latency_p99_ms: list[float]
    errors_4xx: list[float]
    errors_5xx: list[float]
    timestamps: list[str]


class MetricTimeSeries(BaseModel):
    timestamps: list[str]
    values: list[float]


class EndpointMetricsTimeSeries(BaseModel):
    latency: MetricTimeSeries
    invocations: MetricTimeSeries
    errors5xx: MetricTimeSeries


class AlarmState(BaseModel):
    name: str
    state: str
    endpoint: str
    metric: str


class ScaleRequest(BaseModel):
    min_capacity: int = Field(ge=0)
    max_capacity: int = Field(ge=1)


class SfnExecution(BaseModel):
    execution_arn: str
    name: str
    status: str
    start_date: str
    stop_date: str | None = None


@router.get("/endpoints")
async def list_endpoints(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> list[EndpointStatus]:
    if settings.local:
        from mock_data import mock_endpoints

        return [EndpointStatus(**ep) for ep in mock_endpoints()]
    ecs = boto3.client("ecs", region_name=settings.region)
    cluster = f"{settings.prefix}-inference-cluster"
    services = [f"{settings.prefix}-inference-{r}" for r in [512, 768, 1024, "gpu"]]
    endpoints = []
    try:
        resp = ecs.describe_services(cluster=cluster, services=services)
        for svc in resp.get("services", []):
            is_gpu = "gpu" in svc["serviceName"]
            endpoints.append(
                EndpointStatus(
                    name=svc["serviceName"],
                    status="InService" if svc["runningCount"] > 0 else "Loading",
                    instance_type="g5.xlarge" if is_gpu else "inf2.8xlarge",
                    current_instance_count=svc["runningCount"],
                    desired_instance_count=svc["desiredCount"],
                )
            )
    except Exception as e:
        logger.error(f"list_endpoints failed: {e}")
    return endpoints


@router.get("/endpoints/{name}/metrics")
async def get_endpoint_metrics(
    name: str,
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> EndpointMetrics:
    if settings.local:
        from mock_data import mock_metrics

        return EndpointMetrics(**mock_metrics())
    res = name.split("-")[-1] if "-" in name else name
    tg_name = f"{settings.prefix}-inf-{res}-tg"
    cw = boto3.client("cloudwatch", region_name=settings.region)
    elbv2 = boto3.client("elbv2", region_name=settings.region)
    end = datetime.now(UTC)
    start = end - timedelta(hours=1)

    try:
        tgs = elbv2.describe_target_groups(Names=[tg_name])
        tg_arn = tgs["TargetGroups"][0]["TargetGroupArn"]
        tg_suffix = "/".join(tg_arn.split(":")[-1].split("/")[1:])
    except Exception as exc:
        logger.warning(f"silent catch: {exc}")
        return EndpointMetrics(
            invocations=[], latency_p99_ms=[], errors_4xx=[], errors_5xx=[], timestamps=[]
        )

    def _get_metric(metric_name: str, stat: str = "Sum") -> tuple[list[float], list[str]]:
        resp = cw.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "m1",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AWS/ApplicationELB",
                            "MetricName": metric_name,
                            "Dimensions": [
                                {"Name": "TargetGroup", "Value": f"targetgroup/{tg_suffix}"},
                            ],
                        },
                        "Period": 60,
                        "Stat": stat,
                    },
                }
            ],
            StartTime=start,
            EndTime=end,
        )
        results = resp["MetricDataResults"][0]
        return results["Values"], [t.isoformat() for t in results["Timestamps"]]

    invocations, timestamps = _get_metric("RequestCountPerTarget")
    latency, _ = _get_metric("TargetResponseTime", "p99")
    errors_4xx, _ = _get_metric("HTTPCode_Target_4XX_Count")
    errors_5xx, _ = _get_metric("HTTPCode_Target_5XX_Count")

    return EndpointMetrics(
        invocations=invocations,
        latency_p99_ms=[v / 1000 for v in latency],
        errors_4xx=errors_4xx,
        errors_5xx=errors_5xx,
        timestamps=timestamps,
    )


@router.get("/metrics")
async def get_metrics_timeseries(
    endpoint: str,
    user: Annotated[TokenClaims, Depends(_require_admin)],
    hours: int = 6,
) -> EndpointMetricsTimeSeries:
    if settings.local:
        from mock_data import mock_metrics_timeseries

        return EndpointMetricsTimeSeries(**mock_metrics_timeseries(hours))
    res = endpoint.split("-")[-1] if "-" in endpoint else endpoint
    tg_name = f"{settings.prefix}-inf-{res}-tg"
    cw = boto3.client("cloudwatch", region_name=settings.region)
    elbv2_client = boto3.client("elbv2", region_name=settings.region)
    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)

    try:
        tgs = elbv2_client.describe_target_groups(Names=[tg_name])
        tg_arn = tgs["TargetGroups"][0]["TargetGroupArn"]
        tg_suffix = "/".join(tg_arn.split(":")[-1].split("/")[1:])
    except Exception as exc:
        logger.warning(f"silent catch: {exc}")
        empty = MetricTimeSeries(timestamps=[], values=[])
        return EndpointMetricsTimeSeries(latency=empty, invocations=empty, errors5xx=empty)

    tg_dim = [{"Name": "TargetGroup", "Value": f"targetgroup/{tg_suffix}"}]
    response = cw.get_metric_data(
        MetricDataQueries=[
            {
                "Id": "latency",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/ApplicationELB",
                        "MetricName": "TargetResponseTime",
                        "Dimensions": tg_dim,
                    },
                    "Period": 300,
                    "Stat": "p99",
                },
            },
            {
                "Id": "invocations",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/ApplicationELB",
                        "MetricName": "RequestCountPerTarget",
                        "Dimensions": tg_dim,
                    },
                    "Period": 300,
                    "Stat": "Sum",
                },
            },
            {
                "Id": "errors5xx",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/ApplicationELB",
                        "MetricName": "HTTPCode_Target_5XX_Count",
                        "Dimensions": tg_dim,
                    },
                    "Period": 300,
                    "Stat": "Sum",
                },
            },
        ],
        StartTime=start,
        EndTime=end,
    )

    metrics: dict[str, MetricTimeSeries] = {}
    for result in response["MetricDataResults"]:
        metrics[result["Id"]] = MetricTimeSeries(
            timestamps=[t.isoformat() for t in result["Timestamps"]],
            values=result["Values"],
        )

    empty = MetricTimeSeries(timestamps=[], values=[])
    return EndpointMetricsTimeSeries(
        latency=metrics.get("latency", empty),
        invocations=metrics.get("invocations", empty),
        errors5xx=metrics.get("errors5xx", empty),
    )


@router.post("/endpoints/{name}/scale")
async def scale_endpoint(
    name: str,
    req: ScaleRequest,
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> dict:
    if settings.local:
        return {"status": "ok", "service": name, "desired": req.min_capacity}
    if req.max_capacity < req.min_capacity:
        raise HTTPException(400, "max_capacity must be >= min_capacity")
    ecs = boto3.client("ecs", region_name=settings.region)
    cluster = f"{settings.prefix}-inference-cluster"
    service_name = (
        name if name.startswith(settings.prefix) else f"{settings.prefix}-inference-{name}"
    )
    ecs.update_service(cluster=cluster, service=service_name, desiredCount=req.min_capacity)
    audit(
        actor_sub=user.sub,
        actor_email=user.email,
        action="scale_service",
        resource=name,
        desired_count=req.min_capacity,
    )
    return {"status": "ok", "service": name, "desired": req.min_capacity}


@router.delete("/endpoints/{name}")
async def delete_endpoint(
    name: str,
    user: Annotated[TokenClaims, Depends(_require_admin)],
    confirm: str | None = None,
) -> dict:
    _validate_endpoint_name(name)
    if confirm != name:
        raise HTTPException(
            400,
            "Confirmation required: append ?confirm=<service-name> to the DELETE request",
        )
    if settings.local:
        audit(actor_sub=user.sub, actor_email=user.email, action="stop_service", resource=name)
        return {"status": "stopped", "service": name}
    ecs = boto3.client("ecs", region_name=settings.region)
    cluster = f"{settings.prefix}-inference-cluster"
    ecs.update_service(cluster=cluster, service=name, desiredCount=0)
    audit(actor_sub=user.sub, actor_email=user.email, action="stop_service", resource=name)
    return {"status": "stopped", "service": name}


@router.get("/alarms")
async def list_alarms(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> list[AlarmState]:
    if settings.local:
        from mock_data import mock_alarms

        return [AlarmState(**a) for a in mock_alarms()]
    cw = boto3.client("cloudwatch", region_name=settings.region)
    resp = cw.describe_alarms(AlarmNamePrefix="dev-zimg-")
    alarms = []
    for alarm in resp.get("MetricAlarms", []):
        dims = {d["Name"]: d["Value"] for d in alarm.get("Dimensions", [])}
        alarms.append(
            AlarmState(
                name=alarm["AlarmName"],
                state=alarm["StateValue"],
                endpoint=dims.get("EndpointName", ""),
                metric=alarm["MetricName"],
            )
        )
    return alarms


@router.get("/pipeline")
async def get_pipeline_status(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> list[dict]:
    if settings.local:
        from mock_data import mock_pipeline

        return mock_pipeline()
    cp = boto3.client("codepipeline", region_name=settings.region)
    pipeline_name = f"{settings.prefix}-inference-pipeline"
    try:
        resp = cp.get_pipeline_state(name=pipeline_name)
        return [
            {"stage": s["stageName"], "status": s.get("latestExecution", {}).get("status")}
            for s in resp["stageStates"]
        ]
    except cp.exceptions.PipelineNotFoundException:
        return []


@router.get("/sfn")
async def get_sfn_status(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> list[SfnExecution]:
    if settings.local:
        from mock_data import mock_sfn_executions

        return [SfnExecution(**e) for e in mock_sfn_executions()]
    sfn = boto3.client("stepfunctions", region_name=settings.region)
    sfn_arn = _compile_sfn_arn()
    resp = sfn.list_executions(stateMachineArn=sfn_arn, maxResults=10)
    return [
        SfnExecution(
            execution_arn=e["executionArn"],
            name=e["name"],
            status=e["status"],
            start_date=e["startDate"].isoformat(),
            stop_date=e["stopDate"].isoformat() if e.get("stopDate") else None,
        )
        for e in resp["executions"]
    ]


@router.post("/sfn/start")
async def start_sfn(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> dict:
    if settings.local:
        return {"execution_arn": "arn:mock:local", "name": "local-mock"}
    sfn = boto3.client("stepfunctions", region_name=settings.region)
    sfn_arn = _compile_sfn_arn()
    import uuid

    name = f"admin-{uuid.uuid4().hex[:8]}"
    resp = sfn.start_execution(stateMachineArn=sfn_arn, name=name, input="{}")
    audit(
        actor_sub=user.sub,
        actor_email=user.email,
        action="start_sfn",
        resource=sfn_arn,
        execution_name=name,
    )
    return {"execution_arn": resp["executionArn"], "name": name}


@router.post("/sfn/stop")
async def stop_sfn(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> dict:
    if settings.local:
        return {"status": "stopped", "execution_arn": "arn:mock:local"}
    sfn = boto3.client("stepfunctions", region_name=settings.region)
    sfn_arn = _compile_sfn_arn()
    resp = sfn.list_executions(stateMachineArn=sfn_arn, statusFilter="RUNNING", maxResults=1)
    if not resp["executions"]:
        return {"status": "no running execution"}
    exec_arn = resp["executions"][0]["executionArn"]
    sfn.stop_execution(executionArn=exec_arn)
    audit(
        actor_sub=user.sub,
        actor_email=user.email,
        action="stop_sfn",
        resource=exec_arn,
    )
    return {"status": "stopped", "execution_arn": exec_arn}


def _service_names() -> list[str]:
    return [f"{settings.prefix}-inference-{r}" for r in [512, 768, 1024, "gpu"]]


def _validate_endpoint_name(name: str) -> None:
    valid = _service_names() + ["512", "768", "1024", "gpu"]
    if name not in valid:
        raise HTTPException(404, f"Unknown service: {name}")


@lru_cache(maxsize=1)
def _account_id() -> str:
    return boto3.client("sts", region_name=settings.region).get_caller_identity()["Account"]


def _compile_sfn_arn() -> str:
    return (
        f"arn:aws:states:{settings.region}:{_account_id()}:stateMachine:"
        f"{settings.prefix}-compile-sfn"
    )
