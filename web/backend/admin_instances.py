from __future__ import annotations

from typing import Annotated

import boto3
from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from auth import Role, TokenClaims, require_role
from config import settings

router = APIRouter(prefix="/admin/api", tags=["admin"])

_require_admin = require_role(Role.ADMIN)

_RESOLUTIONS = [512, 768, 1024, "gpu"]


class TargetInstance(BaseModel):
    id: str
    availability_zone: str
    health_state: str
    reason: str | None = None
    active_connections: float | None = None


class EndpointInstances(BaseModel):
    name: str
    resolution: str
    running_count: int
    desired_count: int
    targets: list[TargetInstance]


@router.get("/instances")
async def get_instances(
    user: Annotated[TokenClaims, Depends(_require_admin)],
) -> list[EndpointInstances]:
    if settings.local:
        from mock_data import mock_instances

        return [EndpointInstances(**ep) for ep in mock_instances()]
    return _fetch_instance_data()


def _fetch_instance_data() -> list[EndpointInstances]:
    ecs = boto3.client("ecs", region_name=settings.region)
    elbv2 = boto3.client("elbv2", region_name=settings.region)

    cluster = f"{settings.prefix}-inference-cluster"
    service_names = [f"{settings.prefix}-inference-{r}" for r in _RESOLUTIONS]

    try:
        svc_resp = ecs.describe_services(cluster=cluster, services=service_names)
    except Exception as e:
        logger.warning(f"Failed to describe ECS services: {e}")
        return []

    results: list[EndpointInstances] = []

    for svc in svc_resp.get("services", []):
        svc_name = svc["serviceName"]
        resolution = svc_name.split("-")[-1]
        tg_name = f"{settings.prefix}-inf-{resolution}-tg"
        targets = _get_target_health(elbv2, tg_name)

        results.append(
            EndpointInstances(
                name=svc_name,
                resolution=resolution,
                running_count=svc["runningCount"],
                desired_count=svc["desiredCount"],
                targets=targets,
            )
        )

    return results


def _get_target_health(elbv2, tg_name: str) -> list[TargetInstance]:
    try:
        tgs = elbv2.describe_target_groups(Names=[tg_name])
        if not tgs["TargetGroups"]:
            return []
        tg_arn = tgs["TargetGroups"][0]["TargetGroupArn"]
        health_resp = elbv2.describe_target_health(TargetGroupArn=tg_arn)
    except Exception as e:
        logger.debug(f"Could not get target health for {tg_name}: {e}")
        return []

    targets = []
    for desc in health_resp.get("TargetHealthDescriptions", []):
        target = desc["Target"]
        health = desc["TargetHealth"]
        targets.append(
            TargetInstance(
                id=target["Id"],
                availability_zone=target.get("AvailabilityZone", "unknown"),
                health_state=health["State"],
                reason=health.get("Reason"),
            )
        )
    return targets
