from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from auth import TokenClaims

_ADMIN_USER = TokenClaims(sub="admin-1", email="admin@test.com", groups=["admins", "users"])


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("admin.settings") as s:
        s.local = False
        s.region = "us-east-1"
        s.prefix = "dev-zimg"
        s.endpoint_512 = "zimage-endpoint-512"
        s.endpoint_768 = "zimage-endpoint-768"
        s.endpoint_1024 = "zimage-endpoint-1024"
        s.endpoint_gpu = "zimage-endpoint-gpu"
        yield s


@pytest.fixture(autouse=True)
def mock_boto3_client():
    clients = {}

    def _make_client(service, **kwargs):
        if service not in clients:
            mock = MagicMock()
            if service == "codepipeline":
                mock.exceptions = MagicMock()
                mock.exceptions.PipelineNotFoundException = type(
                    "PipelineNotFoundException", (Exception,), {}
                )
            clients[service] = mock
        return clients[service]

    with patch("admin.boto3.client", side_effect=_make_client) as _:
        yield clients


class TestListEndpoints:
    def test_returns_endpoint_statuses(self, mock_boto3_client):
        from admin import list_endpoints

        ecs = MagicMock()
        ecs.describe_services.return_value = {
            "services": [
                {
                    "serviceName": "dev-zimg-inference-512",
                    "runningCount": 1,
                    "desiredCount": 1,
                },
                {
                    "serviceName": "dev-zimg-inference-768",
                    "runningCount": 1,
                    "desiredCount": 1,
                },
                {
                    "serviceName": "dev-zimg-inference-1024",
                    "runningCount": 0,
                    "desiredCount": 1,
                },
            ]
        }
        mock_boto3_client["ecs"] = ecs

        result = _run(list_endpoints(user=_ADMIN_USER))
        assert len(result) == 3
        assert result[0].name == "dev-zimg-inference-512"
        assert result[0].status == "InService"
        assert result[2].status == "Loading"


class TestGetEndpointMetrics:
    def test_returns_metrics(self, mock_boto3_client):
        from admin import get_endpoint_metrics

        cw = MagicMock()
        cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {
                    "Values": [10.0, 20.0],
                    "Timestamps": [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                    ],
                }
            ]
        }
        mock_boto3_client["cloudwatch"] = cw

        result = _run(get_endpoint_metrics(name="dev-zimg-inference-512", user=_ADMIN_USER))
        assert result.invocations == [10.0, 20.0]
        assert len(result.timestamps) == 2
        assert result.latency_p99_ms == [0.01, 0.02]


class TestScaleEndpoint:
    def test_calls_ecs_update_service(self, mock_boto3_client):
        from admin import ScaleRequest, scale_endpoint

        ecs = MagicMock()
        mock_boto3_client["ecs"] = ecs

        req = ScaleRequest(min_capacity=1, max_capacity=4)
        result = _run(scale_endpoint(name="dev-zimg-inference-512", req=req, user=_ADMIN_USER))
        assert result["status"] == "ok"
        assert result["desired"] == 1
        ecs.update_service.assert_called_once()

    def test_rejects_invalid_max_less_than_min(self, mock_boto3_client):
        from admin import scale_endpoint

        req = MagicMock()
        req.min_capacity = 5
        req.max_capacity = 2

        with pytest.raises(HTTPException) as exc_info:
            _run(scale_endpoint(name="dev-zimg-inference-512", req=req, user=_ADMIN_USER))
        assert exc_info.value.status_code == 400
        assert "max_capacity" in exc_info.value.detail


class TestDeleteEndpoint:
    def test_scales_ecs_to_zero(self, mock_boto3_client):
        from admin import delete_endpoint

        ecs = MagicMock()
        mock_boto3_client["ecs"] = ecs

        result = _run(
            delete_endpoint(
                name="dev-zimg-inference-512",
                user=_ADMIN_USER,
                confirm="dev-zimg-inference-512",
            )
        )
        assert result["status"] == "stopped"
        ecs.update_service.assert_called_once()

    def test_rejects_unknown_service(self, mock_boto3_client):
        from admin import delete_endpoint

        with pytest.raises(HTTPException) as exc_info:
            _run(
                delete_endpoint(
                    name="unknown-endpoint",
                    user=_ADMIN_USER,
                    confirm="unknown-endpoint",
                )
            )
        assert exc_info.value.status_code == 404
        assert "Unknown service" in exc_info.value.detail

    def test_rejects_without_confirm(self, mock_boto3_client):
        from admin import delete_endpoint

        with pytest.raises(HTTPException) as exc_info:
            _run(delete_endpoint(name="dev-zimg-inference-512", user=_ADMIN_USER, confirm=None))
        assert exc_info.value.status_code == 400


class TestListAlarms:
    def test_returns_alarm_states(self, mock_boto3_client):
        from admin import list_alarms

        cw = MagicMock()
        cw.describe_alarms.return_value = {
            "MetricAlarms": [
                {
                    "AlarmName": "dev-zimg-high-latency",
                    "StateValue": "ALARM",
                    "MetricName": "ModelLatency",
                    "Dimensions": [{"Name": "EndpointName", "Value": "zimage-endpoint-512"}],
                }
            ]
        }
        mock_boto3_client["cloudwatch"] = cw

        result = _run(list_alarms(user=_ADMIN_USER))
        assert len(result) == 1
        assert result[0].name == "dev-zimg-high-latency"
        assert result[0].state == "ALARM"
        assert result[0].endpoint == "zimage-endpoint-512"
        assert result[0].metric == "ModelLatency"


class TestGetPipelineStatus:
    def test_returns_stage_list(self, mock_boto3_client):
        from admin import get_pipeline_status

        cp = MagicMock()
        cp.get_pipeline_state.return_value = {
            "stageStates": [
                {"stageName": "Source", "latestExecution": {"status": "Succeeded"}},
                {"stageName": "Build", "latestExecution": {"status": "InProgress"}},
            ]
        }
        cp.exceptions = MagicMock()
        cp.exceptions.PipelineNotFoundException = type(
            "PipelineNotFoundException", (Exception,), {}
        )
        mock_boto3_client["codepipeline"] = cp

        result = _run(get_pipeline_status(user=_ADMIN_USER))
        assert len(result) == 2
        assert result[0] == {"stage": "Source", "status": "Succeeded"}
        assert result[1] == {"stage": "Build", "status": "InProgress"}


class TestGetSfnStatus:
    def test_returns_executions(self, mock_boto3_client):
        from admin import get_sfn_status

        sfn = MagicMock()
        sfn.list_executions.return_value = {
            "executions": [
                {
                    "executionArn": "arn:aws:states:us-east-1:123456789012:execution:dev-zimg-compile-sfn:run-1",
                    "name": "run-1",
                    "status": "SUCCEEDED",
                    "startDate": datetime(2024, 1, 1, tzinfo=UTC),
                    "stopDate": datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
                }
            ]
        }
        mock_boto3_client["stepfunctions"] = sfn

        with patch("admin._account_id", return_value="123456789012"):
            result = _run(get_sfn_status(user=_ADMIN_USER))
        assert len(result) == 1
        assert result[0].name == "run-1"
        assert result[0].status == "SUCCEEDED"
        assert result[0].execution_arn.endswith("run-1")


class TestStartSfn:
    def test_starts_execution(self, mock_boto3_client):
        from admin import start_sfn

        sfn = MagicMock()
        sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123456789012:execution:dev-zimg-compile-sfn:admin-abc12345"
        }
        mock_boto3_client["stepfunctions"] = sfn

        with patch("admin._account_id", return_value="123456789012"):
            result = _run(start_sfn(user=_ADMIN_USER))
        assert "execution_arn" in result
        assert "name" in result
        sfn.start_execution.assert_called_once()


class TestStopSfn:
    def test_stops_running_execution(self, mock_boto3_client):
        from admin import stop_sfn

        sfn = MagicMock()
        sfn.list_executions.return_value = {
            "executions": [
                {
                    "executionArn": "arn:aws:states:us-east-1:123456789012:execution:dev-zimg-compile-sfn:run-2"
                }
            ]
        }
        mock_boto3_client["stepfunctions"] = sfn

        with patch("admin._account_id", return_value="123456789012"):
            result = _run(stop_sfn(user=_ADMIN_USER))
        assert result["status"] == "stopped"
        assert "execution_arn" in result
        sfn.stop_execution.assert_called_once()

    def test_returns_message_when_nothing_running(self, mock_boto3_client):
        from admin import stop_sfn

        sfn = MagicMock()
        sfn.list_executions.return_value = {"executions": []}
        mock_boto3_client["stepfunctions"] = sfn

        with patch("admin._account_id", return_value="123456789012"):
            result = _run(stop_sfn(user=_ADMIN_USER))
        assert result["status"] == "no running execution"
