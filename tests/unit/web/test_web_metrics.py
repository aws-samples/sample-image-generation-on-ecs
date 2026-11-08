from __future__ import annotations

import json


def _read_emf(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    last = out.splitlines()[-1]
    return json.loads(last)


class TestEmitMetric:
    def test_emits_emf_format_with_namespace(self, capsys):
        from metrics import emit_metric

        emit_metric("TestMetric", 42)
        emf = _read_emf(capsys)
        assert emf["TestMetric"] == 42
        assert emf["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "ZImageWeb"
        assert emf["_aws"]["CloudWatchMetrics"][0]["Metrics"][0]["Name"] == "TestMetric"

    def test_emits_dimensions_when_provided(self, capsys):
        from metrics import emit_metric

        emit_metric("Counter", 1, action="delete_endpoint")
        emf = _read_emf(capsys)
        assert emf["action"] == "delete_endpoint"
        assert "action" in emf["_aws"]["CloudWatchMetrics"][0]["Dimensions"][0]


class TestSpecializedEmitters:
    def test_auth_failure_includes_reason(self, capsys):
        from metrics import emit_auth_failure

        emit_auth_failure("expired_token")
        emf = _read_emf(capsys)
        assert emf["AuthFailures"] == 1
        assert emf["reason"] == "expired_token"

    def test_admin_action(self, capsys):
        from metrics import emit_admin_action

        emit_admin_action("scale_endpoint")
        emf = _read_emf(capsys)
        assert emf["AdminActions"] == 1
        assert emf["action"] == "scale_endpoint"

    def test_generation_latency_records_milliseconds(self, capsys):
        from metrics import emit_generation_latency

        emit_generation_latency(1234.5, backend="neuron", resolution=768)
        emf = _read_emf(capsys)
        assert emf["GenerationLatencyMs"] == 1234.5
        assert emf["backend"] == "neuron"
        assert emf["resolution"] == "768"
        assert emf["_aws"]["CloudWatchMetrics"][0]["Metrics"][0]["Unit"] == "Milliseconds"
