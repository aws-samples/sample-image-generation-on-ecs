import pytest
from infra.utils.contracts import GpuEndpointConfig, NetworkConfig
from runtime.lib.contracts.endpoint import InferenceRequest
from runtime.lib.contracts.enums import FrameType


class TestGpuEndpointConfig:
    def test_defaults(self) -> None:
        cfg = GpuEndpointConfig()
        assert cfg.max_pixels == 1048576
        assert cfg.min_dimension == 256
        assert cfg.dimension_step == 8
        assert cfg.min_capacity == 1

    def test_max_lt_min_raises(self) -> None:
        with pytest.raises(ValueError):
            GpuEndpointConfig(min_capacity=3, max_capacity=1)


class TestNetworkConfig:
    def test_defaults(self) -> None:
        cfg = NetworkConfig()
        assert cfg.vpc_cidr == "10.20.0.0/16"
        assert cfg.nat_count == 1

    def test_nat_count_bounds(self) -> None:
        with pytest.raises(ValueError):
            NetworkConfig(nat_count=0)
        with pytest.raises(ValueError):
            NetworkConfig(nat_count=4)


class TestInferenceRequest:
    def test_defaults(self) -> None:
        req = InferenceRequest(prompt="a cat")
        assert req.resolution == 512
        assert req.steps == 8
        assert req.seed == 42
        assert req.guidance_scale == 0.0

    def test_invalid_resolution(self) -> None:
        with pytest.raises(ValueError):
            InferenceRequest.model_validate({"prompt": "a cat", "resolution": 999})


class TestFrameType:
    def test_values(self) -> None:
        assert FrameType.INTERMEDIATE == 1
        assert FrameType.FINAL == 2
        assert FrameType.ERROR == 4
        assert FrameType.COMPLETE == 3
