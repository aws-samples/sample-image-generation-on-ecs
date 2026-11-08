import pytest
from infra.utils.enums import Environment
from runtime.lib.contracts.enums import Resolution


class TestEnvironment:
    def test_all_values(self) -> None:
        assert Environment("dev") == "dev"
        assert Environment("staging") == "staging"
        assert Environment("prod") == "prod"

    def test_from_string(self) -> None:
        assert Environment("dev") == Environment("dev")

    def test_custom_env_works(self) -> None:
        assert Environment("gpu") == "gpu"
        assert Environment("neuron") == "neuron"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            Environment("Dev")
        with pytest.raises(ValueError):
            Environment("my-env")


class TestResolution:
    def test_values_are_integers(self) -> None:
        assert Resolution.RES_512 == 512
        assert Resolution.RES_768 == 768
        assert Resolution.RES_1024 == 1024

    def test_iteration(self) -> None:
        resolutions = list(Resolution)
        assert len(resolutions) == 3
