import pytest


class TestDeploySettings:
    def test_loads_dev_defaults(self, monkeypatch) -> None:
        monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
        monkeypatch.setenv("CDK_DEFAULT_REGION", "us-east-1")
        monkeypatch.setenv("ZIMAGE_ENV", "dev")

        from infra.config.settings import DeploySettings

        settings = DeploySettings()
        assert settings.environment.value == "dev"
        assert settings.account == "123456789012"
        assert settings.region == "us-east-1"
        assert settings.inferentia_endpoint is not None
        assert settings.gpu_endpoint is not None

    def test_loads_with_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
        monkeypatch.setenv("CDK_DEFAULT_REGION", "us-east-1")

        from infra.config.settings import DeploySettings

        settings = DeploySettings(env_override="dev")
        assert settings.environment.value == "dev"
        assert settings.prefix == "dev-zimg"

    def test_env_override_parameter(self, monkeypatch) -> None:
        monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "111111111111")
        monkeypatch.setenv("CDK_DEFAULT_REGION", "us-west-2")
        monkeypatch.setenv("ZIMAGE_ENV", "dev")

        from infra.config.settings import DeploySettings

        settings = DeploySettings(env_override="staging")
        assert settings.environment.value == "staging"

    def test_missing_account_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("CDK_DEFAULT_ACCOUNT", raising=False)
        monkeypatch.delenv("CDK_DEFAULT_REGION", raising=False)
        monkeypatch.setenv("ZIMAGE_ENV", "dev")

        from infra.config.settings import DeploySettings

        with pytest.raises(KeyError):
            DeploySettings()

    def test_both_backends_disabled_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
        monkeypatch.setenv("CDK_DEFAULT_REGION", "us-east-1")
        monkeypatch.setenv("ZIMAGE_ENV", "dev")

        from unittest.mock import patch

        from infra.config.settings import DeploySettings

        config = {
            "inferentia_endpoint": {"enabled": False},
            "gpu_endpoint": {"enabled": False},
        }
        with patch("infra.config.settings._load_yaml", return_value=config):
            with pytest.raises(ValueError, match="At least one inference backend"):
                DeploySettings(env_override="dev")
