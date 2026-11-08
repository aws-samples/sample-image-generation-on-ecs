from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def _cdk_env(monkeypatch) -> None:
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
    monkeypatch.setenv("CDK_DEFAULT_REGION", "us-east-1")


@pytest.mark.usefixtures("_cdk_env")
class TestCognitoDomainPrefix:
    def test_dev_default_is_account_scoped_and_valid(self) -> None:
        from infra.config.settings import DeploySettings

        settings = DeploySettings(env_override="dev")
        assert settings.cognito_domain_prefix == "dev-zimg-auth-123456789012"

    def test_other_valid_env_default_passes(self) -> None:
        from infra.config.settings import DeploySettings

        settings = DeploySettings(env_override="staging2")
        assert settings.cognito_domain_prefix == "staging2-zimg-auth-123456789012"

    def test_overlong_env_default_raises(self) -> None:
        from infra.config.settings import DeploySettings

        env = "a" * 41
        settings = DeploySettings(env_override=env)
        assert len(f"{env}-zimg-auth-123456789012") > 63
        with pytest.raises(ValueError, match="computed default") as exc:
            _ = settings.cognito_domain_prefix
        assert "web.cognito_domain_prefix" in str(exc.value)
        assert env in str(exc.value)

    def test_reserved_substring_in_env_default_raises(self) -> None:
        from infra.config.settings import DeploySettings

        settings = DeploySettings(env_override="awsdev")
        with pytest.raises(ValueError, match="awsdev-zimg-auth"):
            _ = settings.cognito_domain_prefix

    @pytest.mark.parametrize("env", ["Dev", "dev_x", "dev-x"])
    def test_env_with_uppercase_or_underscore_raises(self, env: str) -> None:
        from infra.config.settings import DeploySettings

        with pytest.raises(ValueError):
            _ = DeploySettings(env_override=env).cognito_domain_prefix

    def test_valid_yaml_override_wins(self) -> None:
        from infra.config.settings import DeploySettings

        config = {"web": {"cognito_domain_prefix": "my-pinned-auth"}}
        with patch("infra.config.settings._load_yaml", return_value=config):
            settings = DeploySettings(env_override="dev")
        assert settings.cognito_domain_prefix == "my-pinned-auth"

    def test_invalid_yaml_override_rejected_at_load(self) -> None:
        from infra.config.settings import DeploySettings

        config = {"web": {"cognito_domain_prefix": "Bad_Prefix"}}
        with patch("infra.config.settings._load_yaml", return_value=config):
            with pytest.raises(ValueError, match="cognito_domain_prefix"):
                DeploySettings(env_override="dev")
