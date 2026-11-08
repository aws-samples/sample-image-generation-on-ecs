from infra.utils.contracts import TagConfig
from infra.utils.enums import Environment
from infra.utils.tagging import build_tags


class TestBuildTags:
    def test_dev_tags(self) -> None:
        tags = build_tags(Environment("dev"), TagConfig())
        assert tags["Environment"] == "dev"
        assert tags["Project"] == "z-image"
        assert tags["ManagedBy"] == "CDK"

    def test_prod_tags(self) -> None:
        tags = build_tags(Environment("prod"), TagConfig(owner="platform-team"))
        assert tags["Environment"] == "prod"
        assert tags["Owner"] == "platform-team"

    def test_custom_cost_center(self) -> None:
        tags = build_tags(Environment("staging"), TagConfig(cost_center="research"))
        assert tags["CostCenter"] == "research"

    def test_returns_all_required_keys(self) -> None:
        tags = build_tags(Environment("dev"), TagConfig())
        expected_keys = {"App", "Project", "Environment", "Owner", "CostCenter", "ManagedBy"}
        assert set(tags.keys()) == expected_keys
