import pytest
from infra.utils.enums import Environment
from infra.utils.naming import ResourceNamer


class TestResourceNamer:
    def test_dev_endpoint(self) -> None:
        namer = ResourceNamer(Environment("dev"))
        assert namer.name("endpoint-512") == "dev-zimg-endpoint-512"

    def test_prod_artifacts(self) -> None:
        namer = ResourceNamer(Environment("prod"))
        assert (
            namer.name("artifacts-123456789012-us-east-1")
            == "prod-zimg-artifacts-123456789012-us-east-1"
        )

    def test_stack_name(self) -> None:
        namer = ResourceNamer(Environment("staging"))
        assert namer.stack_name("platform") == "staging-zimg-platform"

    def test_rejects_names_over_63_chars(self) -> None:
        namer = ResourceNamer(Environment("dev"))
        with pytest.raises(ValueError, match="exceeds AWS limit"):
            namer.name("a" * 60)

    def test_63_chars_exactly_is_valid(self) -> None:
        namer = ResourceNamer(Environment("dev"))
        result = namer.name("a" * 54)
        assert len(result) == 63
