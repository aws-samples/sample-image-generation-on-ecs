from runtime.lib.logging import configure_logging


class TestConfigureLogging:
    def test_configure_default(self) -> None:
        configure_logging()

    def test_configure_json_mode(self) -> None:
        configure_logging(json=True)

    def test_configure_debug_level(self) -> None:
        configure_logging(level="DEBUG")

    def test_configure_error_level(self) -> None:
        configure_logging(level="ERROR", json=True)
