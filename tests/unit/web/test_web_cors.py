from __future__ import annotations

from unittest.mock import MagicMock


class TestShouldRegisterCors:
    def test_register_in_local_mode_only(self):
        from server import _should_register_cors

        local_settings = MagicMock(is_lambda=False)
        assert _should_register_cors(local_settings) is True

    def test_skip_when_running_in_lambda(self):
        from server import _should_register_cors

        lambda_settings = MagicMock(is_lambda=True)
        assert _should_register_cors(lambda_settings) is False
