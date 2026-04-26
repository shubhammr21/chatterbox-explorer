"""
tests/unit/infrastructure/test_settings.py
==========================================
TDD unit tests for infrastructure/settings.py — RestSettings.
Written before implementation (RED phase).
"""

from __future__ import annotations


class TestRestSettingsDefaults:
    def test_constructs_without_env_file(self) -> None:
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s is not None

    def test_default_server_host(self) -> None:
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.server.host == "0.0.0.0"

    def test_default_server_port(self) -> None:
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.server.port == 7860

    def test_default_log_level_is_info(self) -> None:
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.logging.level == "INFO"

    def test_default_environment_is_local(self) -> None:
        from infrastructure.constants import Environment
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.environment == Environment.LOCAL

    def test_default_hf_token_is_none(self) -> None:
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.huggingface.token is None


class TestRestSettingsFromEnv:
    def test_environment_overridable_via_env(self, monkeypatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "PRODUCTION")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        from infrastructure.constants import Environment

        assert s.environment == Environment.PRODUCTION

    def test_server_port_overridable_via_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SERVER__PORT", "9000")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.server.port == 9000

    def test_hf_token_is_secret_str(self, monkeypatch) -> None:
        monkeypatch.setenv("HUGGINGFACE__TOKEN", "hf_secret123")
        from pydantic import SecretStr

        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert isinstance(s.huggingface.token, SecretStr)

    def test_hf_token_masked_in_str(self, monkeypatch) -> None:
        monkeypatch.setenv("HUGGINGFACE__TOKEN", "hf_secret123")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert "hf_secret123" not in str(s.huggingface.token)

    def test_hf_token_accessible_via_get_secret_value(self, monkeypatch) -> None:
        monkeypatch.setenv("HUGGINGFACE__TOKEN", "hf_actual_value")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.huggingface.token is not None
        assert s.huggingface.token.get_secret_value() == "hf_actual_value"


class TestRestSettingsEnvironmentProperties:
    def test_local_environment_is_debug(self, monkeypatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "LOCAL")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.environment.is_debug is True

    def test_production_environment_is_deployed(self, monkeypatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "PRODUCTION")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.environment.is_deployed is True

    def test_production_uses_json_logs(self, monkeypatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "PRODUCTION")
        from infrastructure.settings import RestSettings

        s = RestSettings()
        assert s.environment.use_json_logs is True
