"""
src/infrastructure/settings.py
================================
RestSettings — pydantic-settings BaseSettings for the FastAPI REST adapter.

IMPORTANT: This module must ONLY be imported inside build_rest_app() (deferred
import inside the function body). Never import at module level — it would break
the compat-patch ordering guarantee.

Available only when the 'rest' optional extra is installed:
    uv sync --extra rest
"""

from __future__ import annotations

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from infrastructure.constants import Environment


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860


class LoggingSettings(BaseModel):
    level: str = "INFO"


class HuggingFaceSettings(BaseModel):
    token: SecretStr | None = None


class RestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    server: ServerSettings = ServerSettings()
    logging: LoggingSettings = LoggingSettings()
    huggingface: HuggingFaceSettings = HuggingFaceSettings()
