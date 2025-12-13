"""Application configuration.

Follows 12-factor app principles with environment variable configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AUDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service info
    service_name: str = "audit-service"
    version: str = "0.1.0"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8084
    debug: bool = False

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Rules configuration
    rules_path: str = "config/rules.yaml"
    patterns_path: str = "config/patterns.yaml"

    # External service URLs (populated in Phase 6+)
    llm_gateway_url: str = "http://localhost:8081"
    semantic_search_url: str = "http://localhost:8082"
    ai_platform_data_url: str = "http://localhost:8083"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
