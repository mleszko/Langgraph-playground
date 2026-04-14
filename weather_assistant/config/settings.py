"""Environment-backed settings for dependency wiring."""

import os
from dataclasses import dataclass


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class WeatherAssistantSettings:
    """Runtime settings used by the composition root."""

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    default_max_attempts: int = DEFAULT_MAX_ATTEMPTS
    database_url: str | None = None

    @classmethod
    def from_env(cls) -> "WeatherAssistantSettings":
        """Load settings from environment variables with safe fallbacks."""
        model = os.getenv("WEATHER_ASSISTANT_MODEL", DEFAULT_MODEL)

        temperature_raw = os.getenv("WEATHER_ASSISTANT_TEMPERATURE")
        try:
            temperature = (
                float(temperature_raw)
                if temperature_raw is not None
                else DEFAULT_TEMPERATURE
            )
        except ValueError:
            temperature = DEFAULT_TEMPERATURE

        max_attempts_raw = os.getenv("WEATHER_ASSISTANT_MAX_ATTEMPTS")
        try:
            default_max_attempts = (
                int(max_attempts_raw)
                if max_attempts_raw is not None
                else DEFAULT_MAX_ATTEMPTS
            )
        except ValueError:
            default_max_attempts = DEFAULT_MAX_ATTEMPTS

        database_url = os.getenv("DATABASE_URL")

        return cls(
            model=model,
            temperature=temperature,
            default_max_attempts=default_max_attempts,
            database_url=database_url or None,
        )

