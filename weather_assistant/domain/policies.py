"""Pure domain policy helpers (no model calls)."""

from typing import Literal

from .models import Intent

WEATHER_ONLY_REFUSAL_TEXT = (
    "I can only help with weather questions. Ask me about the weather in a city."
)

WEATHER_ONLY_SYSTEM_PROMPT = (
    "You are a weather-only assistant. Answer only weather-related questions. "
    "If the user asks about anything else, respond exactly with: "
    f"'{WEATHER_ONLY_REFUSAL_TEXT}'"
)

WEATHER_ONLY_REPAIR_PROMPT = (
    "You are a weather-only assistant. Provide only weather-related answers "
    "and stay grounded in available tool outputs."
)


def route_after_planner(intent: Intent | None) -> Literal["weather_agent", "out_of_scope"]:
    """Route to weather path only for weather intent."""
    return "weather_agent" if intent == "weather" else "out_of_scope"


def route_after_verification(
    *, is_correct: bool, attempts: int, max_attempts: int
) -> Literal["repair", "end"]:
    """Stop when verified or when max attempts is reached."""
    if is_correct or attempts >= max_attempts:
        return "end"
    return "repair"

