"""Application-level helpers orchestrating domain policies."""

from weather_assistant.domain.models import GraphState

DEFAULT_MAX_ATTEMPTS = 2


def with_default_attempt_limits(
    state: GraphState, *, default_max_attempts: int = DEFAULT_MAX_ATTEMPTS
) -> GraphState:
    """Ensure attempt counters are initialized for a run."""
    return {
        **state,
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", default_max_attempts),
    }

