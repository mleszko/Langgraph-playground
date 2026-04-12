from weather_assistant.application.use_cases import with_default_attempt_limits
from weather_assistant.domain.models import VerificationDecision
from weather_assistant.domain.policies import (
    route_after_planner,
    route_after_verification,
)


def test_route_after_planner_defaults_to_out_of_scope() -> None:
    assert route_after_planner("weather") == "weather_agent"
    assert route_after_planner("out_of_scope") == "out_of_scope"
    assert route_after_planner(None) == "out_of_scope"


def test_route_after_verification_stops_for_success_or_limit() -> None:
    assert route_after_verification(is_correct=True, attempts=0, max_attempts=2) == "end"
    assert route_after_verification(is_correct=False, attempts=2, max_attempts=2) == "end"
    assert route_after_verification(is_correct=False, attempts=1, max_attempts=2) == "repair"


def test_with_default_attempt_limits_sets_missing_values() -> None:
    state = with_default_attempt_limits({"messages": []})
    assert state["attempts"] == 0
    assert state["max_attempts"] == 2


def test_with_default_attempt_limits_keeps_explicit_values() -> None:
    state = with_default_attempt_limits({"messages": [], "attempts": 3, "max_attempts": 5})
    assert state["attempts"] == 3
    assert state["max_attempts"] == 5


def test_with_default_attempt_limits_allows_custom_default() -> None:
    state = with_default_attempt_limits({"messages": []}, default_max_attempts=4)
    assert state["attempts"] == 0
    assert state["max_attempts"] == 4


def test_verification_decision_fields() -> None:
    result = VerificationDecision(is_correct=False, feedback="Needs correction")
    assert result.is_correct is False
    assert result.feedback == "Needs correction"

