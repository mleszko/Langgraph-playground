"""Domain models shared by graph/application layers."""

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage

Intent = Literal["weather", "out_of_scope"]


class GraphState(TypedDict):
    """Conversation state passed between graph nodes."""

    messages: list[BaseMessage]
    intent: NotRequired[Intent]
    attempts: NotRequired[int]
    max_attempts: NotRequired[int]
    is_correct: NotRequired[bool]
    verification_feedback: NotRequired[str]


@dataclass(frozen=True)
class VerificationDecision:
    """Result of validating the latest assistant answer."""

    is_correct: bool
    feedback: str

    def to_state_update(self) -> dict[str, Any]:
        """Convert verification result into graph-state update fields."""
        return {
            "is_correct": self.is_correct,
            "verification_feedback": self.feedback,
        }


# Backward-compatible alias while tests/migrations are in progress.
VerificationResult = VerificationDecision

