"""High-level assistant service ports."""

from typing import Any, Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage

from weather_assistant.domain.models import Intent, VerificationDecision


class AssistantAIServicePort(Protocol):
    """Port for AI operations used by graph nodes."""

    def classify_intent(self, user_text: str) -> Intent:
        """Classify user text into domain intent."""

    def respond_weather(
        self, messages: Sequence[BaseMessage], tools: Sequence[Any]
    ) -> AIMessage:
        """Generate weather-focused response, optionally with tools."""

    def verify_answer(
        self, *, user_text: str, ai_text: str, tool_observations: Sequence[str]
    ) -> VerificationDecision:
        """Verify the latest answer and provide repair feedback when needed."""

    def repair_answer(self, *, user_text: str, ai_text: str, feedback: str) -> AIMessage:
        """Generate a corrected assistant answer."""

