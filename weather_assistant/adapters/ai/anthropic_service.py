"""Anthropic-backed implementation of assistant AI services."""

from typing import Any, Sequence, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from weather_assistant.domain.models import Intent, VerificationDecision
from weather_assistant.domain.policies import (
    WEATHER_ONLY_REPAIR_PROMPT,
    WEATHER_ONLY_SYSTEM_PROMPT,
)


class PlannerOutput(BaseModel):
    """Structured planner output."""

    intent: Intent = Field(
        description=(
            '"weather" if the user asks about weather/temperature/forecast in a place; '
            '"out_of_scope" for any request that is not weather-related.'
        )
    )


class VerificationOutput(BaseModel):
    """Structured verifier output."""

    is_correct: bool = Field(
        description="True if the latest AI answer is correct and addresses the user request."
    )
    feedback: str = Field(
        description=(
            "Short feedback for how to fix the answer when incorrect. "
            "Use an empty string when the answer is correct."
        )
    )


class AnthropicAssistantAIService:
    """AI service used by graph nodes via a high-level port."""

    def __init__(self, *, model: str = "claude-sonnet-4-6", temperature: float = 0) -> None:
        self._model = model
        self._temperature = temperature

    def classify_intent(self, user_text: str) -> Intent:
        if not user_text.strip():
            return "out_of_scope"

        llm = ChatAnthropic(
            model=self._model, temperature=self._temperature
        ).with_structured_output(PlannerOutput)
        plan = llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Classify the user's latest message.\n"
                        '- "weather": about weather, temperature, forecast, or conditions somewhere.\n'
                        '- "out_of_scope": anything else.\n\n'
                        f"User message:\n{user_text}"
                    )
                )
            ]
        )
        return plan.intent

    def respond_weather(
        self, messages: Sequence[BaseMessage], tools: Sequence[Any]
    ) -> AIMessage:
        guardrail = SystemMessage(content=WEATHER_ONLY_SYSTEM_PROMPT)
        llm = ChatAnthropic(model=self._model, temperature=self._temperature).bind_tools(
            list(tools)
        )
        msg = llm.invoke([guardrail] + list(messages))
        return cast(AIMessage, msg)

    def verify_answer(
        self, *, user_text: str, ai_text: str, tool_observations: Sequence[str]
    ) -> VerificationDecision:
        if not ai_text.strip():
            return VerificationDecision(
                is_correct=False, feedback="No assistant answer to verify."
            )

        verifier = ChatAnthropic(
            model=self._model, temperature=self._temperature
        ).with_structured_output(VerificationOutput)
        verification = verifier.invoke(
            [
                HumanMessage(
                    content=(
                        "You are validating an assistant answer.\n"
                        "Return is_correct=true only if the answer is accurate and fully addresses the user.\n"
                        "If any tool results are provided, ensure the answer matches them.\n"
                        "If incorrect, provide concise actionable feedback.\n\n"
                        f"User request:\n{user_text}\n\n"
                        f"Assistant answer:\n{ai_text}\n\n"
                        "Tool observations:\n"
                        + (
                            "\n".join(f"- {observation}" for observation in tool_observations)
                            or "- (none)"
                        )
                    )
                )
            ]
        )
        return VerificationDecision(
            is_correct=verification.is_correct, feedback=verification.feedback
        )

    def repair_answer(self, *, user_text: str, ai_text: str, feedback: str) -> AIMessage:
        repair_llm = ChatAnthropic(model=self._model, temperature=self._temperature)
        repaired_msg = repair_llm.invoke(
            [
                SystemMessage(content=WEATHER_ONLY_REPAIR_PROMPT),
                HumanMessage(
                    content=(
                        "Rewrite the assistant answer so it is correct and directly addresses the user.\n"
                        "Incorporate the verifier feedback.\n\n"
                        f"User request:\n{user_text}\n\n"
                        f"Previous assistant answer:\n{ai_text}\n\n"
                        f"Verifier feedback:\n{feedback}"
                    )
                ),
            ]
        )
        return cast(AIMessage, repaired_msg)

