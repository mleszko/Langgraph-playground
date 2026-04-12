from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from weather_assistant.composition import AppContainer, build_default_container
from weather_assistant.config import WeatherAssistantSettings
from weather_assistant.domain.models import VerificationDecision


class DummyAssistant:
    def classify_intent(self, user_text: str) -> str:
        return "weather"

    def respond_weather(self, messages, tools) -> AIMessage:
        return AIMessage(content="Weather summary")

    def verify_answer(self, *, user_text: str, ai_text: str, tool_observations):
        return VerificationDecision(is_correct=True, feedback="")

    def repair_answer(self, *, user_text: str, ai_text: str, feedback: str) -> AIMessage:
        return AIMessage(content="Repaired answer")


def test_settings_from_env_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("WEATHER_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("WEATHER_ASSISTANT_TEMPERATURE", raising=False)
    monkeypatch.delenv("WEATHER_ASSISTANT_MAX_ATTEMPTS", raising=False)

    settings = WeatherAssistantSettings.from_env()

    assert settings.model == "claude-sonnet-4-6"
    assert settings.temperature == 0.0
    assert settings.default_max_attempts == 2


def test_settings_from_env_handles_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("WEATHER_ASSISTANT_MODEL", "custom-model")
    monkeypatch.setenv("WEATHER_ASSISTANT_TEMPERATURE", "not-a-float")
    monkeypatch.setenv("WEATHER_ASSISTANT_MAX_ATTEMPTS", "not-an-int")

    settings = WeatherAssistantSettings.from_env()

    assert settings.model == "custom-model"
    assert settings.temperature == 0.0
    assert settings.default_max_attempts == 2


def test_default_container_uses_provided_settings() -> None:
    settings = WeatherAssistantSettings(
        model="claude-custom", temperature=0.25, default_max_attempts=4
    )
    container = build_default_container(settings=settings)

    assert container.settings == settings
    assert len(container.tools) == 1
    assert container.tools[0].name == "get_weather"
    assert container.conversation_repository is not None


def test_container_build_graph_with_dummy_assistant() -> None:
    settings = WeatherAssistantSettings()
    container = AppContainer(settings=settings, assistant=DummyAssistant(), tools=[])
    graph = container.build_graph()

    final_state = graph.invoke(
        {"messages": [HumanMessage(content="weather in London?")], "attempts": 0, "max_attempts": 2}
    )

    assert final_state["intent"] == "weather"
    assert final_state["is_correct"] is True

