from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage

from weather_assistant.adapters.api import create_app
from weather_assistant.composition import AppContainer
from weather_assistant.config import WeatherAssistantSettings
from weather_assistant.domain.models import VerificationDecision


class DummyAssistant:
    def classify_intent(self, user_text: str) -> str:
        return "weather" if "weather" in user_text.lower() else "out_of_scope"

    def respond_weather(self, messages: list[BaseMessage], tools: list[object]) -> AIMessage:
        return AIMessage(content="Weather in London is Rainy, 55°F")

    def verify_answer(
        self, *, user_text: str, ai_text: str, tool_observations: list[str]
    ) -> VerificationDecision:
        return VerificationDecision(is_correct=True, feedback="")

    def repair_answer(self, *, user_text: str, ai_text: str, feedback: str) -> AIMessage:
        return AIMessage(content="Repaired answer")


def _test_container() -> AppContainer:
    settings = WeatherAssistantSettings(
        model="test-model", temperature=0.0, default_max_attempts=2
    )
    return AppContainer(settings=settings, assistant=DummyAssistant(), tools=[])


def test_health_endpoint() -> None:
    app = create_app(container=_test_container())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "test-model",
        "default_max_attempts": 2,
    }


def test_chat_endpoint_weather_request() -> None:
    app = create_app(container=_test_container())
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"conversation_id": "conv-weather", "message": "What's the weather in London?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "conv-weather"
    assert payload["intent"] == "weather"
    assert payload["verified"] is True
    assert payload["attempts"] == 0
    assert "Rainy, 55" in payload["reply"]
    assert any(m["role"] == "human" for m in payload["messages"])
    assert any(m["role"] == "ai" for m in payload["messages"])


def test_chat_endpoint_out_of_scope_request() -> None:
    app = create_app(container=_test_container())
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"conversation_id": "conv-oos", "message": "Tell me a joke"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "conv-oos"
    assert payload["intent"] == "out_of_scope"
    assert payload["verified"] is True
    assert "only help with weather questions" in payload["reply"]


def test_chat_endpoint_reuses_conversation_history() -> None:
    app = create_app(container=_test_container())
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={
            "conversation_id": "conv-history",
            "message": "What's the weather in London?",
        },
    )
    second = client.post(
        "/chat",
        json={
            "conversation_id": "conv-history",
            "message": "And what about Tokyo?",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["conversation_id"] == "conv-history"
    assert len(second_payload["messages"]) >= 4
    assert any(
        msg["role"] == "human" and "weather in London" in msg["content"]
        for msg in second_payload["messages"]
    )


def test_delete_conversation_endpoint() -> None:
    app = create_app(container=_test_container())
    client = TestClient(app)

    client.post(
        "/chat",
        json={"conversation_id": "conv-delete", "message": "What's the weather in London?"},
    )
    delete_response = client.delete("/conversations/conv-delete")
    delete_again_response = client.delete("/conversations/conv-delete")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"conversation_id": "conv-delete", "deleted": True}
    assert delete_again_response.status_code == 200
    assert delete_again_response.json() == {"conversation_id": "conv-delete", "deleted": False}

