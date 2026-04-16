from langchain_core.messages import AIMessage, HumanMessage

from weather_assistant.adapters.repositories.postgres import (
    PostgresConversationStateRepository,
)


def test_postgres_repository_serialization_round_trip() -> None:
    # Bypass __init__ to avoid requiring a live database for this unit test.
    repo = PostgresConversationStateRepository.__new__(PostgresConversationStateRepository)
    state = {
        "messages": [
            HumanMessage(content="weather in london?"),
            AIMessage(content="Rainy, 55°F"),
        ],
        "intent": "weather",
        "attempts": 1,
        "max_attempts": 2,
        "is_correct": True,
        "verification_feedback": "",
    }

    payload = repo._serialize_state(state)
    restored = repo._deserialize_state(payload)

    assert len(restored["messages"]) == 2
    assert str(restored["messages"][0].content) == "weather in london?"
    assert str(restored["messages"][1].content) == "Rainy, 55°F"
    assert restored["intent"] == "weather"
    assert restored["attempts"] == 1
