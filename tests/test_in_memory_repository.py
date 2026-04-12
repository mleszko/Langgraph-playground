from langchain_core.messages import HumanMessage

from weather_assistant.adapters.repositories import InMemoryConversationStateRepository


def test_in_memory_repository_round_trip() -> None:
    repo = InMemoryConversationStateRepository()
    state = {"messages": [HumanMessage(content="hello")], "attempts": 0, "max_attempts": 2}

    repo.upsert("conv-1", state)
    loaded = repo.get("conv-1")

    assert loaded is not None
    assert len(loaded["messages"]) == 1
    assert str(loaded["messages"][0].content) == "hello"


def test_in_memory_repository_returns_copied_state() -> None:
    repo = InMemoryConversationStateRepository()
    repo.upsert(
        "conv-1",
        {"messages": [HumanMessage(content="first")], "attempts": 0, "max_attempts": 2},
    )

    loaded = repo.get("conv-1")
    assert loaded is not None
    loaded["messages"].append(HumanMessage(content="mutated"))  # local mutation only

    loaded_again = repo.get("conv-1")
    assert loaded_again is not None
    assert len(loaded_again["messages"]) == 1


def test_in_memory_repository_delete_is_idempotent() -> None:
    repo = InMemoryConversationStateRepository()
    repo.upsert(
        "conv-1",
        {"messages": [HumanMessage(content="first")], "attempts": 0, "max_attempts": 2},
    )

    repo.delete("conv-1")
    repo.delete("conv-1")

    assert repo.get("conv-1") is None

