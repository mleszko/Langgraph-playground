"""In-memory repository for conversation state."""

from copy import deepcopy
from threading import RLock

from weather_assistant.domain.models import GraphState


class InMemoryConversationStateRepository:
    """Simple thread-safe in-memory store keyed by conversation ID."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._state_by_conversation_id: dict[str, GraphState] = {}

    def get(self, conversation_id: str) -> GraphState | None:
        with self._lock:
            state = self._state_by_conversation_id.get(conversation_id)
            return deepcopy(state) if state is not None else None

    def upsert(self, conversation_id: str, state: GraphState) -> None:
        with self._lock:
            self._state_by_conversation_id[conversation_id] = deepcopy(state)

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._state_by_conversation_id.pop(conversation_id, None)

