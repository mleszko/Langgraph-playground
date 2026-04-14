"""Repository ports for conversation state persistence."""

from typing import Protocol

from weather_assistant.domain.models import GraphState


class ConversationStateRepositoryPort(Protocol):
    """Persistence interface for conversation graph state."""

    def get(self, conversation_id: str) -> GraphState | None:
        """Return stored state for a conversation, if present."""

    def upsert(self, conversation_id: str, state: GraphState) -> None:
        """Create or replace the state for a conversation."""

    def delete(self, conversation_id: str) -> None:
        """Delete state for a conversation if present."""

