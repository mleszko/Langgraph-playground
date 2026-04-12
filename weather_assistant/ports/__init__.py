"""Interfaces for adapters (LLMs, tools, persistence)."""

from .assistant import AssistantAIServicePort
from .repository import ConversationStateRepositoryPort

__all__ = ["AssistantAIServicePort", "ConversationStateRepositoryPort"]

