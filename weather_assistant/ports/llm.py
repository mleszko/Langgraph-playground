"""LLM-facing ports used by application logic."""

from typing import Generic, Protocol, Sequence, TypeVar

from langchain_core.messages import BaseMessage

TResult = TypeVar("TResult")


class StructuredLLMPort(Protocol, Generic[TResult]):
    """LLM that returns validated structured output."""

    def invoke(self, messages: Sequence[BaseMessage]) -> TResult:
        """Generate typed output from a message sequence."""


class ChatLLMPort(Protocol):
    """LLM chat interface returning a message."""

    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        """Generate a chat message from input messages."""

