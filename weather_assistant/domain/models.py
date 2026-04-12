"""Domain models shared by graph/application layers."""

from typing import Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage

Intent = Literal["weather", "out_of_scope"]


class GraphState(TypedDict):
    """Conversation state passed between graph nodes."""

    messages: list[BaseMessage]
    intent: NotRequired[Intent]
    attempts: NotRequired[int]
    max_attempts: NotRequired[int]
    is_correct: NotRequired[bool]
    verification_feedback: NotRequired[str]

