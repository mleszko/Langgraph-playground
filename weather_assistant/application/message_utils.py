"""Message-centric helpers for graph/application layers."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from weather_assistant.domain.models import GraphState


def last_human_text(state: GraphState) -> str:
    """Return latest human message content, if any."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def last_ai_text(state: GraphState) -> str:
    """Return latest AI message content, if any."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def tool_observations(state: GraphState) -> list[str]:
    """Return rendered tool outputs seen in state."""
    return [str(msg.content) for msg in state["messages"] if isinstance(msg, ToolMessage)]

