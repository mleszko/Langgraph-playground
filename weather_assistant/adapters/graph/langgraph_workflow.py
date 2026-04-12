"""LangGraph workflow adapter for the weather assistant."""

from typing import Any, Iterable, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from weather_assistant.application.message_utils import (
    last_ai_text,
    last_human_text,
    tool_observations,
)
from weather_assistant.domain.models import GraphState
from weather_assistant.domain.policies import (
    WEATHER_ONLY_REFUSAL_TEXT,
    route_after_planner,
    route_after_verification,
)
from weather_assistant.ports.assistant import AssistantAIServicePort


def route_after_llm(state: GraphState) -> Literal["tools", "verify"]:
    """Route to tool execution when the latest assistant message requests tools."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "verify"


class LangGraphWeatherWorkflow:
    """Build and run the LangGraph workflow with injected dependencies."""

    def __init__(
        self, *, assistant: AssistantAIServicePort, tools: Iterable[BaseTool]
    ) -> None:
        self._assistant = assistant
        self._tools = list(tools)
        self._tools_by_name = {tool.name: tool for tool in self._tools}

    def _planner_node(self, state: GraphState) -> dict[str, Any]:
        return {"intent": self._assistant.classify_intent(last_human_text(state))}

    def _route_after_planner(
        self, state: GraphState
    ) -> Literal["weather_agent", "out_of_scope"]:
        return route_after_planner(state.get("intent"))

    def _route_after_verification(self, state: GraphState) -> Literal["repair", "end"]:
        return route_after_verification(
            is_correct=state.get("is_correct", False),
            attempts=state.get("attempts", 0),
            max_attempts=state.get("max_attempts", 2),
        )

    def _weather_agent_node(self, state: GraphState) -> GraphState:
        msg = self._assistant.respond_weather(state["messages"], tools=self._tools)
        return {"messages": state["messages"] + [msg]}

    def _out_of_scope_node(self, state: GraphState) -> GraphState:
        refusal = AIMessage(content=WEATHER_ONLY_REFUSAL_TEXT)
        return {
            "messages": state["messages"] + [refusal],
            "is_correct": True,
            "verification_feedback": "",
        }

    def _tools_node(self, state: GraphState) -> GraphState:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return state

        out: list[BaseMessage] = []
        for call in last.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tool_call_id = call.get("id") or ""

            tool = self._tools_by_name.get(name or "")
            result = (
                tool.invoke(args)  # type: ignore[arg-type]
                if tool is not None
                else f"Unknown tool: {name}"
            )
            out.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

        return {"messages": state["messages"] + out}

    def _verify_answer_node(self, state: GraphState) -> dict[str, Any]:
        verification = self._assistant.verify_answer(
            user_text=last_human_text(state),
            ai_text=last_ai_text(state),
            tool_observations=tool_observations(state),
        )
        return verification.to_state_update()

    def _repair_answer_node(self, state: GraphState) -> dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        repaired_msg = self._assistant.repair_answer(
            user_text=last_human_text(state),
            ai_text=last_ai_text(state),
            feedback=state.get("verification_feedback", ""),
        )
        return {"messages": state["messages"] + [repaired_msg], "attempts": attempts}

    def build(self):
        """Compile and return workflow graph."""
        g = StateGraph(GraphState)
        g.add_node("planner", self._planner_node)
        g.add_node("weather_agent", self._weather_agent_node)
        g.add_node("tools", self._tools_node)
        g.add_node("out_of_scope", self._out_of_scope_node)
        g.add_node("verify", self._verify_answer_node)
        g.add_node("repair", self._repair_answer_node)

        g.add_edge(START, "planner")
        g.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"weather_agent": "weather_agent", "out_of_scope": "out_of_scope"},
        )
        g.add_conditional_edges(
            "weather_agent",
            route_after_llm,
            {"tools": "tools", "verify": "verify"},
        )
        g.add_edge("tools", "weather_agent")
        g.add_edge("out_of_scope", END)
        g.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {"repair": "repair", "end": END},
        )
        g.add_edge("repair", "verify")

        return g.compile()

