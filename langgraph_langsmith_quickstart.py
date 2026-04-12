import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import END, START, StateGraph
from weather_assistant.adapters.ai import AnthropicAssistantAIService
from weather_assistant.application.message_utils import (
    last_ai_text,
    last_human_text,
    tool_observations,
)
from weather_assistant.application.use_cases import with_default_attempt_limits
from weather_assistant.domain.models import GraphState
from weather_assistant.domain.policies import (
    WEATHER_ONLY_REFUSAL_TEXT,
    route_after_planner as policy_route_after_planner,
    route_after_verification as policy_route_after_verification,
)

# Load .env from this directory reliably (works under debugpy too).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "San Francisco": "Foggy, 62°F",
        "New York": "Sunny, 75°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Clear, 68°F",
    }
    return weather_data.get(city, "Weather data not available")


_assistant = AnthropicAssistantAIService()


def _planner_node(state: GraphState) -> dict[str, Any]:
    return {"intent": _assistant.classify_intent(last_human_text(state))}


def _route_after_planner(state: GraphState) -> Literal["weather_agent", "out_of_scope"]:
    return policy_route_after_planner(state.get("intent"))


def _route_after_llm(state: GraphState) -> Literal["tools", "verify"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "verify"


def _route_after_verification(state: GraphState) -> Literal["repair", "end"]:
    return policy_route_after_verification(
        is_correct=state.get("is_correct", False),
        attempts=state.get("attempts", 0),
        max_attempts=state.get("max_attempts", 2),
    )


def _weather_agent_node(state: GraphState) -> GraphState:
    msg = _assistant.respond_weather(state["messages"], tools=[get_weather])
    return {"messages": state["messages"] + [msg]}


def _out_of_scope_node(state: GraphState) -> GraphState:
    refusal = AIMessage(content=WEATHER_ONLY_REFUSAL_TEXT)
    return {
        "messages": state["messages"] + [refusal],
        "is_correct": True,
        "verification_feedback": "",
    }


def _tools_node(state: GraphState) -> GraphState:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return state

    out: list[BaseMessage] = []
    for call in last.tool_calls:
        name = call.get("name")
        args = call.get("args") or {}
        tool_call_id = call.get("id") or ""

        if name == "get_weather":
            result = get_weather.invoke(args)  # type: ignore[arg-type]
        else:
            result = f"Unknown tool: {name}"

        out.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

    return {"messages": state["messages"] + out}


def _verify_answer_node(state: GraphState) -> dict[str, Any]:
    verification = _assistant.verify_answer(
        user_text=last_human_text(state),
        ai_text=last_ai_text(state),
        tool_observations=tool_observations(state),
    )
    return verification.to_state_update()


def _repair_answer_node(state: GraphState) -> dict[str, Any]:
    attempts = state.get("attempts", 0) + 1
    repaired_msg = _assistant.repair_answer(
        user_text=last_human_text(state),
        ai_text=last_ai_text(state),
        feedback=state.get("verification_feedback", ""),
    )
    return {"messages": state["messages"] + [repaired_msg], "attempts": attempts}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("planner", _planner_node)
    g.add_node("weather_agent", _weather_agent_node)
    g.add_node("tools", _tools_node)
    g.add_node("out_of_scope", _out_of_scope_node)
    g.add_node("verify", _verify_answer_node)
    g.add_node("repair", _repair_answer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"weather_agent": "weather_agent", "out_of_scope": "out_of_scope"},
    )
    g.add_conditional_edges(
        "weather_agent",
        _route_after_llm,
        {"tools": "tools", "verify": "verify"},
    )
    g.add_edge("tools", "weather_agent")
    g.add_edge("out_of_scope", END)
    g.add_conditional_edges(
        "verify",
        _route_after_verification,
        {"repair": "repair", "end": END},
    )
    g.add_edge("repair", "verify")

    return g.compile()


def main() -> None:
    graph = build_graph()
    # Try also: "Just say hi in one friendly sentence."
    question = "What's the weather like in San Francisco and Tokyo?"
    initial_state: GraphState = with_default_attempt_limits(
        {"messages": [HumanMessage(content=question)]}
    )

    # Makes LangSmith grouping explicit in one trace (still needs LANGSMITH_* / LANGCHAIN_* env).
    with tracing_v2_enabled():
        final_state = graph.invoke(initial_state)

    print("\n--- ROUTING ---")
    print("intent:", final_state.get("intent", "(missing)"))  # weather or out_of_scope
    print("verified:", final_state.get("is_correct", "(missing)"))
    print("attempts:", final_state.get("attempts", 0))
    print("\n--- FINAL MESSAGES ---")
    for m in final_state["messages"]:
        if isinstance(m, HumanMessage):
            print("Human:", m.content)
        elif isinstance(m, ToolMessage):
            print("Tool:", m.content)
        elif isinstance(m, AIMessage):
            print("AI:", m.content)
        else:
            print(type(m).__name__ + ":", getattr(m, "content", m))


if __name__ == "__main__":
    main()
