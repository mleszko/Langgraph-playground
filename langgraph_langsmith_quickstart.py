import os
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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


class PlannerOutput(BaseModel):
    """Structured plan: which branch of the graph should run."""

    intent: Literal["weather", "general"] = Field(
        description=(
            '"weather" if the user asks about weather/temperature/forecast in a place; '
            '"general" for greetings, small talk, or unrelated topics.'
        )
    )


class GraphState(TypedDict):
    messages: list[BaseMessage]
    intent: NotRequired[Literal["weather", "general"]]


def _last_human_text(state: GraphState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _planner_node(state: GraphState) -> dict[str, Any]:
    human_text = _last_human_text(state)
    if not human_text.strip():
        return {"intent": "general"}

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).with_structured_output(
        PlannerOutput
    )
    plan = llm.invoke(
        [
            HumanMessage(
                content=(
                    "Classify the user's latest message.\n"
                    '- "weather": about weather, temperature, forecast, or conditions somewhere.\n'
                    '- "general": anything else.\n\n'
                    f"User message:\n{human_text}"
                )
            )
        ]
    )
    return {"intent": plan.intent}


def _route_after_planner(state: GraphState) -> Literal["weather_agent", "chitchat"]:
    return "weather_agent" if state.get("intent") == "weather" else "chitchat"


def _route_after_llm(state: GraphState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def _weather_agent_node(state: GraphState) -> GraphState:
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools([get_weather])
    msg = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [msg]}


def _chitchat_node(state: GraphState) -> GraphState:
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    msg = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [msg]}


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


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("planner", _planner_node)
    g.add_node("weather_agent", _weather_agent_node)
    g.add_node("tools", _tools_node)
    g.add_node("chitchat", _chitchat_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"weather_agent": "weather_agent", "chitchat": "chitchat"},
    )
    g.add_conditional_edges(
        "weather_agent",
        _route_after_llm,
        {"tools": "tools", "end": END},
    )
    g.add_edge("tools", "weather_agent")
    g.add_edge("chitchat", END)

    return g.compile()


def main() -> None:
    graph = build_graph()
    # Try also: "Just say hi in one friendly sentence — no weather."
    question = "What's the weather like in San Francisco and Tokyo?"
    initial_state: GraphState = {"messages": [HumanMessage(content=question)]}

    # Makes LangSmith grouping explicit in one trace (still needs LANGSMITH_* / LANGCHAIN_* env).
    with tracing_v2_enabled():
        final_state = graph.invoke(initial_state)

    print("\n--- ROUTING ---")
    print("intent:", final_state.get("intent", "(missing)"))
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
