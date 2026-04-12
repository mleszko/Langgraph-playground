import os
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
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

    intent: Literal["weather", "out_of_scope"] = Field(
        description=(
            '"weather" if the user asks about weather/temperature/forecast in a place; '
            '"out_of_scope" for any request that is not weather-related.'
        )
    )


class VerificationOutput(BaseModel):
    """Structured verification of the latest assistant answer."""

    is_correct: bool = Field(
        description="True if the latest AI answer is correct and addresses the user request."
    )
    feedback: str = Field(
        description=(
            "Short feedback for how to fix the answer when incorrect. "
            "Use an empty string when the answer is correct."
        )
    )


class GraphState(TypedDict):
    messages: list[BaseMessage]
    intent: NotRequired[Literal["weather", "out_of_scope"]]
    attempts: NotRequired[int]
    max_attempts: NotRequired[int]
    is_correct: NotRequired[bool]
    verification_feedback: NotRequired[str]


def _last_human_text(state: GraphState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _last_ai_text(state: GraphState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


def _planner_node(state: GraphState) -> dict[str, Any]:
    human_text = _last_human_text(state)
    if not human_text.strip():
        return {"intent": "out_of_scope"}

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).with_structured_output(
        PlannerOutput
    )
    plan = llm.invoke(
        [
            HumanMessage(
                content=(
                    "Classify the user's latest message.\n"
                    '- "weather": about weather, temperature, forecast, or conditions somewhere.\n'
                    '- "out_of_scope": anything else.\n\n'
                    f"User message:\n{human_text}"
                )
            )
        ]
    )
    return {"intent": plan.intent}


def _route_after_planner(state: GraphState) -> Literal["weather_agent", "out_of_scope"]:
    return "weather_agent" if state.get("intent") == "weather" else "out_of_scope"


def _route_after_llm(state: GraphState) -> Literal["tools", "verify"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "verify"


def _route_after_verification(state: GraphState) -> Literal["repair", "end"]:
    if state.get("is_correct", False):
        return "end"
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 2)
    if attempts >= max_attempts:
        return "end"
    return "repair"


def _weather_agent_node(state: GraphState) -> GraphState:
    guardrail = SystemMessage(
        content=(
            "You are a weather-only assistant. Answer only weather-related questions. "
            "If the user asks about anything else, respond exactly with: "
            "'I can only help with weather questions. Ask me about the weather in a city.'"
        )
    )
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools([get_weather])
    msg = llm.invoke([guardrail] + state["messages"])
    return {"messages": state["messages"] + [msg]}


def _out_of_scope_node(state: GraphState) -> GraphState:
    refusal = AIMessage(
        content="I can only help with weather questions. Ask me about the weather in a city."
    )
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
    user_text = _last_human_text(state)
    ai_text = _last_ai_text(state)
    tool_observations = [
        str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)
    ]

    if not ai_text.strip():
        return {
            "is_correct": False,
            "verification_feedback": "No assistant answer to verify.",
        }

    verifier = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).with_structured_output(
        VerificationOutput
    )
    verification = verifier.invoke(
        [
            HumanMessage(
                content=(
                    "You are validating an assistant answer.\n"
                    "Return is_correct=true only if the answer is accurate and fully addresses the user.\n"
                    "If any tool results are provided, ensure the answer matches them.\n"
                    "If incorrect, provide concise actionable feedback.\n\n"
                    f"User request:\n{user_text}\n\n"
                    f"Assistant answer:\n{ai_text}\n\n"
                    "Tool observations:\n"
                    + ("\n".join(f"- {obs}" for obs in tool_observations) or "- (none)")
                )
            )
        ]
    )
    return {
        "is_correct": verification.is_correct,
        "verification_feedback": verification.feedback,
    }


def _repair_answer_node(state: GraphState) -> dict[str, Any]:
    user_text = _last_human_text(state)
    ai_text = _last_ai_text(state)
    feedback = state.get("verification_feedback", "")
    attempts = state.get("attempts", 0) + 1
    repair_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    repaired_msg = repair_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a weather-only assistant. Provide only weather-related answers "
                    "and stay grounded in available tool outputs."
                )
            ),
            HumanMessage(
                content=(
                    "Rewrite the assistant answer so it is correct and directly addresses the user.\n"
                    "Incorporate the verifier feedback.\n\n"
                    f"User request:\n{user_text}\n\n"
                    f"Previous assistant answer:\n{ai_text}\n\n"
                    f"Verifier feedback:\n{feedback}"
                )
            )
        ]
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
    initial_state: GraphState = {
        "messages": [HumanMessage(content=question)],
        "attempts": 0,
        "max_attempts": 2,
    }

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
