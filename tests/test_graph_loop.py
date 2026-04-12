from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from weather_assistant.adapters.graph import LangGraphWeatherWorkflow, route_after_llm
from weather_assistant.adapters.tools import get_weather
from weather_assistant.domain.models import GraphState, Intent, VerificationDecision


class FakeAssistant:
    def __init__(
        self,
        *,
        intent: Intent = "weather",
        verify_sequence: Sequence[bool] = (True,),
        use_tool_path: bool = False,
    ) -> None:
        self._intent = intent
        self._verify_sequence = list(verify_sequence)
        self._verify_idx = 0
        self._weather_calls = 0
        self._repair_calls = 0
        self._use_tool_path = use_tool_path

    def classify_intent(self, user_text: str) -> Intent:
        return self._intent

    def respond_weather(self, messages: Sequence[BaseMessage], tools: Sequence[Any]) -> AIMessage:
        self._weather_calls += 1
        if self._use_tool_path and self._weather_calls == 1:
            return AIMessage(
                content="Calling weather tool.",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "San Francisco"},
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content=f"Weather answer attempt {self._weather_calls}")

    def verify_answer(
        self, *, user_text: str, ai_text: str, tool_observations: Sequence[str]
    ) -> VerificationDecision:
        idx = min(self._verify_idx, len(self._verify_sequence) - 1)
        self._verify_idx += 1
        is_correct = self._verify_sequence[idx]
        return VerificationDecision(
            is_correct=is_correct,
            feedback="" if is_correct else "Please improve accuracy.",
        )

    def repair_answer(self, *, user_text: str, ai_text: str, feedback: str) -> AIMessage:
        self._repair_calls += 1
        return AIMessage(content=f"Repaired answer {self._repair_calls}")


def _build_graph(*, assistant: FakeAssistant):
    return LangGraphWeatherWorkflow(assistant=assistant, tools=[get_weather]).build()


def test_route_after_llm_to_tools_and_verify() -> None:
    tool_call_state: GraphState = {
        "messages": [
            HumanMessage(content="weather"),
            AIMessage(content="tool", tool_calls=[{"name": "get_weather", "args": {}, "id": "x"}]),
        ]
    }
    plain_answer_state: GraphState = {
        "messages": [HumanMessage(content="hello"), AIMessage(content="hi")]
    }

    assert route_after_llm(tool_call_state) == "tools"
    assert route_after_llm(plain_answer_state) == "verify"


def test_tool_execution_from_workflow() -> None:
    assistant = FakeAssistant(intent="weather", verify_sequence=(True,), use_tool_path=True)
    graph = _build_graph(assistant=assistant)

    final_state = graph.invoke(
        {"messages": [HumanMessage(content="What's the weather?")], "attempts": 0, "max_attempts": 2}
    )

    assert any(isinstance(m, ToolMessage) for m in final_state["messages"])
    assert any("Foggy, 62" in str(m.content) for m in final_state["messages"])


def test_graph_retries_until_verified() -> None:
    assistant = FakeAssistant(intent="weather", verify_sequence=(False, True), use_tool_path=True)
    graph = _build_graph(assistant=assistant)
    initial_state: GraphState = {
        "messages": [HumanMessage(content="What's the weather?")],
        "attempts": 0,
        "max_attempts": 2,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["intent"] == "weather"
    assert final_state["is_correct"] is True
    assert final_state["attempts"] == 1
    assert any(isinstance(m, ToolMessage) for m in final_state["messages"])
    assert any(
        isinstance(m, AIMessage) and "Repaired answer 1" in str(m.content)
        for m in final_state["messages"]
    )


def test_graph_stops_when_max_attempts_reached() -> None:
    assistant = FakeAssistant(intent="weather", verify_sequence=(False, False, False))
    graph = _build_graph(assistant=assistant)
    initial_state: GraphState = {
        "messages": [HumanMessage(content="What's the weather?")],
        "attempts": 0,
        "max_attempts": 2,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["is_correct"] is False
    assert final_state["attempts"] == 2


def test_graph_out_of_scope_flow_ends_without_verification() -> None:
    assistant = FakeAssistant(intent="out_of_scope", verify_sequence=(False, False, True))
    graph = _build_graph(assistant=assistant)
    initial_state: GraphState = {
        "messages": [HumanMessage(content="Hi!")],
        "attempts": 0,
        "max_attempts": 2,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["intent"] == "out_of_scope"
    assert final_state["attempts"] == 0
    assert final_state["is_correct"] is True
    assert any(
        isinstance(m, AIMessage) and "only help with weather questions" in str(m.content)
        for m in final_state["messages"]
    )
