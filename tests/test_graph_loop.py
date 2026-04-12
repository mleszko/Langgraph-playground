from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import langgraph_langsmith_quickstart as app


def _patch_graph_nodes(
    monkeypatch: Any,
    *,
    intent: str = "weather",
    verify_sequence: list[bool] | None = None,
    use_tool_path: bool = False,
) -> None:
    verify_sequence = verify_sequence or [True]
    verify_idx = {"value": 0}
    weather_calls = {"value": 0}

    def fake_planner_node(_state: app.GraphState) -> dict[str, str]:
        return {"intent": intent}

    def fake_weather_agent_node(state: app.GraphState) -> app.GraphState:
        weather_calls["value"] += 1
        if use_tool_path and weather_calls["value"] == 1:
            msg = AIMessage(
                content="Calling weather tool.",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "San Francisco"},
                        "id": "call-1",
                    }
                ],
            )
        else:
            msg = AIMessage(content=f"Weather answer attempt {weather_calls['value']}")
        return {"messages": state["messages"] + [msg]}

    def fake_out_of_scope_node(state: app.GraphState) -> app.GraphState:
        return {
            "messages": state["messages"]
            + [AIMessage(content="I can only help with weather questions.")]
        }

    def fake_verify_answer_node(_state: app.GraphState) -> dict[str, Any]:
        idx = verify_idx["value"]
        verify_idx["value"] += 1
        is_correct = verify_sequence[min(idx, len(verify_sequence) - 1)]
        return {
            "is_correct": is_correct,
            "verification_feedback": "" if is_correct else "Please improve accuracy.",
        }

    def fake_repair_answer_node(state: app.GraphState) -> dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        repaired = AIMessage(content=f"Repaired answer {attempts}")
        return {"messages": state["messages"] + [repaired], "attempts": attempts}

    monkeypatch.setattr(app, "_planner_node", fake_planner_node)
    monkeypatch.setattr(app, "_weather_agent_node", fake_weather_agent_node)
    monkeypatch.setattr(app, "_out_of_scope_node", fake_out_of_scope_node)
    monkeypatch.setattr(app, "_verify_answer_node", fake_verify_answer_node)
    monkeypatch.setattr(app, "_repair_answer_node", fake_repair_answer_node)


def test_route_after_llm_to_tools_and_verify() -> None:
    tool_call_state: app.GraphState = {
        "messages": [
            HumanMessage(content="weather"),
            AIMessage(content="tool", tool_calls=[{"name": "get_weather", "args": {}, "id": "x"}]),
        ]
    }
    plain_answer_state: app.GraphState = {
        "messages": [HumanMessage(content="hello"), AIMessage(content="hi")]
    }

    assert app._route_after_llm(tool_call_state) == "tools"
    assert app._route_after_llm(plain_answer_state) == "verify"


def test_tool_execution_from_weather_assistant_service() -> None:
    state: app.GraphState = {
        "messages": [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="Calling tool",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "San Francisco"},
                        "id": "call-123",
                    }
                ],
            ),
        ]
    }

    new_state = app._tools_node(state)

    assert len(new_state["messages"]) == 3
    assert isinstance(new_state["messages"][-1], ToolMessage)
    assert "Foggy, 62" in str(new_state["messages"][-1].content)


def test_graph_retries_until_verified(monkeypatch: Any) -> None:
    _patch_graph_nodes(monkeypatch, verify_sequence=[False, True], use_tool_path=True)
    graph = app.build_graph()
    initial_state: app.GraphState = {
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


def test_graph_stops_when_max_attempts_reached(monkeypatch: Any) -> None:
    _patch_graph_nodes(monkeypatch, verify_sequence=[False, False, False])
    graph = app.build_graph()
    initial_state: app.GraphState = {
        "messages": [HumanMessage(content="What's the weather?")],
        "attempts": 0,
        "max_attempts": 2,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["is_correct"] is False
    assert final_state["attempts"] == 2


def test_graph_out_of_scope_flow_ends_without_verification(monkeypatch: Any) -> None:
    _patch_graph_nodes(monkeypatch, intent="out_of_scope", verify_sequence=[False, False, True])
    graph = app.build_graph()
    initial_state: app.GraphState = {
        "messages": [HumanMessage(content="Hi!")],
        "attempts": 0,
        "max_attempts": 2,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["intent"] == "out_of_scope"
    assert final_state["attempts"] == 0
    assert "is_correct" not in final_state
    assert any(
        isinstance(m, AIMessage) and "only help with weather questions" in str(m.content)
        for m in final_state["messages"]
    )
