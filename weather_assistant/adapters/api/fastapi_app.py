"""FastAPI adapter exposing health and chat endpoints."""

from __future__ import annotations

from uuid import uuid4
from typing import Any

from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from weather_assistant.application.use_cases import with_default_attempt_limits
from weather_assistant.composition import AppContainer, build_default_container
from weather_assistant.domain.models import GraphState


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(description="User input message")
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional conversation identifier. "
            "If omitted, server starts a new conversation."
        ),
    )
    max_attempts: int | None = Field(
        default=None, description="Optional verification retry cap override"
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    conversation_id: str
    intent: str | None
    verified: bool | None
    attempts: int
    reply: str
    messages: list[dict[str, str]]


def _render_messages_for_response(state: GraphState) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, AIMessage):
            role = "ai"
        else:
            role = type(message).__name__.lower()
        rendered.append({"role": role, "content": str(message.content)})
    return rendered


def create_app(*, container: AppContainer | None = None) -> FastAPI:
    """Create FastAPI app with injected or default container."""
    app_container = container or build_default_container()
    graph = app_container.build_graph()
    repository = app_container.conversation_repository

    app = FastAPI(title="Weather Assistant API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": app_container.settings.model,
            "default_max_attempts": app_container.settings.default_max_attempts,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        existing_state = repository.get(conversation_id)
        prior_messages = existing_state["messages"] if existing_state is not None else []
        initial_state: GraphState = with_default_attempt_limits(
            {"messages": [*prior_messages, HumanMessage(content=payload.message)]},
            default_max_attempts=(
                payload.max_attempts
                if payload.max_attempts is not None
                else app_container.settings.default_max_attempts
            ),
        )
        final_state = graph.invoke(initial_state)
        repository.upsert(conversation_id, final_state)
        rendered_messages = _render_messages_for_response(final_state)
        ai_replies = [m["content"] for m in rendered_messages if m["role"] == "ai"]
        reply = ai_replies[-1] if ai_replies else ""

        return ChatResponse(
            conversation_id=conversation_id,
            intent=final_state.get("intent"),
            verified=final_state.get("is_correct"),
            attempts=final_state.get("attempts", 0),
            reply=reply,
            messages=rendered_messages,
        )

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        existed = repository.get(conversation_id) is not None
        repository.delete(conversation_id)
        return {"conversation_id": conversation_id, "deleted": existed}

    return app


app = create_app()

