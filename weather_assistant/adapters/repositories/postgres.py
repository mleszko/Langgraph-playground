"""PostgreSQL-backed repository for conversation state."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import messages_from_dict, messages_to_dict
from psycopg import connect, sql
from psycopg.types.json import Jsonb

from weather_assistant.domain.models import GraphState


class PostgresConversationStateRepository:
    """Persist conversation graph state in a PostgreSQL table."""

    def __init__(self, dsn: str, *, table_name: str = "conversation_state") -> None:
        self._dsn = dsn
        self._table_name = table_name
        self._ensure_table()

    def _ensure_table(self) -> None:
        with connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            conversation_id TEXT PRIMARY KEY,
                            state JSONB NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    ).format(sql.Identifier(self._table_name))
                )

    def get(self, conversation_id: str) -> GraphState | None:
        with connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT state FROM {} WHERE conversation_id = %s").format(
                        sql.Identifier(self._table_name)
                    ),
                    (conversation_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        raw_state = cast(dict[str, Any], row[0])
        return self._deserialize_state(raw_state)

    def upsert(self, conversation_id: str, state: GraphState) -> None:
        serialized_state = self._serialize_state(state)
        with connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (conversation_id, state, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (conversation_id)
                        DO UPDATE SET state = EXCLUDED.state, updated_at = NOW()
                        """
                    ).format(sql.Identifier(self._table_name)),
                    (conversation_id, Jsonb(serialized_state)),
                )

    def delete(self, conversation_id: str) -> None:
        with connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE conversation_id = %s").format(
                        sql.Identifier(self._table_name)
                    ),
                    (conversation_id,),
                )

    def _serialize_state(self, state: GraphState) -> dict[str, Any]:
        payload = dict(state)
        payload["messages"] = messages_to_dict(state["messages"])
        return payload

    def _deserialize_state(self, payload: dict[str, Any]) -> GraphState:
        state: dict[str, Any] = dict(payload)
        state["messages"] = messages_from_dict(payload.get("messages", []))
        return cast(GraphState, state)

