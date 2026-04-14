"""Dependency composition for workflow runtime."""

from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.tools import BaseTool

from weather_assistant.adapters.ai import AnthropicAssistantAIService
from weather_assistant.adapters.graph import LangGraphWeatherWorkflow
from weather_assistant.adapters.repositories import InMemoryConversationStateRepository
from weather_assistant.adapters.tools import get_weather
from weather_assistant.config import WeatherAssistantSettings
from weather_assistant.ports.assistant import AssistantAIServicePort
from weather_assistant.ports.repository import ConversationStateRepositoryPort


@dataclass
class AppContainer:
    """Container object that assembles concrete runtime dependencies."""

    settings: WeatherAssistantSettings
    assistant: AssistantAIServicePort
    tools: list[BaseTool] = field(default_factory=list)
    conversation_repository: ConversationStateRepositoryPort = field(
        default_factory=InMemoryConversationStateRepository
    )

    def build_workflow(self) -> LangGraphWeatherWorkflow:
        """Build graph workflow adapter with injected dependencies."""
        return LangGraphWeatherWorkflow(assistant=self.assistant, tools=self.tools)

    def build_graph(self):
        """Build and compile executable graph."""
        return self.build_workflow().build()


def build_default_container(
    *,
    settings: WeatherAssistantSettings | None = None,
    tools: Iterable[BaseTool] | None = None,
) -> AppContainer:
    """Create default app container from settings and standard adapters."""
    resolved_settings = settings or WeatherAssistantSettings.from_env()
    assistant = AnthropicAssistantAIService(
        model=resolved_settings.model,
        temperature=resolved_settings.temperature,
    )
    resolved_tools = list(tools) if tools is not None else [get_weather]
    return AppContainer(
        settings=resolved_settings,
        assistant=assistant,
        tools=resolved_tools,
        conversation_repository=InMemoryConversationStateRepository(),
    )

