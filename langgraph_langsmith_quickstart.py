import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tracers.context import tracing_v2_enabled

from weather_assistant.adapters.ai import AnthropicAssistantAIService
from weather_assistant.adapters.graph import LangGraphWeatherWorkflow
from weather_assistant.adapters.tools import get_weather
from weather_assistant.application.use_cases import with_default_attempt_limits
from weather_assistant.domain.models import GraphState

# Load .env from this directory reliably (works under debugpy too).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def main() -> None:
    assistant = AnthropicAssistantAIService()
    graph = LangGraphWeatherWorkflow(assistant=assistant, tools=[get_weather]).build()
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
