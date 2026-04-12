# LangGraph + LangSmith Quickstart

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-7f52ff)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-0ea5e9)
![Anthropic](https://img.shields.io/badge/Model-Anthropic-111827)

This repository contains a single Python quickstart script:

- `langgraph_langsmith_quickstart.py`

It demonstrates a small **LangGraph workflow** that:

1. Classifies the latest user message as either:
   - `weather`
   - `out_of_scope`
2. Routes to:
   - a weather-capable agent (with tool use), or
   - a weather-only refusal response
3. Uses a simple local tool (`get_weather`) when weather data is requested
4. Verifies the generated answer and, if needed, retries with a repair loop
5. Wraps execution with LangSmith tracing (`tracing_v2_enabled`) so runs are visible in LangSmith when configured.

## Graph Flow (weather-only with verification loop)

The graph uses these main nodes:

- `planner`: classifies user intent (`weather` or `out_of_scope`)
- `weather_agent`: responds to weather prompts and may call tools
- `tools`: executes `get_weather` tool calls
- `out_of_scope`: returns a fixed weather-only refusal for non-weather prompts
- `verify`: checks whether the latest AI answer is correct and complete
- `repair`: rewrites the answer based on verifier feedback when needed

Loop behavior:

- Weather-path answers pass through `verify`.
- If `verify` marks the answer as correct, the graph ends.
- If incorrect, the graph routes to `repair`, then back to `verify`.
- The loop is bounded by `attempts` and `max_attempts` to prevent infinite retries.
- Out-of-scope prompts skip verification/repair and end after refusal.

## Prerequisites

- Python 3.11+
- An Anthropic API key
- (Recommended) A LangSmith API key for tracing

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Cursor Cloud Agent setup

This repository includes `.cursor/environment.json` so cloud agents automatically run:

```bash
python3 -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the repository root:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_key

# LangSmith (optional but recommended)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGCHAIN_PROJECT=langgraph-langsmith-quickstart
```

Notes:
- The script loads `.env` from the same directory as the script file.
- LangSmith tracing context is enabled in code (`with tracing_v2_enabled():`), so tracing-related env vars should be set if you want traces recorded.
- Optional runtime tuning via environment variables:
  - `WEATHER_ASSISTANT_MODEL` (default: `claude-sonnet-4-6`)
  - `WEATHER_ASSISTANT_TEMPERATURE` (default: `0.0`)
  - `WEATHER_ASSISTANT_MAX_ATTEMPTS` (default: `2`)

## Run

```bash
python langgraph_langsmith_quickstart.py
```

By default, the script asks:

> What's the weather like in San Francisco and Tokyo?

You can edit the `question` variable in `main()` to try other inputs.

## Run as API (FastAPI)

Start the HTTP service:

```bash
uvicorn weather_assistant.adapters.api.fastapi_app:create_app --factory --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /chat`
- `DELETE /conversations/{conversation_id}`

`POST /chat` request body:

- `message` (required)
- `conversation_id` (optional; if provided, conversation history is reused)
- `max_attempts` (optional verification retry cap for that request)

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo-1","message":"What is the weather in Tokyo?"}'
```

Delete conversation state:

```bash
curl -X DELETE http://localhost:8000/conversations/demo-1
```

## Test

```bash
pytest
```

## What to Expect

The script prints:

- Selected route/intent (`weather` or `out_of_scope`)
- Verification status (`verified: True/False`)
- Number of repair attempts used
- Full message flow (Human, Tool, AI messages)

For non-weather prompts, the assistant returns:

> I can only help with weather questions. Ask me about the weather in a city.

For weather questions, the model may call `get_weather`, which returns mock weather data for:

- San Francisco
- New York
- London
- Tokyo

## Project Structure

```text
.
├── weather_assistant/
│   ├── adapters/
│   │   ├── ai/
│   │   ├── api/
│   │   ├── graph/
│   │   ├── repositories/
│   │   └── tools/
│   ├── application/
│   ├── composition/
│   ├── config/
│   ├── domain/
│   └── ports/
├── langgraph_langsmith_quickstart.py
├── requirements.txt
└── tests/
    ├── test_api_app.py
    ├── test_composition_container.py
    ├── test_domain_policies.py
    ├── test_graph_loop.py
    └── test_in_memory_repository.py
```

The root script is now a thin CLI entrypoint. Graph construction lives in
`weather_assistant.adapters.graph.LangGraphWeatherWorkflow`, Anthropic-specific AI behavior lives in
`weather_assistant.adapters.ai.AnthropicAssistantAIService`, HTTP wiring lives in
`weather_assistant.adapters.api.fastapi_app`, and runtime dependency wiring is handled by
`weather_assistant.composition.AppContainer`. Conversation state persistence is abstracted through a
repository port with an in-memory adapter, which keeps storage concerns swappable for a future
production database in a microservice deployment.

