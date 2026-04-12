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
   - `general`
2. Routes to:
   - a weather-capable agent (with tool use), or
   - a general chitchat agent
3. Uses a simple local tool (`get_weather`) when weather data is requested
4. Wraps execution with LangSmith tracing (`tracing_v2_enabled`) so runs are visible in LangSmith when configured.

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

## Run

```bash
python langgraph_langsmith_quickstart.py
```

By default, the script asks:

> What's the weather like in San Francisco and Tokyo?

You can edit the `question` variable in `main()` to try other inputs.

## Test

```bash
pytest
```

## What to Expect

The script prints:

- Selected route/intent (`weather` or `general`)
- Full message flow (Human, Tool, AI messages)

For weather questions, the model may call `get_weather`, which returns mock weather data for:

- San Francisco
- New York
- London
- Tokyo

## Project Structure

```text
.
├── langgraph_langsmith_quickstart.py
├── requirements.txt
└── README.md
```

