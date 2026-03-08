# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env

# Run (dev)
.venv/Scripts/python -m uvicorn app.main:app --reload

# Test
.venv/Scripts/python -m unittest discover -s tests -v

# Single test
.venv/Scripts/python -m unittest tests.test_api.TestClassName.test_method_name -v

# Syntax check
.venv/Scripts/python -m compileall app tests

# Docker
docker build -f docker/Dockerfile -t agent-router .
docker run -p 8000:8000 agent-router
```

## Architecture

This is a **multi-model LLM routing backend** (FastAPI + LiteLLM). It classifies incoming chat requests and routes them to the most appropriate LLM provider.

### Request Pipeline (`app/agent/orchestrator.py`)

```
ChatRequest → TaskClassifier → FeatureExtractor → RuleRouter → LiteLLMGateway → OutputValidator → fallback chain → SQLite log
```

1. **Classify** — keyword-based task type: `general_qa`, `summarization`, `coding`, `reasoning`, `rag`
2. **Extract features** — code presence, context length, tool needs, JSON requirements, reasoning complexity
3. **Route** — rule-based selection of one of 5 logical model aliases
4. **Invoke** — LiteLLM call (or mock if `MOCK_LLM_RESPONSES=true`)
5. **Validate** — quality check; trigger fallback chain on failure
6. **Persist** — log task, execution, evaluation to SQLite

### Model Aliases (configured via `.env`)

| Alias | Default | Use case |
|---|---|---|
| `fast_general` | Qwen/GPT-4o-mini | General QA |
| `strong_reasoning` | Claude 3.5 Sonnet | Reasoning |
| `code_specialist` | GPT-4o-mini | Coding |
| `long_context_rag` | Claude 3.5 Sonnet | RAG / long context |
| `local_fallback` | mock | Offline / last resort |

Fallback chains: `long_context_rag → strong_reasoning → fast_general → local_fallback`

### Key Modules

- `app/router/` — Classification, feature extraction, routing rules, fallback policy
- `app/models/` — Model registry, LiteLLM gateway, provider adapters
- `app/agent/orchestrator.py` — Main pipeline coordinator
- `app/storage/` — SQLAlchemy entities + repositories (tasks, executions, evaluations)
- `app/core/config.py` — Settings loaded from `.env`

### Development Notes

- `MOCK_LLM_RESPONSES=true` (default) enables offline development without real API keys
- Every routing decision must include a concrete reason (explainability requirement from `AGENTS.md`)
- No formatter/linter configured — manual code review only
- Database: SQLite at `agent_router.db` (auto-created on startup)
- API docs available at `http://127.0.0.1:8000/docs` when running locally
