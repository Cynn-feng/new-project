# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `app/`. Keep API entrypoints in `app/api/`, orchestration logic in `app/agent/`, routing rules in `app/router/`, provider integrations in `app/models/`, validation in `app/evaluator/`, and persistence code in `app/storage/`. Shared configuration and database setup belong in `app/core/`. Request/response contracts live in `app/schemas/`. Tests are in `tests/`, and container files are in `docker/`. Runtime artifacts such as `agent_router.db` are generated locally and should not be treated as source files.

## Build, Test, and Development Commands

- `python -m venv .venv`: create a local virtual environment.
- `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`: install FastAPI, LiteLLM, SQLAlchemy, and runtime dependencies.
- `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload`: run the API locally with auto-reload.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`: run the test suite.
- `.\.venv\Scripts\python.exe -m compileall app tests`: quick syntax validation before opening a PR.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, explicit type hints, and small focused modules. Prefer `snake_case` for functions, variables, and module names; use `PascalCase` for Pydantic and SQLAlchemy models. Keep routing decisions explainable: every model choice should include a concrete reason. Follow existing package boundaries instead of adding new top-level modules.

No formatter or linter is configured in-repo yet. Keep imports grouped logically, avoid unused code, and preserve ASCII unless a file already contains non-ASCII text.

## Testing Guidelines

Tests use `unittest` with `fastapi.testclient.TestClient`. Name test files `test_*.py` and keep API behavior checks close to request/response expectations. Cover at least health checks, model registry output, and one end-to-end chat path when changing routing, gateway, or storage code.

## Commit & Pull Request Guidelines

This workspace does not include Git history, so no local commit convention can be inferred. Use short imperative commit messages such as `feat(router): add fallback chain` or `fix(storage): initialize tables in tests`.

PRs should include a concise summary, touched areas (`app/router`, `app/models`, etc.), test evidence, and any required environment changes. Include sample request/response payloads when API behavior changes.

## Security & Configuration Tips

Copy `.env.example` to `.env` for local development. Do not commit API keys or provider secrets. Keep `MOCK_LLM_RESPONSES=true` for offline development; switch it off only when real provider credentials are configured.
