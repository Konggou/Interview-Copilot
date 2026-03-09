# Agent Guide for rag-bot-fastapi

This repository is a Python FastAPI backend with a Streamlit client.
Use this file as the operating guide for agentic changes in this repo.

## Scope
- Applies to the entire repository unless a nested AGENTS.md exists.
- No other AGENTS.md files were found in this repo.
- No Cursor or Copilot rules were found.

## Repo Layout
- `client/`: Streamlit UI and client utilities.
- `server/`: FastAPI backend, core logic, vectorstores, config.
- `assets/`: Images and gifs for README.

## Quick Start
- Create virtual env: `python3 -m venv venv`
- Activate (bash): `source venv/bin/activate`
- Install client deps: `pip3 install -r client/requirements.txt`
- Install server deps: `pip3 install -r server/requirements.txt`

## Run Commands
- Backend (FastAPI): `cd server && uvicorn main:app --reload`
- Frontend (Streamlit): `cd client && streamlit run app.py`

## Build / Lint / Test
- No build system is configured.
- No lint config detected (no `pyproject.toml`, `setup.cfg`, `ruff.toml`, or `.flake8`).
- No test framework config detected.
- If adding tests, use `pytest` and document how to run single tests.

### Suggested Test Commands (if you add pytest)
- Run all: `pytest`
- Run a file: `pytest path/to/test_file.py`
- Run a test: `pytest path/to/test_file.py -k test_name`

## Code Style and Conventions
Follow existing patterns in `client/` and `server/`.

### Python Formatting
- Indentation is 2 spaces (not 4).
- Use trailing commas for multi-line arguments.
- Keep line length moderate; no formatter configured.

### Imports
- Prefer absolute imports from package roots within each app.
  - Client: `from utils.helpers import ...`
  - Server: `from core.vector_database import ...`
- Group imports with a blank line between stdlib, third-party, local.

### Typing
- Type hints are used selectively; keep them where present.
- Prefer built-in generics (`list[str]`, `dict[str, Any]`).
- Use `Optional[...]` from `typing` when needed.

### Naming
- Functions: `snake_case`.
- Variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Files and modules: `snake_case.py`.

### Error Handling
- Use explicit error messages and log with `utils.logger.logger` on server.
- On the client, surface failures via Streamlit `st.error`.
- Avoid silent exceptions; always log or display an error.

### Logging
- Server uses `utils/logger.py` with a global `logger`.
- Use `logger.debug/info/warning/error/exception` as appropriate.

## Server-Side Patterns (FastAPI)
- Routes are defined in `server/api/routes.py` using `APIRouter`.
- Request/response models are in `server/api/schemas.py`.
- Core logic is in `server/core/` (LLM chain, vectorstore, PDFs).
- Config and env vars are in `server/config/settings.py`.

## Client-Side Patterns (Streamlit)
- `client/app.py` is the main entrypoint.
- UI components live in `client/components/`.
- Session state helpers are in `client/state/`.
- API calls are in `client/utils/api.py`.
- Orchestration helpers are in `client/utils/helpers.py`.

## Environment Variables
- `GROQ_API_KEY` for Groq.
- `GOOGLE_API_KEY` for Gemini.
- `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` for DeepSeek.
- Local `.env` is loaded in `server/config/settings.py`.

## Data and Storage
- Vectorstores persist under `server/data/`.
- Uploaded PDFs are saved under `server/temp/`.
- Do not commit generated data, temp files, or `.env`.

## Dependencies
- Backend: FastAPI, LangChain, Chroma, embeddings.
- Frontend: Streamlit, Requests, Pandas.

## When Adding Features
- Keep changes minimal and local to the relevant layer.
- Prefer updating helpers over duplicating logic.
- Match existing error messages and UI patterns.
- Avoid refactors unrelated to the task.

## Documentation
- Update the appropriate README when behavior changes.
- Do not add new docs unless requested.

## Validation Checklist
- Run backend and frontend locally if behavior changes.
- If tests are added, run targeted tests first.
- Ensure no secrets are added to the repo.

## Notes for Agents
- The project uses 2-space indentation.
- This repo is split into `client/` and `server/` with separate deps.
- No existing lint/test tools are configured.
- If you introduce tooling, update this file.
