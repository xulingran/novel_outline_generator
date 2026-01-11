# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-11
**Commit:** 9a57cdb
**Branch:** main

## OVERVIEW
AI-powered novel outline generator using LLM APIs (OpenAI, Gemini, Zhipu, AiHubMix). Async-first architecture with dual-mode entry (CLI/Web UI), intelligent text splitting, parallel processing, and resume capability.

## STRUCTURE
```
novel_outline_generator/
├── main.py                    # Dual-mode entry: Web UI (mode 1) / Processing (mode 2)
├── web_api.py                 # FastAPI backend (standalone entry point)
├── config.py                  # API + processing configuration (env-based)
├── splitter.py                # Intelligent text chunking (chapter/paragraph/token-based)
├── tokenizer.py               # Token counting utilities
├── prompts.py                 # LLM prompt templates (merge, compact, array)
├── exceptions.py              # Custom exception hierarchy
├── validators.py              # Input validation (paths, filenames)
├── utils.py                   # Logging, helpers
├── services/                  # Business logic layer (8 modules)
│   ├── llm_service.py        # LLM abstraction base + 4 provider impls
│   ├── novel_processing_service.py  # Core orchestration
│   ├── task_queue.py         # Async queue for batch jobs
│   ├── progress_service.py   # Save/load resume capability
│   ├── eta_estimator.py      # Time estimation
│   ├── file_service.py       # File I/O
│   └── token_estimator.py   # Pre-processing cost estimation
├── models/                   # Data models (processing_state, outline, character)
├── tests/                    # Pytest test suite (11 files)
└── ui/index.html             # SPA frontend (file:// URI, calls FastAPI)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Entry points | `main.py` (CLI selector), `web_api.py` (FastAPI) | Two independent entry points |
| API calls | `services/llm_service.py` | Provider-specific implementations |
| Core logic | `services/novel_processing_service.py` | Split → Generate → Merge flow |
| Text splitting | `splitter.py` | Chapter/paragraph/token detection |
| Config management | `config.py` | `get_api_config()`, `get_processing_config()` |
| Testing | `tests/` | pytest, fixtures in `conftest.py` |
| Frontend | `ui/index.html` | SPA, file:// URI, HTTP to localhost:8000 |

## CONVENTIONS (Deviations from Standard)

**Project Structure:**
- **Flat package** (no `src/` directory) - Core modules at root level
- `services/` and `models/` are only organized subdirectories
- Entry point `main.py` launches uvicorn internally for Web UI mode (not separate process)

**Code Style:**
- Line limit: 100 characters (Black/Ruff)
- Import order: standard → third-party → local (strict)
- Type annotations required for all public APIs (modern `str | None`)
- Chinese docstrings for all public classes/methods

**Async Programming:**
- I/O operations must use `async/await`
- HTTP client: `httpx.AsyncClient` (shared, cleaned on shutdown)
- Sync third-party calls wrapped in `loop.run_in_executor`

**Configuration:**
- Access via `config.get_api_config()` and `config.get_processing_config()`
- Environment variables in `.env` (never hardcode keys)
- Call `_refresh_config_cache()` or restart after config changes

**Output Format:**
- Final outline: **plain text only** (no markdown symbols)
- Enforced in `prompts.py` merge prompts

**Package Management:**
- Primary: `uv` (modern Python package manager)
- Legacy: `requirements.lock` (still present for compatibility)
- Build system: hatchling

## ANTI-PATTERNS (Forbidden)

1. **Type error suppression**: Never use `as any`, `@ts-ignore`, `@ts-expect-error`
2. **Test deletion**: Never skip or delete failing tests
3. **Config bypassing**: Never hardcode API keys or placeholder values
4. **Empty catch blocks**: Never swallow exceptions silently
5. **Import violations**: Never reorder (std→3rd→local)

## COMMANDS
```bash
# Virtual environment (using uv)
.venv/Scripts/python    # Windows
.venv/bin/python        # Linux/Mac

# Code quality (MUST execute in order)
.venv/Scripts/python -m ruff check . --fix    # Lint
.venv/Scripts/python -m black .               # Format
.venv/Scripts/python -m pytest tests/ -v       # Test

# Run application
python -Xutf8 main.py              # CLI mode selector
uvicorn web_api:app --reload       # FastAPI standalone
```

## NOTES

**Gotchas:**
- Windows Chinese support: Use `-Xutf8` flag for proper encoding
- Frontend served via `file://` protocol (not static files in FastAPI)
- Dual processing modes: Direct (`/process`) and Queued (`/queue/add`)
- Resume capability: Progress saved via `progress_service.py`
- Circuit breaker pattern: `CircuitBreaker` in `llm_service.py` for fault tolerance

**API Provider Selection:**
- Set `API_PROVIDER` env var: `openai`, `gemini`, `zhipu`, `aihubmix`
- All providers use unified `LLMService` interface

**Test Execution:**
- All tests async: Use `@pytest.mark.asyncio`
- Fixtures: `conftest.py` with `autouse=True`
- Run single test: `.venv/Scripts/python -m pytest tests/test_module.py::TestClass::test_method -v`
