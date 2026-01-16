# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-14
**Commit:** 50ea618
**Branch:** dev

## OVERVIEW
AI-powered novel outline generator using LLM APIs (OpenAI, Gemini, Zhipu, AiHubMix). Async-first architecture with dual-mode entry (CLI/Web UI), intelligent text splitting, parallel processing, partial chunk completion, and resume capability.

## STRUCTURE
```
novel_outline_generator/
├── main.py                    # Dual-mode entry: Web UI (mode 1) / Processing (mode 2)
├── web_api.py                 # FastAPI backend (standalone entry point)
├── config.py                  # API + processing configuration (env-based, lazy validation)
├── splitter.py                # Intelligent text chunking (chapter/paragraph/token-based)
├── tokenizer.py               # Token counting utilities (tiktoken with fallback)
├── prompts.py                 # LLM prompt templates (merge, compact, array)
├── exceptions.py              # Custom exception hierarchy
├── validators.py              # Input validation (paths, filenames, security)
├── utils.py                   # Logging, helpers, ProgressTracker batching
├── services/                  # Business logic layer (8 modules)
│   ├── llm_service.py        # LLM abstraction base + 4 provider impls + CircuitBreaker
│   ├── novel_processing_service.py  # Core orchestration (Split → Generate → Merge)
│   ├── task_queue.py         # Async queue for batch jobs
│   ├── progress_service.py   # Save/load resume capability with partial chunk support
│   ├── eta_estimator.py      # Time estimation (moving avg + outlier rejection)
│   ├── file_service.py       # File I/O with encoding detection
│   └── token_estimator.py   # Pre-processing cost estimation
├── models/                   # Data models (processing_state, outline, character)
├── tests/                    # Pytest test suite (12 files, 151 tests)
└── ui/index.html             # SPA frontend (file:// URI, calls FastAPI)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Entry points | `main.py` (CLI selector), `web_api.py` (FastAPI) | Two independent entry points |
| API calls | `services/llm_service.py` | Provider-specific implementations with CircuitBreaker |
| Core logic | `services/novel_processing_service.py` | Split → Generate → Merge flow |
| Partial completion | `services/novel_processing_service.py::_split_chunk_into_five()` | Fallback for failed chunks |
| Progress resume | `services/progress_service.py` | JSON persistence + backup recovery |
| Text splitting | `splitter.py` | Chapter/paragraph/token detection |
| Config management | `config.py` | `get_api_config()`, lazy validation |
| Token counting | `tokenizer.py` | Tiktoken-based with fallback encoder |
| Testing | `tests/` | pytest, fixtures in `conftest.py` (dual autouse) |
| Frontend | `ui/index.html` | SPA, file:// URI, HTTP to localhost:8000 |
| Batch processing | `services/task_queue.py` | Async queue with concurrency limits |

## CONVENTIONS (Deviations from Standard)

**Project Structure:**
- **Flat package** (no `src/` directory) - Core modules at root level
- `services/` and `models/` are only organized subdirectories
- Entry point `main.py` launches uvicorn internally for Web UI mode (not separate process)

**Code Style:**
- Line limit: 100 characters (Black/Ruff, non-standard default)
- Import order: standard → third-party → local (strict)
- Type annotations required for all public APIs (modern `str | None`)
- Chinese docstrings for all public classes/methods

**Async Programming:**
- I/O operations must use `async/await`
- HTTP client: `httpx.AsyncClient` (shared, cleaned on shutdown)
- Sync third-party calls wrapped in `loop.run_in_executor`

**Configuration:**
- Access via `config.get_api_config()` and `config.get_processing_config()`
- **Lazy validation**: API keys validated on first access, not at import time
- Environment variables in `.env` (never hardcode keys)
- Call `_refresh_config_cache()` or restart after config changes

**Output Format:**
- Final outline: **plain text only** (no markdown symbols)
- Enforced in `prompts.py` merge prompts

**Error Handling:**
- Partial chunk completion: Failed chunks split into 5 sub-chunks, merged if some succeed
- Progress recovery: Supports resume from partial completion states
- Circuit breaker: `CircuitBreaker` in `llm_service.py` for fault tolerance

**Package Management:**
- Primary: `uv` (modern Python package manager, Rust-based)
- Legacy: `requirements.lock` (binary lock file, not pip compatible)
- Build system: hatchling

## ANTI-PATTERNS (Forbidden)

1. **Type error suppression**: Never use `as any`, `@ts-ignore`, `@ts-expect-error`
2. **Test deletion**: Never skip or delete failing tests
3. **Config bypassing**: Never hardcode API keys or placeholder values
4. **Empty catch blocks**: Never swallow exceptions silently
5. **Import violations**: Never reorder (std→3rd→local)
6. **Synchronous HTTP**: Never use sync `httpx.Client` in services layer
7. **Progress spam**: Never emit progress updates for every operation (use batching)

## COMMANDS
```bash
# Virtual environment (using uv)
.venv/Scripts/python    # Windows
.venv/bin/python        # Linux/Mac

# Code quality (MUST execute in order)
.venv/Scripts/python -m ruff check . --fix    # Lint
.venv/Scripts/python -m black .               # Format
.venv/Scripts/python -m mypy .                # Type check
.venv/Scripts/python -m pytest tests/ -v       # Test

# Run application
python -Xutf8 main.py              # CLI mode selector (Windows UTF-8)
python main.py                     # CLI mode selector (Linux/Mac)
uvicorn web_api:app --reload       # FastAPI standalone

# Using uv (modern package manager)
uv venv                            # Create virtual environment
uv pip install -e ".[dev]"         # Install dependencies
uv run python main.py               # Run with uv
```

## NOTES

**Gotchas:**
- Windows Chinese support: Use `-Xutf8` flag for proper encoding
- Frontend served via `file://` protocol (not static files in FastAPI)
- Dual processing modes: Direct (`/process`) and Queued (`/queue/add`)
- Resume capability: Progress saved via `progress_service.py`
- Circuit breaker pattern: `CircuitBreaker` in `llm_service.py` for fault tolerance
- Partial completion: Failed chunks split into 5 sub-chunks for recovery

**API Provider Selection:**
- Set `API_PROVIDER` env var: `openai`, `gemini`, `zhipu`, `aihubmix`
- All providers use unified `LLMService` interface
- Lazy validation: API keys checked on first use, not import

**Test Execution:**
- All tests async: Use `@pytest.mark.asyncio`
- Fixtures: `conftest.py` with dual autouse fixtures (setup_test_env + disable_dotenv)
- Custom `DummyService` for LLM mocking (inherits from `LLMService`)
- Run single test: `.venv/Scripts/python -m pytest tests/test_module.py::TestClass::test_method -v`

**CI/CD:**
- GitHub Actions only triggers on `master` branch (intentional restriction)
- Dual workflows: lint.yml (Ruff, Black, Mypy) + test.yml (pytest + coverage)
- Coverage uploaded to Codecov

**Partial Chunk Completion:**
- When chunk processing fails, split into 5 sub-chunks (`SUB_CHUNK_COUNT`)
- Successful sub-chunks merged into partial outline (`is_partial: true`)
- Resume logic automatically merges partial results
- Progress file contains `partial_chunks` count
