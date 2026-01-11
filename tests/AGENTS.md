# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-11
**Commit:** 9a57cdb
**Branch:** main

## OVERVIEW
Pytest test suite with async support covering all modules (11 test files).

## WHERE TO LOOK
| Category | Test Files | Coverage |
|----------|------------|----------|
| **API/Integration** | `test_web_api.py` | FastAPI endpoints, rate limiting, file upload, jobs, queue |
| **LLM Services** | `test_llm_service.py` | Retry logic, circuit breaker, rate limit handling, provider init |
| **Core Processing** | `test_eta_estimator.py`, `test_task_queue.py`, `test_progress_service.py` | Time estimation, async queue management, state persistence |
| **Text Processing** | `test_splitter.py`, `test_tokenizer.py` | Chunking logic, token counting, sentence detection |
| **Validation/Config** | `test_validators.py`, `test_processing_state.py` | Path validation, API provider check, data models |
| **Utilities** | `test_utils.py` | Helper functions, logging |

## CONVENTIONS

**Test Framework:**
- All async tests: `@pytest.mark.asyncio` decorator required
- Global fixtures in `conftest.py`: `setup_test_env`, `disable_dotenv` (both `autouse=True`)
- TestClient from `fastapi.testclient` for API integration tests

**Mocking Pattern:**
- LLM tests: Custom `DummyService` inheriting `LLMService` with controlled responses
- `monkeypatch` for environment variables and external dependencies
- Use `types.SimpleNamespace` for mocking module imports

**Test Isolation:**
- `test_web_api.py`: Clear `web_api.JOBS` dict, reset rate limiter per test
- Global queue: Clean up via `queue.clear_queue()` after tests
- Temporary files: `tmp_path` fixture, explicit cleanup

**Fixture Usage:**
- Module-level fixtures for common test setup
- Queue tests: Use `queue()` fixture returning `TaskQueue(max_concurrent=1)`

## ANTI-PATTERNS

1. **Test deletion/skipping**: Never remove or skip failing tests - fix them
2. **Real API keys**: Never use actual API keys in tests (use `test-key-for-ci`)
3. **Sleep-based timing**: Prefer `monkeypatch` to fake `asyncio.sleep` over real delays
4. **Unmocked I/O**: Never make real network calls or file writes outside `tmp_path`
5. **Global state mutation**: Always reset global state between tests (jobs, rate limiter, queue)
6. **Hardcoded paths**: Never use absolute paths - always use `tmp_path` or relative paths
7. **Test interdependence**: Tests must run independently - no dependency on test execution order
