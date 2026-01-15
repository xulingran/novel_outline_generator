# SERVICES/ KNOWLEDGE BASE

**Generated:** 2026-01-14
**Commit:** 50ea618
**Branch:** dev

## OVERVIEW
Business logic layer: LLM abstraction, core orchestration, async queue, progress tracking, ETA estimation, partial chunk completion.

## STRUCTURE
```
services/
├── llm_service.py              # Base LLMService + 4 provider impls + CircuitBreaker
├── novel_processing_service.py  # Orchestration: Split → Generate → Merge + partial completion
├── task_queue.py               # Async queue for batch processing
├── progress_service.py         # Persistence + backup recovery
├── eta_estimator.py            # Time estimation (moving avg + outlier rejection)
├── file_service.py             # File I/O with encoding detection
├── token_estimator.py          # Pre-processing cost estimation
└── __init__.py                 # Exports: create_llm_service()
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| LLM provider selection | `llm_service.py` | Factory: `create_llm_service()` |
| API call flow | `llm_service.py::call_llm()` | CircuitBreaker check → HTTP → retry |
| Split logic | `novel_processing_service.py::_split_text_into_chunks()` | Delegates to `splitter.py` |
| Parallel generation | `novel_processing_service.py::_process_chunks()` | Semaphore-limited |
| Partial completion | `novel_processing_service.py::_split_chunk_into_five()` | Failed chunks → 5 sub-chunks |
| Progress persistence | `progress_service.py::save_progress()` | JSON + backup |
| ETA calculation | `eta_estimator.py::estimate_remaining()` | Windowed average |
| Queue management | `task_queue.py::TaskQueue` | asyncio.Queue with status tracking |

## CONVENTIONS

**Service Initialization:**
- Call `get_api_config()` / `get_processing_config()` in `__init__`
- Lazy LLMService creation via `create_llm_service()` factory
- Shared `httpx.AsyncClient`: store in `llm_service.py`, clean on shutdown

**Async Patterns:**
- All public methods async (I/O)
- Use `asyncio.Semaphore` for parallelism limits
- `asyncio.Event` for cancellation support (pass to service constructors)

**Error Handling:**
- Raise `exceptions.APIError`, `APIKeyError`, `RateLimitError`, `ProcessingError`
- Never catch `Exception` without re-raising
- Log errors before raising

**Progress Tracking:**
- Batch progress updates (ProgressTracker with `batch_size=5`)
- Always call `_emit_progress()` after state changes
- Save progress after each completed chunk

**Partial Completion Pattern:**
- Failed chunks: Split into 5 sub-chunks (`SUB_CHUNK_COUNT`)
- Partial results: Mark with `is_partial: true`, store `partial_outlines`
- Resume: Merge partial outlines on recovery
- Token counting: Accumulate across both full chunks and sub-chunks

## ANTI-PATTERNS (Forbidden)

1. **Synchronous HTTP**: Never use sync `httpx.Client` in services layer
2. **Direct exception handling**: Never catch `Exception` broadly without logging
3. **Progress spam**: Never emit progress updates for every single operation (use batching)
4. **Hardcoded config**: Never bypass `config.py` - always use getters
5. **Silent failures**: Never suppress file I/O errors - log and propagate
6. **Blocking loops**: Never use `for` loops without `await` in async methods
7. **Circuit breaker bypass**: Never call provider APIs without `CircuitBreaker.call_allowed()` check
