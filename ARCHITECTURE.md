# Architecture

## Overview
This project generates novel outlines from large text files using an LLM provider.
It offers a CLI entrypoint and a FastAPI web API that share the same service layer.

## Key modules
- `main.py`: CLI entrypoint and interactive flow.
- `web_api.py`: FastAPI endpoints for upload, processing, and status.
- `services/novel_processing_service.py`: Orchestrates splitting, LLM calls, merging, and persistence.
- `services/llm_service.py`: Provider-specific LLM implementations with retry/circuit breaker logic.
- `services/file_service.py`: File I/O, validation, and metadata helpers.
- `services/progress_service.py`: Progress persistence and resume logic.
- `models/`: Data models for chunks and processing state.
- `splitter.py` and `tokenizer.py`: Text splitting and token estimation.

## Processing flow
1. Load and validate the input file.
2. Split the text into chunks based on token budget.
3. Call the configured LLM provider for each chunk (with retries and rate limits).
4. Merge chunk outlines into a final outline.
5. Save outputs and update progress state.

## Web API flow
- `/upload` stores a text file in `outputs/uploads`.
- `/process` starts an async job and returns a job id.
- `/jobs/{job_id}` returns status and logs.
- `/estimate` returns token usage estimates for a given file.
