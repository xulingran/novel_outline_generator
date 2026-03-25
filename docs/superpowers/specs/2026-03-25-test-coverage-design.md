# 测试覆盖率提升设计文档

**日期**: 2026-03-25
**状态**: 待实施
**目标**: 整体测试覆盖率从 58% 提升至 85%

---

## 背景

经过三层渐进式重构（commit c17b243），项目服务层已完成拆分：
- `ChunkProcessor`、`OutlineMerger`、`ProgressTracker` 已从 `NovelProcessingService` 提取
- 当前整体覆盖率 58%，有多个关键文件覆盖率严重不足

## 覆盖率目标

| 文件 | 当前覆盖率 | 目标覆盖率 | 优先级 |
|---|---|---|---|
| `services/llm_service.py` | 39% | 80% | 高 |
| `tokenizer.py` | 69% | 95% | 高 |
| `services/outline_merger.py` | 69% | 90% | 高 |
| `web_api.py` | 74% | 85% | 中 |
| `validators.py` | 78% | 90% | 中 |
| `services/progress_service.py` | 75% | 88% | 中 |

---

## 方案：分文件逐个补测（方案 A）

按覆盖率从低到高，逐文件独立补充测试，每个文件独立提交，风险最低。

---

## 详细设计

### 1. llm_service.py（39% → 80%）

**策略**：各 Provider 分别 mock 对应 SDK + 可选集成测试（分两个文件）

> **重要**：各 Provider 使用不同的 HTTP 客户端，不能统一 mock httpx：
> - `OpenAIService` / `ZhipuService`：mock `self.client.chat.completions.create`
> - `GeminiService`：mock `google.generativeai.GenerativeModel.generate_content`
> - `AiHubMixService`：mock `requests.post`（同步调用，用 `run_in_executor` 包装）

**扩充 `tests/test_llm_service.py`**

#### 组 1 - 熔断器状态机（`CircuitBreaker`，纯逻辑）

覆盖三种状态转换：

```python
def test_circuit_breaker_initial_state_closed()
def test_circuit_breaker_stays_closed_below_threshold()   # 未达阈值时保持 CLOSED
def test_circuit_breaker_opens_after_threshold()          # CLOSED → OPEN
def test_circuit_breaker_half_open_after_timeout()        # OPEN → HALF_OPEN
def test_circuit_breaker_closes_on_success()              # HALF_OPEN → CLOSED
def test_circuit_breaker_reopens_on_half_open_fail()      # HALF_OPEN → OPEN
def test_circuit_breaker_blocks_calls_when_open()
```

#### 组 2 - Provider 层 mock（各 SDK 分别 mock）

```python
# OpenAI Provider
async def test_openai_provider_parse_response()           # mock client.chat.completions.create
async def test_openai_provider_handles_rate_limit()       # 抛出 openai.RateLimitError
async def test_openai_provider_handles_timeout()          # 抛出 openai.APITimeoutError
async def test_openai_provider_no_retry_on_content_filter()

# Gemini Provider
async def test_gemini_provider_parse_response()           # mock GenerativeModel.generate_content

# Zhipu Provider
async def test_zhipu_provider_parse_response()            # mock client.chat.completions.create

# 通用重试逻辑（通过 DummyService 子类覆盖 _call_api，沿用现有测试模式）
async def test_provider_retry_on_transient_error()
async def test_provider_exhausts_retries_raises()
```

#### 组 3 - 注册机制

```python
def test_provider_registry_registers_correctly()
def test_get_llm_service_with_known_provider()
def test_get_llm_service_with_unknown_provider_raises()
```

**新增 `tests/test_llm_integration.py`**

```python
import os
import pytest

@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="需设置 RUN_INTEGRATION_TESTS=1 以运行集成测试"
)
class TestLLMIntegration:
    async def test_real_openai_call()
    async def test_real_gemini_call()
```

**依赖**：无需新增库，各 Provider 使用对应 SDK 的 mock

---

### 2. tokenizer.py（69% → 95%）

**策略**：纯逻辑，无外部依赖，直接覆盖全部分支

**扩充 `tests/test_tokenizer.py`**

```python
def test_count_tokens_empty_string()
def test_count_tokens_chinese_text()
def test_count_tokens_mixed_language()
def test_count_tokens_special_characters()
def test_estimate_tokens_from_chars_basic()           # 补充 estimate_tokens_from_chars() 覆盖
def test_estimate_tokens_from_chars_empty_string()
def test_truncate_by_tokens_truncates_correctly()     # truncate_by_tokens() 而非 count_tokens 截断
def test_truncate_by_tokens_no_truncate_when_short()
# 注：tokenizer 使用模块级单例 _encoder，fallback 测试需通过 importlib.reload 重置状态
def test_fallback_encoder_used_when_tiktoken_missing()
```

---

### 3. outline_merger.py（69% → 90%）

**策略**：新建测试文件，mock LLM 调用

**新增 `tests/test_outline_merger.py`**

```python
@pytest.fixture
def merger_fixture():
    """完整的 OutlineMerger fixture，包含所有必需依赖"""
    llm_service = AsyncMock()
    llm_service.generate.return_value = LLMResponse(
        content="merged outline",
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    processing_state = MagicMock()
    cancel_event = asyncio.Event()
    emit_progress_fn = AsyncMock()
    accumulate_tokens_fn = MagicMock()
    merger = OutlineMerger(llm_service=llm_service, cancel_event=cancel_event)
    return merger, processing_state, emit_progress_fn, accumulate_tokens_fn

class TestOutlineMerger:
    async def test_merge_single_chunk_returns_as_is()
    async def test_merge_two_chunks_calls_llm_once()
    async def test_merge_recursive_batching()
    # 触发条件：mock count_tokens 使 input_tokens > model_max_tokens * MAX_INPUT_TOKEN_RATIO
    async def test_merge_respects_cancel_event()        # cancel_event.set() 后抛出 CancelledError
    async def test_merge_partial_outlines_list()
    async def test_merge_handles_empty_input()          # 测试 merge_outlines_recursive([]) 返回 ""
    async def test_merge_accumulates_token_usage()
```

---

### 4. web_api.py（74% → 85%）

**策略**：`TestClient` + mock 所有服务依赖

**扩充 `tests/test_web_api.py`**

```python
from fastapi.testclient import TestClient

@pytest.fixture
def client(monkeypatch):
    # mock NovelProcessingService, TaskQueue, JobManager
    from web_api import app
    return TestClient(app)

class TestUploadEndpoint:
    def test_upload_single_file_success()
    def test_upload_rejects_non_txt_file()
    def test_upload_multiple_files()

class TestProcessEndpoint:
    def test_process_creates_job_and_returns_id()
    def test_process_with_missing_file_returns_404()
    def test_process_queues_task_correctly()

class TestJobsEndpoint:
    def test_get_job_status_running()
    def test_get_job_status_completed()
    def test_get_job_not_found_returns_404()

class TestQueueEndpoints:
    def test_queue_list()
    def test_queue_cancel()
    def test_queue_clear()
    def test_estimate_tokens()
```

---

### 5. validators.py（78% → 90%）

**策略**：补全边界分支和安全场景

**扩充 `tests/test_validators.py`**

```python
def test_validate_path_with_traversal_attack()        # ../../../etc/passwd
def test_validate_path_double_slash()                 # /etc//passwd（替代 URL 编码变体，更符合实际）
def test_validate_parallel_limit_at_boundary_20()    # limit=20 触发 warning
def test_validate_parallel_limit_exceeds_boundary()  # limit=21
def test_validate_empty_filename_raises()
def test_validate_oversized_file_raises()
def test_validate_unsupported_extension_raises()
```

---

### 6. progress_service.py（75% → 88%）

**策略**：mock 文件 I/O，覆盖持久化和恢复路径

**扩充 `tests/test_progress_service.py`**

```python
def test_save_progress_creates_file(tmp_path)
def test_load_progress_returns_saved_state(tmp_path)
def test_load_progress_missing_file_returns_none()
def test_load_progress_corrupted_file_tries_bak_recovery(tmp_path)
# 注：损坏文件时会尝试 .bak 备份恢复，并生成 .corrupt_* 文件，非简单返回 None
def test_has_partial_progress_returns_true_when_exists()
def test_has_partial_progress_returns_false_when_missing()
# 注：ProgressService 不提供 cleanup_expired 方法，过期清理属于 JobManager 职责
```

---

## 执行顺序

1. `tokenizer.py` + `outline_merger.py`（纯逻辑，风险最低）
2. `validators.py` + `progress_service.py`（有限外部依赖）
3. `llm_service.py`（最复杂，单独攻克）
4. `web_api.py`（依赖前面服务稳定后再做）

每个文件完成后运行以下命令验证覆盖率提升，再提交：

```bash
.venv/bin/python -m pytest tests/ --cov=. --cov-report=term-missing -q
```

> GUI 测试（`tests/test_gui/`）使用 mock customtkinter，可在无图形环境运行，纳入统计。

---

## 验收标准

- `pytest tests/ -q` 全部通过，无新增失败
- `pytest tests/ --cov --cov-report=term-missing` 显示整体覆盖率 ≥ 85%
- 集成测试默认跳过（`RUN_INTEGRATION_TESTS` 未设置时）
- `ruff check . && black . && mypy .` 全部通过
