# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ 代码质量检查（必须执行）

**每次写完代码后，必须依次执行以下检查，确保全部通过：**

```bash
# 1. Ruff 检查并修复
.venv/Scripts/python -m ruff check . --fix

# 2. Black 格式化
.venv/Scripts/python -m black .

# 3. Pytest 测试
.venv/Scripts/python -m pytest tests/ -v
```

## 常用命令

### 虚拟环境（使用项目内 .venv）
```bash
# Windows
.venv\Scripts\python

# Linux/Mac
.venv/bin/python
```

### 测试与代码质量
```bash
# 运行所有测试
.venv/Scripts/python -m pytest tests/ -v

# 运行单个测试文件
.venv/Scripts/python -m pytest tests/test_llm_service.py -v

# 运行单个测试
.venv/Scripts/python -m pytest tests/test_splitter.py::TestTextSplitter::test_split_short_text -v

# Ruff 代码检查
.venv/Scripts/python -m ruff check .

# Black 格式化检查
.venv/Scripts/python -m black --check .

# 自动修复 Ruff 问题
.venv/Scripts/python -m ruff check . --fix

# 自动格式化代码
.venv/Scripts/python -m black .
```

### 运行应用
```bash
# 命令行模式（Windows，启动后选择模式 2）
python -Xutf8 main.py

# Web UI 模式（Windows，启动后选择模式 1）
python -Xutf8 main.py

# Linux/Mac
python main.py

# 直接启动 Web API（开发模式）
.venv/Scripts/python -m uvicorn web_api:app --reload --port 8000
```

## 项目架构

### 核心设计模式

**1. 多提供商 LLM 抽象**
- `services/llm_service.py` 定义 `LLMService` 抽象基类
- 具体实现：`OpenAIService`、`GeminiService`、`ZhipuService`、`AiHubMixService`
- 工厂函数 `create_llm_service()` 根据 `API_PROVIDER` 环境变量创建对应服务
- 所有服务共享统一的错误处理、重试机制、熔断器模式

**2. 分层处理流程**
```
用户上传 → 文本分块 → 并行生成大纲 → 递归合并 → 输出
   ↓           ↓            ↓            ↓         ↓
file_service  splitter  llm_service  merge_prompt  最终txt
```

- **分块**：`splitter.py` 的 `TextSplitter` 按句子边界智能切分，控制每块 token 数量
- **生成**：`novel_processing_service.py` 并发调用 LLM 处理每个块
- **合并**：递归两两合并，直到得到单一总纲

**3. 状态管理与恢复**
- `models/processing_state.py` 定义 `ProcessingState` 和 `ProgressData`
- 支持断点恢复：通过 JSON 文件持久化进度
- `ETAEstimator` 根据历史处理时间动态预估剩余时间（过滤异常值，使用加权平均）

### 关键模块

| 模块 | 职责 |
|------|------|
| `config.py` | 使用环境变量的配置管理，支持多 API 提供商 |
| `prompts.py` | 提示词模板，支持通过环境变量自定义 |
| `services/novel_processing_service.py` | 核心业务逻辑：分块、调用 LLM、合并、保存 |
| `services/llm_service.py` | LLM 调用统一接口，包含重试、熔断、速率限制处理 |
| `services/progress_service.py` | 进度更新与日志记录 |
| `services/eta_estimator.py` | ETA 估算（滑动窗口 + 异常值过滤 + 加权平均） |
| `splitter.py` | 智能文本分块，按句子边界切分 |
| `tokenizer.py` | Token 计数，基于 tiktoken |
| `web_api.py` | FastAPI 后端，提供文件上传、处理、进度查询接口 |
| `main.py` | CLI 入口，选择 Web UI 或文件处理模式 |

### 数据流

1. **输入**：txt/md 文本文件
2. **分块**：按 `TARGET_TOKENS_PER_CHUNK` 切分
3. **并行处理**：根据 `PARALLEL_LIMIT` 并发调用 LLM
4. **递归合并**：
   - 第一层：合并相邻块的大纲
   - 后续层：继续合并直到得到单一总纲
5. **输出**：`{原文件名}-提纲.{扩展名}` 纯文本文件

### API 提供商配置

通过 `API_PROVIDER` 环境变量选择：
- `openai`：需配置 `OPENAI_API_KEY`、可选 `OPENAI_API_BASE`
- `gemini`：需配置 `GEMINI_API_KEY`
- `zhipu`：需配置 `ZHIPU_API_KEY`
- `aihubmix`：需配置 `AIHUBMIX_API_KEY`

### 输出格式

最终大纲输出为**纯文本格式**，不包含 markdown 符号。这在 `prompts.py` 的 `merge_prompt` 和 `merge_text_prompt` 中明确要求。

### 错误处理

- `APIKeyError`：API 密钥未配置或无效
- `RateLimitError`：速率限制，支持 `retry-after` 头
- `APIError`：通用 API 错误，支持重试标记
- `ContentFilterError`：内容审查错误（不触发熔断器）
- 熔断器：连续失败达到阈值后暂停请求

### 失败块重试机制

当某个文本块处理失败时，系统会：
1. 先进行指数退避重试（最多 `MAX_RETRY` 次）
2. 若仍失败，将该块拆分为 5 个小块（`SUB_CHUNK_COUNT`）
3. 逐个处理小块，成功的小块会被合并为部分完成的大纲
4. 部分完成的块标记为 `is_partial: true`，包含 `partial_outlines` 字段
5. 进度恢复时会自动合并部分完成的小块

### 断点恢复机制

- 进度文件：`outputs/progress.json`
- 包含内容：已完成块、部分完成块、失败块、处理时间等
- 验证机制：基于文件路径和内容哈希（`ProgressData.calculate_chunks_hash`）
- 恢复时会合并部分完成的小块为完整大纲

### Web API 关键端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/env` | GET | 查看环境配置（API Key 已脱敏） |
| `/upload` | POST | 单文件上传 |
| `/upload-multiple` | POST | 批量文件上传 |
| `/process` | POST | 启动处理任务 |
| `/jobs/{job_id}` | GET | 查询任务状态和进度 |
| `/jobs/{job_id}/logs` | GET | 获取任务日志（支持增量） |
| `/queue/status` | GET | 查看任务队列状态 |
| `/force-complete/{job_id}` | POST | 强制完成任务（忽略未完成块） |

### 任务队列

- 全局单例队列：`get_global_queue()`
- 并发限制：默认 1 个任务同时运行（`max_concurrent`）
- 任务优先级：支持优先级排序（默认 FIFO）
- 自动清理：24 小时后清理完成的任务

### 测试覆盖

重要测试文件：
- `test_llm_service.py`：LLM 服务和熔断器测试
- `test_splitter.py`：文本分块逻辑测试
- `test_processing_state.py`：处理状态管理测试
- `test_partial_completion.py`：部分完成机制测试
- `test_resume_processing.py`：断点恢复测试
- `test_task_queue.py`：任务队列测试
- `test_web_api.py`：Web API 端点测试
- `test_logging_config.py`：日志配置和轮转测试

## 开发注意事项

### 配置修改后的重启

修改 `.env` 文件后，必须**重启服务**（Web API 或 CLI）才能生效。配置在启动时加载，运行时不会自动重载。

### Token 计数

- 使用 `tiktoken` 库（`cl100k_base` 编码）
- 所有 token 计数通过 `tokenizer.py` 的 `count_tokens()` 统一处理
- 分块时按 `TARGET_TOKENS_PER_CHUNK` 控制大小

### 提示词自定义

提示词定义在 `prompts.py` 中：
- `chunk_prompt()`: 处理单个文本块的提示词
- `merge_prompt()`: 合并 JSON 格式大纲
- `merge_text_prompt()`: 合并文本格式大纲

**重要**：所有合并提示词都明确要求输出**纯文本格式**，不使用 Markdown 符号。

### 虚拟环境路径

项目使用项目内虚拟环境 `.venv`，所有命令都应使用：
- Windows: `.venv\Scripts\python`
- Linux/Mac: `.venv/bin/python`

避免使用系统 Python 或全局虚拟环境。

### 日志系统

项目使用按天自动轮转的日志系统：
- **日志目录**：`logs/`（默认，可通过 `LOG_DIR` 环境变量配置）
- **当前日志**：`novel_outline.log`（当天的日志）
- **历史日志**：`novel_outline.log.YYYY-MM-DD`（如 `novel_outline.log.2026-01-16`）
- **轮转规则**：每天午夜自动轮转，当前日志重命名为带日期后缀的文件
- **日志保留**：默认保留 30 天，可通过 `LOG_BACKUP_DAYS` 环境变量配置
- **日志级别**：默认 INFO，可通过 `LOG_LEVEL` 环境变量配置（DEBUG, INFO, WARNING, ERROR）

**环境变量配置**（在 `.env` 文件中）：
```bash
LOG_LEVEL=INFO          # 日志级别
LOG_DIR=logs            # 日志目录
LOG_BACKUP_DAYS=30      # 保留天数
```

**实现细节**：
- 使用 `TimedRotatingFileHandler` 实现按天轮转
- 自动清理超过保留天数的旧日志文件
- 同时输出到文件（所有级别）和控制台（INFO 及以上）
- 日志格式：`YYYY-MM-DD HH:MM:SS - 模块名 - 级别 - 消息`

