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
# 命令行模式
python -Xutf8 main.py

# Web UI 模式（启动后选择模式 1）
python -Xutf8 main.py
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
- 熔断器：连续失败达到阈值后暂停请求
