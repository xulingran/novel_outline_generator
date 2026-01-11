# AGENTS.md - 代理开发指南

本文档为在此项目中进行代码开发的AI代理提供指导。

## 构建与测试命令

### 虚拟环境（使用项目内 .venv）
```bash
# Windows
.venv\Scripts\python

# Linux/Mac
.venv/bin/python
```

### 代码质量检查（必须按顺序执行）
```bash
# 1. Ruff 代码检查并自动修复
.venv/Scripts/python -m ruff check . --fix

# 2. Black 代码格式化
.venv/Scripts/python -m black .

# 3. Pytest 运行测试
.venv/Scripts/python -m pytest tests/ -v
```

### 运行单个测试
```bash
# 运行单个测试文件
.venv/Scripts/python -m pytest tests/test_llm_service.py -v

# 运行单个测试方法
.venv/Scripts/python -m pytest tests/test_splitter.py::TestTextSplitter::test_split_short_text -v

# 运行带有特定标记的测试
.venv/Scripts/python -m pytest tests/ -v -m "标记名"
```

### 运行应用
```bash
# 命令行模式
python -Xutf8 main.py

# Web UI 模式（启动后选择模式 1）
python -Xutf8 main.py
```

## 代码风格指南

### 导入顺序（严格遵循）
1. 标准库导入
2. 第三方库导入
3. 本地项目导入

示例：
```python
# 标准库
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

# 第三方库
import httpx
from openai import AsyncOpenAI

# 本地导入
from config import get_api_config
from exceptions import APIError
```

### 类型注解
- 必须为所有公共函数和方法添加类型注解
- 使用现代Python联合类型语法（`str | None` 而非 `Optional[str]`）
- 使用 `from __future__ import annotations` 或直接使用类型注解

```python
from typing import Any

def process_text(content: str, max_tokens: int | None = None) -> dict[str, Any]:
    """处理文本内容"""
    pass

class LLMService(ABC):
    """LLM服务基类"""

    async def call(self, prompt: str, chunk_id: int | None = None) -> LLMResponse:
        """统一的调用接口"""
        pass
```

### 命名约定
- **类名**：PascalCase（如 `LLMService`, `APIConfig`）
- **函数/方法名**：snake_case（如 `get_api_config`, `process_text`）
- **常量**：UPPER_SNAKE_CASE（如 `SUPPORTED_API_PROVIDERS`, `MAX_RETRY`）
- **私有方法**：前导下划线（如 `_init_client`, `_call_api`）

### 错误处理
- 使用自定义异常类（继承自 `exceptions.NovelOutlineError`）
- 为API错误提供 `is_retryable` 标志
- 在错误消息中包含上下文信息

```python
from exceptions import APIError, APIKeyError

def validate_api_key(self) -> None:
    if not self.api_key:
        raise APIKeyError("API密钥未配置")

async def call_api(self, prompt: str) -> LLMResponse:
    try:
        response = await self.client.chat.completions.create(...)
    except Exception as e:
        if "rate" in str(e).lower():
            raise APIError("速率限制", is_retryable=True) from e
        raise APIError("API调用失败", is_retryable=False) from e
```

### 文档字符串
- 所有公共类和方法必须有中文docstring
- 使用三引号格式
- 简洁描述功能，必要时包含参数和返回值说明

```python
class CircuitBreaker:
    """熔断器模式实现"""

    def call_allowed(self) -> bool:
        """检查是否允许调用"""
        pass
```

### 异步编程
- I/O密集型操作必须使用 `async/await`
- 同步第三方库调用需用 `loop.run_in_executor` 包裹
- HTTP客户端使用 `httpx.AsyncClient`

```python
import asyncio

async def _call_sync_library(self) -> str:
    """在异步上下文中调用同步库"""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, self.sync_method)
    return result
```

### 配置管理
- 所有配置通过 `config.py` 的 `get_api_config()` 和 `get_processing_config()` 访问
- 使用环境变量 `.env` 文件管理敏感信息
- 修改配置后需调用 `_refresh_config_cache()` 或重启服务

### 日志记录
- 使用 `logging` 模块，每个模块创建独立logger
- 日志级别：DEBUG（调试信息）、INFO（正常流程）、WARNING（警告）、ERROR（错误）
- 在关键操作前后添加日志

```python
import logging

logger = logging.getLogger(__name__)

def process_data(self, data: Any) -> None:
    logger.debug(f"开始处理数据: {len(data)}")
    try:
        # 处理逻辑
        logger.info("数据处理完成")
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
```

### 测试编写
- 使用 `pytest` 框架
- 测试文件命名：`test_<module>.py`
- 测试类命名：`Test<ClassName>`
- 测试方法命名：`test_<functionality>`

```python
import pytest

class TestLLMService:
    """LLM服务测试"""

    @pytest.mark.asyncio
    async def test_call_success(self, mock_llm_service):
        """测试API调用成功"""
        response = await mock_llm_service.call("测试提示")
        assert response.content is not None
```

## 项目架构要点

### 核心模块职责
- `config.py`: 配置管理（API密钥、处理参数）
- `services/llm_service.py`: LLM抽象基类和具体实现（OpenAI、Gemini、Zhipu、AiHubMix）
- `services/novel_processing_service.py`: 小说处理主逻辑（分块、LLM调用、合并）
- `services/progress_service.py`: 进度更新与日志记录
- `services/eta_estimator.py`: ETA估算
- `splitter.py`: 智能文本分块
- `models/processing_state.py`: 状态数据模型

### 数据流
```
用户上传 → 文本分块 → 并行生成大纲 → 递归合并 → 输出txt
```

### API提供商支持
通过 `API_PROVIDER` 环境变量选择：`openai`、`gemini`、`zhipu`、`aihubmix`

### 注意事项
- **禁止**使用 `as any`、`@ts-ignore`、`@ts-expect-error` 等类型错误抑制
- **禁止**删除或跳过失败的测试
- **必须**在完成代码更改后运行全部检查命令并确保通过
