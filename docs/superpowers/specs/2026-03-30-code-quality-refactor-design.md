# 代码质量与模块化重构设计

日期：2026-03-30
分支：`feat/test-coverage-85` 基础上继续

## 概述

对 novel_outline_generator 项目进行方案 B 级别的综合改进：精准清理 + 重点重构，提升代码可维护性和可测试性。

## 第一部分：精准清理

### 1.1 删除未使用的 `models/character.py`

- `Character` 和 `Relationship` 两个类从未被项目代码引用
- TODO 注释标注"2026 年 Q3 前未使用考虑移除"，现已到期
- 同时清理 `models/__init__.py` 中可能的导入
- 删除相关测试文件（如存在）

### 1.2 清理 `requirements.txt` 重复条目

当前文件第 22-28 行和第 31-39 行完全重复：
- `customtkinter>=5.2.0`（重复）
- `pillow>=10.0.0`（重复）
- `pyinstaller>=6.0.0`（重复）
- `pywin32>=305`（重复）

保留第一份，删除第二段重复内容。

### 1.3 拆分 `utils.py`（500 行 → 3 个模块）

将 `utils.py` 转为 `utils/` 包，按职责拆分：

| 新模块 | 内容 | 行数约 |
|--------|------|--------|
| `utils/logging_config.py` | `setup_logging()`、`init_logging()`、`_logging_configured` 标志 | ~100 |
| `utils/file_ops.py` | `_is_plausible_text`、`_ensure_directory`、`_create_backup`、`_write_temp_file`、`atomic_write_json`、`atomic_write_text`、`safe_read_json`、`safe_read_text`、`detect_text_encoding`、`format_file_size`、`get_file_info` | ~300 |
| `utils/text.py` | `truncate_text`、`ProgressTracker` | ~60 |

**兼容性**：`utils/__init__.py` 全部 re-export，保持现有 `from utils import xxx` 导入不变。

## 第二部分：拆分 `web_api.py`（821 行 → 4 个模块）

### 2.1 当前问题

`web_api.py` 混合了多种职责：
- FastAPI 路由定义
- `RateLimiter` 限流器实现
- `Job` 数据类与存储管理
- 回退 `JobManager` 实现（70 行）
- 上传文件处理与清理逻辑
- 进度回调处理
- CORS 配置

### 2.2 拆分方案

```
web_api/
├── __init__.py        # 导出 app，保持 uvicorn web_api:app 兼容
├── rate_limiter.py    # ~30 行
├── job_storage.py     # ~180 行
├── upload_handler.py  # ~120 行
└── routes.py          # ~350 行
```

#### `web_api/rate_limiter.py`
- `RateLimiter` 类
- `rate_limiter` 模块级实例

#### `web_api/job_storage.py`
- 回退 `JobManager` 实现（保持 TYPE_CHECKING 导入模式）
- `Job` 数据类
- `job_manager` 实例、`JOBS` 别名
- 常量：`MAX_JOBS`、`JOB_MAX_AGE_HOURS`
- 清理函数：`cleanup_expired_jobs`、`cleanup_excess_jobs`、`_periodic_job_cleanup`、`startup_cleanup_task`
- `_update_progress_from_info` 进度更新辅助
- `format_token_usage_log` 日志格式化

#### `web_api/upload_handler.py`
- `UPLOAD_DIR`、`_UPLOAD_ROOT` 常量
- `_resolve_upload_path` 路径验证
- `cleanup_uploads` 清理逻辑
- `load_env_file` 本地包装

#### `web_api/routes.py`
- `_load_cors_origins`、`CORS_ORIGINS`
- `_SENSITIVE_KEYWORDS`、`_mask_sensitive_value` 掩码工具（仅被 `/env` 端点使用）
- Pydantic 模型：`ProcessRequest`、`MultipleFilesRequest`
- `lifespan` 上下文管理器
- FastAPI `app` 实例 + 所有路由
- `_run_job`、`run_queue_task` 任务执行函数

#### `web_api/__init__.py`
```python
from web_api.routes import app
from web_api.job_storage import Job, job_manager, JOBS

__all__ = ["app", "Job", "job_manager", "JOBS"]
```

确保 `uvicorn web_api:app` 和测试中 `from web_api import JOBS` 继续工作。

## 第三部分：改进 `config.py`

### 3.1 消除 lambda 滥用 — `env_field` 辅助函数

引入 `env_field()` 替代 14 个重复的 `field(default_factory=lambda: os.getenv(...))` 模式：

```python
def env_field(env_var: str, default: str | None = None, cast: type = str):
    """从环境变量创建 dataclass field。"""
    def factory():
        raw = os.getenv(env_var, default)
        if raw is None:
            return None
        return cast(raw) if cast is not str else raw
    return field(default_factory=factory)
```

使用示例：
```python
@dataclass
class APIConfig:
    provider: str = env_field("API_PROVIDER", "openai")
    openai_key: str | None = env_field("OPENAI_API_KEY")
    openai_model: str = env_field("OPENAI_MODEL", "gpt-4o-mini")
    model_max_tokens: int = env_field("MODEL_MAX_TOKENS", "200000", cast=int)
```

### 3.2 统一 provider 配置映射

合并当前 `_PROVIDER_KEY_CONFIG`（用于验证）和 `api_key`/`base_url`/`model_name` 属性中的 if-elif 链为统一映射：

```python
_PROVIDER_REGISTRY: ClassVar[dict[str, dict[str, str | None]]] = {
    "openai": {
        "key_field": "openai_key",
        "base_field": "openai_base",
        "model_field": "openai_model",
        "name": "OpenAI API",
        "env_var": "OPENAI_API_KEY",
        "hint": "提示：OpenAI API Key 通常以 'sk-' 开头",
    },
    "gemini": {
        "key_field": "gemini_key",
        "base_field": None,
        "model_field": "gemini_model",
        "name": "Gemini API",
        "env_var": "GEMINI_API_KEY",
        "hint": "",
    },
    "zhipu": {
        "key_field": "zhipu_key",
        "base_field": "zhipu_base",
        "model_field": "zhipu_model",
        "name": "智谱API",
        "env_var": "ZHIPU_API_KEY",
        "hint": "",
    },
    "aihubmix": {
        "key_field": "aihubmix_api_key",
        "base_field": "aihubmix_api_base",
        "model_field": "aihubmix_model",
        "name": "AiHubMix API",
        "env_var": "AIHUBMIX_API_KEY",
        "hint": "",
    },
}
```

`api_key`、`base_url`、`model_name` 属性改为查表实现，消除 if-elif 链。

### 3.3 统一错误消息格式

当前 `_validate_api_key` 中 OpenAI 和其他 provider 错误消息格式不一致（OpenAI 多一个换行和 hint）。统一为同一模板：

```python
msg = f"使用{name}时必须设置{env_var}环境变量。\n当前值看起来像是占位符，请在 .env 文件中填入真实的 API Key"
if hint:
    msg += f"\n{hint}"
```

注意：这会改变 OpenAI 的错误消息格式，需要同步更新对应测试的断言。

### 3.4 保留单例模式

`get_api_config()` / `get_processing_config()` / `reset_all_configs()` 接口不变。原因：
- 项目中大量调用点（services、web_api、gui）
- 全面改为依赖注入改动面过大，超出本次范围
- 现有 reset 函数已足够支持测试

## 风险评估

| 改动 | 风险 | 缓解措施 |
|------|------|---------|
| 删除 character.py | 极低 | 确认无任何引用 |
| 清理 requirements.txt | 极低 | 仅删除重复行 |
| 拆分 utils.py → utils/ 包 | 低 | `__init__.py` re-export 保持所有导入兼容 |
| 拆分 web_api.py → web_api/ 包 | 中 | `__init__.py` re-export；测试中 monkeypatch 路径可能需要更新 |
| 改进 config.py | 中 | `env_field` 行为与原 lambda 完全等价；统一错误消息需更新测试断言 |

## 验收标准

1. `ruff check .` 无错误
2. `black --check .` 通过
3. `mypy .` 无新增错误
4. `pytest tests/ -v` 全部通过
5. 所有现有的 `from utils import xxx` 和 `from web_api import xxx` 导入无需修改
6. `uvicorn web_api:app` 正常启动
