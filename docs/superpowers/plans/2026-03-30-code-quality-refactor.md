# 代码质量与模块化重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过精准清理和重点重构，提升项目代码可维护性和可测试性。

**Architecture:** 将单体文件 `utils.py`（500行）和 `web_api.py`（821行）拆分为职责聚焦的包，改进 `config.py` 消除重复模式。所有拆分通过 `__init__.py` re-export 保持现有导入兼容。

**Tech Stack:** Python 3.10+, FastAPI, dataclasses, pytest

---

### Task 1: 删除未使用的 `models/character.py`

**Files:**
- Delete: `models/character.py`
- Delete: `tests/test_models_character.py`
- Modify: `models/__init__.py`

- [ ] **Step 1: 确认无引用**

Run: `.venv/bin/python -m ruff check . 2>&1 | head -5` (确认当前无错误)
Run: `grep -r "from models.character import\|from models import.*Character\|from models import.*Relationship" --include="*.py" .`

预期：仅 `models/__init__.py` 和 `tests/test_models_character.py` 有引用。

- [ ] **Step 2: 删除文件**

删除 `models/character.py` 和 `tests/test_models_character.py`。

- [ ] **Step 3: 清理 `models/__init__.py`**

修改 `models/__init__.py`，移除 Character 和 Relationship 的导入和导出：

```python
"""
数据模型模块
定义项目中使用的各种数据结构
"""

from .outline import OutlineData, TextChunk
from .processing_state import ProcessingState, ProgressData

__all__ = [
    "OutlineData",
    "TextChunk",
    "ProcessingState",
    "ProgressData",
]
```

- [ ] **Step 4: 运行测试验证**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: 全部通过（character 相关测试已删除）

- [ ] **Step 5: 提交**

```bash
git add -A models/character.py tests/test_models_character.py models/__init__.py
git commit -m "refactor: 删除未使用的 models/character.py 及其测试"
```

---

### Task 2: 清理 `requirements.txt` 重复条目

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 删除重复段落**

`requirements.txt` 第 30-40 行是第 22-28 行的重复。删除第 30 行（空行）到第 40 行之间的重复内容，保留第 42 行开始的测试依赖和代码质量工具部分。

修改后的完整文件：

```
# 核心依赖
openai>=1.0.0
tiktoken>=0.5.0
httpx>=0.24.0
python-dotenv>=1.0.0

# Gemini API 支持（可选，仅在 API_PROVIDER="gemini" 时需要）
google-generativeai>=0.3.0

# 智谱清言API 支持（可选，仅在 API_PROVIDER="zhipu" 时需要）
zhipuai>=2.0.0

# AiHubMix API 支持（可选，仅在 API_PROVIDER="aihubmix" 时需要）
requests>=2.28.0


# Web API
fastapi>=0.115.0
uvicorn>=0.23.0
python-multipart>=0.0.9

# GUI 框架
customtkinter>=5.2.0
pillow>=10.0.0

# 打包工具
pyinstaller>=6.0.0
pywin32>=305; sys_platform == "win32"


# 测试依赖
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.0.0

# 代码质量工具
black>=23.0.0
mypy>=1.0.0
ruff>=0.1.0
pytest-cov>=4.0.0
types-requests>=2.28.0
```

- [ ] **Step 2: 提交**

```bash
git add requirements.txt
git commit -m "chore: 清理 requirements.txt 重复条目"
```

---

### Task 3: 拆分 `utils.py` 为 `utils/` 包

**Files:**
- Delete: `utils.py`
- Create: `utils/__init__.py`
- Create: `utils/logging_config.py`
- Create: `utils/file_ops.py`
- Create: `utils/text.py`

- [ ] **Step 1: 创建 `utils/logging_config.py`**

从 `utils.py` 第 1-114 行提取日志相关代码：

```python
"""日志配置模块"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_logging_configured = False


def setup_logging(level=None, log_dir="logs", log_backup_days=30):
    """统一配置日志系统，支持按天自动轮转

    Args:
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
        log_dir: 日志目录，默认从环境变量 LOG_DIR 读取
        log_backup_days: 日志保留天数，默认从环境变量 LOG_BACKUP_DAYS 读取
    """
    global _logging_configured
    if _logging_configured:
        return

    # 从环境变量读取配置
    log_dir = os.getenv("LOG_DIR", log_dir)
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
    try:
        log_backup_days = int(os.getenv("LOG_BACKUP_DAYS", str(log_backup_days)))
    except ValueError:
        log_backup_days = 30

    # 确保日志目录存在
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # 如果创建目录失败，回退到项目根目录
        print(f"警告：无法创建日志目录 {log_dir}，将使用项目根目录: {e}")
        log_path = Path.cwd()

    # 日志文件名：当前日志使用基础名称，轮转后自动添加日期后缀
    # 例如：novel_outline.log（当前）-> novel_outline.log.2026-01-16（历史）
    log_filename = "novel_outline.log"
    log_filepath = log_path / log_filename

    # 创建按天轮转的文件处理器
    try:
        file_handler = TimedRotatingFileHandler(
            log_filepath,
            when="midnight",
            interval=1,
            backupCount=log_backup_days,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(level)
    except (OSError, ValueError) as e:
        print(f"警告：无法创建日志文件处理器: {e}")
        file_handler = None

    # 控制台处理器（只显示 INFO 及以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if file_handler:
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _logging_configured = True


def init_logging(level=None, log_dir="logs", log_backup_days=30):
    """显式初始化日志系统（应在应用入口处调用）

    Args:
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
        log_dir: 日志目录，默认从环境变量 LOG_DIR 读取
        log_backup_days: 日志保留天数，默认从环境变量 LOG_BACKUP_DAYS 读取

    Returns:
        logging.Logger: 根日志器
    """
    setup_logging(level, log_dir, log_backup_days)
    return logging.getLogger()
```

- [ ] **Step 2: 创建 `utils/file_ops.py`**

从 `utils.py` 第 114-406 行提取文件操作代码：

```python
"""文件操作工具模块"""

import json
import logging
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, Any, cast

logger = logging.getLogger(__name__)


def _is_plausible_text(content: str) -> bool:
    """判断解码结果是否像正常文本，避免将二进制误判为文本。"""
    if not content:
        return True

    suspicious_chars = 0
    total_chars = len(content)
    allowed_controls = {"\n", "\r", "\t"}

    for char in content:
        if char in allowed_controls:
            continue
        category = unicodedata.category(char)
        if category.startswith("C"):
            suspicious_chars += 1

    return suspicious_chars / total_chars <= 0.1


def _ensure_directory(file_path: Path) -> None:
    """确保父目录存在"""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _create_backup(file_path: Path) -> bool:
    """创建备份文件，成功返回True"""
    if not file_path.exists():
        return False
    backup_path = file_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
    try:
        shutil.copy2(file_path, backup_path)
        logger.debug(f"创建备份文件: {backup_path}")
        return True
    except Exception as e:
        logger.warning(f"创建备份文件失败: {e}")
        return False


def _write_temp_file(
    file_path: Path, write_func: Callable[[IO[str]], None], encoding: str = "utf-8"
) -> None:
    """原子性写入临时文件并替换

    Args:
        file_path: 目标文件路径
        write_func: 写入函数，接收文件对象
        encoding: 文件编码
    """
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=file_path.name + "_", dir=file_path.parent
    )

    try:
        with os.fdopen(temp_fd, "w", encoding=encoding) as f:
            write_func(f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, file_path)
        logger.debug(f"原子性写入成功: {file_path}")

    except Exception as e:
        try:
            os.unlink(temp_path)
        except (OSError, FileNotFoundError):
            pass
        logger.error(f"写入文件失败: {file_path}, 错误: {e}")
        raise


def atomic_write_json(
    file_path: str | Path,
    data: dict[str, Any] | list[Any],
    backup: bool = True,
    indent: int = 2,
) -> None:
    """原子性写入JSON文件

    Args:
        file_path: 目标文件路径
        data: 要写入的数据
        backup: 是否创建备份文件
        indent: JSON缩进

    Raises:
        IOError: 文件操作失败
        json.JSONDecodeError: JSON编码失败
    """
    file_path = Path(file_path)
    _ensure_directory(file_path)
    if backup:
        _create_backup(file_path)

    def write_json(f: IO[str]) -> None:
        json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=True)

    _write_temp_file(file_path, write_json, encoding="utf-8")


def atomic_write_text(
    file_path: str | Path, content: str, backup: bool = True, encoding: str = "utf-8"
) -> None:
    """原子性写入文本文件

    Args:
        file_path: 目标文件路径
        content: 文件内容
        backup: 是否创建备份文件
        encoding: 文件编码
    """
    file_path = Path(file_path)
    _ensure_directory(file_path)
    if backup:
        _create_backup(file_path)

    def write_text(f: IO[str]) -> None:
        f.write(content)

    _write_temp_file(file_path, write_text, encoding=encoding)


def safe_read_json(
    file_path: str | Path,
    default: dict[str, Any] | None = None,
    backup_on_corruption: bool = True,
) -> dict[str, Any]:
    """安全读取JSON文件

    Args:
        file_path: 文件路径
        default: 默认值（如果文件不存在或读取失败）
        backup_on_corruption: 是否在文件损坏时创建备份

    Returns:
        Dict: 解析后的JSON数据
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return default or {}

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            return default or {}

    except json.JSONDecodeError as e:
        logger.error(f"JSON文件损坏: {file_path}, 错误: {e}")

        if backup_on_corruption:
            backup_path = file_path.with_suffix(
                f".corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            try:
                shutil.copy2(file_path, backup_path)
                logger.info(f"损坏的文件已备份: {backup_path}")
            except Exception as backup_error:
                logger.warning(f"备份损坏文件失败: {backup_error}")

        return default or {}

    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        return default or {}


def safe_read_text(
    file_path: str | Path,
    encoding: str = "utf-8",
    fallback_encodings: list[str] | None = None,
) -> tuple[str, str]:
    """安全读取文本文件，支持多种编码

    Args:
        file_path: 文件路径
        encoding: 首选编码
        fallback_encodings: 备选编码列表

    Returns:
        Tuple[str, str]: (文件内容, 实际使用的编码)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    encodings = [encoding]
    if fallback_encodings:
        encodings.extend(fallback_encodings)

    with open(file_path, "rb") as file_obj:
        file_header = file_obj.read(4)

    last_error = None
    for enc in encodings:
        normalized_enc = enc.lower().replace("_", "-")
        if normalized_enc == "utf-16" and not (
            file_header.startswith(b"\xff\xfe") or file_header.startswith(b"\xfe\xff")
        ):
            logger.debug(f"跳过编码 {enc}: 缺少 UTF-16 BOM")
            continue
        if normalized_enc == "utf-16-le" and not file_header.startswith(b"\xff\xfe"):
            logger.debug(f"跳过编码 {enc}: 缺少 UTF-16 LE BOM")
            continue
        if normalized_enc == "utf-16-be" and not file_header.startswith(b"\xfe\xff"):
            logger.debug(f"跳过编码 {enc}: 缺少 UTF-16 BE BOM")
            continue

        try:
            with open(file_path, encoding=enc) as file_obj:
                content = file_obj.read()
            if not _is_plausible_text(content):
                raise UnicodeDecodeError(enc, b"", 0, 1, "解码结果疑似二进制数据")
            logger.debug(f"成功读取文件 {file_path}，使用编码: {enc}")
            return content, enc
        except UnicodeDecodeError as e:
            last_error = e
            logger.debug(f"编码 {enc} 失败: {e}")
            continue
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 编码: {enc}, 错误: {e}")
            raise

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        f"无法使用任何编码读取文件: {', '.join(encodings)}",
    )


def detect_text_encoding(
    file_path: str | Path,
    encodings: list[str],
    sample_size: int = 64 * 1024,
) -> str:
    """探测文本编码，仅读取文件前缀样本。"""
    file_path = Path(file_path)
    with open(file_path, "rb") as file_obj:
        raw_sample = file_obj.read(sample_size)
    if not raw_sample:
        return encodings[0]

    last_error = None
    for enc in encodings:
        normalized_enc = enc.lower().replace("_", "-")
        if normalized_enc == "utf-16" and not (
            raw_sample.startswith(b"\xff\xfe") or raw_sample.startswith(b"\xfe\xff")
        ):
            continue
        if normalized_enc == "utf-16-le" and not raw_sample.startswith(b"\xff\xfe"):
            continue
        if normalized_enc == "utf-16-be" and not raw_sample.startswith(b"\xfe\xff"):
            continue

        try:
            decoded = raw_sample.decode(enc)
            if not _is_plausible_text(decoded):
                raise UnicodeDecodeError(
                    enc, raw_sample, 0, min(len(raw_sample), 1), "解码结果疑似二进制数据"
                )
            return enc
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        f"无法探测文件编码: {', '.join(encodings)}",
    )


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def get_file_info(file_path: str | Path) -> dict[str, Any]:
    """获取文件信息"""
    file_path = Path(file_path)

    if not file_path.exists():
        return {"exists": False}

    stat = file_path.stat()

    return {
        "exists": True,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "extension": file_path.suffix,
        "name": file_path.name,
        "absolute_path": str(file_path.absolute()),
    }
```

- [ ] **Step 3: 创建 `utils/text.py`**

从 `utils.py` 第 426-501 行提取文本工具和 ProgressTracker：

```python
"""文本处理工具模块"""

import logging
from typing import Any


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后的后缀

    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


class ProgressTracker:
    """进度跟踪器（带批量更新功能）"""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.pending_updates: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__ + ".ProgressTracker")

    def add_update(self, update: dict[str, Any]) -> None:
        """添加进度更新（批量保存）"""
        self.pending_updates.append(update)

        if len(self.pending_updates) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """刷新待处理的更新"""
        if not self.pending_updates:
            return

        self.logger.debug(f"批量更新进度: {len(self.pending_updates)} 项")
        self.pending_updates.clear()

    def force_flush(self) -> None:
        """强制刷新（用于程序退出前）"""
        if self.pending_updates:
            self.flush()
```

- [ ] **Step 4: 创建 `utils/__init__.py`**

Re-export 所有公共符号以保持向后兼容：

```python
"""
通用工具模块
包含原子文件操作、JSON处理等实用功能
"""

from utils.file_ops import (
    atomic_write_json,
    atomic_write_text,
    detect_text_encoding,
    format_file_size,
    get_file_info,
    safe_read_json,
    safe_read_text,
)
from utils.logging_config import _logging_configured, init_logging, setup_logging
from utils.text import ProgressTracker, truncate_text

__all__ = [
    # logging_config
    "setup_logging",
    "init_logging",
    "_logging_configured",
    # file_ops
    "atomic_write_json",
    "atomic_write_text",
    "safe_read_json",
    "safe_read_text",
    "detect_text_encoding",
    "format_file_size",
    "get_file_info",
    # text
    "truncate_text",
    "ProgressTracker",
]
```

- [ ] **Step 5: 删除原 `utils.py`**

删除项目根目录的 `utils.py` 文件。

- [ ] **Step 6: 修复测试中 `utils._logging_configured` 引用**

`tests/test_logging_config.py` 中通过 `import utils; utils._logging_configured = False` 访问全局标志。由于 `__init__.py` re-export 了 `_logging_configured`，但赋值操作不会传播到子模块。需要修改测试中的引用：

将 `tests/test_logging_config.py` 中的：
```python
import utils
utils._logging_configured = False
```
改为：
```python
import utils.logging_config as logging_config_mod
logging_config_mod._logging_configured = False
```

同时更新 fixture 中两处 `utils._logging_configured = False` 为 `logging_config_mod._logging_configured = False`。

导入行 `from utils import setup_logging` 保持不变。

- [ ] **Step 7: 运行全量测试**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: 全部通过

- [ ] **Step 8: 运行代码质量检查**

Run: `.venv/bin/python -m ruff check . --fix && .venv/bin/python -m black . && .venv/bin/python -m mypy .`
Expected: 无错误（或仅有 mypy 已知问题）

- [ ] **Step 9: 提交**

```bash
git add utils/ tests/test_logging_config.py
git rm utils.py
git commit -m "refactor: 拆分 utils.py 为 utils/ 包（logging_config/file_ops/text）"
```

---

### Task 4: 拆分 `web_api.py` 为 `web_api/` 包

**Files:**
- Delete: `web_api.py`
- Create: `web_api/__init__.py`
- Create: `web_api/rate_limiter.py`
- Create: `web_api/job_storage.py`
- Create: `web_api/upload_handler.py`
- Create: `web_api/routes.py`
- Modify: `tests/test_web_api.py`

- [ ] **Step 1: 创建 `web_api/rate_limiter.py`**

```python
"""简单内存限流器"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException


class RateLimiter:
    """简单内存限流器，按 IP 在时间窗口内计数。"""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check_rate_limit(self, client_ip: str, max_requests: int, window_seconds: int) -> None:
        now = time.time()
        window_start = now - window_seconds
        bucket = self._requests[client_ip]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= max_requests:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        bucket.append(now)


rate_limiter = RateLimiter()
```

- [ ] **Step 2: 创建 `web_api/job_storage.py`**

从原 `web_api.py` 提取 Job 存储相关代码。包含：回退 JobManager、Job 数据类、job_manager 实例、清理函数、进度更新辅助函数。

```python
"""Job 存储与生命周期管理"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.job_manager import JobManager
else:
    try:
        from services.job_manager import JobManager
    except ModuleNotFoundError:

        class JobManager:
            """内存任务管理器（回退实现）。"""

            def __init__(self, max_jobs: int = 100, max_age_hours: int = 24) -> None:
                self.max_jobs = max_jobs
                self.max_age_hours = max_age_hours
                self.jobs: dict[str, Any] = {}

            def get(self, job_id: str) -> Any | None:
                return self.jobs.get(job_id)

            def set(self, job_id: str, job: Any) -> None:
                self.jobs[job_id] = job

            def values(self):
                return self.jobs.values()

            def cleanup_expired(self, now: float) -> int:
                cutoff_time = now - self.max_age_hours * 3600
                expired_ids = []
                for job_id, job in self.jobs.items():
                    created_at: float
                    if isinstance(job, dict):
                        created_at = job.get("created_at", 0.0)
                    else:
                        created_at = getattr(job, "created_at", 0.0)
                    if created_at < cutoff_time:
                        expired_ids.append(job_id)
                for job_id in expired_ids:
                    del self.jobs[job_id]
                return len(expired_ids)

            def cleanup_excess(self) -> int:
                if len(self.jobs) <= self.max_jobs:
                    return 0

                over_limit = len(self.jobs) - self.max_jobs
                removed = 0

                def _by_age(statuses: tuple[str, ...]) -> list[tuple[str, Any]]:
                    def _get_status(job: Any) -> str:
                        if isinstance(job, dict):
                            return job.get("status", "")
                        return getattr(job, "status", "")

                    def _get_created_at(job: Any) -> float:
                        if isinstance(job, dict):
                            return job.get("created_at", 0.0)
                        return getattr(job, "created_at", 0.0)

                    return sorted(
                        (
                            (job_id, job)
                            for job_id, job in self.jobs.items()
                            if _get_status(job) in statuses
                        ),
                        key=lambda item: _get_created_at(item[1]),
                    )

                for statuses in (("success", "error"), ("pending",), ("running",)):
                    if over_limit <= 0:
                        break
                    for job_id, _ in _by_age(statuses):
                        if over_limit <= 0:
                            break
                        del self.jobs[job_id]
                        over_limit -= 1
                        removed += 1

                return removed


logger = logging.getLogger(__name__)


def format_token_usage_log(token_usage: dict[str, Any], prefix: str = "合并完成，") -> str:
    """格式化 token 使用日志"""
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    return f"{prefix}Token统计: 输入={prompt_tokens:,}, 输出={completion_tokens:,}, 总计={total_tokens:,}"


@dataclass
class Job:
    id: str
    file_path: str = ""
    status: str = "pending"
    message: str = ""
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    log_offset: int = 0
    token_logged: bool = False
    created_at: float = field(default_factory=time.time)

    def log(self, text: str) -> None:
        """Append a log line and keep list size bounded."""
        self.logs.append(text)
        if len(self.logs) > 200:
            overflow = len(self.logs) - 200
            del self.logs[:overflow]
            self.log_offset += overflow


MAX_JOBS = 100
JOB_MAX_AGE_HOURS = 24

job_manager = JobManager(max_jobs=MAX_JOBS, max_age_hours=JOB_MAX_AGE_HOURS)
JOBS: dict[str, Job] = job_manager.jobs
_cleanup_task: asyncio.Task | None = None


def _update_progress_from_info(
    info: dict[str, Any],
    target: "Job | Any",
) -> None:
    """从进度信息更新目标对象（Job 或 QueueTask）"""
    target.progress = info.get("progress", target.progress)

    result_fields = [
        "total_chunks",
        "completed_chunks",
        "failed_chunks",
        "partial_chunks",
        "partial_info",
        "eta_seconds",
        "eta_confidence",
        "eta_method",
        "phase",
        "merge_level",
        "merge_batch_current",
        "merge_batch_total",
        "merge_outlines_count",
    ]

    for field_name in result_fields:
        if info.get(field_name) is not None:
            target.result[field_name] = info[field_name]

    if info.get("last_chunk_id") is not None:
        if info.get("last_error"):
            target.log(f"块 {info['last_chunk_id']} 失败: {info['last_error']}")
        else:
            target.log(f"块 {info['last_chunk_id']} 完成")

    if info.get("token_usage") and not target.token_logged:
        token_usage = info["token_usage"]
        target.result["token_usage"] = token_usage
        target.log(format_token_usage_log(token_usage, "合并完成，"))
        target.token_logged = True


async def _periodic_job_cleanup() -> None:
    """定期清理过期和过多的job任务"""
    while True:
        try:
            await asyncio.sleep(60)
            cleanup_expired_jobs()
            cleanup_excess_jobs()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"定期清理job失败: {e}")


def startup_cleanup_task() -> None:
    """启动后台清理任务"""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_periodic_job_cleanup())


def cleanup_expired_jobs() -> None:
    """清理超过最大存活时间的job"""
    job_manager.max_age_hours = JOB_MAX_AGE_HOURS
    expired_count = job_manager.cleanup_expired(now=time.time())
    if expired_count:
        logger.debug(f"清理了 {expired_count} 个过期job")


def cleanup_excess_jobs() -> None:
    """清理过多的job，防止内存泄漏"""
    job_manager.max_jobs = MAX_JOBS
    job_manager.cleanup_excess()
```

- [ ] **Step 3: 创建 `web_api/upload_handler.py`**

```python
"""上传文件处理与清理"""

import logging
import shutil
from pathlib import Path

from config import load_env_file as load_env_file_from_config

logger = logging.getLogger(__name__)

ENV_PATH = Path(".env")
UPLOAD_DIR = Path("outputs/uploads")
_UPLOAD_ROOT = UPLOAD_DIR.resolve()


def _resolve_upload_path(path: str) -> Path | None:
    """返回在 uploads 目录下的路径，其他情况返回 None。"""
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return None

    if resolved == _UPLOAD_ROOT:
        return None

    try:
        resolved.relative_to(_UPLOAD_ROOT)
        return resolved
    except ValueError:
        return None


def cleanup_uploads(protected_paths: set[Path] | None = None) -> int:
    """删除上传目录中的内容，保留目录本身。"""
    if not UPLOAD_DIR.exists():
        return 0

    keep = {p.resolve() for p in protected_paths} if protected_paths else set()

    cleaned = 0
    for item in UPLOAD_DIR.iterdir():
        item_path = item.resolve()
        if any(item_path == kept_path or item_path in kept_path.parents for kept_path in keep):
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
            cleaned += 1
        except Exception as e:
            logger.warning(f"清理上传文件失败: {e}")
    return cleaned


def load_env_file() -> dict[str, str]:
    """读取 .env 原始配置值（不掩码）。"""
    return load_env_file_from_config(str(ENV_PATH))
```

- [ ] **Step 4: 创建 `web_api/routes.py`**

从原 `web_api.py` 提取路由层。这是最大的文件，包含 CORS 配置、FastAPI app、所有路由端点、任务执行函数。

```python
"""FastAPI 路由定义

启动方式：
  uvicorn web_api:app --reload --port 8000
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_processing_config, init_config
from services.novel_processing_service import NovelProcessingService
from services.task_queue import QueueTask, get_global_queue
from services.token_estimator import estimate_tokens
from utils import init_logging
from web_api.job_storage import (
    Job,
    _update_progress_from_info,
    cleanup_excess_jobs,
    cleanup_expired_jobs,
    format_token_usage_log,
    job_manager,
    startup_cleanup_task,
)
from web_api.rate_limiter import rate_limiter
from web_api.upload_handler import (
    UPLOAD_DIR,
    _UPLOAD_ROOT,
    _resolve_upload_path,
    cleanup_uploads,
    load_env_file,
)

logger = logging.getLogger(__name__)

# 先加载 .env 再读取 CORS 配置
init_config(create_env_if_missing=False)

UPLOAD_FILE_PARAM = File(...)
UPLOAD_FILES_PARAM = File(...)

# 敏感信息关键词（用于掩码处理）
_SENSITIVE_KEYWORDS: set[str] = {"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"}


def _mask_sensitive_value(key: str, value: str) -> str:
    """对敏感值进行掩码处理"""
    key_upper = key.upper()
    if not any(keyword in key_upper for keyword in _SENSITIVE_KEYWORDS):
        return value
    if not value:
        return ""
    if len(value) <= 8:
        return "********"
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _load_cors_origins() -> list[str]:
    """Load CORS origins."""
    is_production = os.getenv("PRODUCTION", "false").lower() == "true"

    if is_production:
        raw = os.getenv("CORS_ORIGINS", "")
        if not raw:
            logger.warning("生产环境未配置CORS_ORIGINS，将拒绝所有跨域请求")
            return []
    else:
        raw = os.getenv(
            "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,null"
        )

    origins: list[str] = []
    for origin in raw.split(","):
        origin = origin.strip()
        if not origin:
            continue
        if origin == "file://":
            origin = "null"
        if is_production and origin in ("null", "*"):
            logger.warning(f"生产环境忽略不安全的CORS来源: {origin}")
            continue
        origins.append(origin)

    return list(dict.fromkeys(origins))


CORS_ORIGINS = _load_cors_origins()


class ProcessRequest(BaseModel):
    file_path: str
    resume: bool = True


class MultipleFilesRequest(BaseModel):
    """批量文件请求"""

    file_paths: list[str]


_cleanup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_logging()
    init_config(create_env_if_missing=False)
    startup_cleanup_task()

    queue = get_global_queue()
    queue.set_callback(run_queue_task)

    yield
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    from services.llm_service import OpenAIService

    await OpenAIService.close_http_clients()


app = FastAPI(title="Novel Outline API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/env")
def get_env() -> dict[str, Any]:
    data = load_env_file()
    masked = {k: _mask_sensitive_value(k, v) for k, v in data.items()}
    return {"env": masked}


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = UPLOAD_FILE_PARAM):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 10, 60)
    if file.content_type not in ("text/plain", "text/markdown", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="仅支持文本文件")

    processing_config = get_processing_config()

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in processing_config.allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"仅支持 {', '.join(processing_config.allowed_extensions)} 文件"
        )

    from validators import sanitize_filename

    safe_filename = sanitize_filename(file.filename)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / safe_filename

    try:
        dest_resolved = dest.resolve()
        upload_root_resolved = _UPLOAD_ROOT.resolve()
        dest_resolved.relative_to(upload_root_resolved)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"无效的文件路径: {safe_filename}") from e

    content = await file.read()
    max_size_bytes = processing_config.max_upload_file_size_mb * 1024 * 1024
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=400, detail=f"文件过大，限制{processing_config.max_upload_file_size_mb}MB"
        )
    dest.write_bytes(content)
    return {"file_path": str(dest)}


@app.post("/upload-multiple")
async def upload_multiple_files(request: Request, files: list[UploadFile] = UPLOAD_FILES_PARAM):
    """批量上传多个文件"""
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 10, 60)

    processing_config = get_processing_config()

    from validators import sanitize_filename

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_files = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in processing_config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"仅支持 {', '.join(processing_config.allowed_extensions)} 文件: {file.filename}",
            )

        safe_filename = sanitize_filename(file.filename)
        dest = UPLOAD_DIR / safe_filename
        content = await file.read()
        max_size_bytes = processing_config.max_upload_file_size_mb * 1024 * 1024
        if len(content) > max_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大，限制{processing_config.max_upload_file_size_mb}MB: {file.filename}",
            )

        dest.write_bytes(content)
        uploaded_files.append(str(dest))

    return {"file_paths": uploaded_files}


async def _run_job(job: Job, req: ProcessRequest):
    job.status = "running"
    job.file_path = req.file_path
    job.progress = 0.0
    job.result = {}
    job.log(f"开始处理文件: {req.file_path}")

    def handle_progress(info: dict[str, Any]) -> None:
        _update_progress_from_info(info, job)
        if info.get("merge_batch_current") is not None:
            job.result["merge_batch_current"] = info["merge_batch_current"]
        if info.get("merge_batch_total") is not None:
            job.result["merge_batch_total"] = info["merge_batch_total"]
        if info.get("merge_outlines_count") is not None:
            job.result["merge_outlines_count"] = info["merge_outlines_count"]
        if info.get("last_chunk_id") is not None:
            if info.get("last_error"):
                job.log(f"块 {info['last_chunk_id']} 失败: {info['last_error']}")
            else:
                job.log(f"块 {info['last_chunk_id']} 完成")
        if info.get("token_usage") and not job.token_logged:
            token_usage = info["token_usage"]
            job.result["token_usage"] = token_usage
            job.log(format_token_usage_log(token_usage, "合并完成，"))
            job.token_logged = True

    try:
        service = NovelProcessingService(progress_callback=handle_progress)
        result = await service.process_novel(req.file_path, resume=req.resume)
        job.result.update(result)
        job.progress = 1.0
        job.status = "success"

        if "token_usage" in result and not job.token_logged:
            token_usage = result["token_usage"]
            job.log(format_token_usage_log(token_usage))
            job.token_logged = True

        job.log("处理完成")
        try:
            current_upload = _resolve_upload_path(req.file_path)
            if current_upload:
                active_uploads: set[Path] = set()
                for other_job in job_manager.values():
                    if other_job.id == job.id:
                        continue
                    if other_job.status not in {"pending", "running"}:
                        continue
                    if not other_job.file_path:
                        continue
                    upload_path = _resolve_upload_path(other_job.file_path)
                    if upload_path:
                        active_uploads.add(upload_path)

                cleaned = cleanup_uploads(protected_paths=active_uploads)
                if cleaned:
                    job.log(f"已清理上传文件 {cleaned} 个")
        except Exception as cleanup_err:
            job.log(f"清理上传文件失败: {cleanup_err}")

    except Exception as e:
        logger.exception("Job %s failed with error: %s", job.id, e)
        job.status = "error"
        job.message = str(e)
        job.log(f"错误: {e}")


async def run_queue_task(task: QueueTask) -> None:
    """运行队列任务（由 TaskQueue 调用）"""
    task.log(f"开始处理文件: {task.file_path}")

    def handle_progress(info: dict[str, Any]) -> None:
        _update_progress_from_info(info, task)

    try:
        service = NovelProcessingService(
            progress_callback=handle_progress, cancel_event=task.cancel_event
        )
        if task.should_force_complete:
            service.force_complete = True
            logger.info(f"任务 {task.id} 启用强制完成模式")

        result = await service.process_novel(task.file_path, resume=True)
        task.result.update(result)
        task.progress = 1.0
        task.status = "success"

        if "token_usage" in result and not task.token_logged:
            token_usage = result["token_usage"]
            task.log(format_token_usage_log(token_usage))
            task.token_logged = True

        task.log("处理完成")

    except asyncio.CancelledError:
        if task.should_force_complete and len(task.result.get("outlines", [])) > 0:
            logger.info(f"任务 {task.id} 强制完成模式：继续合并已有结果")
            task.status = "success"
            task.message = "强制完成（部分结果已合并）"
            task.log("强制完成：将合并已有部分结果")
        else:
            task.status = "cancelled"
            task.message = "任务被取消"
            task.log("任务被取消")
    except Exception as e:
        logger.exception("Task %s failed with error: %s", task.id, e)
        task.status = "error"
        task.message = str(e)
        task.log(f"错误: {e}")


@app.post("/process")
async def start_process(request: Request, req: ProcessRequest):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 5, 60)
    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    cleanup_excess_jobs()

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, file_path=req.file_path)
    job_manager.set(job_id, job)

    async def _run_job_wrapper(job: Job, req: ProcessRequest) -> None:
        try:
            await _run_job(job, req)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Job %s 启动失败: %s", job.id, e)
            job.status = "error"
            job.message = f"启动失败: {str(e)}"
            job.log(f"启动失败: {e}")

    asyncio.create_task(_run_job_wrapper(job, req))
    return {"job_id": job_id}


@app.get("/estimate")
def estimate(file_path: str):
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return estimate_tokens(file_path)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {
        "id": job.id,
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "result": job.result,
        "logs": job.logs,
        "log_offset": job.log_offset,
    }


@app.get("/queue/list")
async def list_queue():
    queue = get_global_queue()
    tasks = await queue.list_tasks()
    return {"tasks": tasks}


@app.post("/queue/add")
async def add_to_queue(request: Request, req: ProcessRequest):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 5, 60)

    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    queue = get_global_queue()
    task_id = await queue.add_task(req.file_path)
    return {"task_id": task_id, "message": "任务已添加到队列"}


@app.post("/queue/add-multiple")
async def add_multiple_to_queue(request: Request, req: MultipleFilesRequest):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 5, 60)

    if not req.file_paths:
        raise HTTPException(status_code=400, detail="file_paths 不能为空")

    queue = get_global_queue()
    task_ids = []

    for file_path in req.file_paths:
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path 不能为空")
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
        task_id = await queue.add_task(file_path)
        task_ids.append(task_id)

    return {
        "task_ids": task_ids,
        "count": len(task_ids),
        "message": f"已将 {len(task_ids)} 个文件添加到队列",
    }


@app.post("/queue/cancel")
async def cancel_queue_task(request: Request, task_id: str):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 10, 60)

    queue = get_global_queue()
    success = await queue.cancel_task(task_id)

    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或无法取消")

    return {"success": True, "message": "任务已取消"}


@app.post("/queue/clear")
async def clear_queue():
    queue = get_global_queue()
    count = await queue.clear_queue()
    return {"success": True, "cancelled_count": count}


@app.post("/queue/force-complete/{task_id}")
async def force_complete_queue_task(request: Request, task_id: str):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 10, 60)

    queue = get_global_queue()
    success = await queue.force_complete_task(task_id)

    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或无法强制完成")

    return {"success": True, "message": "已强制完成，将合并已有结果"}


@app.get("/queue/stats")
async def get_queue_stats():
    queue = get_global_queue()
    stats = await queue.get_stats()
    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 5: 创建 `web_api/__init__.py`**

```python
"""FastAPI 后端接口包

向后兼容导出：确保 `import web_api` 和 `uvicorn web_api:app` 继续工作。
"""

import time  # noqa: F401 — tests monkeypatch web_api.time
import asyncio  # noqa: F401 — tests monkeypatch web_api.asyncio

from web_api.job_storage import (
    JOBS,
    JOB_MAX_AGE_HOURS,
    MAX_JOBS,
    Job,
    _update_progress_from_info,
    cleanup_excess_jobs,
    cleanup_expired_jobs,
    format_token_usage_log,
    job_manager,
)
from web_api.rate_limiter import RateLimiter, rate_limiter
from web_api.routes import (
    CORS_ORIGINS,
    NovelProcessingService,
    _load_cors_origins,
    _mask_sensitive_value,
    _run_job,
    app,
    get_global_queue,
    run_queue_task,
)
from web_api.upload_handler import (
    ENV_PATH,
    UPLOAD_DIR,
    _UPLOAD_ROOT,
    _resolve_upload_path,
    cleanup_uploads,
    load_env_file,
)

__all__ = [
    "app",
    "Job",
    "job_manager",
    "JOBS",
    "UPLOAD_DIR",
    "_UPLOAD_ROOT",
    "rate_limiter",
    "RateLimiter",
    "_run_job",
    "run_queue_task",
    "_update_progress_from_info",
    "format_token_usage_log",
    "cleanup_expired_jobs",
    "cleanup_excess_jobs",
    "cleanup_uploads",
    "_resolve_upload_path",
    "_mask_sensitive_value",
    "_load_cors_origins",
    "CORS_ORIGINS",
    "load_env_file",
    "ENV_PATH",
    "MAX_JOBS",
    "JOB_MAX_AGE_HOURS",
    "NovelProcessingService",
    "get_global_queue",
]
```

- [ ] **Step 6: 删除原 `web_api.py`**

删除项目根目录的 `web_api.py` 文件。

- [ ] **Step 7: 运行全量测试**

Run: `.venv/bin/python -m pytest tests/test_web_api.py -v --tb=short 2>&1 | tail -30`
Expected: 全部通过（`import web_api` 和所有 `monkeypatch.setattr(web_api, ...)` 通过 `__init__.py` re-export 继续工作）

如果有 monkeypatch 失败（因为 setattr 在 `__init__` 模块上设置属性不会传播到子模块内部引用），需要在 `routes.py` 中将直接引用改为通过 import 读取。具体修复方法：在 `routes.py` 中使用 `from web_api import upload_handler` 然后用 `upload_handler.UPLOAD_DIR` 而非局部变量。

- [ ] **Step 8: 运行代码质量检查**

Run: `.venv/bin/python -m ruff check . --fix && .venv/bin/python -m black . && .venv/bin/python -m mypy .`
Expected: 无新增错误

- [ ] **Step 9: 提交**

```bash
git add web_api/ tests/test_web_api.py
git rm web_api.py
git commit -m "refactor: 拆分 web_api.py 为 web_api/ 包（rate_limiter/job_storage/upload_handler/routes）"
```

---

### Task 5: 改进 `config.py` — `env_field` 辅助函数

**Files:**
- Modify: `config.py`

- [ ] **Step 1: 添加 `env_field` 辅助函数**

在 `config.py` 的 `load_env_file` 函数之后、`APIConfig` 之前，添加：

```python
def env_field(
    env_var: str,
    default: str | None = None,
    *,
    cast: type = str,
) -> Any:
    """从环境变量创建 dataclass field。

    Args:
        env_var: 环境变量名
        default: 默认值（字符串形式）
        cast: 类型转换函数（int、bool 等）
    """
    def factory():
        raw = os.getenv(env_var, default)
        if raw is None:
            return None
        if cast is bool:
            return raw.lower() == "true"
        return cast(raw) if cast is not str else raw
    return field(default_factory=factory)
```

- [ ] **Step 2: 用 `env_field` 重写 `APIConfig` 字段**

替换所有 lambda 默认值：

```python
@dataclass
class APIConfig:
    """API配置类"""

    provider: str = env_field("API_PROVIDER", "openai")
    openai_key: str | None = env_field("OPENAI_API_KEY")
    openai_base: str | None = env_field("OPENAI_API_BASE")
    openai_model: str = env_field("OPENAI_MODEL", "gpt-4o-mini")
    gemini_key: str | None = env_field("GEMINI_API_KEY")
    gemini_model: str = env_field("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_safety: str = env_field("GEMINI_SAFETY_SETTINGS", "BLOCK_NONE")
    zhipu_key: str | None = env_field("ZHIPU_API_KEY")
    zhipu_base: str | None = env_field("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    zhipu_model: str = env_field("ZHIPU_MODEL", "glm-4-flash")
    aihubmix_api_key: str | None = env_field("AIHUBMIX_API_KEY")
    aihubmix_model: str = env_field("AIHUBMIX_MODEL", "gpt-3.5-turbo")
    aihubmix_api_base: str | None = env_field("AIHUBMIX_API_BASE", "https://aihubmix.com/v1")
    _validated: bool = field(default=False, init=False)
```

- [ ] **Step 3: 用 `env_field` 重写 `ProcessingConfig` 字段**

```python
@dataclass
class ProcessingConfig:
    """处理配置类"""

    default_txt_file: str = field(default="novel.txt")
    output_dir: str = field(default="outputs")
    progress_file: str = field(init=False)
    allowed_extensions: list[str] = field(default_factory=lambda: [".txt", ".md", ".text"])
    max_upload_file_size_mb: int = env_field("MAX_UPLOAD_FILE_SIZE_MB", "100", cast=int)

    encodings: list[str] = field(
        default_factory=lambda: [
            "utf-8", "gbk", "gb2312", "gb18030", "big5",
            "utf-16", "utf-16-le", "utf-16-be", "latin1", "cp1252",
        ]
    )

    model_max_tokens: int = env_field("MODEL_MAX_TOKENS", "200000", cast=int)
    target_tokens_per_chunk: int = env_field("TARGET_TOKENS_PER_CHUNK", "64000", cast=int)
    parallel_limit: int = env_field("PARALLEL_LIMIT", "5", cast=int)
    max_retry: int = env_field("MAX_RETRY", "5", cast=int)
    log_every: int = env_field("LOG_EVERY", "1", cast=int)
    sub_chunk_count: int = env_field("SUB_CHUNK_COUNT", "5", cast=int)
    retry_backoff_base: int = env_field("RETRY_BACKOFF_BASE", "1", cast=int)
    stream_split_threshold_mb: int = env_field("STREAM_SPLIT_THRESHOLD_MB", "20", cast=int)

    use_proxy: bool = env_field("USE_PROXY", "false", cast=bool)
    proxy_url: str = env_field("PROXY_URL", "http://127.0.0.1:7897")
```

注意：`allowed_extensions` 和 `encodings` 是硬编码列表，不从环境变量读取，保留原有 `field(default_factory=lambda: [...])` 形式。`default_txt_file`、`output_dir` 也是固定值，保留 `field(default=...)` 形式。

- [ ] **Step 4: 运行测试验证行为等价**

Run: `.venv/bin/python -m pytest tests/test_config.py -v --tb=short`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add config.py
git commit -m "refactor: 引入 env_field 辅助函数，消除 config.py 中的 lambda 滥用"
```

---

### Task 6: 统一 provider 配置映射

**Files:**
- Modify: `config.py`
- Modify: `tests/test_config.py`（如需更新断言）

- [ ] **Step 1: 合并 `_PROVIDER_KEY_CONFIG` 为 `_PROVIDER_REGISTRY`**

在 `APIConfig` 类中，替换 `_PROVIDER_KEY_CONFIG` 字段为类变量 `_PROVIDER_REGISTRY`：

```python
    _PROVIDER_REGISTRY: dict[str, dict[str, str | None]] = {
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

注意：这是一个**类变量**（不使用 `field()`），放在 `_validated` 之后，不参与 dataclass 的 `__init__`、`__repr__` 等。

- [ ] **Step 2: 重写 `_validate_api_key` 和 `validate` 方法**

统一错误消息格式，删除 OpenAI 特殊分支：

```python
    def _validate_api_key(self, key_value: str | None, config: dict[str, str | None]) -> None:
        """验证单个 API 密钥"""
        if not key_value or "your_" in key_value.lower() or "here" in key_value.lower():
            name = config["name"]
            env_var = config["env_var"]
            hint = config.get("hint", "")
            msg = (
                f"使用{name}时必须设置{env_var}环境变量。\n"
                "当前值看起来像是占位符，请在 .env 文件中填入真实的 API Key"
            )
            if hint:
                msg += f"\n{hint}"
            raise ConfigurationError(msg)

    def validate(self) -> None:
        """验证配置（延迟到实际使用时）"""
        if self._validated:
            return

        if self.provider not in SUPPORTED_API_PROVIDERS:
            raise ConfigurationError(
                f"不支持的API提供商: {self.provider}. "
                f"支持的提供商: {', '.join(SUPPORTED_API_PROVIDERS)}"
            )

        provider_config = self._PROVIDER_REGISTRY.get(self.provider)
        if provider_config:
            key_value = getattr(self, provider_config["key_field"])
            self._validate_api_key(key_value, provider_config)

        self._validated = True
```

- [ ] **Step 3: 重写 `api_key`、`base_url`、`model_name` 属性**

用查表替代 if-elif 链：

```python
    @property
    def api_key(self) -> str:
        """获取当前API密钥"""
        self.validate()
        config = self._PROVIDER_REGISTRY[self.provider]
        value = getattr(self, config["key_field"])
        if not value:
            raise APIKeyError(f"{config['name']}密钥未配置")
        return value

    @property
    def base_url(self) -> str | None:
        """获取API基础URL"""
        config = self._PROVIDER_REGISTRY.get(self.provider)
        if not config or not config.get("base_field"):
            return None
        return getattr(self, config["base_field"])

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        config = self._PROVIDER_REGISTRY[self.provider]
        return getattr(self, config["model_field"])
```

- [ ] **Step 4: 删除旧的 `_PROVIDER_KEY_CONFIG` 字段**

删除 `_PROVIDER_KEY_CONFIG` 的 `field(default_factory=...)` 定义（原 config.py 第 67-96 行）。

- [ ] **Step 5: 更新测试断言**

`tests/test_config.py` 中对 API 密钥错误消息的断言使用 `match="使用.*时必须设置"` 模式，这些不需要修改（正则匹配兼容）。

但 `APIKeyError` 的消息格式变了。旧的消息如 `"OpenAI API密钥未配置"` 变为 `"OpenAI API密钥未配置"`（来自 `config['name'] + '密钥未配置'`）。检查 `tests/test_config.py` 中是否有精确匹配 APIKeyError 消息的断言，如有则更新。

Run: `grep -n "APIKeyError\|密钥未配置" tests/test_config.py` 检查。

- [ ] **Step 6: 运行测试验证**

Run: `.venv/bin/python -m pytest tests/test_config.py -v --tb=short`
Expected: 全部通过

- [ ] **Step 7: 运行全量测试**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: 全部通过

- [ ] **Step 8: 运行代码质量检查**

Run: `.venv/bin/python -m ruff check . --fix && .venv/bin/python -m black . && .venv/bin/python -m mypy .`
Expected: 无新增错误

- [ ] **Step 9: 提交**

```bash
git add config.py tests/test_config.py
git commit -m "refactor: 统一 provider 配置映射，消除 if-elif 链"
```

---

### Task 7: 最终验证

**Files:** 无新文件修改

- [ ] **Step 1: 运行完整代码质量检查**

```bash
.venv/bin/python -m ruff check . --fix
.venv/bin/python -m black .
.venv/bin/python -m mypy .
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: 全部通过

- [ ] **Step 2: 验证导入兼容性**

```bash
.venv/bin/python -c "from utils import init_logging, setup_logging, atomic_write_json, ProgressTracker; print('utils OK')"
.venv/bin/python -c "from web_api import app, Job, job_manager, JOBS; print('web_api OK')"
.venv/bin/python -c "from config import APIConfig, ProcessingConfig, get_api_config; print('config OK')"
```

Expected: 三行均输出 OK

- [ ] **Step 3: 验证 uvicorn 启动**

```bash
timeout 5 .venv/bin/python -m uvicorn web_api:app --port 18999 2>&1 || true
```

Expected: 输出包含 "Uvicorn running on"（5 秒后超时退出正常）

- [ ] **Step 4: 提交代码质量修复（如有）**

如果 Step 1 中 ruff/black 修复了任何文件：

```bash
git add -A
git commit -m "style: 代码格式修复"
```
