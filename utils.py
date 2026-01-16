"""
通用工具模块
包含原子文件操作、JSON处理等实用功能
"""

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import IO, Any, cast

# 日志配置函数
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
            when="midnight",  # 每天午夜轮转
            interval=1,  # 间隔1天
            backupCount=log_backup_days,  # 保留天数
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"  # 轮转文件名后缀（例如 .2026-01-16）
        file_handler.setLevel(level)
    except (OSError, ValueError) as e:
        # 如果无法创建文件处理器，至少保留控制台输出
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
    root_logger.handlers.clear()  # 清除现有处理器

    if file_handler:
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _logging_configured = True


# 自动配置日志（首次导入时）
setup_logging()

logger = logging.getLogger(__name__)


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

        # 原子性重命名
        os.replace(temp_path, file_path)
        logger.debug(f"原子性写入成功: {file_path}")

    except Exception as e:
        # 清理临时文件
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

    # 确保目录存在
    _ensure_directory(file_path)

    # 创建备份
    if backup:
        _create_backup(file_path)

    # 写入临时文件
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

    # 确保目录存在
    _ensure_directory(file_path)

    # 创建备份
    if backup:
        _create_backup(file_path)

    # 写入临时文件
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

    last_error = None
    for enc in encodings:
        try:
            with open(file_path, encoding=enc) as f:
                content = f.read()
            logger.debug(f"成功读取文件 {file_path}，使用编码: {enc}")
            return content, enc
        except UnicodeDecodeError as e:
            last_error = e
            logger.debug(f"编码 {enc} 失败: {e}")
            continue
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 编码: {enc}, 错误: {e}")
            raise

    # 所有编码都失败
    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        f"无法使用任何编码读取文件: {', '.join(encodings)}",
    )


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        str: 格式化的大小字符串
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


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


def get_file_info(file_path: str | Path) -> dict[str, Any]:
    """获取文件信息

    Args:
        file_path: 文件路径

    Returns:
        Dict: 文件信息
    """
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

        # 记录批量更新并清空队列
        # 如需持久化保存，可在此处扩展文件写入逻辑
        self.logger.debug(f"批量更新进度: {len(self.pending_updates)} 项")
        self.pending_updates.clear()

    def force_flush(self) -> None:
        """强制刷新（用于程序退出前）"""
        if self.pending_updates:
            self.flush()
