"""
通用工具模块
包含原子文件操作、JSON处理等实用功能
"""
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import logging
from datetime import datetime

# 日志配置函数
_logging_configured = False

def setup_logging(level=None, log_file='novel_outline.log'):
    """统一配置日志系统，避免重复配置
    
    Args:
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取，若未设置则使用 INFO
        log_file: 日志文件路径
    """
    global _logging_configured
    if _logging_configured:
        return
    
    # 支持通过环境变量控制日志级别
    if level is None:
        level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置，即使已经配置过
    )
    _logging_configured = True

# 自动配置日志（首次导入时）
setup_logging()

logger = logging.getLogger(__name__)


def atomic_write_json(file_path: Union[str, Path],
                     data: Dict[str, Any],
                     backup: bool = True,
                     indent: int = 2) -> None:
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
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建备份
    if backup and file_path.exists():
        backup_path = file_path.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak')
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"创建备份文件: {backup_path}")
        except Exception as e:
            logger.warning(f"创建备份文件失败: {e}")

    # 写入临时文件
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=file_path.name + '_',
        dir=file_path.parent
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # 强制写入磁盘

        # 原子性重命名
        os.replace(temp_path, file_path)
        logger.debug(f"原子性写入成功: {file_path}")

    except Exception as e:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except (OSError, FileNotFoundError):
            # 忽略清理临时文件时的错误
            pass
        logger.error(f"写入文件失败: {file_path}, 错误: {e}")
        raise


def atomic_write_text(file_path: Union[str, Path],
                     content: str,
                     backup: bool = True,
                     encoding: str = 'utf-8') -> None:
    """原子性写入文本文件

    Args:
        file_path: 目标文件路径
        content: 文件内容
        backup: 是否创建备份文件
        encoding: 文件编码
    """
    file_path = Path(file_path)

    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建备份
    if backup and file_path.exists():
        backup_path = file_path.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak')
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"创建备份文件: {backup_path}")
        except Exception as e:
            logger.warning(f"创建备份文件失败: {e}")

    # 写入临时文件
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=file_path.name + '_',
        dir=file_path.parent
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # 强制写入磁盘

        # 原子性重命名
        os.replace(temp_path, file_path)
        logger.debug(f"原子性写入成功: {file_path}")

    except Exception as e:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except (OSError, FileNotFoundError):
            # 忽略清理临时文件时的错误
            pass
        logger.error(f"写入文件失败: {file_path}, 错误: {e}")
        raise


def safe_read_json(file_path: Union[str, Path],
                  default: Optional[Dict[str, Any]] = None,
                  backup_on_corruption: bool = True) -> Dict[str, Any]:
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
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        logger.error(f"JSON文件损坏: {file_path}, 错误: {e}")

        if backup_on_corruption:
            backup_path = file_path.with_suffix(f'.corrupt_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            try:
                shutil.copy2(file_path, backup_path)
                logger.info(f"损坏的文件已备份: {backup_path}")
            except Exception as backup_error:
                logger.warning(f"备份损坏文件失败: {backup_error}")

        return default or {}

    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        return default or {}


def safe_read_text(file_path: Union[str, Path],
                  encoding: str = 'utf-8',
                  fallback_encodings: Optional[List[str]] = None) -> str:
    """安全读取文本文件，支持多种编码

    Args:
        file_path: 文件路径
        encoding: 首选编码
        fallback_encodings: 备选编码列表

    Returns:
        str: 文件内容
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
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            logger.debug(f"成功读取文件 {file_path}，使用编码: {enc}")
            return content
        except UnicodeDecodeError as e:
            last_error = e
            logger.debug(f"编码 {enc} 失败: {e}")
            continue
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 编码: {enc}, 错误: {e}")
            raise

    # 所有编码都失败
    raise UnicodeDecodeError(
        last_error.encoding if last_error else 'unknown',
        last_error.object if last_error else b'',
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        f"无法使用任何编码读取文件: {', '.join(encodings)}"
    )


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        str: 格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


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
    return text[:max_length-len(suffix)] + suffix


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
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
        "absolute_path": str(file_path.absolute())
    }


def cleanup_old_backups(directory: Union[str, Path],
                       pattern: str = "*.bak",
                       max_files: int = 10) -> None:
    """清理旧的备份文件

    Args:
        directory: 目录路径
        pattern: 文件模式
        max_files: 保留的最大文件数
    """
    directory = Path(directory)

    if not directory.exists():
        return

    # 查找备份文件
    backup_files = list(directory.glob(pattern))

    # 按修改时间排序（最新的在前）
    backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # 删除多余的备份
    for old_backup in backup_files[max_files:]:
        try:
            old_backup.unlink()
            logger.debug(f"删除旧备份: {old_backup}")
        except Exception as e:
            logger.warning(f"删除旧备份失败: {old_backup}, 错误: {e}")


class ProgressTracker:
    """进度跟踪器（带批量更新功能）"""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.pending_updates = []
        self.logger = logging.getLogger(__name__ + '.ProgressTracker')

    def add_update(self, update: Dict[str, Any]) -> None:
        """添加进度更新（批量保存）"""
        self.pending_updates.append(update)

        if len(self.pending_updates) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """刷新待处理的更新"""
        if not self.pending_updates:
            return

        # 这里可以添加实际的保存逻辑
        # 例如：将更新合并并写入进度文件
        self.logger.debug(f"批量更新进度: {len(self.pending_updates)} 项")
        self.pending_updates.clear()

    def force_flush(self) -> None:
        """强制刷新（用于程序退出前）"""
        if self.pending_updates:
            self.flush()