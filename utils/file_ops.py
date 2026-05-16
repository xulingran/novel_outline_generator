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


def _check_bom_for_encoding(enc: str, header: bytes) -> bool:
    """检查文件头是否匹配编码所需的 BOM。不匹配时返回 False。"""
    normalized = enc.lower().replace("_", "-")
    if normalized == "utf-16" and not (
        header.startswith(b"\xff\xfe") or header.startswith(b"\xfe\xff")
    ):
        return False
    if normalized == "utf-16-le" and not header.startswith(b"\xff\xfe"):
        return False
    if normalized == "utf-16-be" and not header.startswith(b"\xfe\xff"):
        return False
    return True


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
        if not _check_bom_for_encoding(enc, file_header):
            logger.debug(f"跳过编码 {enc}: 缺少对应 BOM")
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
        if not _check_bom_for_encoding(enc, raw_sample[:4]):
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
