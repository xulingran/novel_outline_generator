"""
输入验证模块
提供各种输入验证功能，确保安全性
"""

import logging
import os
import re
from pathlib import Path

from config import MAX_INPUT_TOKEN_RATIO, RECOMMENDED_MAX_PARALLEL_LIMIT, SUPPORTED_API_PROVIDERS
from exceptions import FileValidationError

logger = logging.getLogger(__name__)


def _validate_path_safety(file_path: str | Path) -> Path:
    """规范化路径后检查 `..` 分段，返回解析后的绝对路径。

    注意：此函数仅拦截规范化后仍包含 `..` 分段的路径（如相对路径中的 `../../etc`）。
    若调用方需要将路径限制在特定目录内，应在此函数之上额外做基础目录锚定检查。

    Args:
        file_path: 文件路径

    Returns:
        Path: 解析后的绝对路径

    Raises:
        FileValidationError: 路径规范化后仍含有 `..` 分段
    """
    # 在解析前检查原始路径，防止路径遍历
    path_str = str(file_path)
    # 规范化路径分隔符后检查 ".." 模式
    normalized_input = os.path.normpath(path_str)
    if ".." in normalized_input.split(os.sep):
        raise FileValidationError("检测到不安全的路径遍历")

    # 解析为绝对路径
    path_obj = Path(file_path).resolve()

    return path_obj


def validate_file_path(
    file_path: str | Path,
    allowed_extensions: list[str] | None = None,
    max_size_mb: int | None = None,
    allowed_dir: str | Path | None = None,
) -> Path:
    """验证文件路径的安全性

    Args:
        file_path: 文件路径
        allowed_extensions: 允许的文件扩展名列表，如['.txt', '.md']
        max_size_mb: 最大文件大小（MB）
        allowed_dir: 允许的基础目录，如果提供，文件路径必须位于此目录内

    Returns:
        Path: 验证后的Path对象

    Raises:
        FileValidationError: 文件验证失败
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    if not file_path:
        raise FileValidationError("文件路径不能为空")

    # 使用安全的路径验证（绝对路径比较）
    path_obj = _validate_path_safety(file_path)

    # 检查路径遍历攻击
    if ".." in path_obj.parts:
        raise FileValidationError("检测到不安全的路径遍历")

    if allowed_dir:
        try:
            resolved_path = path_obj.resolve()
            resolved_allowed = Path(allowed_dir).resolve()
            if not resolved_path.is_relative_to(resolved_allowed):
                raise FileValidationError("检测到不安全的路径遍历：超出允许目录")
        except Exception as e:
            if not isinstance(e, FileValidationError):
                raise FileValidationError(f"路径解析失败: {e}") from e
            raise

    # 检查文件是否存在
    if not path_obj.exists():
        raise FileValidationError(f"文件不存在: {file_path}")

    # 检查是否为文件
    if not path_obj.is_file():
        raise FileValidationError(f"路径不是文件: {file_path}")

    # 检查扩展名
    if allowed_extensions:
        ext = path_obj.suffix.lower()
        if ext not in allowed_extensions:
            raise FileValidationError(
                f"不支持的文件扩展名: {ext}. 支持的扩展名: {', '.join(allowed_extensions)}"
            )

    # 检查文件大小
    if max_size_mb is not None:
        size_mb = path_obj.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise FileValidationError(f"文件过大: {size_mb:.2f}MB. 最大允许: {max_size_mb}MB")

    return path_obj


def validate_output_dir(output_dir: str | Path, allowed_dir: str | Path | None = None) -> Path:
    """验证并创建输出目录

    Args:
        output_dir: 输出目录路径
        allowed_dir: 允许的基础目录，如果提供，输出目录必须位于此目录内

    Returns:
        Path: 验证后的Path对象
    """
    if not output_dir or not isinstance(output_dir, (str, Path)):
        raise FileValidationError("输出目录路径不能为空")

    # 使用安全的路径验证（绝对路径比较）
    path_obj = _validate_path_safety(output_dir)

    # 检查路径遍历攻击
    if ".." in path_obj.parts:
        raise FileValidationError("检测到不安全的路径遍历")

    if allowed_dir:
        try:
            resolved_path = path_obj.resolve()
            resolved_allowed = Path(allowed_dir).resolve()
            if not resolved_path.is_relative_to(resolved_allowed):
                raise FileValidationError("检测到不安全的路径遍历：超出允许目录")
        except Exception as e:
            if not isinstance(e, FileValidationError):
                raise FileValidationError(f"路径解析失败: {e}") from e
            raise

    # 创建目录（如果不存在）
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise FileValidationError(f"没有权限创建目录: {output_dir}") from e
    except Exception as e:
        raise FileValidationError(f"创建目录失败: {output_dir}, 错误: {str(e)}") from e

    return path_obj


def validate_encoding_list(encodings: list[str]) -> list[str]:
    """验证编码列表

    Args:
        encodings: 编码列表

    Returns:
        list: 验证后的编码列表

    Raises:
        FileValidationError: 编码列表无效
    """
    if not encodings or not isinstance(encodings, list):
        raise FileValidationError("编码列表不能为空")

    # 常见的有效编码
    valid_encodings = {
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "gbk",
        "gb2312",
        "gb18030",
        "big5",
        "latin1",
        "cp1252",
        "ascii",
    }

    validated_encodings: list[str] = []
    for encoding in encodings:
        if not isinstance(encoding, str):
            logger.warning(f"跳过无效的编码类型: {encoding!r} (类型: {type(encoding).__name__})")
            continue

        encoding_lower = encoding.lower().replace("_", "-")
        if encoding_lower in valid_encodings or encoding.lower() in valid_encodings:
            validated_encodings.append(encoding)
        else:
            logger.warning(f"跳过不支持的编码: {encoding}")

    if not validated_encodings:
        raise FileValidationError("没有找到有效的编码")

    # 如果有编码被跳过，记录信息
    if len(validated_encodings) < len(encodings):
        logger.info(f"编码列表已过滤: {len(encodings)} -> {len(validated_encodings)} 个有效编码")

    return validated_encodings


def validate_api_provider(provider: str) -> str:
    """验证API提供商

    Args:
        provider: API提供商名称

    Returns:
        str: 验证后的提供商名称
    """
    if not provider or not isinstance(provider, str):
        raise FileValidationError("API提供商不能为空")

    provider = provider.lower()
    if provider not in SUPPORTED_API_PROVIDERS:
        raise FileValidationError(
            f"不支持的API提供商: {provider}. 支持的提供商: {', '.join(SUPPORTED_API_PROVIDERS)}"
        )

    return provider


def validate_positive_int(value: int, name: str) -> int:
    """验证正整数

    Args:
        value: 要验证的值
        name: 参数名称（用于错误信息）

    Returns:
        int: 验证后的值
    """
    if not isinstance(value, int) or value <= 0:
        raise FileValidationError(f"{name}必须是正整数，当前值: {value}")

    return value


def validate_non_negative_int(value: int, name: str) -> int:
    """验证非负整数

    Args:
        value: 要验证的值
        name: 参数名称（用于错误信息）

    Returns:
        int: 验证后的值
    """
    if not isinstance(value, int) or value < 0:
        raise FileValidationError(f"{name}必须是非负整数，当前值: {value}")

    return value


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全字符

    Args:
        filename: 原始文件名

    Returns:
        str: 清理后的文件名
    """
    if not filename:
        return "output"

    # 移除路径分隔符和其他不安全字符
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # 移除控制字符
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)

    # 限制长度
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: 255 - len(ext)] + ext

    # 确保不为空
    if not sanitized or sanitized in [".", ".."]:
        sanitized = "output"

    return sanitized


def validate_chunk_size(chunk_size: int, max_tokens: int) -> int:
    """验证块大小参数

    Args:
        chunk_size: 目标块大小（token数）
        max_tokens: 模型最大token数

    Returns:
        int: 验证后的块大小
    """
    chunk_size = validate_positive_int(chunk_size, "TARGET_TOKENS_PER_CHUNK")
    max_tokens = validate_positive_int(max_tokens, "MODEL_MAX_TOKENS")

    if chunk_size >= max_tokens:
        raise FileValidationError(f"块大小({chunk_size})必须小于模型最大token数({max_tokens})")

    # 建议块大小不超过最大token数的 MAX_INPUT_TOKEN_RATIO
    if chunk_size > max_tokens * MAX_INPUT_TOKEN_RATIO:
        import warnings

        warnings.warn(
            f"块大小({chunk_size})接近模型限制({max_tokens})，建议设置为最大值的80%以下",
            UserWarning,
            stacklevel=2,
        )

    return chunk_size


def validate_parallel_limit(limit: int) -> int:
    """验证并发限制

    Args:
        limit: 并发限制数

    Returns:
        int: 验证后的并发限制
    """
    limit = validate_positive_int(limit, "PARALLEL_LIMIT")

    # 建议合理的并发限制
    if limit > RECOMMENDED_MAX_PARALLEL_LIMIT:
        import warnings

        warnings.warn(
            f"并发限制({limit})较高，可能导致API速率限制或资源不足",
            UserWarning,
            stacklevel=2,
        )

    return limit
