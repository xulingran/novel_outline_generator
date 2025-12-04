"""
Token计数器模块
使用tiktoken库计算文本的token数量
"""
import tiktoken
from typing import Optional
import logging

from exceptions import EncodingError

logger = logging.getLogger(__name__)

# 全局编码器实例
_encoder: Optional[tiktoken.Encoding] = None


def get_encoder() -> tiktoken.Encoding:
    """获取编码器实例（单例模式）"""
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.get_encoding("cl100k_base")
            logger.debug("初始化tiktoken编码器: cl100k_base")
        except Exception as e:
            logger.error(f"初始化编码器失败: {e}")
            raise EncodingError(f"无法初始化token编码器: {str(e)}")
    return _encoder


def count_tokens(text: str) -> int:
    """
    计算文本的token数量

    Args:
        text: 要计算的文本

    Returns:
        int: token数量

    Raises:
        EncodingError: 编码失败
    """
    if not isinstance(text, str):
        raise ValueError("输入必须是字符串")

    if not text:
        return 0

    try:
        encoder = get_encoder()
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"计算token失败: {e}")
        raise EncodingError(f"无法计算token数量: {str(e)}") from e


def count_tokens_batch(texts: list[str]) -> dict[int, int]:
    """
    批量计算多个文本的token数量

    Args:
        texts: 文本列表

    Returns:
        Dict[int, int]: 索引到token数量的映射

    Raises:
        EncodingError: 编码失败
    """
    if not isinstance(texts, list):
        raise ValueError("输入必须是列表")

    try:
        encoder = get_encoder()
        result = {}
        for idx, text in enumerate(texts):
            if isinstance(text, str):
                tokens = encoder.encode(text)
                result[idx] = len(tokens)
            else:
                result[idx] = 0
        return result
    except Exception as e:
        logger.error(f"批量计算token失败: {e}")
        raise EncodingError(f"无法批量计算token数量: {str(e)}") from e


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """
    根据token数量截断文本

    Args:
        text: 原始文本
        max_tokens: 最大token数

    Returns:
        str: 截断后的文本
    """
    try:
        encoder = get_encoder()
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # 截断tokens
        truncated_tokens = tokens[:max_tokens]
        # 解码回文本
        return encoder.decode(truncated_tokens)
    except Exception as e:
        logger.error(f"截断文本失败: {e}")
        # 如果无法使用token截断，使用字符截断作为后备
        avg_chars_per_token = 4  # 估算值
        max_chars = max_tokens * avg_chars_per_token
        return text[:max_chars]


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    从字符数估算token数

    Args:
        char_count: 字符数

    Returns:
        int: 估算的token数
    """
    # 英文平均约4个字符=1个token
    # 中文平均约2.5个字符=1个token
    # 使用保守估算
    return max(1, char_count // 3)
