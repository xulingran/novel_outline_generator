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
