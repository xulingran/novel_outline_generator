"""
数据模型模块
定义项目中使用的各种数据结构
"""

from .character import Character, Relationship
from .outline import OutlineData, TextChunk
from .processing_state import ProcessingState, ProgressData

__all__ = [
    "OutlineData",
    "TextChunk",
    "Character",
    "Relationship",
    "ProcessingState",
    "ProgressData",
]
