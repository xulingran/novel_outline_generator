"""
人物相关的数据模型
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Relationship:
    """人物关系模型"""
    character_a: str
    character_b: str
    description: str
    weight: int = 1
    contexts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理"""
        if self.character_a == self.character_b:
            raise ValueError("人物关系的两端不能是同一个人")

    def __str__(self) -> str:
        return f"{self.character_a} - {self.character_b}: {self.description}"

    def add_context(self, context: str) -> None:
        """添加关系上下文"""
        if context and context not in self.contexts:
            self.contexts.append(context)

    @property
    def pair(self) -> Tuple[str, str]:
        """返回排序后的人物对"""
        return tuple(sorted([self.character_a, self.character_b]))


@dataclass
class Character:
    """人物模型"""
    name: str
    aliases: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    appearances: List[int] = field(default_factory=list)  # 出现的块ID
    first_appearance: Optional[int] = None
    relationship_count: int = 0
    importance_score: float = 0.0

    def __str__(self) -> str:
        return f"Character({self.name}, appearances={len(self.appearances)})"

    def add_appearance(self, chunk_id: int) -> None:
        """记录人物出现的文本块"""
        if chunk_id not in self.appearances:
            self.appearances.append(chunk_id)
            self.appearances.sort()

            if self.first_appearance is None or chunk_id < self.first_appearance:
                self.first_appearance = chunk_id

    def add_alias(self, alias: str) -> None:
        """添加别名"""
        if alias and alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)

    def add_description(self, description: str) -> None:
        """添加描述"""
        if description and description not in self.descriptions:
            self.descriptions.append(description)

    @property
    def total_appearances(self) -> int:
        """返回出现次数"""
        return len(self.appearances)

    def calculate_importance(self, total_chunks: int) -> float:
        """计算人物重要性分数"""
        if total_chunks == 0:
            return 0.0

        # 基于出现频率
        frequency = self.total_appearances / total_chunks

        # 基于关系数量
        relationship_factor = min(self.relationship_count / 10, 1.0)

        # 基于描述长度
        description_factor = min(len(self.descriptions) / 5, 1.0)

        # 综合评分
        self.importance_score = (frequency * 0.5 +
                                relationship_factor * 0.3 +
                                description_factor * 0.2)

        return self.importance_score