"""
大纲相关的数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TextChunk:
    """文本块模型"""

    id: int
    content: str
    token_count: int
    start_position: int
    end_position: int
    chapter_title: str | None = None

    def __str__(self) -> str:
        return f"TextChunk(id={self.id}, tokens={self.token_count})"


@dataclass
class OutlineData:
    """大纲数据模型"""

    chunk_id: int
    events: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    relationships: list[list[str]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    raw_response: str | None = None
    processing_time: float | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        return {
            "chunk_id": self.chunk_id,
            "events": self.events,
            "characters": self.characters,
            "relationships": self.relationships,
            "conflicts": self.conflicts,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutlineData":
        """从字典创建实例（用于JSON反序列化）"""
        outline = cls(
            chunk_id=data.get("chunk_id", 0),
            raw_response=data.get("raw_response"),
            processing_time=data.get("processing_time"),
        )

        outline.events = data.get("events", [])
        outline.characters = data.get("characters", [])
        outline.relationships = data.get("relationships", [])
        outline.conflicts = data.get("conflicts", [])

        if "created_at" in data and data["created_at"]:
            outline.created_at = datetime.fromisoformat(data["created_at"])

        return outline

    def validate(self) -> bool:
        """验证数据完整性"""
        if self.chunk_id < 0:
            return False
        if not self.events and not self.characters and not self.conflicts:
            return False
        return True

    @property
    def character_count(self) -> int:
        """返回人物数量"""
        return len(self.characters)

    @property
    def relationship_count(self) -> int:
        """返回关系数量"""
        return len(self.relationships)

    def merge_with(self, other: "OutlineData") -> None:
        """合并另一个大纲数据"""
        if other.chunk_id != self.chunk_id:
            raise ValueError("Cannot merge outlines with different chunk IDs")

        self.events.extend(other.events)
        self.characters.extend([c for c in other.characters if c not in self.characters])
        self.relationships.extend([r for r in other.relationships if r not in self.relationships])
        self.conflicts.extend([c for c in other.conflicts if c not in self.conflicts])

        if other.raw_response and not self.raw_response:
            self.raw_response = other.raw_response
