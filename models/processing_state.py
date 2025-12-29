"""
处理状态相关的数据模型
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ProgressData:
    """进度数据模型"""

    txt_file: str
    total_chunks: int
    completed_count: int
    completed_indices: set[int]
    outlines: list[dict[str, Any]]
    last_update: datetime
    chunks_hash: str
    processing_times: list[float] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        """计算完成率"""
        if self.total_chunks == 0:
            return 0.0
        return self.completed_count / self.total_chunks

    @property
    def average_processing_time(self) -> float:
        """计算平均处理时间"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def add_error(self, chunk_id: int, error_message: str) -> None:
        """添加错误记录"""
        self.errors.append(
            {"chunk_id": chunk_id, "error": error_message, "timestamp": datetime.now().isoformat()}
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        return {
            "txt_file": self.txt_file,
            "total_chunks": self.total_chunks,
            "completed_count": self.completed_count,
            "completed_indices": sorted(self.completed_indices),
            "outlines": self.outlines,
            "last_update": self.last_update.isoformat(),
            "chunks_hash": self.chunks_hash,
            "processing_times": self.processing_times,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressData":
        """从字典创建实例（用于JSON反序列化）"""
        return cls(
            txt_file=data.get("txt_file", ""),
            total_chunks=data.get("total_chunks", 0),
            completed_count=data.get("completed_count", 0),
            completed_indices=set(data.get("completed_indices", [])),
            outlines=data.get("outlines", []),
            last_update=datetime.fromisoformat(data.get("last_update", datetime.now().isoformat())),
            chunks_hash=data.get("chunks_hash", ""),
            processing_times=data.get("processing_times", []),
            errors=data.get("errors", []),
        )

    @staticmethod
    def calculate_chunks_hash(chunks: list[str]) -> str:
        """计算文本块的哈希值"""
        content = str(sorted(chunks))
        return hashlib.md5(content.encode("utf-8")).hexdigest()


@dataclass
class ProcessingState:
    """处理状态模型"""

    file_path: str
    total_chunks: int
    processed_chunks: int = 0
    failed_chunks: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    current_phase: str = "initialization"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        """计算已用时间（秒）"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def progress_percentage(self) -> float:
        """计算进度百分比"""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        total_attempted = self.processed_chunks + self.failed_chunks
        if total_attempted == 0:
            return 0.0
        return (self.processed_chunks / total_attempted) * 100

    def update_progress(self, processed: int, failed: int = 0) -> None:
        """更新进度"""
        self.processed_chunks += processed
        self.failed_chunks += failed

    def add_error(self, error: str) -> None:
        """添加错误信息"""
        self.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error}")

    def add_warning(self, warning: str) -> None:
        """添加警告信息"""
        self.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {warning}")

    def complete(self) -> None:
        """标记处理完成"""
        self.end_time = datetime.now()
        self.current_phase = "completed"

    def fail(self, error: str) -> None:
        """标记处理失败"""
        self.end_time = datetime.now()
        self.current_phase = "failed"
        self.add_error(error)

    def get_summary(self) -> dict[str, Any]:
        """获取处理摘要"""
        return {
            "file_path": self.file_path,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "progress_percentage": round(self.progress_percentage, 2),
            "success_rate": round(self.success_rate, 2),
            "elapsed_time": round(self.elapsed_time, 2),
            "current_phase": self.current_phase,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
        }
