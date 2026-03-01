"""
Job 管理服务模块
负责内存中的任务存储与清理策略
"""

from collections.abc import Mapping
from typing import Any


class JobManager:
    """内存任务管理器。"""

    def __init__(self, max_jobs: int = 100, max_age_hours: int = 24) -> None:
        self.max_jobs = max_jobs
        self.max_age_hours = max_age_hours
        self.jobs: dict[str, Any] = {}

    def get(self, job_id: str) -> Any | None:
        """获取任务。"""
        return self.jobs.get(job_id)

    def set(self, job_id: str, job: Any) -> None:
        """设置任务。"""
        self.jobs[job_id] = job

    def delete(self, job_id: str) -> None:
        """删除任务。"""
        if job_id in self.jobs:
            del self.jobs[job_id]

    def values(self):
        """获取任务值视图。"""
        return self.jobs.values()

    def cleanup_expired(self, now: float) -> int:
        """清理过期任务。"""
        cutoff_time = now - self.max_age_hours * 3600
        expired_ids = [job_id for job_id, job in self.jobs.items() if job.created_at < cutoff_time]
        for job_id in expired_ids:
            del self.jobs[job_id]
        return len(expired_ids)

    def cleanup_excess(self) -> int:
        """清理超出上限的任务。"""
        if len(self.jobs) <= self.max_jobs:
            return 0

        removed = 0
        over_limit = len(self.jobs) - self.max_jobs

        def by_age(statuses: tuple[str, ...]) -> list[tuple[str, Any]]:
            return sorted(
                ((job_id, job) for job_id, job in self.jobs.items() if job.status in statuses),
                key=lambda item: item[1].created_at,
            )

        for statuses in (("success", "error"), ("pending",), ("running",)):
            if over_limit <= 0:
                break
            for job_id, _job in by_age(statuses):
                if over_limit <= 0:
                    break
                del self.jobs[job_id]
                over_limit -= 1
                removed += 1

        return removed

    def snapshot(self) -> Mapping[str, Any]:
        """获取任务快照。"""
        return dict(self.jobs)
