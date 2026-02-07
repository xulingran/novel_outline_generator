"""
JobManager 模块测试
"""

from dataclasses import dataclass

from services.job_manager import JobManager


@dataclass
class DummyJob:
    id: str
    status: str
    created_at: float


def test_cleanup_expired():
    manager = JobManager(max_jobs=100, max_age_hours=24)
    manager.set("old", DummyJob(id="old", status="success", created_at=0.0))
    manager.set("new", DummyJob(id="new", status="running", created_at=24 * 3600))

    removed = manager.cleanup_expired(now=24 * 3600 + 1)
    assert removed == 1
    assert manager.get("old") is None
    assert manager.get("new") is not None


def test_cleanup_excess_priority():
    manager = JobManager(max_jobs=2, max_age_hours=24)
    manager.set("s1", DummyJob(id="s1", status="success", created_at=1.0))
    manager.set("e1", DummyJob(id="e1", status="error", created_at=2.0))
    manager.set("p1", DummyJob(id="p1", status="pending", created_at=3.0))
    manager.set("r1", DummyJob(id="r1", status="running", created_at=4.0))

    removed = manager.cleanup_excess()
    assert removed == 2
    assert len(manager.jobs) == 2
    # 优先删除 success/error，至少保留一个未完成任务
    assert any(job.status in {"pending", "running"} for job in manager.values())
