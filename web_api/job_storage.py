"""任务存储与管理"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.job_manager import JobManager
else:
    try:
        from services.job_manager import JobManager
    except ModuleNotFoundError:
        # 回退实现：防止部署遗漏模块时 API 因导入失败无法启动
        class JobManager:
            """内存任务管理器（回退实现）。"""

            def __init__(self, max_jobs: int = 100, max_age_hours: int = 24) -> None:
                self.max_jobs = max_jobs
                self.max_age_hours = max_age_hours
                self.jobs: dict[str, Any] = {}

            def get(self, job_id: str) -> Any | None:
                return self.jobs.get(job_id)

            def set(self, job_id: str, job: Any) -> None:
                self.jobs[job_id] = job

            def values(self):
                return self.jobs.values()

            def cleanup_expired(self, now: float) -> int:
                cutoff_time = now - self.max_age_hours * 3600
                expired_ids = []
                for job_id, job in self.jobs.items():
                    # 安全地获取created_at，支持对象属性或字典访问
                    created_at: float
                    if isinstance(job, dict):
                        created_at = job.get("created_at", 0.0)
                    else:
                        created_at = getattr(job, "created_at", 0.0)
                    if created_at < cutoff_time:
                        expired_ids.append(job_id)
                for job_id in expired_ids:
                    del self.jobs[job_id]
                return len(expired_ids)

            def cleanup_excess(self) -> int:
                if len(self.jobs) <= self.max_jobs:
                    return 0

                over_limit = len(self.jobs) - self.max_jobs
                removed = 0

                def _by_age(statuses: tuple[str, ...]) -> list[tuple[str, Any]]:
                    def _get_status(job: Any) -> str:
                        if isinstance(job, dict):
                            return job.get("status", "")
                        return getattr(job, "status", "")

                    def _get_created_at(job: Any) -> float:
                        if isinstance(job, dict):
                            return job.get("created_at", 0.0)
                        return getattr(job, "created_at", 0.0)

                    return sorted(
                        (
                            (job_id, job)
                            for job_id, job in self.jobs.items()
                            if _get_status(job) in statuses
                        ),
                        key=lambda item: _get_created_at(item[1]),
                    )

                for statuses in (("success", "error"), ("pending",), ("running",)):
                    if over_limit <= 0:
                        break
                    for job_id, _ in _by_age(statuses):
                        if over_limit <= 0:
                            break
                        del self.jobs[job_id]
                        over_limit -= 1
                        removed += 1

                return removed


logger = logging.getLogger(__name__)


def format_token_usage_log(token_usage: dict[str, Any], prefix: str = "合并完成，") -> str:
    """格式化 token 使用日志

    Args:
        token_usage: 包含 token 使用信息的字典
        prefix: 日志前缀

    Returns:
        格式化后的日志字符串
    """
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    return f"{prefix}Token统计: 输入={prompt_tokens:,}, 输出={completion_tokens:,}, 总计={total_tokens:,}"


@dataclass
class Job:
    id: str
    file_path: str = ""
    status: str = "pending"  # pending|running|success|error
    message: str = ""
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    log_offset: int = 0
    token_logged: bool = False
    created_at: float = field(default_factory=time.time)

    def log(self, text: str) -> None:
        """Append a log line and keep list size bounded."""
        self.logs.append(text)
        if len(self.logs) > 200:
            # 只保留最近 200 条，避免内存增长过快
            # 使用 del 删除而非重新赋值，避免 list 对象变化导致前端轮询失效
            overflow = len(self.logs) - 200
            del self.logs[:overflow]
            self.log_offset += overflow


# 常量定义
MAX_JOBS = 100
JOB_MAX_AGE_HOURS = 24

job_manager = JobManager(max_jobs=MAX_JOBS, max_age_hours=JOB_MAX_AGE_HOURS)
# 向后兼容：JOBS 是 job_manager.jobs 的别名，测试代码使用
JOBS: dict[str, Job] = job_manager.jobs
_cleanup_task: asyncio.Task | None = None


def _update_progress_from_info(
    info: dict[str, Any],
    target: Any,  # Job 或 QueueTask
) -> None:
    """从进度信息更新目标对象（Job 或 QueueTask）

    Args:
        info: 进度信息字典
        target: 目标对象（Job 或 QueueTask）
    """
    target.progress = info.get("progress", target.progress)

    # 更新结果字典中的字段
    result_fields = [
        "total_chunks",
        "completed_chunks",
        "failed_chunks",
        "partial_chunks",
        "partial_info",
        "eta_seconds",
        "eta_confidence",
        "eta_method",
        "phase",
        "merge_level",
        "merge_batch_current",
        "merge_batch_total",
        "merge_outlines_count",
    ]

    for field_name in result_fields:
        if info.get(field_name) is not None:
            target.result[field_name] = info[field_name]

    # 处理块完成/失败日志
    if info.get("last_chunk_id") is not None:
        if info.get("last_error"):
            target.log(f"块 {info['last_chunk_id']} 失败: {info['last_error']}")
        else:
            target.log(f"块 {info['last_chunk_id']} 完成")

    # 处理 token 使用统计（只记录一次）
    if info.get("token_usage") and not target.token_logged:
        token_usage = info["token_usage"]
        target.result["token_usage"] = token_usage
        target.log(format_token_usage_log(token_usage, "合并完成，"))
        target.token_logged = True


async def _periodic_job_cleanup() -> None:
    """定期清理过期和过多的job任务"""
    import web_api

    while True:
        try:
            await asyncio.sleep(60)  # 每60秒检查一次
            web_api.cleanup_expired_jobs()
            web_api.cleanup_excess_jobs()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"定期清理job失败: {e}")


def startup_cleanup_task() -> None:
    """启动后台清理任务"""
    import web_api

    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_periodic_job_cleanup())
    # 同步到 web_api 包级别，使测试 monkeypatch 可见
    web_api._cleanup_task = _cleanup_task


def cleanup_expired_jobs() -> None:
    """清理超过最大存活时间的job"""
    import web_api

    job_manager.max_age_hours = web_api.JOB_MAX_AGE_HOURS
    expired_count = job_manager.cleanup_expired(now=time.time())
    if expired_count:
        logger.debug(f"清理了 {expired_count} 个过期job")


def cleanup_excess_jobs() -> None:
    """清理过多的job，防止内存泄漏"""
    import web_api

    job_manager.max_jobs = web_api.MAX_JOBS
    job_manager.cleanup_excess()
