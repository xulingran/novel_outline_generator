"""
FastAPI 后端接口：
- /env   GET/POST 读取与更新 .env（可视化配置编辑）
- /upload POST 上传文本文件
- /process POST 启动小说处理（基于 NovelProcessingService）
- /jobs/{job_id} GET 查询处理状态

启动方式：
  uvicorn web_api:app --reload --port 8000
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.novel_processing_service import NovelProcessingService
from services.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)

ENV_PATH = Path(".env")
UPLOAD_DIR = Path("outputs/uploads")
_UPLOAD_ROOT = UPLOAD_DIR.resolve()


# CORS 允许的来源，可通过环境变量配置
# 默认允许本地开发常用端口，生产环境应配置具体域名
def _load_cors_origins() -> list[str]:
    """Load CORS origins, translating file:// to null for browser Origin headers."""
    raw = os.getenv(
        "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,null"
    )
    origins: list[str] = []
    for origin in raw.split(","):
        origin = origin.strip()
        if not origin:
            continue
        if origin == "file://":
            origin = "null"
        origins.append(origin)
    # Preserve order but drop duplicates
    return list(dict.fromkeys(origins))


CORS_ORIGINS = _load_cors_origins()

ALLOWED_KEYS = {
    "API_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_API_BASE",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_SAFETY_SETTINGS",
    "ZHIPU_API_KEY",
    "ZHIPU_MODEL",
    "ZHIPU_API_BASE",
    "AIHUBMIX_API_KEY",
    "AIHUBMIX_MODEL",
    "AIHUBMIX_API_BASE",
    "MODEL_MAX_TOKENS",
    "TARGET_TOKENS_PER_CHUNK",
    "PARALLEL_LIMIT",
    "MAX_RETRY",
    "LOG_EVERY",
    "LOG_LEVEL",
    "USE_PROXY",
    "PROXY_URL",
    "CORS_ORIGINS",
    "OUTLINE_PROMPT_TEMPLATE",
}


class RateLimiter:
    """简单内存限流器，按 IP 在时间窗口内计数。"""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check_rate_limit(self, client_ip: str, max_requests: int, window_seconds: int) -> None:
        now = time.time()
        window_start = now - window_seconds
        bucket = self._requests[client_ip]
        # 清理窗口外的请求时间戳
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= max_requests:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        bucket.append(now)


rate_limiter = RateLimiter()
UPLOAD_FILE_PARAM = File(...)


def load_env_file() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    data: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def save_env_file(updates: dict[str, str]) -> None:
    existing = load_env_file()
    existing.update(updates)
    for key, value in updates.items():
        os.environ[key] = value
    lines: list[str] = []
    for k, v in existing.items():
        lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask_value(key: str, value: str) -> str:
    """对敏感值（如 API Key）进行掩码处理"""
    if "KEY" in key.upper():
        if not value:
            return ""
        return "*" * max(4, len(value) - 4) + value[-4:]
    return value


class ProcessRequest(BaseModel):
    file_path: str
    resume: bool = True


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


JOBS: dict[str, Job] = {}
MAX_JOBS = 100
MAX_RUNNING_JOBS = 20
JOB_MAX_AGE_HOURS = 24
_cleanup_task: asyncio.Task | None = None


async def _periodic_job_cleanup() -> None:
    """定期清理过期和过多的job任务"""
    while True:
        try:
            await asyncio.sleep(60)  # 每60秒检查一次
            cleanup_expired_jobs()
            cleanup_excess_jobs()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"定期清理job失败: {e}")


def startup_cleanup_task() -> None:
    """启动后台清理任务"""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_periodic_job_cleanup())


def cleanup_expired_jobs() -> None:
    """清理超过最大存活时间的job"""
    cutoff_time = time.time() - JOB_MAX_AGE_HOURS * 3600
    expired_ids = [job_id for job_id, job in JOBS.items() if job.created_at < cutoff_time]
    for job_id in expired_ids:
        del JOBS[job_id]
    if expired_ids:
        logger.debug(f"清理了 {len(expired_ids)} 个过期job")


def cleanup_excess_jobs() -> None:
    """清理过多的job，防止内存泄漏"""
    if len(JOBS) <= MAX_JOBS:
        return

    over_limit = len(JOBS) - MAX_JOBS

    def _by_age(statuses):
        return sorted(
            ((job_id, job) for job_id, job in JOBS.items() if job.status in statuses),
            key=lambda x: x[1].created_at,
        )

    for statuses in (("success", "error"), ("pending",), ("running",)):
        if over_limit <= 0:
            break
        for job_id, _ in _by_age(statuses):
            if over_limit <= 0:
                break
            del JOBS[job_id]
            over_limit -= 1


def cleanup_old_jobs():
    """清理旧的 job 记录，确保 JOBS 不会无限增长"""
    if len(JOBS) <= MAX_JOBS:
        return

    # 优先清理已结束的任务，其次未开始的，最后才是仍在运行的
    over_limit = len(JOBS) - MAX_JOBS

    def _by_age(statuses):
        return sorted(
            ((job_id, job) for job_id, job in JOBS.items() if job.status in statuses),
            key=lambda x: x[1].created_at,
        )

    for statuses in (("success", "error"), ("pending",), ("running",)):
        if over_limit <= 0:
            break
        for job_id, _ in _by_age(statuses):
            if over_limit <= 0:
                break
            del JOBS[job_id]
            over_limit -= 1


def _resolve_upload_path(path: str) -> Path | None:
    """返回在 uploads 目录下的路径，其他情况返回 None。"""
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return None

    if resolved == _UPLOAD_ROOT:
        return None

    try:
        resolved.relative_to(_UPLOAD_ROOT)
        return resolved
    except ValueError:
        return None


def cleanup_uploads(protected_paths: set[Path] | None = None) -> int:
    """删除上传目录中的内容，保留目录本身。

    Args:
        protected_paths: 需要保留的上传文件路径集合（已解析的绝对路径）
    """
    if not UPLOAD_DIR.exists():
        return 0

    keep = {p.resolve() for p in protected_paths} if protected_paths else set()

    cleaned = 0
    for item in UPLOAD_DIR.iterdir():
        item_path = item.resolve()
        if any(item_path == kept_path or item_path in kept_path.parents for kept_path in keep):
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
            cleaned += 1
        except Exception:
            # 忽略清理失败，避免影响主流程
            continue
    return cleaned


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_cleanup_task()
    yield
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    from services.llm_service import OpenAIService

    await OpenAIService.close_http_clients()


app = FastAPI(title="Novel Outline API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/env")
def get_env() -> dict[str, Any]:
    data = load_env_file()
    # 只返回掩码后的数据，防止API密钥泄露
    masked = {k: mask_value(k, v) for k, v in data.items()}
    return {"env": masked}


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = UPLOAD_FILE_PARAM):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 10, 60)
    if file.content_type not in ("text/plain", "text/markdown", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="仅支持文本文件")

    # 验证文件名存在
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".md"):
        raise HTTPException(status_code=400, detail="仅支持.txt 或 .md 文件")

    # 使用安全文件名，防止路径遍历攻击
    from validators import sanitize_filename

    safe_filename = sanitize_filename(file.filename)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / safe_filename
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件过大，限制100MB")
    dest.write_bytes(content)
    return {"file_path": str(dest)}


async def _run_job(job: Job, req: ProcessRequest):
    job.status = "running"
    job.file_path = req.file_path
    job.progress = 0.0
    job.result = {}
    job.log(f"开始处理文件: {req.file_path}")

    def handle_progress(info: dict[str, Any]) -> None:
        job.progress = info.get("progress", job.progress)
        if "total_chunks" in info:
            job.result["chunk_count"] = info["total_chunks"]
        if "completed_chunks" in info:
            job.result["completed_chunks"] = info["completed_chunks"]
        if "failed_chunks" in info:
            job.result["failed_chunks"] = info["failed_chunks"]
        if info.get("eta_seconds") is not None:
            job.result["eta_seconds"] = info["eta_seconds"]
        if info.get("eta_confidence"):
            job.result["eta_confidence"] = info["eta_confidence"]
        if info.get("eta_method"):
            job.result["eta_method"] = info["eta_method"]
        if info.get("phase"):
            job.result["phase"] = info["phase"]
        if info.get("merge_level") is not None:
            job.result["merge_level"] = info["merge_level"]
        if info.get("merge_batch_current") is not None:
            job.result["merge_batch_current"] = info["merge_batch_current"]
        if info.get("merge_batch_total") is not None:
            job.result["merge_batch_total"] = info["merge_batch_total"]
        if info.get("merge_outlines_count") is not None:
            job.result["merge_outlines_count"] = info["merge_outlines_count"]
        if info.get("last_chunk_id") is not None:
            if info.get("last_error"):
                job.log(f"块 {info['last_chunk_id']} 失败: {info['last_error']}")
            else:
                job.log(f"块 {info['last_chunk_id']} 完成")
        if info.get("token_usage") and not job.token_logged:
            token_usage = info["token_usage"]
            job.result["token_usage"] = token_usage
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            job.log(
                "合并完成，Token统计: "
                f"输入={prompt_tokens:,}, 输出={completion_tokens:,}, 总计={total_tokens:,}"
            )
            job.token_logged = True

    try:
        service = NovelProcessingService(progress_callback=handle_progress)
        result = await service.process_novel(req.file_path, resume=req.resume)
        job.result.update(result)
        job.progress = 1.0
        job.status = "success"

        # 输出token统计
        if "token_usage" in result and not job.token_logged:
            token_usage = result["token_usage"]
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)

            job.log(
                f"Token统计: 输入={prompt_tokens:,}, 输出={completion_tokens:,}, 总计={total_tokens:,}"
            )
            job.token_logged = True

        job.log("处理完成")
        try:
            current_upload = _resolve_upload_path(req.file_path)
            if current_upload:
                active_uploads: set[Path] = set()
                for other_job in JOBS.values():
                    if other_job.id == job.id:
                        continue
                    if other_job.status not in {"pending", "running"}:
                        continue
                    if not other_job.file_path:
                        continue
                    upload_path = _resolve_upload_path(other_job.file_path)
                    if upload_path:
                        active_uploads.add(upload_path)

                cleaned = cleanup_uploads(protected_paths=active_uploads)
                if cleaned:
                    job.log(f"已清理上传文件 {cleaned} 个")
        except Exception as cleanup_err:  # noqa: BLE001
            job.log(f"清理上传文件失败: {cleanup_err}")

    except Exception as e:  # noqa: BLE001
        job.status = "error"
        job.message = str(e)
        job.log(f"错误: {e}")


@app.post("/process")
async def start_process(request: Request, req: ProcessRequest):
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.check_rate_limit(client_host, 5, 60)
    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    # 清理旧job
    cleanup_old_jobs()

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, file_path=req.file_path)
    JOBS[job_id] = job

    asyncio.create_task(_run_job(job, req))
    return {"job_id": job_id}


@app.get("/estimate")
def estimate(file_path: str):
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return estimate_tokens(file_path)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {
        "id": job.id,
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "result": job.result,
        "logs": job.logs,
        "log_offset": job.log_offset,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=True)
