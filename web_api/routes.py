"""FastAPI 路由定义与应用实例"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_processing_config, init_config
from services.token_estimator import estimate_tokens
from utils import init_logging
from web_api.job_storage import (
    Job,
    _update_progress_from_info,
    format_token_usage_log,
)

if TYPE_CHECKING:
    from services.task_queue import QueueTask

logger = logging.getLogger(__name__)

# 先加载 .env 再读取 CORS 配置，避免导入时拿到过期默认值。
# 注意：lifespan 中也会调用一次 init_config，用于确保 ASGI worker 启动后
# 读取到最新的环境变量（如容器注入的环境变量）。
init_config(create_env_if_missing=False)

UPLOAD_FILE_PARAM = File(...)
UPLOAD_FILES_PARAM = File(...)

# 敏感信息关键词（用于掩码处理）
_SENSITIVE_KEYWORDS: set[str] = {"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"}


def _mask_sensitive_value(key: str, value: str) -> str:
    """对敏感值（如 API Key、密码等）进行掩码处理

    Args:
        key: 配置项的键名
        value: 配置项的值

    Returns:
        掩码后的值（如果是敏感信息）或原值
    """
    key_upper = key.upper()
    if not any(keyword in key_upper for keyword in _SENSITIVE_KEYWORDS):
        return value

    if not value:
        return ""

    # 对敏感值进行掩码：保留前4和后4位，中间用*替换
    if len(value) <= 8:
        return "********"

    return value[:4] + "*" * (len(value) - 8) + value[-4:]


# CORS 允许的来源，可通过环境变量配置
# 默认允许本地开发常用端口，生产环境应配置具体域名
def _load_cors_origins() -> list[str]:
    """Load CORS origins, translating file:// to null for browser Origin headers.

    生产环境配置：
    - 设置 PRODUCTION=true 启用严格模式
    - 设置 CORS_ORIGINS 为具体的生产域名
    """
    is_production = os.getenv("PRODUCTION", "false").lower() == "true"

    if is_production:
        # 生产环境：只读取环境变量，不允许默认值
        raw = os.getenv("CORS_ORIGINS", "")
        if not raw:
            logger.warning("生产环境未配置CORS_ORIGINS，将拒绝所有跨域请求")
            return []
    else:
        # 开发环境：使用默认配置
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
        # 生产环境安全检查：过滤掉危险的来源
        if is_production and origin in ("null", "*"):
            logger.warning(f"生产环境忽略不安全的CORS来源: {origin}")
            continue
        origins.append(origin)

    # Preserve order but drop duplicates
    return list(dict.fromkeys(origins))


CORS_ORIGINS = _load_cors_origins()


class ProcessRequest(BaseModel):
    file_path: str
    resume: bool = True


class MultipleFilesRequest(BaseModel):
    """批量文件请求"""

    file_paths: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import web_api

    # 显式初始化日志系统
    init_logging()
    # 再次加载配置，确保 ASGI worker 启动后注入的环境变量（如容器 env）被拾取。
    # 模块级已调用过一次（用于 CORS 计算），此处刷新单例缓存。
    init_config(create_env_if_missing=False)
    web_api.startup_cleanup_task()

    # Initialize the queue callback
    queue = web_api.get_global_queue()
    queue.set_callback(run_queue_task)

    yield
    if web_api._cleanup_task:
        web_api._cleanup_task.cancel()
        try:
            await web_api._cleanup_task
        except asyncio.CancelledError:
            pass
    from services.connection_pool import get_default_connection_pool

    await get_default_connection_pool().close_all()


app = FastAPI(title="Novel Outline API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# 获取前端 HTML 文件路径
_UI_HTML_PATH = Path(__file__).parent.parent / "ui" / "index.html"


@app.get("/")
async def get_index():
    """返回前端 HTML 页面"""
    if _UI_HTML_PATH.exists():
        from fastapi.responses import HTMLResponse

        html_content = _UI_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    else:
        raise HTTPException(status_code=404, detail=f"前端文件未找到: {_UI_HTML_PATH}")


@app.get("/env")
def get_env() -> dict[str, Any]:
    import web_api

    data = web_api.load_env_file()
    # 只返回掩码后的数据，防止API密钥泄露
    masked = {k: _mask_sensitive_value(k, v) for k, v in data.items()}
    return {"env": masked}


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = UPLOAD_FILE_PARAM):
    import web_api

    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 10, 60)
    if file.content_type not in ("text/plain", "text/markdown", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="仅支持文本文件")

    # 获取配置
    processing_config = get_processing_config()

    # 验证文件名存在
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in processing_config.allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"仅支持 {', '.join(processing_config.allowed_extensions)} 文件"
        )

    # 使用安全文件名，防止路径遍历攻击
    from validators import sanitize_filename

    safe_filename = sanitize_filename(file.filename)
    web_api.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = web_api.UPLOAD_DIR / safe_filename

    # 验证最终路径仍在上传目录内（防止路径遍历攻击）
    try:
        dest_resolved = dest.resolve()
        upload_root_resolved = web_api._UPLOAD_ROOT.resolve()
        dest_resolved.relative_to(upload_root_resolved)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"无效的文件路径: {safe_filename}") from e

    content = await file.read()
    max_size_bytes = processing_config.max_upload_file_size_mb * 1024 * 1024
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=400, detail=f"文件过大，限制{processing_config.max_upload_file_size_mb}MB"
        )
    dest.write_bytes(content)
    return {"file_path": str(dest)}


# ... 其他导入 ...


@app.post("/upload-multiple")
async def upload_multiple_files(request: Request, files: list[UploadFile] = UPLOAD_FILES_PARAM):
    """批量上传多个文件"""
    import web_api

    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 10, 60)

    # 获取配置
    processing_config = get_processing_config()

    from validators import sanitize_filename

    web_api.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_files = []

    for file in files:
        # 验证文件名存在
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in processing_config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"仅支持 {', '.join(processing_config.allowed_extensions)} 文件: {file.filename}",
            )

        safe_filename = sanitize_filename(file.filename)
        dest = web_api.UPLOAD_DIR / safe_filename
        content = await file.read()
        max_size_bytes = processing_config.max_upload_file_size_mb * 1024 * 1024
        if len(content) > max_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大，限制{processing_config.max_upload_file_size_mb}MB: {file.filename}",
            )

        dest.write_bytes(content)
        uploaded_files.append(str(dest))

    return {"file_paths": uploaded_files}


async def _run_job(job: Job, req: ProcessRequest):
    import web_api

    job.status = "running"
    job.file_path = req.file_path
    job.progress = 0.0
    job.result = {}
    job.log(f"开始处理文件: {req.file_path}")

    def handle_progress(info: dict[str, Any]) -> None:
        _update_progress_from_info(info, job)
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
            job.log(format_token_usage_log(token_usage, "合并完成，"))
            job.token_logged = True

    try:
        service = web_api.NovelProcessingService(progress_callback=handle_progress)
        result = await service.process_novel(req.file_path, resume=req.resume)
        job.result.update(result)
        job.progress = 1.0
        job.status = "success"

        # 输出token统计
        if "token_usage" in result and not job.token_logged:
            token_usage = result["token_usage"]
            job.log(format_token_usage_log(token_usage))
            job.token_logged = True

        job.log("处理完成")
        try:
            current_upload = web_api._resolve_upload_path(req.file_path)
            if current_upload:
                active_uploads: set[Path] = set()
                for other_job in web_api.job_manager.values():
                    if other_job.id == job.id:
                        continue
                    if other_job.status not in {"pending", "running"}:
                        continue
                    if not other_job.file_path:
                        continue
                    upload_path = web_api._resolve_upload_path(other_job.file_path)
                    if upload_path:
                        active_uploads.add(upload_path)

                cleaned = web_api.cleanup_uploads(protected_paths=active_uploads)
                if cleaned:
                    job.log(f"已清理上传文件 {cleaned} 个")
        except Exception as cleanup_err:
            job.log(f"清理上传文件失败: {cleanup_err}")

    except Exception as e:
        logger.exception("Job %s failed with error: %s", job.id, e)
        job.status = "error"
        job.message = str(e)
        job.log(f"错误: {e}")


async def run_queue_task(task: "QueueTask") -> None:
    """运行队列任务（由 TaskQueue 调用）"""
    import web_api

    task.log(f"开始处理文件: {task.file_path}")

    def handle_progress(info: dict[str, Any]) -> None:
        _update_progress_from_info(info, task)

    try:
        service = web_api.NovelProcessingService(
            progress_callback=handle_progress, cancel_event=task.cancel_event
        )
        # 检查是否强制完成
        if task.should_force_complete:
            service.force_complete = True
            logger.info(f"任务 {task.id} 启用强制完成模式")

        result = await service.process_novel(task.file_path, resume=True)
        task.result.update(result)
        task.progress = 1.0
        task.status = "success"

        # 输出token统计
        if "token_usage" in result and not task.token_logged:
            token_usage = result["token_usage"]
            task.log(format_token_usage_log(token_usage))
            task.token_logged = True

        task.log("处理完成")

    except asyncio.CancelledError:
        # 如果是强制完成模式，继续处理已有结果
        if task.should_force_complete and len(task.result.get("outlines", [])) > 0:
            logger.info(f"任务 {task.id} 强制完成模式：继续合并已有结果")
            task.status = "success"
            task.message = "强制完成（部分结果已合并）"
            task.log("强制完成：将合并已有部分结果")
        else:
            task.status = "cancelled"
            task.message = "任务被取消"
            task.log("任务被取消")
    except Exception as e:
        logger.exception("Task %s failed with error: %s", task.id, e)
        task.status = "error"
        task.message = str(e)
        task.log(f"错误: {e}")


@app.post("/process")
async def start_process(request: Request, req: ProcessRequest):
    import web_api

    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 5, 60)
    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    # 清理旧job
    web_api.cleanup_excess_jobs()

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, file_path=req.file_path)
    web_api.job_manager.set(job_id, job)

    async def _run_job_wrapper(job: Job, req: ProcessRequest) -> None:
        """_run_job 的包装器，捕获启动异常并记录"""
        try:
            await web_api._run_job(job, req)
        except asyncio.CancelledError:
            # 取消错误需要向上传播
            raise
        except Exception as e:
            logger.exception("Job %s 启动失败: %s", job.id, e)
            job.status = "error"
            job.message = f"启动失败: {str(e)}"
            job.log(f"启动失败: {e}")

    asyncio.create_task(_run_job_wrapper(job, req))
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
    import web_api

    job = web_api.job_manager.get(job_id)
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


@app.get("/queue/list")
async def list_queue():
    import web_api

    """列出所有队列任务"""
    queue = web_api.get_global_queue()
    tasks = await queue.list_tasks()
    return {"tasks": tasks}


@app.post("/queue/add")
async def add_to_queue(request: Request, req: ProcessRequest):
    import web_api

    """添加任务到队列"""
    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 5, 60)

    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    queue = web_api.get_global_queue()
    task_id = await queue.add_task(req.file_path)
    return {"task_id": task_id, "message": "任务已添加到队列"}


@app.post("/queue/add-multiple")
async def add_multiple_to_queue(request: Request, req: MultipleFilesRequest):
    import web_api

    """批量添加文件到队列"""
    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 5, 60)

    if not req.file_paths:
        raise HTTPException(status_code=400, detail="file_paths 不能为空")

    queue = web_api.get_global_queue()
    task_ids = []

    for file_path in req.file_paths:
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path 不能为空")
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
        task_id = await queue.add_task(file_path)
        task_ids.append(task_id)

    return {
        "task_ids": task_ids,
        "count": len(task_ids),
        "message": f"已将 {len(task_ids)} 个文件添加到队列",
    }


@app.post("/queue/cancel")
async def cancel_queue_task(request: Request, task_id: str):
    import web_api

    """取消队列任务"""
    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 10, 60)

    queue = web_api.get_global_queue()
    success = await queue.cancel_task(task_id)

    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或无法取消")

    return {"success": True, "message": "任务已取消"}


@app.post("/queue/clear")
async def clear_queue():
    import web_api

    """清空队列（仅取消未开始的任务）"""
    queue = web_api.get_global_queue()
    count = await queue.clear_queue()
    return {"success": True, "cancelled_count": count}


@app.post("/queue/force-complete/{task_id}")
async def force_complete_queue_task(request: Request, task_id: str):
    import web_api

    """强制完成任务（忽略未完成的块，直接合并已有结果）"""
    client_host = request.client.host if request.client else "unknown"
    web_api.rate_limiter.check_rate_limit(client_host, 10, 60)

    queue = web_api.get_global_queue()
    success = await queue.force_complete_task(task_id)

    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或无法强制完成")

    return {"success": True, "message": "已强制完成，将合并已有结果"}


@app.get("/queue/stats")
async def get_queue_stats():
    import web_api

    """获取队列统计信息"""
    queue = web_api.get_global_queue()
    stats = await queue.get_stats()
    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=True)
