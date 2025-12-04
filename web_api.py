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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.novel_processing_service import NovelProcessingService
from services.token_estimator import estimate_tokens

ENV_PATH = Path(".env")
UPLOAD_DIR = Path("outputs/uploads")
ALLOWED_KEYS = {
    "API_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_API_BASE",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "ZHIPU_API_KEY",
    "ZHIPU_MODEL",
    "ZHIPU_API_BASE",
    "MODEL_MAX_TOKENS",
    "TARGET_TOKENS_PER_CHUNK",
    "PARALLEL_LIMIT",
    "MAX_RETRY",
    "LOG_EVERY",
    "OUTLINE_PROMPT_TEMPLATE",
}


def load_env_file() -> Dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    data: Dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def save_env_file(updates: Dict[str, str]) -> None:
    existing = load_env_file()
    existing.update(updates)
    for key, value in updates.items():
        os.environ[key] = value
    lines: List[str] = []
    for k, v in existing.items():
        lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask_value(key: str, value: str) -> str:
    if "KEY" in key.upper():
        if not value:
            return ""
        return "*" * max(4, len(value) - 4) + value[-4:]
    return value


class EnvUpdate(BaseModel):
    updates: Dict[str, str]


class ProcessRequest(BaseModel):
    file_path: str
    resume: bool = True


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending|running|success|error
    message: str = ""
    progress: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def log(self, text: str) -> None:
        """Append a log line and keep list size bounded."""
        self.logs.append(text)
        if len(self.logs) > 200:
            # 只保留最近 200 条，避免内存增长过快
            self.logs = self.logs[-200:]


JOBS: Dict[str, Job] = {}

def cleanup_uploads() -> int:
    """删除上传目录中的内容，保留目录本身"""
    if not UPLOAD_DIR.exists():
        return 0
    cleaned = 0
    for item in UPLOAD_DIR.iterdir():
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

app = FastAPI(title="Novel Outline API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/env")
def get_env() -> Dict[str, Any]:
    data = load_env_file()
    masked = {k: mask_value(k, v) for k, v in data.items()}
    return {"env": data, "masked": masked}


@app.post("/env")
def update_env(body: EnvUpdate):
    bad_keys = [k for k in body.updates if k not in ALLOWED_KEYS]
    if bad_keys:
        raise HTTPException(status_code=400, detail=f"不允许修改的键: {bad_keys}")
    save_env_file(body.updates)
    return {"ok": True}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ("text/plain", "text/markdown", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="仅支持文本文件")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件过大，限制100MB")
    dest.write_bytes(content)
    return {"file_path": str(dest)}


async def _run_job(job: Job, req: ProcessRequest):
    job.status = "running"
    job.progress = 0.0
    job.result = {}
    job.log(f"开始处理文件: {req.file_path}")

    def handle_progress(info: Dict[str, Any]) -> None:
        job.progress = info.get("progress", job.progress)
        if "total_chunks" in info:
            job.result["chunk_count"] = info["total_chunks"]
        if "completed_chunks" in info:
            job.result["completed_chunks"] = info["completed_chunks"]
        if "failed_chunks" in info:
            job.result["failed_chunks"] = info["failed_chunks"]
        if info.get("eta_seconds") is not None:
            job.result["eta_seconds"] = info["eta_seconds"]
        if info.get("phase"):
            job.result["phase"] = info["phase"]
        if info.get("last_chunk_id") is not None:
            if info.get("last_error"):
                job.log(f"块 {info['last_chunk_id']} 失败: {info['last_error']}")
            else:
                job.log(f"块 {info['last_chunk_id']} 完成")

    try:
        service = NovelProcessingService(progress_callback=handle_progress)
        result = await service.process_novel(req.file_path, resume=req.resume)
        job.result.update(result)
        job.progress = 1.0
        job.status = "success"
        job.log("处理完成")
        try:
            cleaned = cleanup_uploads()
            if cleaned:
                job.log(f"已清理上传文件 {cleaned} 个")
        except Exception as cleanup_err:  # noqa: BLE001
            job.log(f"清理上传文件失败: {cleanup_err}")

    except Exception as e:  # noqa: BLE001
        job.status = "error"
        job.message = str(e)
        job.log(f"错误: {e}")


@app.post("/process")
async def start_process(req: ProcessRequest):
    if not req.file_path:
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    job_id = str(uuid.uuid4())
    job = Job(id=job_id)
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
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=True)
