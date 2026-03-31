"""
FastAPI 后端接口：
- /env   GET 读取 .env（可视化配置查看）
- /upload POST 上传文本文件
- /process POST 启动小说处理（基于 NovelProcessingService）
- /jobs/{job_id} GET 查询处理状态

启动方式：
  uvicorn web_api:app --reload --port 8000
"""

import asyncio  # noqa: F401 - 测试 monkeypatch web_api.asyncio
import time  # noqa: F401 - 测试 monkeypatch web_api.time

# routes 中通过延迟导入使用 NovelProcessingService，
# 测试 monkeypatch web_api.NovelProcessingService，因此需要在包级别可见
from services.novel_processing_service import NovelProcessingService  # noqa: F401
from services.task_queue import get_global_queue  # noqa: F401

# 从子模块 re-export 所有被测试 monkeypatch 或直接访问的符号
# _cleanup_task 是可变状态，从 job_storage 引入初始值
from web_api.job_storage import (  # noqa: F401
    JOB_MAX_AGE_HOURS,
    JOBS,
    MAX_JOBS,
    Job,
    _cleanup_task,  # noqa: F401
    _periodic_job_cleanup,
    _update_progress_from_info,
    cleanup_excess_jobs,
    cleanup_expired_jobs,
    format_token_usage_log,
    job_manager,
    startup_cleanup_task,
)
from web_api.rate_limiter import RateLimiter, rate_limiter  # noqa: F401
from web_api.routes import (  # noqa: F401
    CORS_ORIGINS,
    MultipleFilesRequest,
    ProcessRequest,
    _load_cors_origins,
    _mask_sensitive_value,
    _run_job,
    app,
    run_queue_task,
    upload_file,
    upload_multiple_files,
)
from web_api.upload_handler import (  # noqa: F401
    _UPLOAD_ROOT,
    ENV_PATH,
    UPLOAD_DIR,
    _resolve_upload_path,
    cleanup_uploads,
    load_env_file,
)
