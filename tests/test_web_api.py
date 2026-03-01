import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import web_api
from services.task_queue import QueueTask


@pytest.fixture(autouse=True)
def isolate_web_api_state(monkeypatch, tmp_path):
    web_api.job_manager.jobs.clear()
    monkeypatch.setattr(web_api, "rate_limiter", web_api.RateLimiter())
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(web_api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(web_api, "_UPLOAD_ROOT", upload_dir.resolve())
    yield


def test_get_env():
    client = TestClient(web_api.app)
    response = client.get("/env")
    assert response.status_code == 200
    data = response.json()
    assert "env" in data
    # 只返回掩码后的数据，不返回原始数据
    env_data = data["env"]
    # 验证API密钥已被掩码处理
    if "OPENAI_API_KEY" in env_data:
        masked_key = env_data["OPENAI_API_KEY"]
        # 掩码后的密钥应该以 * 开头
        assert masked_key.startswith("*")
        # 掩码后的密钥不应该以 sk- 开头（原始格式）
        assert not masked_key.startswith("sk-")


def test_upload_file_success():
    client = TestClient(web_api.app)
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 200
    file_path = Path(response.json()["file_path"])
    assert file_path.exists()
    assert file_path.read_bytes() == b"hello"


def test_upload_file_rejects_invalid_content_type():
    client = TestClient(web_api.app)
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"hello", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_file_rejects_invalid_extension():
    client = TestClient(web_api.app)
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rate_limit():
    client = TestClient(web_api.app)
    for index in range(10):
        response = client.post(
            "/upload",
            files={"file": (f"sample_{index}.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 200

    response = client.post(
        "/upload",
        files={"file": ("sample_10.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 429


def test_process_and_job_status(monkeypatch, tmp_path):
    async def fake_run_job(job, req, started_event=None):
        if started_event:
            started_event.set()
        job.status = "success"
        job.progress = 1.0
        job.result = {"ok": True}
        job.log("done")

    monkeypatch.setattr(web_api, "_run_job", fake_run_job)

    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    client = TestClient(web_api.app)
    response = client.post(
        "/process",
        json={"file_path": str(input_path), "resume": False},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    time.sleep(0.05)

    job_response = client.get(f"/jobs/{job_id}")
    assert job_response.status_code == 200
    payload = job_response.json()
    assert payload["id"] == job_id
    assert payload["status"] in {"pending", "running", "success"}


def test_process_rate_limit(monkeypatch, tmp_path):
    async def fake_run_job(job, req, started_event=None):
        if started_event:
            started_event.set()
        job.status = "success"

    monkeypatch.setattr(web_api, "_run_job", fake_run_job)

    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    client = TestClient(web_api.app)
    for _ in range(5):
        response = client.post(
            "/process",
            json={"file_path": str(input_path), "resume": False},
        )
        assert response.status_code == 200

    response = client.post(
        "/process",
        json={"file_path": str(input_path), "resume": False},
    )
    assert response.status_code == 429


def test_process_missing_file(tmp_path):
    client = TestClient(web_api.app)
    missing_path = tmp_path / "missing.txt"
    response = client.post(
        "/process",
        json={"file_path": str(missing_path), "resume": False},
    )
    assert response.status_code == 404


def test_estimate_success(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    client = TestClient(web_api.app)
    response = client.get("/estimate", params={"file_path": str(input_path)})
    assert response.status_code == 200
    data = response.json()
    assert "total_tokens" in data
    assert "chunk_count" in data


def test_estimate_missing_file(tmp_path):
    client = TestClient(web_api.app)
    missing_path = tmp_path / "missing.txt"
    response = client.get("/estimate", params={"file_path": str(missing_path)})
    assert response.status_code == 404


def test_job_not_found():
    client = TestClient(web_api.app)
    response = client.get("/jobs/missing")
    assert response.status_code == 404


# ============ Queue Related Tests ============


def test_queue_add_success(tmp_path):
    """测试单个文件添加到队列成功"""
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add",
        json={"file_path": str(input_path), "resume": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["message"] == "任务已添加到队列"


def test_queue_add_missing_file(tmp_path):
    """测试添加不存在的文件"""
    missing_path = tmp_path / "missing.txt"
    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add",
        json={"file_path": str(missing_path), "resume": False},
    )
    assert response.status_code == 404
    assert "不存在" in response.json()["detail"]


def test_queue_add_empty_path():
    """测试添加空文件路径"""
    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add",
        json={"file_path": "", "resume": False},
    )
    assert response.status_code == 400
    assert "不能为空" in response.json()["detail"]


def test_queue_add_multiple_success(tmp_path):
    """测试批量添加文件到队列成功"""
    input_path1 = tmp_path / "input1.txt"
    input_path2 = tmp_path / "input2.txt"
    input_path3 = tmp_path / "input3.txt"

    input_path1.write_text("hello1", encoding="utf-8")
    input_path2.write_text("hello2", encoding="utf-8")
    input_path3.write_text("hello3", encoding="utf-8")

    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add-multiple",
        json={"file_paths": [str(input_path1), str(input_path2), str(input_path3)]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_ids" in data
    assert len(data["task_ids"]) == 3
    assert data["count"] == 3
    assert "已将 3 个文件添加到队列" == data["message"]


def test_queue_add_multiple_empty_list():
    """测试批量添加 - 空列表"""
    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add-multiple",
        json={"file_paths": []},
    )
    assert response.status_code == 400
    assert "不能为空" in response.json()["detail"]


def test_queue_add_multiple_missing_file(tmp_path):
    """测试批量添加 - 文件不存在"""
    input_path1 = tmp_path / "input1.txt"
    input_path1.write_text("hello1", encoding="utf-8")

    missing_path = tmp_path / "missing.txt"

    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add-multiple",
        json={"file_paths": [str(input_path1), str(missing_path)]},
    )
    assert response.status_code == 404
    assert "不存在" in response.json()["detail"]


def test_queue_add_multiple_empty_string(tmp_path):
    """测试批量添加 - 空字符串路径"""
    client = TestClient(web_api.app)
    response = client.post(
        "/queue/add-multiple",
        json={"file_paths": ["", "file2.txt"]},
    )
    # 空字符串可能导致不同的验证错误，所以只检查不是200
    assert response.status_code != 200


def test_queue_list(tmp_path):
    """测试列出队列"""
    client = TestClient(web_api.app)
    response = client.get("/queue/list")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert isinstance(data["tasks"], list)


def test_queue_stats():
    """测试获取队列统计"""
    client = TestClient(web_api.app)
    response = client.get("/queue/stats")
    assert response.status_code == 200
    data = response.json()
    assert "pending" in data
    assert "running" in data
    assert "total" in data
    assert data["pending"] >= 0
    assert data["running"] >= 0


def test_queue_clear(tmp_path):
    """测试清空队列"""
    input_path1 = tmp_path / "input1.txt"
    input_path2 = tmp_path / "input2.txt"
    input_path1.write_text("hello1", encoding="utf-8")
    input_path2.write_text("hello2", encoding="utf-8")

    client = TestClient(web_api.app)

    # 先添加几个任务
    client.post("/queue/add", json={"file_path": str(input_path1), "resume": False})
    client.post("/queue/add", json={"file_path": str(input_path2), "resume": False})

    # 清空队列
    response = client.post("/queue/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "cancelled_count" in data
    assert data["cancelled_count"] >= 0


def test_load_cors_origins_normalization(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS",
        " http://a.com , file://,http://a.com,,null ",
    )
    origins = web_api._load_cors_origins()
    assert origins == ["http://a.com", "null"]


def test_load_env_file_and_mask(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test-key\n#comment\nINVALID_LINE\nOTHER=value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_api, "ENV_PATH", env_file)

    data = web_api.load_env_file()
    assert data["OPENAI_API_KEY"] == "sk-test-key"
    assert data["OTHER"] == "value"
    assert web_api.mask_value("OPENAI_API_KEY", "sk-test-key").startswith("*")
    assert web_api.mask_value("OTHER", "value") == "value"


def test_rate_limiter_window_cleanup(monkeypatch):
    limiter = web_api.RateLimiter()
    sequence = iter([100.0, 101.0, 102.0, 200.0])
    monkeypatch.setattr(web_api.time, "time", lambda: next(sequence))
    limiter.check_rate_limit("127.0.0.1", max_requests=2, window_seconds=10)
    limiter.check_rate_limit("127.0.0.1", max_requests=2, window_seconds=10)
    with pytest.raises(HTTPException) as exc:
        limiter.check_rate_limit("127.0.0.1", max_requests=2, window_seconds=10)
    assert exc.value.status_code == 429
    # 时间窗口过去后应恢复可用
    limiter.check_rate_limit("127.0.0.1", max_requests=2, window_seconds=10)


def test_job_log_bounded():
    job = web_api.Job(id="job-1")
    for i in range(205):
        job.log(f"log-{i}")
    assert len(job.logs) == 200
    assert job.log_offset == 5
    assert job.logs[0] == "log-5"


def test_update_progress_from_info_token_logged_once():
    target = web_api.Job(id="job-2")
    info = {
        "progress": 0.2,
        "completed_chunks": 1,
        "last_chunk_id": 1,
        "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    web_api._update_progress_from_info(info, target)
    assert target.progress == 0.2
    assert target.result["completed_chunks"] == 1
    first_log_count = len(target.logs)
    web_api._update_progress_from_info(info, target)
    assert len(target.logs) == first_log_count + 1  # 只增加块完成日志，token日志不重复


def test_cleanup_expired_jobs(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(web_api.time, "time", lambda: now)
    old_job = web_api.Job(id="old", created_at=now - web_api.JOB_MAX_AGE_HOURS * 3600 - 1)
    new_job = web_api.Job(id="new", created_at=now)
    web_api.job_manager.jobs["old"] = old_job
    web_api.job_manager.jobs["new"] = new_job
    web_api.cleanup_expired_jobs()
    assert "old" not in web_api.job_manager.jobs
    assert "new" in web_api.job_manager.jobs


def test_cleanup_excess_jobs_prioritizes_finished(monkeypatch):
    monkeypatch.setattr(web_api, "MAX_JOBS", 2)
    web_api.job_manager.jobs.clear()
    web_api.job_manager.jobs["s1"] = web_api.Job(id="s1", status="success", created_at=1.0)
    web_api.job_manager.jobs["e1"] = web_api.Job(id="e1", status="error", created_at=2.0)
    web_api.job_manager.jobs["p1"] = web_api.Job(id="p1", status="pending", created_at=3.0)
    web_api.job_manager.jobs["r1"] = web_api.Job(id="r1", status="running", created_at=4.0)
    web_api.cleanup_excess_jobs()
    assert len(web_api.job_manager.jobs) == 2
    assert "p1" in web_api.job_manager.jobs or "r1" in web_api.job_manager.jobs


def test_resolve_upload_path_security(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(web_api, "_UPLOAD_ROOT", upload_root.resolve())

    inside_file = upload_root / "a.txt"
    inside_file.write_text("x", encoding="utf-8")
    outside_file = tmp_path / "b.txt"
    outside_file.write_text("x", encoding="utf-8")

    assert web_api._resolve_upload_path(str(inside_file)) == inside_file.resolve()
    assert web_api._resolve_upload_path(str(upload_root)) is None
    assert web_api._resolve_upload_path(str(outside_file)) is None


def test_cleanup_uploads_with_protected_paths(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(web_api, "UPLOAD_DIR", upload_root)
    monkeypatch.setattr(web_api, "_UPLOAD_ROOT", upload_root.resolve())

    keep_file = upload_root / "keep.txt"
    keep_file.write_text("keep", encoding="utf-8")
    remove_file = upload_root / "remove.txt"
    remove_file.write_text("remove", encoding="utf-8")
    remove_dir = upload_root / "folder"
    remove_dir.mkdir()
    (remove_dir / "x.txt").write_text("x", encoding="utf-8")

    cleaned = web_api.cleanup_uploads(protected_paths={keep_file})
    assert cleaned == 2
    assert keep_file.exists()
    assert not remove_file.exists()
    assert not remove_dir.exists()


@pytest.mark.asyncio
async def test_upload_file_empty_filename():
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    upload = SimpleNamespace(
        filename=None, content_type="text/plain", read=AsyncMock(return_value=b"x")
    )
    with pytest.raises(HTTPException) as exc:
        await web_api.upload_file(request, upload)
    assert exc.value.status_code == 400
    assert "文件名不能为空" in exc.value.detail


def test_upload_multiple_invalid_extension():
    client = TestClient(web_api.app)
    response = client.post(
        "/upload-multiple",
        files=[
            ("files", ("ok.txt", b"a", "text/plain")),
            ("files", ("bad.pdf", b"b", "text/plain")),
        ],
    )
    assert response.status_code == 400


def test_queue_cancel_not_found(monkeypatch):
    async def fake_cancel_task(task_id: str) -> bool:
        return False

    queue = SimpleNamespace(cancel_task=fake_cancel_task)
    monkeypatch.setattr(web_api, "get_global_queue", lambda: queue)
    client = TestClient(web_api.app)
    response = client.post("/queue/cancel", params={"task_id": "not-exist"})
    assert response.status_code == 404


def test_force_complete_not_found(monkeypatch):
    async def fake_force_complete_task(task_id: str) -> bool:
        return False

    queue = SimpleNamespace(force_complete_task=fake_force_complete_task)
    monkeypatch.setattr(web_api, "get_global_queue", lambda: queue)
    client = TestClient(web_api.app)
    response = client.post("/queue/force-complete/not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_job_success_with_cleanup_log(monkeypatch, tmp_path):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event
            self.force_complete = False

        async def process_novel(self, file_path: str, resume: bool = True):
            if self.progress_callback:
                self.progress_callback(
                    {
                        "progress": 0.5,
                        "last_chunk_id": 1,
                        "token_usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                    }
                )
            return {
                "success": True,
                "token_usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15},
            }

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)
    monkeypatch.setattr(web_api, "cleanup_uploads", lambda protected_paths=None: 1)

    upload_file = web_api.UPLOAD_DIR / "current.txt"
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.write_text("x", encoding="utf-8")

    job = web_api.Job(id="job-success")
    web_api.JOBS[job.id] = job
    req = web_api.ProcessRequest(file_path=str(upload_file), resume=False)
    await web_api._run_job(job, req)

    assert job.status == "success"
    assert job.progress == 1.0
    assert "token_usage" in job.result
    assert any("已清理上传文件 1 个" in line for line in job.logs)


@pytest.mark.asyncio
async def test_run_job_cleanup_exception_logged(monkeypatch):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback

        async def process_novel(self, file_path: str, resume: bool = True):
            return {"success": True}

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)

    def _raise_cleanup(protected_paths=None):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(web_api, "cleanup_uploads", _raise_cleanup)

    upload_file = web_api.UPLOAD_DIR / "current.txt"
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.write_text("x", encoding="utf-8")

    job = web_api.Job(id="job-cleanup-fail")
    req = web_api.ProcessRequest(file_path=str(upload_file), resume=False)
    await web_api._run_job(job, req)

    assert job.status == "success"
    assert any("清理上传文件失败" in line for line in job.logs)


@pytest.mark.asyncio
async def test_run_job_error_path(monkeypatch):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback

        async def process_novel(self, file_path: str, resume: bool = True):
            raise RuntimeError("boom")

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)
    job = web_api.Job(id="job-error")
    req = web_api.ProcessRequest(file_path="no-matter.txt", resume=True)
    await web_api._run_job(job, req)

    assert job.status == "error"
    assert job.message == "boom"
    assert any("错误: boom" in line for line in job.logs)


@pytest.mark.asyncio
async def test_run_queue_task_success(monkeypatch):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event
            self.force_complete = False

        async def process_novel(self, file_path: str, resume: bool = True):
            if self.progress_callback:
                self.progress_callback({"progress": 0.8, "last_chunk_id": 1})
            return {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)
    task = QueueTask(id="task-ok", file_path="demo.txt")
    await web_api.run_queue_task(task)
    assert task.status == "success"
    assert task.progress == 1.0
    assert "token_usage" in task.result


@pytest.mark.asyncio
async def test_run_queue_task_cancelled_without_force(monkeypatch):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event
            self.force_complete = False

        async def process_novel(self, file_path: str, resume: bool = True):
            raise asyncio.CancelledError()

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)
    task = QueueTask(id="task-cancel", file_path="demo.txt")
    await web_api.run_queue_task(task)
    assert task.status == "cancelled"
    assert "任务被取消" in task.message


@pytest.mark.asyncio
async def test_run_queue_task_cancelled_force_complete(monkeypatch):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event
            self.force_complete = False

        async def process_novel(self, file_path: str, resume: bool = True):
            raise asyncio.CancelledError()

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)
    task = QueueTask(id="task-force", file_path="demo.txt")
    task.should_force_complete = True
    task.result["outlines"] = [{"chunk_id": 1}]
    await web_api.run_queue_task(task)
    assert task.status == "success"
    assert "强制完成" in task.message


@pytest.mark.asyncio
async def test_run_queue_task_error(monkeypatch):
    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event
            self.force_complete = False

        async def process_novel(self, file_path: str, resume: bool = True):
            raise RuntimeError("queue boom")

    monkeypatch.setattr(web_api, "NovelProcessingService", FakeService)
    task = QueueTask(id="task-err", file_path="demo.txt")
    await web_api.run_queue_task(task)
    assert task.status == "error"
    assert task.message == "queue boom"
