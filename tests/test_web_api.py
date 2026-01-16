import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web_api


@pytest.fixture(autouse=True)
def isolate_web_api_state(monkeypatch, tmp_path):
    web_api.JOBS.clear()
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
    async def fake_run_job(job, req):
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
    async def fake_run_job(job, req):
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
