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
    assert "masked" in data


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
