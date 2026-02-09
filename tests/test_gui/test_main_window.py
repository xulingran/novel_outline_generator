"""测试 GUI 主窗口逻辑。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gui.main_window import MainWindow


class _FakeWorker:
    def __init__(self, coro, **_kwargs):
        self.coro = coro
        self.started = False
        self._alive = False

    def start(self):
        self.started = True
        self._alive = True
        self.coro.close()
        self._alive = False

    def stop(self):
        self._alive = False
        return None

    def is_alive(self):
        return self._alive


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr("gui.main_window.AsyncWorker", _FakeWorker)
    return MainWindow()


def test_resume_prompt_when_progress_exists(window, monkeypatch):
    window.current_file = window.current_file or Path("novel.txt")

    progress_service = MagicMock()
    progress_service.load_progress.return_value = SimpleNamespace(txt_file="novel.txt")
    monkeypatch.setattr("services.progress_service.ProgressService", lambda: progress_service)

    ask = MagicMock(return_value=True)
    monkeypatch.setattr(window, "_confirm_resume_dialog", ask)

    assert window._ask_resume_preference() is True
    ask.assert_called_once()


def test_no_resume_prompt_for_different_file(window, monkeypatch):
    window.current_file = Path("a.txt")

    progress_service = MagicMock()
    progress_service.load_progress.return_value = SimpleNamespace(txt_file="b.txt")
    monkeypatch.setattr("services.progress_service.ProgressService", lambda: progress_service)

    ask = MagicMock(return_value=True)
    monkeypatch.setattr(window, "_confirm_resume_dialog", ask)

    assert window._ask_resume_preference() is False
    ask.assert_not_called()


def test_on_start_processing_uses_resume_flag(window, monkeypatch, tmp_path):
    test_file = tmp_path / "novel.txt"
    test_file.write_text("content", encoding="utf-8")
    window.current_file = test_file

    monkeypatch.setattr(window, "_ask_resume_preference", lambda: True)

    captured = {}

    async def _dummy_coro():
        return {"ok": True}

    def _build(resume: bool):
        captured["resume"] = resume
        return _dummy_coro()

    monkeypatch.setattr(window, "_build_processing_coro", _build)
    window.on_start_processing()

    assert captured["resume"] is True
    assert isinstance(window.async_worker, _FakeWorker)
    assert window.async_worker.started is True


@pytest.mark.asyncio
async def test_build_processing_coro_runs_service_and_updates_progress(
    window, monkeypatch, tmp_path
):
    test_file = tmp_path / "novel.txt"
    test_file.write_text("content", encoding="utf-8")
    window.current_file = test_file
    window.cancel_event = __import__("asyncio").Event()

    captured: dict[str, object] = {}
    monkeypatch.setattr(window, "_is_window_alive", lambda: False)

    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            captured["progress_callback"] = progress_callback
            captured["cancel_event"] = cancel_event

        async def process_novel(self, file_path: str, resume: bool):
            captured["file_path"] = file_path
            captured["resume"] = resume
            progress_callback = captured["progress_callback"]
            if callable(progress_callback):
                progress_callback(
                    {
                        "completed_chunks": 2,
                        "total_chunks": 4,
                        "failed_chunks": 0,
                        "partial_chunks": 0,
                        "phase": "processing",
                    }
                )
            return {"output_dir": "outputs"}

    monkeypatch.setattr("services.novel_processing_service.NovelProcessingService", FakeService)

    result = await window._build_processing_coro(resume=True)
    window._drain_ui_events()
    assert result["output_dir"] == "outputs"
    assert captured["file_path"] == str(test_file)
    assert captured["resume"] is True
    assert window.progress_bar_widget.completed_chunks == 2
    assert window.progress_bar_widget.total_chunks == 4
    assert window.progress_bar_widget.current_phase == "processing"


@pytest.mark.asyncio
async def test_build_processing_coro_propagates_service_errors(window, monkeypatch, tmp_path):
    test_file = tmp_path / "novel.txt"
    test_file.write_text("content", encoding="utf-8")
    window.current_file = test_file
    window.cancel_event = __import__("asyncio").Event()

    class FakeService:
        def __init__(self, progress_callback=None, cancel_event=None):
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event

        async def process_novel(self, file_path: str, resume: bool):
            raise RuntimeError("service failed")

    monkeypatch.setattr("services.novel_processing_service.NovelProcessingService", FakeService)

    with pytest.raises(RuntimeError, match="service failed"):
        await window._build_processing_coro(resume=False)


def test_on_cancel_processing_sets_event_and_stops_worker(window, monkeypatch):
    window.current_file = Path("novel.txt")
    cancel_event = __import__("asyncio").Event()
    window.cancel_event = cancel_event
    window.async_worker = MagicMock()
    window.async_worker.is_alive.side_effect = [True, False]
    window.start_button.configure(state="disabled")
    window.cancel_button.configure(state="normal")

    window.on_cancel_processing()
    assert cancel_event.is_set() is True

    assert window.async_worker is None
    assert window.start_button.cget("state") == "normal"
    assert window.cancel_button.cget("state") == "disabled"


def test_on_start_processing_ignores_when_worker_alive(window, monkeypatch, tmp_path):
    test_file = tmp_path / "novel.txt"
    test_file.write_text("content", encoding="utf-8")
    window.current_file = test_file

    class BusyWorker:
        def is_alive(self):
            return True

    window.async_worker = BusyWorker()

    called = {"build": False}

    async def _dummy_coro():
        return {"ok": True}

    def _build(_resume: bool):
        called["build"] = True
        return _dummy_coro()

    monkeypatch.setattr(window, "_build_processing_coro", _build)
    window.on_start_processing()

    assert called["build"] is False


def test_async_callbacks_enqueue_and_dispatch(window, monkeypatch):
    complete = MagicMock()
    error = MagicMock()
    progress = MagicMock()
    monkeypatch.setattr(window, "_is_window_alive", lambda: False)
    monkeypatch.setattr(window, "_do_progress_update", progress)
    monkeypatch.setattr(window, "_do_processing_complete", complete)
    monkeypatch.setattr(window, "_do_processing_error", error)

    window._on_progress_update({"completed_chunks": 1, "total_chunks": 2})
    window._on_processing_complete({"output_dir": "outputs"})
    window._on_processing_error(RuntimeError("boom"))

    window._drain_ui_events()
    progress.assert_called_once()
    complete.assert_called_once()
    error.assert_called_once()
