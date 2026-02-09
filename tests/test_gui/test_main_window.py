"""
主窗口回归测试。
"""

from gui.main_window import MainWindow


class _DummyButton:
    def __init__(self):
        self.state = "normal"

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]


class _DummyProcessPage:
    def __init__(self):
        self._current_file = "demo.txt"
        self._start_button = _DummyButton()
        self._cancel_button = _DummyButton()
        self.logs: list[str] = []

    def append_log(self, message: str):
        self.logs.append(message)


class _DummyCancelEvent:
    def __init__(self):
        self.set_called = False

    def set(self):
        self.set_called = True


class _DummyWorker:
    def __init__(self):
        self.stop_called = False

    def is_alive(self):
        return True

    def stop(self):
        self.stop_called = True


class TestMainWindowCancel:
    """测试取消处理逻辑。"""

    def test_cancel_does_not_force_stop_event_loop(self):
        """取消时不应调用 worker.stop。"""
        window = object.__new__(MainWindow)
        page = _DummyProcessPage()
        event = _DummyCancelEvent()
        worker = _DummyWorker()

        window._cancel_requested = False
        window._cancel_event = event
        window._async_worker = worker
        window.get_process_page = lambda: page

        window._on_cancel_processing()

        assert window._cancel_requested is True
        assert event.set_called is True
        assert worker.stop_called is False
        assert page._cancel_button.state == "disabled"

    def test_finish_processing_treats_cancel_as_non_error(self):
        """取消收尾不应走失败分支。"""
        window = object.__new__(MainWindow)
        page = _DummyProcessPage()

        window._cancel_requested = True
        window._cancel_event = _DummyCancelEvent()
        window._async_worker = _DummyWorker()
        window.get_process_page = lambda: page

        window._finish_processing(result=None, error=RuntimeError("should be ignored"))

        assert window._cancel_requested is False
        assert "处理已取消" in page.logs
        assert page._start_button.state == "normal"
        assert page._cancel_button.state == "disabled"


class _MockProcessPageWithProgress:
    """Mock ProcessPage for testing progress updates."""

    def __init__(self):
        self.updates = []

    def update_progress(self, **kwargs):
        self.updates.append(kwargs)


class TestMainWindowMergeProgress:
    """测试主窗口合并进度参数传递。"""

    def test_progress_callback_passes_merge_params(self):
        """进度回调应传递合并相关参数到 ProcessPage。"""
        window = object.__new__(MainWindow)
        page = _MockProcessPageWithProgress()
        window.get_process_page = lambda: page

        # 模拟合并阶段的进度数据
        progress_data = {
            "completed_chunks": 100,
            "total_chunks": 100,
            "failed_chunks": 0,
            "partial_chunks": 0,
            "phase": "merging",
            "eta_seconds": 0,
            "merge_level": 2,
            "merge_batch_current": 1,
            "merge_batch_total": 3,
            "merge_outlines_count": 34,
        }

        window._apply_progress_update(progress_data)

        # 验证合并参数已传递
        assert len(page.updates) == 1
        update = page.updates[0]
        assert update["merge_level"] == 2
        assert update["merge_batch_current"] == 1
        assert update["merge_batch_total"] == 3
        assert update["merge_outlines_count"] == 34
