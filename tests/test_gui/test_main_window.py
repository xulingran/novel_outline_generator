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
