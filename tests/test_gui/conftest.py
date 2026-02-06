"""GUI 测试共用 fixtures。"""

import asyncio
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest


class _BaseWidget:
    def __init__(self, *args, **kwargs):
        self._config = dict(kwargs)
        self._after_jobs: dict[str, tuple[int, object]] = {}

    def pack(self, *args, **kwargs):
        return None

    def pack_forget(self):
        return None

    def configure(self, **kwargs):
        self._config.update(kwargs)

    def cget(self, key):
        return self._config.get(key)

    def after(self, _delay, callback):
        job_id = f"job-{len(self._after_jobs) + 1}"
        self._after_jobs[job_id] = (_delay, callback)
        return job_id

    def after_cancel(self, job_id):
        self._after_jobs.pop(job_id, None)


class MockCTkModule(types.ModuleType):
    class CTk(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._title = ""
            self._geometry = ""
            self._protocols = {}

        def title(self, text=None):
            if text is not None:
                self._title = text
            return self._title

        def geometry(self, text=None):
            if text is not None:
                self._geometry = text
            return self._geometry

        def protocol(self, name, callback):
            self._protocols[name] = callback

        def mainloop(self):
            return None

        def grab_set(self):
            return None

        def quit(self):
            return None

        def destroy(self):
            return None

    class CTkToplevel(CTk):
        pass

    class CTkFrame(_BaseWidget):
        pass

    class CTkScrollableFrame(CTkFrame):
        pass

    class CTkTabview(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tabs = {}

        def add(self, name):
            self._tabs[name] = MockCTkModule.CTkFrame()

        def tab(self, name):
            return self._tabs[name]

    class CTkLabel(_BaseWidget):
        pass

    class CTkButton(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.command = kwargs.get("command")

    class CTkEntry(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.textvariable = kwargs.get("textvariable")

        def get(self):
            return self.textvariable.get() if self.textvariable else ""

    class CTkProgressBar(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._value = 0.0

        def set(self, value):
            self._value = value

        def get(self):
            return self._value

        def start(self):
            return None

        def stop(self):
            return None

    class CTkOptionMenu(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.variable = kwargs.get("variable")

        def set(self, value):
            if self.variable:
                self.variable.set(value)

    class CTkCheckBox(_BaseWidget):
        pass

    class CTkTextbox(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.content = ""

        def insert(self, _index, text):
            self.content += text

        def delete(self, _start, _end):
            self.content = ""

        def get(self, _start, _end):
            return self.content

        def see(self, _index):
            return None

    class CTkFont:
        def __init__(self, *args, **kwargs):
            self.size = kwargs.get("size", 12)

    class StringVar:
        def __init__(self, value=""):
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value

    class BooleanVar(StringVar):
        def __init__(self, value=False):
            super().__init__(value)

        def get(self):
            return bool(self._value)

    @staticmethod
    def set_appearance_mode(_mode):
        return None

    @staticmethod
    def set_default_color_theme(_theme):
        return None


ctk = MockCTkModule("customtkinter")
sys.modules.setdefault("customtkinter", ctk)


@pytest.fixture
def skip_if_no_gui():
    """兼容旧测试接口。"""
    return None


@pytest.fixture
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """提供独立事件循环。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """创建临时日志文件。"""
    log_file = tmp_path / "test.log"
    log_file.write_text(
        "2025-01-31 10:00:00 - test - INFO - Test message\n"
        "2025-01-31 10:00:01 - test - ERROR - Error message\n",
        encoding="utf-8",
    )
    return log_file


@pytest.fixture
def temp_test_file(tmp_path: Path) -> Path:
    """创建临时文本文件。"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("第一章\n这是一段测试文本。\n", encoding="utf-8")
    return file_path


def create_mock_progress_data() -> dict[str, object]:
    """构造模拟进度数据。"""
    return {
        "total_chunks": 10,
        "completed_chunks": 5,
        "failed_chunks": 1,
        "partial_chunks": 1,
        "phase": "processing",
        "eta_seconds": 60,
        "eta_confidence": 0.8,
    }
