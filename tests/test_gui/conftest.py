"""
GUI 测试配置和 fixtures

提供 GUI 测试所需的共享 fixtures 和工具函数。
"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator, Iterator

import pytest
from pytest_mock import MockerFixture


# 创建模拟的 customtkinter 模块（用于没有 tkinter 的环境）
class MockCTk:
    """Mock CustomTkinter for testing without GUI"""

    class CTk:
        def __init__(self, *args, **kwargs):
            self._title = ""
            self._geometry = ""

        def mainloop(self):
            pass

        def quit(self):
            pass

        def title(self, title: str = None):
            if title is not None:
                self._title = title
            return self._title

        def geometry(self, geometry: str = None):
            if geometry is not None:
                self._geometry = geometry
            return self._geometry

        def grab_set(self):
            pass

        def pack(self, *args, **kwargs):
            """Add pack method to base CTk class for all widgets"""
            pass

        def pack_forget(self):
            pass

    class CTkToplevel(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.title = ""

        def destroy(self):
            pass

    class CTkFrame(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.children = []

        def pack(self, *args, **kwargs):
            pass

        def pack_forget(self):
            pass

    class CTkScrollableFrame(CTkFrame):
        pass

    class CTkTabview(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tabs = {}

        def add(self, name: str):
            self.tabs[name] = None

        def tab(self, name: str):
            return self.tabs.get(name)

    class CTkLabel(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.text = kwargs.get("text", "")

        def configure(self, **kwargs):
            if "text" in kwargs:
                self.text = kwargs["text"]

    class CTkButton(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.command = kwargs.get("command", None)
            self.state = kwargs.get("state", "normal")

        def configure(self, **kwargs):
            if "state" in kwargs:
                self.state = kwargs["state"]

    class CTkEntry(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.textvariable = kwargs.get("textvariable", None)

        def get(self):
            return self.textvariable.get() if self.textvariable else ""

    class CTkProgressBar(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._value = 0.0
            self.mode = "determinate"

        def set(self, value: float):
            self._value = value

        def get(self):
            return self._value

        def configure(self, **kwargs):
            if "mode" in kwargs:
                self.mode = kwargs["mode"]

        def start(self):
            pass

        def stop(self):
            pass

    class CTkOptionMenu(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.values = kwargs.get("values", [])
            self.variable = kwargs.get("variable", None)

        def set(self, value: str):
            if self.variable:
                self.variable.set(value)

    class CTkCheckBox(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.variable = kwargs.get("variable", None)

    class CTkTextbox(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.content = ""

        def insert(self, index: str, text: str):
            self.content += text

        def delete(self, start: str, end: str):
            self.content = ""

        def get(self, start: str, end: str):
            return self.content

        def see(self, index: str):
            pass

        def index(self, index: str):
            return "1.0"

    class CTkFont:
        def __init__(self, *args, **kwargs):
            self.size = kwargs.get("size", 12)
            self.weight = kwargs.get("weight", "normal")

    @staticmethod
    def set_appearance_mode(mode: str):
        pass

    @staticmethod
    def set_default_color_theme(theme: str):
        pass

    class StringVar:
        def __init__(self, value: str = ""):
            self._value = value

        def set(self, value: str):
            self._value = value

        def get(self):
            return self._value

    class IntVar(StringVar):
        def __init__(self, value: int = 0):
            super().__init__(str(value))

        def get(self):
            return int(self._value)

    class DoubleVar(StringVar):
        def __init__(self, value: float = 0.0):
            super().__init__(str(value))

        def get(self):
            return float(self._value)

    class BooleanVar(StringVar):
        def __init__(self, value: bool = False):
            super().__init__(str(value))

        def get(self):
            return self._value.lower() == "true"

        def set(self, value: bool):
            self._value = str(value)


# 检测 GUI 是否可用
GUI_AVAILABLE = True

try:
    import tkinter
    # 测试 tkinter 是否真正可用
    test_tk = tkinter.Tk()
    test_tk.destroy()
    # 尝试导入 customtkinter
    import customtkinter as ctk
except Exception:
    GUI_AVAILABLE = False
    # 如果不可用，将模拟模块添加到 sys.modules
    sys.modules["customtkinter"] = MockCTk
    ctk = MockCTk


@pytest.fixture
def skip_if_no_gui():
    """如果 GUI 不可用则跳过测试"""
    if not GUI_AVAILABLE:
        pytest.skip("GUI (customtkinter/tkinter) not available")


@pytest.fixture
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """创建事件循环用于异步测试"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """创建临时日志文件"""
    log_file = tmp_path / "test.log"
    log_file.write_text(
        "2025-01-31 10:00:00 - test - INFO - Test message 1\n"
        "2025-01-31 10:00:01 - test - DEBUG - Debug message\n"
        "2025-01-31 10:00:02 - test - WARNING - Warning message\n"
        "2025-01-31 10:00:03 - test - ERROR - Error message\n"
        "2025-01-31 10:00:04 - test - INFO - Test message 2\n",
        encoding="utf-8",
    )
    return log_file


@pytest.fixture
def temp_test_file(tmp_path: Path) -> Path:
    """创建临时测试文件"""
    test_file = tmp_path / "test_novel.txt"
    test_file.write_text(
        "第一章 开始\n这是第一章的内容。\n\n"
        "第二章 发展\n这是第二章的内容。\n\n"
        "第三章 高潮\n这是第三章的内容。\n\n"
        "第四章 结局\n这是第四章的内容。\n",
        encoding="utf-8",
    )
    return test_file


def create_mock_progress_data() -> dict:
    """创建模拟的进度数据"""
    return {
        "total_chunks": 10,
        "completed_chunks": 5,
        "failed_chunks": 1,
        "partial_chunks": 0,
        "phase": "处理中",
        "progress": 0.5,
        "eta_seconds": 120,
        "eta_confidence": 0.8,
    }
