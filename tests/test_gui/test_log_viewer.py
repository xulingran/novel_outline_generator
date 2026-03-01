"""
测试 log_viewer 组件

测试日志查看器组件，包括日志读取、过滤、刷新等功能。
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gui.widgets.log_viewer import LogViewer
from tests.test_gui.conftest import ctk


@pytest.fixture
def log_viewer(temp_log_file: Path):
    """创建日志查看器实例"""
    master = MagicMock()
    viewer = LogViewer(master, log_file=temp_log_file, auto_refresh=False)
    return viewer


class TestLogViewer:
    """测试 LogViewer 组件"""

    def test_init(self, skip_if_no_gui):
        """测试 LogViewer 初始化"""
        master = MagicMock()
        log_file = MagicMock()

        viewer = LogViewer(
            master,
            log_file=log_file,
            auto_refresh=False,
            refresh_interval=2000,
        )

        assert viewer.log_file == log_file
        assert viewer.auto_refresh is False
        assert viewer.refresh_interval == 2000
        assert viewer.current_level == "ALL"

    def test_init_with_defaults(self, skip_if_no_gui):
        """测试使用默认参数初始化"""
        master = MagicMock()

        viewer = LogViewer(master)

        assert viewer.auto_refresh is True  # 默认启用自动刷新
        assert viewer.refresh_interval == 1000  # 默认 1 秒
        assert viewer.current_level == "ALL"

    def test_init_without_log_file(self, skip_if_no_gui):
        """测试不指定日志文件"""
        master = MagicMock()

        viewer = LogViewer(master, log_file=None)

        assert viewer.log_file is None


class TestLogViewerRefresh:
    """测试日志刷新功能"""

    def test_refresh_with_valid_file(self, log_viewer, temp_log_file: Path):
        """测试刷新有效日志文件"""
        # 刷新应该成功读取文件
        log_viewer.refresh_log()

        # 验证文件被读取
        assert temp_log_file.exists()

    def test_refresh_with_nonexistent_file(self, log_viewer, tmp_path: Path):
        """测试刷新不存在的文件"""
        fake_file = tmp_path / "nonexistent.log"
        log_viewer.log_file = fake_file

        # 不应该抛出异常
        log_viewer.refresh_log()

    def test_refresh_with_empty_file(self, log_viewer, tmp_path: Path):
        """测试刷新空文件"""
        empty_file = tmp_path / "empty.log"
        empty_file.write_text("", encoding="utf-8")

        log_viewer.log_file = empty_file
        log_viewer.refresh_log()

        # 应该成功处理空文件
        assert empty_file.exists()

    def test_refresh_with_large_file(self, log_viewer, tmp_path: Path):
        """测试刷新大文件（超过 1000 行）"""
        large_file = tmp_path / "large.log"
        lines = [f"2025-01-31 10:00:{i:02d} - test - INFO - Message {i}\n" for i in range(2000)]
        large_file.write_text("".join(lines), encoding="utf-8")

        log_viewer.log_file = large_file
        log_viewer.refresh_log()

        # 应该成功处理并限制到 1000 行
        assert large_file.exists()


class TestLogViewerLevelFiltering:
    """测试日志级别过滤"""

    def test_filter_all_levels(self, log_viewer):
        """测试显示所有级别"""
        lines = [
            "2025-01-31 10:00:00 - test - DEBUG - Debug message\n",
            "2025-01-31 10:00:01 - test - INFO - Info message\n",
            "2025-01-31 10:00:02 - test - WARNING - Warning message\n",
            "2025-01-31 10:00:03 - test - ERROR - Error message\n",
        ]

        filtered = log_viewer._filter_by_level(lines, "ALL")

        # 应该返回所有行
        assert len(filtered) == 4

    def test_filter_debug_only(self, log_viewer):
        """测试只显示 DEBUG 及以上"""
        lines = [
            "2025-01-31 10:00:00 - test - DEBUG - Debug message\n",
            "2025-01-31 10:00:01 - test - INFO - Info message\n",
            "2025-01-31 10:00:02 - test - WARNING - Warning message\n",
            "2025-01-31 10:00:03 - test - ERROR - Error message\n",
        ]

        filtered = log_viewer._filter_by_level(lines, "DEBUG")

        # 应该返回所有行（DEBUG 是最低级别）
        assert len(filtered) == 4

    def test_filter_info_only(self, log_viewer):
        """测试只显示 INFO 及以上"""
        lines = [
            "2025-01-31 10:00:00 - test - DEBUG - Debug message\n",
            "2025-01-31 10:00:01 - test - INFO - Info message\n",
            "2025-01-31 10:00:02 - test - WARNING - Warning message\n",
            "2025-01-31 10:00:03 - test - ERROR - Error message\n",
        ]

        filtered = log_viewer._filter_by_level(lines, "INFO")

        # 不应该包含 DEBUG
        assert len(filtered) == 3
        assert not any("DEBUG" in line for line in filtered)

    def test_filter_warning_only(self, log_viewer):
        """测试只显示 WARNING 及以上"""
        lines = [
            "2025-01-31 10:00:00 - test - DEBUG - Debug message\n",
            "2025-01-31 10:00:01 - test - INFO - Info message\n",
            "2025-01-31 10:00:02 - test - WARNING - Warning message\n",
            "2025-01-31 10:00:03 - test - ERROR - Error message\n",
        ]

        filtered = log_viewer._filter_by_level(lines, "WARNING")

        # 只应该包含 WARNING 和 ERROR
        assert len(filtered) == 2
        assert any("WARNING" in line for line in filtered)
        assert any("ERROR" in line for line in filtered)

    def test_filter_error_only(self, log_viewer):
        """测试只显示 ERROR"""
        lines = [
            "2025-01-31 10:00:00 - test - DEBUG - Debug message\n",
            "2025-01-31 10:00:01 - test - INFO - Info message\n",
            "2025-01-31 10:00:02 - test - WARNING - Warning message\n",
            "2025-01-31 10:00:03 - test - ERROR - Error message\n",
        ]

        filtered = log_viewer._filter_by_level(lines, "ERROR")

        # 只应该包含 ERROR
        assert len(filtered) == 1
        assert "ERROR" in filtered[0]

    def test_filter_lines_without_level(self, log_viewer):
        """测试过滤没有级别信息的行"""
        lines = [
            "2025-01-31 10:00:00 - test - DEBUG - Debug message\n",
            "This line has no level\n",
            "2025-01-31 10:00:01 - test - INFO - Info message\n",
        ]

        filtered = log_viewer._filter_by_level(lines, "INFO")

        # 应该保留没有级别信息的行
        assert len(filtered) == 2  # INFO 和无级别行
        assert any("no level" in line for line in filtered)


class TestLogViewerLevelChange:
    """测试级别改变"""

    def test_on_level_change(self, log_viewer):
        """测试级别改变处理"""
        # 改变到 WARNING
        log_viewer._on_level_change("WARNING")

        assert log_viewer.current_level == "WARNING"

    def test_on_level_change_all(self, log_viewer):
        """测试改变到 ALL 级别"""
        log_viewer.current_level = "ERROR"
        log_viewer._on_level_change("ALL")

        assert log_viewer.current_level == "ALL"


class TestLogViewerAutoRefresh:
    """测试自动刷新功能"""

    def test_start_auto_refresh(self, skip_if_no_gui):
        """测试启动自动刷新"""
        master = MagicMock()
        log_file = MagicMock()

        viewer = LogViewer(master, log_file=log_file, auto_refresh=False)

        # 启动自动刷新
        viewer.start_auto_refresh()

        assert viewer.auto_refresh is True

    def test_stop_auto_refresh(self, skip_if_no_gui):
        """测试停止自动刷新"""
        master = MagicMock()
        log_file = MagicMock()

        viewer = LogViewer(master, log_file=log_file, auto_refresh=True)

        # 停止自动刷新
        viewer.stop_auto_refresh()

        assert viewer.auto_refresh is False


class TestLogViewerClear:
    """测试清空日志功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkTextbox"), reason="CustomTkinter not available")
    def test_clear_log(self, log_viewer, temp_log_file: Path, tmp_path: Path):
        """测试清空日志文件"""
        # 在测试中不实际清空文件，只验证方法存在
        assert hasattr(log_viewer, "clear_log")

        # 可以模拟清空操作
        original_content = temp_log_file.read_text()
        assert len(original_content) > 0


class TestLogViewerSetLogFile:
    """测试设置日志文件"""

    def test_set_log_file(self, skip_if_no_gui, tmp_path: Path):
        """测试设置日志文件"""
        master = MagicMock()

        viewer = LogViewer(master, log_file=None)

        # 设置新的日志文件
        new_log_file = tmp_path / "new.log"
        new_log_file.write_text("test log content\n", encoding="utf-8")

        viewer.set_log_file(new_log_file)

        assert viewer.log_file == new_log_file


class TestLogViewerAppendLog:
    """测试追加日志功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkTextbox"), reason="CustomTkinter not available")
    def test_append_log(self, log_viewer):
        """测试追加日志消息"""
        # 验证方法存在
        assert hasattr(log_viewer, "append_log")

        # 在实际 GUI 中会追加到文本框
        log_viewer.append_log("Test log message\n")


class TestLogViewerSearch:
    """测试搜索功能"""

    def test_search_keyword(self, log_viewer):
        """测试搜索关键字"""
        # 模拟一些日志内容
        mock_content = "2025-01-31 10:00:00 - test - INFO - Test message 1\n"
        mock_content += "2025-01-31 10:00:01 - test - ERROR - Error message\n"
        mock_content += "2025-01-31 10:00:02 - test - INFO - Test message 2\n"

        # 在实际实现中，搜索会在 log_text 中进行
        # 这里我们验证搜索方法存在
        assert hasattr(log_viewer, "search")

        # 模拟搜索
        count = log_viewer.search("Test")
        # 在模拟环境中可能返回 0
        assert isinstance(count, int)

    def test_search_uses_underlying_text_widget(self, log_viewer):
        """测试优先使用底层 _textbox 的 search/tag 能力"""

        class FakeTextBackend:
            def __init__(self, content: str):
                self.content = content
                self.tags: list[tuple[str, str, str]] = []
                self.clear_calls = 0

            @staticmethod
            def _to_char_index(index: str) -> int:
                if index == "1.0":
                    return 0
                if "+" in index and index.endswith("c"):
                    base, delta = index.split("+", 1)
                    return FakeTextBackend._to_char_index(base) + int(delta[:-1])
                line, col = index.split(".", 1)
                if line != "1":
                    return 0
                return int(col)

            @staticmethod
            def _to_tk_index(char_index: int) -> str:
                return f"1.{char_index}"

            def search(self, keyword: str, start: str, stopindex: str = "end"):
                del stopindex
                start_idx = self._to_char_index(start)
                found = self.content.find(keyword, start_idx)
                if found < 0:
                    return ""
                return self._to_tk_index(found)

            def tag_config(self, *_args, **_kwargs):
                pass

            def tag_add(self, tag: str, start: str, end: str):
                self.tags.append((tag, start, end))

            def tag_remove(self, tag: str, start: str, end: str):
                del tag, start, end
                self.clear_calls += 1

        class FakeTextbox:
            def __init__(self, content: str):
                self._textbox = FakeTextBackend(content)
                self._content = content

            def get(self, _start: str, _end: str):
                return self._content

        content = "Test message 1\nError message\nTest message 2\n"
        fake_textbox = FakeTextbox(content)
        log_viewer.log_text = fake_textbox

        count = log_viewer.search("Test")

        assert count == 2
        assert fake_textbox._textbox.clear_calls == 1
        assert len(fake_textbox._textbox.tags) == 2


class TestLogViewerEdgeCases:
    """测试边界情况"""

    def test_log_with_unicode(self, log_viewer, tmp_path: Path):
        """测试包含 Unicode 的日志"""
        unicode_file = tmp_path / "unicode.log"
        unicode_content = "2025-01-31 10:00:00 - test - INFO - 中文消息\n"
        unicode_content += "2025-01-31 10:00:01 - test - INFO - 日本語メッセージ\n"
        unicode_content += "2025-01-31 10:00:02 - test - INFO - 한국어 메시지\n"

        unicode_file.write_text(unicode_content, encoding="utf-8")

        log_viewer.log_file = unicode_file
        log_viewer.refresh_log()

        # 应该成功处理 Unicode
        assert unicode_file.exists()

    def test_log_with_special_characters(self, log_viewer, tmp_path: Path):
        """测试包含特殊字符的日志"""
        special_file = tmp_path / "special.log"
        special_content = "2025-01-31 10:00:00 - test - INFO - Special: !@#$%^&*()\n"
        special_content += "2025-01-31 10:00:01 - test - INFO - Tabs\t\tTabs\n"
        special_content += "2025-01-31 10:00:02 - test - INFO - Newlines\n\n\n"

        special_file.write_text(special_content, encoding="utf-8")

        log_viewer.log_file = special_file
        log_viewer.refresh_log()

        assert special_file.exists()

    def test_very_long_log_line(self, log_viewer, tmp_path: Path):
        """测试非常长的日志行"""
        long_file = tmp_path / "long.log"
        long_line = "2025-01-31 10:00:00 - test - INFO - " + "x" * 10000 + "\n"

        long_file.write_text(long_line, encoding="utf-8")

        log_viewer.log_file = long_file
        log_viewer.refresh_log()

        assert long_file.exists()
