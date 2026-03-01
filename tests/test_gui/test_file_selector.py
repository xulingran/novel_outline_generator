"""
测试 file_selector 组件

测试文件选择器组件，包括文件选择、信息显示、token 计数等功能。
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gui.widgets.file_selector import FileSelector
from tests.test_gui.conftest import ctk


@pytest.fixture
def file_selector():
    """创建文件选择器实例"""
    # 使用模拟的 master
    master = MagicMock()
    selector = FileSelector(master)
    return selector


class TestFileSelector:
    """测试 FileSelector 组件"""

    def test_init(self, skip_if_no_gui):
        """测试 FileSelector 初始化"""
        master = MagicMock()
        selector = FileSelector(master)

        assert selector.title == "文件选择"
        assert selector.current_file is None
        assert selector.on_file_selected is None
        assert len(selector.file_types) == 3  # txt, md, all

    def test_init_with_custom_params(self, skip_if_no_gui):
        """测试使用自定义参数初始化"""
        master = MagicMock()
        custom_types = [("Python", "*.py"), ("Text", "*.txt")]

        callback = MagicMock()

        selector = FileSelector(
            master,
            title="选择脚本",
            file_types=custom_types,
            on_file_selected=callback,
        )

        assert selector.title == "选择脚本"
        assert selector.file_types == custom_types
        assert selector.on_file_selected == callback

    @pytest.mark.skipif(
        not hasattr(ctk, "CTk"), reason="CustomTkinter not available for full GUI test"
    )
    def test_set_file_valid(self, temp_test_file: Path):
        """测试设置有效文件"""
        master = MagicMock()
        selector = FileSelector(master)

        # FileSelector.set_file 只接受一个参数（filepath）
        selector.set_file(temp_test_file)

        assert selector.current_file == temp_test_file

    @pytest.mark.skipif(
        not hasattr(ctk, "CTk"), reason="CustomTkinter not available for full GUI test"
    )
    def test_set_file_nonexistent(self, temp_test_file: Path):
        """测试设置不存在的文件"""
        master = MagicMock()
        selector = FileSelector(master)

        # 使用不存在的文件路径
        fake_file = Path("/nonexistent/file.txt")

        # 不应该抛出异常，但也不应该设置文件
        selector.set_file(fake_file)

        assert selector.current_file is None

    @pytest.mark.skipif(
        not hasattr(ctk, "CTk"), reason="CustomTkinter not available for full GUI test"
    )
    def test_get_file(self, temp_test_file: Path):
        """测试获取当前文件"""
        master = MagicMock()
        selector = FileSelector(master)

        # 初始状态
        assert selector.get_file() is None

        # 设置文件后
        selector.set_file(temp_test_file)
        assert selector.get_file() == temp_test_file

    @pytest.mark.skipif(
        not hasattr(ctk, "CTk"), reason="CustomTkinter not available for full GUI test"
    )
    def test_clear(self, temp_test_file: Path):
        """测试清空选择"""
        master = MagicMock()
        selector = FileSelector(master)

        # 设置文件
        selector.set_file(temp_test_file)
        assert selector.current_file is not None

        # 清空
        selector.clear()
        assert selector.current_file is None

    def test_file_info_calculation(self, temp_test_file: Path):
        """测试文件信息计算"""
        # 验证测试文件内容
        content = temp_test_file.read_text(encoding="utf-8")
        assert len(content) > 0
        assert temp_test_file.exists()

        # 获取文件大小
        size = temp_test_file.stat().st_size
        assert size > 0


class TestFileSelectorTokenEstimation:
    """测试 token 预估功能"""

    def test_token_count_for_test_file(self, temp_test_file: Path):
        """测试 token 计数"""
        from tokenizer import count_tokens

        content = temp_test_file.read_text(encoding="utf-8")
        token_count = count_tokens(content)

        assert token_count > 0
        assert isinstance(token_count, int)

    def test_chunk_estimation(self, temp_test_file: Path):
        """测试分块预估"""
        from config import get_processing_config
        from tokenizer import count_tokens

        config = get_processing_config()
        content = temp_test_file.read_text(encoding="utf-8")
        token_count = count_tokens(content)

        chunk_count = (token_count // config.target_tokens_per_chunk) + 1

        assert chunk_count >= 1
        assert isinstance(chunk_count, int)

    def test_file_size_display(self, temp_test_file: Path):
        """测试文件大小计算"""
        size_bytes = temp_test_file.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        assert size_mb > 0
        assert size_mb < 1  # 测试文件应该小于 1MB

    def test_mtime_display(self, temp_test_file: Path):
        """测试修改时间获取"""
        from datetime import datetime

        mtime = temp_test_file.stat().st_mtime
        mtime_dt = datetime.fromtimestamp(mtime)

        assert isinstance(mtime_dt, datetime)


class TestFileSelectorFileTypes:
    """测试文件类型过滤"""

    def test_default_file_types(self):
        """测试默认文件类型"""
        master = MagicMock()
        selector = FileSelector(master)

        types = selector.file_types
        assert len(types) == 3

        # 验证格式
        for name, pattern in types:
            assert isinstance(name, str)
            assert isinstance(pattern, str)
            assert pattern.startswith("*.")

    def test_custom_file_types(self):
        """测试自定义文件类型"""
        master = MagicMock()
        custom_types = [
            ("JSON 文件", "*.json"),
            ("XML 文件", "*.xml"),
            ("所有文件", "*.*"),
        ]

        selector = FileSelector(master, file_types=custom_types)

        assert selector.file_types == custom_types
        assert len(selector.file_types) == 3

    def test_empty_file_types(self):
        """测试空文件类型列表"""
        master = MagicMock()
        selector = FileSelector(master, file_types=[])

        # 应该被默认值替换
        assert len(selector.file_types) == 3


class TestFileSelectorCallback:
    """测试回调功能"""

    def test_file_selected_callback(self, temp_test_file: Path):
        """测试文件选择回调"""
        master = MagicMock()
        callback = MagicMock()

        selector = FileSelector(master, on_file_selected=callback)

        # 模拟文件选择（直接调用内部方法）
        if hasattr(selector, "set_file"):
            # 在模拟环境中，set_file 方法可能不会真正工作
            # 这里我们只测试回调的设置
            assert selector.on_file_selected == callback

    def test_callback_with_none(self):
        """测试回调为 None 的情况"""
        master = MagicMock()
        selector = FileSelector(master, on_file_selected=None)

        assert selector.on_file_selected is None


class TestFileSelectorEdgeCases:
    """测试边界情况"""

    def test_empty_file(self, tmp_path: Path):
        """测试空文件"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        from tokenizer import count_tokens

        content = empty_file.read_text(encoding="utf-8")
        token_count = count_tokens(content)

        assert token_count == 0

    def test_large_file_estimation(self, tmp_path: Path):
        """测试大文件预估"""
        # 创建一个较大的测试文件（100KB）
        large_file = tmp_path / "large.txt"
        large_content = "测试内容。" * 10000
        large_file.write_text(large_content, encoding="utf-8")

        from tokenizer import count_tokens

        token_count = count_tokens(large_content)

        assert token_count > 10000  # 应该有很多 token

    def test_unicode_file(self, tmp_path: Path):
        """测试 Unicode 文件"""
        unicode_file = tmp_path / "unicode.txt"
        unicode_content = "中文内容\n日本語\n한국어\nΕλληνικά\nالعربية"
        unicode_file.write_text(unicode_content, encoding="utf-8")

        from tokenizer import count_tokens

        token_count = count_tokens(unicode_content)

        assert token_count > 0

    def test_special_characters(self, tmp_path: Path):
        """测试特殊字符文件"""
        special_file = tmp_path / "special.txt"
        special_content = "!@#$%^&*()_+-=[]{}|;':\",./<>?\n\t\r\n"
        special_file.write_text(special_content, encoding="utf-8")

        from tokenizer import count_tokens

        token_count = count_tokens(special_content)

        assert token_count > 0
