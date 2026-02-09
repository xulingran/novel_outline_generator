"""
ProcessPage 组件测试
"""

from unittest.mock import MagicMock, patch

from gui.pages.process_page import ProcessPage


def test_merge_state_variables_initialized():
    """合并相关状态变量应在初始化时设置"""
    # Mock _setup_ui 以避免实际的 UI 初始化
    with patch.object(ProcessPage, "_setup_ui"):
        # 创建一个 mock master
        mock_master = MagicMock()

        # 初始化 ProcessPage
        page = ProcessPage(mock_master)

        # 验证合并状态变量已初始化
        assert hasattr(page, "_last_phase")
        assert hasattr(page, "_initial_outline_count")
        assert hasattr(page, "_is_merge_phase")

        # 验证初始值
        assert page._last_phase == ""
        assert page._initial_outline_count == 0
        assert page._is_merge_phase is False
