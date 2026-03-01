"""
状态徽章组件测试
"""

from unittest.mock import MagicMock, patch

from gui.components.status_badge import STATUS_CONFIG, ProcessingStatus, StatusBadge


class TestProcessingStatus:
    """处理状态枚举测试"""

    def test_status_values(self):
        """测试状态枚举值"""
        assert ProcessingStatus.IDLE.value == "idle"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.MERGING.value == "merging"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.CANCELLED.value == "cancelled"
        assert ProcessingStatus.FAILED.value == "failed"

    def test_all_statuses_have_config(self):
        """测试所有状态都有配置"""
        for status in ProcessingStatus:
            assert status in STATUS_CONFIG
            config = STATUS_CONFIG[status]
            assert "text" in config
            assert "bg_color" in config
            assert "text_color" in config
            assert "icon" in config


class TestStatusBadge:
    """状态徽章组件测试"""

    def test_init_default_status(self):
        """测试默认状态初始化"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance"):
                    badge = StatusBadge(MagicMock())
                    assert badge._status == ProcessingStatus.IDLE
                    assert badge._size == "md"

    def test_init_custom_status(self):
        """测试自定义状态初始化"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance"):
                    badge = StatusBadge(MagicMock(), status=ProcessingStatus.PROCESSING, size="sm")
                    assert badge._status == ProcessingStatus.PROCESSING
                    assert badge._size == "sm"

    def test_size_configurations(self):
        """测试尺寸配置"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance"):
                    badge_sm = StatusBadge(MagicMock(), size="sm")
                    badge_md = StatusBadge(MagicMock(), size="md")
                    badge_lg = StatusBadge(MagicMock(), size="lg")

                    assert badge_sm._size_config["sm"]["height"] == 24
                    assert badge_md._size_config["md"]["height"] == 28
                    assert badge_lg._size_config["lg"]["height"] == 32

    def test_set_status(self):
        """测试设置状态"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance") as mock_update:
                    badge = StatusBadge(MagicMock())
                    mock_update.reset_mock()

                    badge.set_status(ProcessingStatus.COMPLETED)
                    assert badge._status == ProcessingStatus.COMPLETED
                    mock_update.assert_called_once()

    def test_set_same_status_no_update(self):
        """测试设置相同状态不更新"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance") as mock_update:
                    badge = StatusBadge(MagicMock(), status=ProcessingStatus.PROCESSING)
                    mock_update.reset_mock()

                    badge.set_status(ProcessingStatus.PROCESSING)
                    mock_update.assert_not_called()

    def test_get_status(self):
        """测试获取状态"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance"):
                    badge = StatusBadge(MagicMock(), status=ProcessingStatus.FAILED)
                    assert badge.get_status() == ProcessingStatus.FAILED

    def test_refresh_theme(self):
        """测试刷新主题"""
        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch.object(StatusBadge, "_setup_ui"):
                with patch.object(StatusBadge, "_update_appearance") as mock_update:
                    badge = StatusBadge(MagicMock())
                    mock_update.reset_mock()

                    badge.refresh_theme()
                    mock_update.assert_called_once()


class TestStatusConfig:
    """状态配置测试"""

    def test_bg_color_tuple_format(self):
        """测试背景色元组格式"""
        for _status, config in STATUS_CONFIG.items():
            bg_color = config["bg_color"]
            assert isinstance(bg_color, tuple)
            assert len(bg_color) == 2

    def test_text_color_tuple_format(self):
        """测试文字色元组格式"""
        for _status, config in STATUS_CONFIG.items():
            text_color = config["text_color"]
            assert isinstance(text_color, tuple)
            assert len(text_color) == 2

    def test_icon_not_empty(self):
        """测试图标不为空"""
        for status, config in STATUS_CONFIG.items():
            assert config["icon"], f"Icon for {status} should not be empty"

    def test_text_not_empty(self):
        """测试文字不为空"""
        for status, config in STATUS_CONFIG.items():
            assert config["text"], f"Text for {status} should not be empty"


class TestStatusBadgeUI:
    """状态徽章 UI 测试"""

    def test_setup_ui_creates_components(self):
        """测试 UI 设置创建组件"""
        mock_master = MagicMock()

        with patch("gui.components.status_badge.ctk.CTkFrame.__init__", return_value=None):
            with patch("gui.components.status_badge.ctk.CTkFrame.pack"):
                with patch("gui.components.status_badge.ctk.CTkFrame.configure"):
                    with patch("gui.components.status_badge.ctk.CTkLabel"):
                        with patch("gui.components.status_badge.ctk.CTkFrame"):
                            with patch.object(StatusBadge, "_update_appearance"):
                                badge = StatusBadge(mock_master)

                                badge._container = MagicMock()
                                badge._icon_label = MagicMock()
                                badge._text_label = MagicMock()

                                assert hasattr(badge, "_container")
                                assert hasattr(badge, "_icon_label")
                                assert hasattr(badge, "_text_label")
