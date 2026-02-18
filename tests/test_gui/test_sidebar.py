"""
侧边栏组件测试
"""

from unittest.mock import MagicMock, patch

from gui.components.sidebar import NAV_ITEMS, NavItem, NavItemSpec, Sidebar


class TestNavItem:
    """导航项枚举测试"""

    def test_nav_item_values(self):
        """测试导航项枚举值"""
        assert NavItem.PROCESS.value == "process"
        assert NavItem.CONFIG.value == "config"
        assert NavItem.LOG.value == "log"
        assert NavItem.ABOUT.value == "about"


class TestNavItemSpec:
    """导航项规格测试"""

    def test_nav_item_spec_creation(self):
        """测试创建导航项规格"""
        spec = NavItemSpec(
            id=NavItem.PROCESS,
            label="处理",
            icon="rocket",
            badge="NEW",
        )
        assert spec.id == NavItem.PROCESS
        assert spec.label == "处理"
        assert spec.icon == "rocket"
        assert spec.badge == "NEW"

    def test_nav_item_spec_default_badge(self):
        """测试默认徽章为 None"""
        spec = NavItemSpec(
            id=NavItem.CONFIG,
            label="配置",
            icon="sliders",
        )
        assert spec.badge is None


class TestNavItems:
    """导航项定义测试"""

    def test_nav_items_count(self):
        """测试导航项数量"""
        assert len(NAV_ITEMS) == 4

    def test_nav_items_have_required_fields(self):
        """测试导航项都有必需字段"""
        for item in NAV_ITEMS:
            assert isinstance(item.id, NavItem)
            assert item.label
            assert item.icon

    def test_nav_items_unique_ids(self):
        """测试导航项 ID 唯一"""
        ids = [item.id for item in NAV_ITEMS]
        assert len(ids) == len(set(ids))


class TestSidebarConstants:
    """侧边栏常量测试"""

    def test_width_constants(self):
        """测试宽度常量"""
        assert Sidebar.MIN_WIDTH == 180
        assert Sidebar.MAX_WIDTH == 320
        assert Sidebar.COLLAPSED_WIDTH == 60

    def test_width_constraints(self):
        """测试宽度约束"""
        assert Sidebar.MIN_WIDTH < Sidebar.MAX_WIDTH
        assert Sidebar.COLLAPSED_WIDTH < Sidebar.MIN_WIDTH


class TestSidebarInit:
    """侧边栏初始化测试"""

    def test_init_default_params(self):
        """测试默认参数初始化"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)

                        assert sidebar._width == 240
                        assert sidebar._active_item == NavItem.PROCESS
                        assert sidebar._on_navigation is None
                        assert sidebar._collapsed is False
                        assert sidebar._dragging is False

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        mock_master = MagicMock()
        callback = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(
                            mock_master,
                            width=300,
                            active_item=NavItem.CONFIG,
                            on_navigation=callback,
                        )

                        assert sidebar._width == 300
                        assert sidebar._active_item == NavItem.CONFIG
                        assert sidebar._on_navigation == callback


class TestSidebarCollapse:
    """侧边栏折叠测试"""

    def test_toggle_collapse_to_collapsed(self):
        """测试折叠"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar.configure = MagicMock()
                        sidebar._hide_labels = MagicMock()

                        sidebar.toggle_collapse()

                        assert sidebar._collapsed is True
                        sidebar.configure.assert_called_once()
                        sidebar._hide_labels.assert_called_once()

    def test_toggle_collapse_to_expanded(self):
        """测试展开"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._collapsed = True
                        sidebar.configure = MagicMock()
                        sidebar._show_labels = MagicMock()

                        sidebar.toggle_collapse()

                        assert sidebar._collapsed is False
                        sidebar.configure.assert_called_once()
                        sidebar._show_labels.assert_called_once()

    def test_hide_labels(self):
        """测试隐藏标签"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._logo_label = MagicMock()
                        sidebar._logo_label.pack_forget = MagicMock()
                        sidebar._nav_buttons = {
                            NavItem.PROCESS: MagicMock(_label=MagicMock(pack_forget=MagicMock())),
                        }

                        sidebar._hide_labels()

                        sidebar._logo_label.pack_forget.assert_called_once()

    def test_show_labels(self):
        """测试显示标签"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        mock_label = MagicMock()
                        mock_label.winfo_exists = MagicMock(return_value=True)
                        mock_label.pack = MagicMock()
                        sidebar._logo_label = mock_label
                        sidebar._nav_buttons = {}

                        sidebar._show_labels()

                        mock_label.pack.assert_called_once()


class TestSidebarResize:
    """侧边栏调整宽度测试"""

    def test_start_resize(self):
        """测试开始调整"""
        mock_master = MagicMock()
        mock_event = MagicMock()
        mock_event.x_root = 100

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._resize_handle = MagicMock()

                        sidebar._start_resize(mock_event)

                        assert sidebar._dragging is True
                        assert sidebar._drag_start_x == 100
                        assert sidebar._drag_start_width == 240

    def test_do_resize_not_dragging(self):
        """测试调整中（未拖拽）"""
        mock_master = MagicMock()
        mock_event = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._dragging = False
                        sidebar.configure = MagicMock()

                        sidebar._do_resize(mock_event)

                        sidebar.configure.assert_not_called()

    def test_do_resize_dragging(self):
        """测试调整中（拖拽中）"""
        mock_master = MagicMock()
        mock_event = MagicMock()
        mock_event.x_root = 150

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._dragging = True
                        sidebar._drag_start_x = 100
                        sidebar._drag_start_width = 240
                        sidebar.configure = MagicMock()

                        sidebar._do_resize(mock_event)

                        sidebar.configure.assert_called_once()
                        new_width = sidebar.configure.call_args[1]["width"]
                        assert Sidebar.MIN_WIDTH <= new_width <= Sidebar.MAX_WIDTH

    def test_do_resize_clamps_to_min(self):
        """测试调整宽度限制到最小值"""
        mock_master = MagicMock()
        mock_event = MagicMock()
        mock_event.x_root = 0

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._dragging = True
                        sidebar._drag_start_x = 100
                        sidebar._drag_start_width = 240
                        sidebar.configure = MagicMock()

                        sidebar._do_resize(mock_event)

                        new_width = sidebar.configure.call_args[1]["width"]
                        assert new_width == Sidebar.MIN_WIDTH

    def test_do_resize_clamps_to_max(self):
        """测试调整宽度限制到最大值"""
        mock_master = MagicMock()
        mock_event = MagicMock()
        mock_event.x_root = 1000

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._dragging = True
                        sidebar._drag_start_x = 100
                        sidebar._drag_start_width = 240
                        sidebar.configure = MagicMock()

                        sidebar._do_resize(mock_event)

                        new_width = sidebar.configure.call_args[1]["width"]
                        assert new_width == Sidebar.MAX_WIDTH

    def test_end_resize(self):
        """测试结束调整"""
        mock_master = MagicMock()
        mock_event = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._dragging = True
                        sidebar._resize_handle = MagicMock()

                        sidebar._end_resize(mock_event)

                        assert sidebar._dragging is False


class TestSidebarActiveItem:
    """侧边栏激活项测试"""

    def test_set_active_item(self):
        """测试设置激活项"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)
                        sidebar._nav_buttons = {}
                        sidebar._update_active_state = MagicMock()

                        sidebar.set_active_item(NavItem.LOG)

                        assert sidebar._active_item == NavItem.LOG
                        sidebar._update_active_state.assert_called_once()

    def test_get_active_item(self):
        """测试获取激活项"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master, active_item=NavItem.ABOUT)

                        result = sidebar.get_active_item()

                        assert result == NavItem.ABOUT


class TestSidebarNavigation:
    """侧边栏导航测试"""

    def test_on_nav_click_with_callback(self):
        """测试导航点击（有回调）"""
        mock_master = MagicMock()
        mock_callback = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master, on_navigation=mock_callback)

                        sidebar._on_nav_click(NavItem.CONFIG)

                        mock_callback.assert_called_once_with(NavItem.CONFIG)

    def test_on_nav_click_without_callback(self):
        """测试导航点击（无回调）"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)

                        sidebar._on_nav_click(NavItem.CONFIG)


class TestSidebarTheme:
    """侧边栏主题测试"""

    def test_get_theme_label(self):
        """测试获取主题标签"""
        mock_master = MagicMock()

        with patch.object(Sidebar, "_setup_ui"):
            with patch.object(Sidebar, "_setup_resize_handle"):
                with patch.object(Sidebar, "_update_active_state"):
                    with patch("gui.components.sidebar.ctk.CTkFrame.__init__", return_value=None):
                        sidebar = Sidebar(mock_master)

                        assert sidebar._get_theme_label("light") == "浅色"
                        assert sidebar._get_theme_label("dark") == "深色"
                        assert sidebar._get_theme_label("system") == "跟随系统"
                        assert sidebar._get_theme_label("unknown") == "unknown"
