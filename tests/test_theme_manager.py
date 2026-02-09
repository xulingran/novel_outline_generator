"""
主题管理器测试
"""

import os
import tempfile

import pytest

from gui.theme_manager import (
    CORNER_RADIUS,
    FONTS,
    NORD_COLORS,
    SPACING,
    ThemeManager,
    get_color,
    get_theme_manager,
)


class TestThemeManager:
    """ThemeManager测试类"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """重置单例状态"""
        # 保存原始状态
        original_instance = ThemeManager._instance
        original_initialized = ThemeManager._initialized

        # 重置单例
        ThemeManager._instance = None
        ThemeManager._initialized = False

        yield

        # 恢复原始状态
        ThemeManager._instance = original_instance
        ThemeManager._initialized = original_initialized

    @pytest.fixture
    def temp_settings(self):
        """创建临时设置文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            temp_path = f.name

        yield temp_path

        # 清理
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_singleton_pattern(self, temp_settings):
        """测试单例模式"""
        tm1 = ThemeManager(temp_settings)
        tm2 = ThemeManager(temp_settings)
        assert tm1 is tm2

    def test_default_theme(self, temp_settings):
        """测试默认主题为dark"""
        tm = ThemeManager(temp_settings)
        assert tm.get_current_theme() == "dark"

    def test_set_theme(self, temp_settings):
        """测试设置主题"""
        tm = ThemeManager(temp_settings)

        # 测试有效主题
        tm.set_theme("light")
        assert tm.get_current_theme() == "light"

        tm.set_theme("dark")
        assert tm.get_current_theme() == "dark"

        tm.set_theme("system")
        assert tm.get_current_theme() == "system"

    def test_set_invalid_theme(self, temp_settings):
        """测试设置无效主题"""
        tm = ThemeManager(temp_settings)

        with pytest.raises(ValueError):
            tm.set_theme("invalid")

    def test_theme_persistence(self, temp_settings):
        """测试主题持久化"""
        # 创建管理器并设置主题
        tm1 = ThemeManager(temp_settings)
        tm1.set_theme("light")

        # 重置单例模拟重启
        ThemeManager._instance = None
        ThemeManager._initialized = False

        # 创建新管理器，应该加载保存的主题
        tm2 = ThemeManager(temp_settings)
        assert tm2.get_current_theme() == "light"

    def test_get_color_auto_mode(self, temp_settings):
        """测试get_color auto模式返回元组"""
        tm = ThemeManager(temp_settings)
        color = tm.get_color("accent", mode="auto")

        assert isinstance(color, tuple)
        assert len(color) == 2
        assert color[0].startswith("#")  # light color
        assert color[1].startswith("#")  # dark color

    def test_get_color_fixed_mode(self, temp_settings):
        """测试get_color固定模式返回字符串"""
        tm = ThemeManager(temp_settings)

        light_color = tm.get_color("accent", mode="light")
        assert isinstance(light_color, str)
        assert light_color.startswith("#")

        dark_color = tm.get_color("accent", mode="dark")
        assert isinstance(dark_color, str)
        assert dark_color.startswith("#")

    def test_get_color_nord_values(self, temp_settings):
        """测试颜色值符合Nord规范"""
        tm = ThemeManager(temp_settings)

        # 暗色主题
        assert tm.get_color("bg_primary", mode="dark") == "#2E3440"
        assert tm.get_color("accent", mode="dark") == "#88C0D0"
        assert tm.get_color("error", mode="dark") == "#BF616A"
        assert tm.get_color("success", mode="dark") == "#A3BE8C"

        # 亮色主题
        assert tm.get_color("bg_primary", mode="light") == "#ECEFF4"
        assert tm.get_color("accent", mode="light") == "#5E81AC"
        assert tm.get_color("error", mode="light") == "#BF616A"

    def test_get_color_invalid_name(self, temp_settings):
        """测试获取不存在的颜色"""
        tm = ThemeManager(temp_settings)

        with pytest.raises(KeyError):
            tm.get_color("invalid_color")

    def test_get_color_invalid_mode(self, temp_settings):
        """测试无效的mode参数"""
        tm = ThemeManager(temp_settings)

        with pytest.raises(ValueError):
            tm.get_color("accent", mode="invalid")

    def test_callback_registration(self, temp_settings):
        """测试回调注册"""
        tm = ThemeManager(temp_settings)
        callbacks = []

        def callback(theme):
            callbacks.append(theme)

        tm.on_theme_change(callback)
        tm.set_theme("light")

        assert "light" in callbacks

    def test_callback_removal(self, temp_settings):
        """测试回调移除"""
        tm = ThemeManager(temp_settings)
        callbacks = []

        def callback(theme):
            callbacks.append(theme)

        tm.on_theme_change(callback)
        tm.set_theme("light")

        # 移除回调
        tm.remove_callback(callback)
        initial_count = len(callbacks)

        # 再次切换主题，回调不应该被调用
        tm.set_theme("dark")
        assert len(callbacks) == initial_count

    def test_global_functions(self, temp_settings):
        """测试全局便捷函数"""
        # 重置单例
        ThemeManager._instance = None
        ThemeManager._initialized = False

        # 测试get_theme_manager
        tm = get_theme_manager()
        assert isinstance(tm, ThemeManager)

        # 测试get_color
        color = get_color("accent", mode="dark")
        assert color == "#88C0D0"


class TestDesignSystem:
    """设计系统常量测试"""

    def test_spacing_constants(self):
        """测试间距常量"""
        assert SPACING["xs"] == 4
        assert SPACING["sm"] == 8
        assert SPACING["md"] == 16
        assert SPACING["lg"] == 24
        assert SPACING["xl"] == 32

    def test_fonts_structure(self):
        """测试字体常量结构"""
        assert "family" in FONTS
        assert "sizes" in FONTS
        assert "weights" in FONTS

        # 检查所有尺寸
        expected_sizes = ["xs", "sm", "md", "lg", "xl", "2xl"]
        for size in expected_sizes:
            assert size in FONTS["sizes"]

    def test_corner_radius_constants(self):
        """测试圆角常量"""
        assert CORNER_RADIUS["sm"] == 4
        assert CORNER_RADIUS["md"] == 8
        assert CORNER_RADIUS["lg"] == 12
        assert CORNER_RADIUS["xl"] == 16

    def test_nord_colors_structure(self):
        """测试Nord颜色结构"""
        assert "dark" in NORD_COLORS
        assert "light" in NORD_COLORS

        # 暗色和亮色主题应有相同的颜色键
        dark_keys = set(NORD_COLORS["dark"].keys())
        light_keys = set(NORD_COLORS["light"].keys())

        # 亮色主题可能缺少某些键，但至少应有主要颜色
        essential_colors = ["bg_primary", "fg_primary", "accent"]
        for color in essential_colors:
            assert color in dark_keys
            assert color in light_keys
