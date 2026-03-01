"""
工具提示组件测试
"""


class TestTooltipBasic:
    """工具提示基本测试"""

    def test_tooltip_module_imports(self):
        """测试模块可以导入"""
        from gui.components import tooltip

        assert hasattr(tooltip, "Tooltip")
        assert hasattr(tooltip, "add_tooltip")

    def test_tooltip_class_exists(self):
        """测试类存在"""
        from gui.components.tooltip import Tooltip

        assert Tooltip is not None

    def test_add_tooltip_function_exists(self):
        """测试函数存在"""
        from gui.components.tooltip import add_tooltip

        assert callable(add_tooltip)

    def test_tooltip_has_required_methods(self):
        """测试有必需方法"""
        from gui.components.tooltip import Tooltip

        assert hasattr(Tooltip, "_on_enter")
        assert hasattr(Tooltip, "_on_leave")
        assert hasattr(Tooltip, "_schedule_show")
        assert hasattr(Tooltip, "_cancel_show")
        assert hasattr(Tooltip, "_show")
        assert hasattr(Tooltip, "_hide")
        assert hasattr(Tooltip, "_resolve_color")
        assert hasattr(Tooltip, "update_text")
