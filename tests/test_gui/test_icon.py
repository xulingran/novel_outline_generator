"""
图标组件回归测试。
"""

from gui.components.icon import Icon, IconWeight


class TestIconDispatch:
    """测试图标分发与可用性。"""

    def test_hyphenated_icon_name_dispatches_to_underscore_method(self):
        """连字符名称应映射到下划线绘制方法。"""
        icon = object.__new__(Icon)
        icon._name = "file-text"
        icon._weight = IconWeight.REGULAR
        icon._color = "#000000"
        icon._size = 24
        icon._padding = 2

        called = {"drawn": False, "placeholder": False}

        icon.delete = lambda *_args, **_kwargs: None
        icon._get_color = lambda: "#000000"
        icon._get_line_width = lambda: 1.5
        icon._draw_placeholder = lambda *_args, **_kwargs: called.__setitem__("placeholder", True)
        icon._draw_file_text = lambda *_args, **_kwargs: called.__setitem__("drawn", True)

        Icon._draw(icon)

        assert called["drawn"] is True
        assert called["placeholder"] is False

    def test_rocket_icon_is_available(self):
        """火箭图标应在可用列表中。"""
        assert "rocket" in Icon.AVAILABLE_ICONS


class TestIconMoonColor:
    """测试月亮图标颜色解析。"""

    def test_draw_moon_resolves_auto_bg_color_to_single_string(self, monkeypatch):
        """moon 遮挡层应传入单一字符串颜色，避免 Tk 报错。"""
        icon = object.__new__(Icon)

        captured_kwargs = []

        def fake_create_oval(*_args, **kwargs):
            captured_kwargs.append(kwargs)

        icon.create_oval = fake_create_oval
        icon.cget = lambda _key: "transparent"

        monkeypatch.setattr(
            "gui.components.icon.ctk.get_appearance_mode", lambda: "Dark", raising=False
        )

        Icon._draw_moon(icon, pad=2, size=24, color="#ffffff", line_width=1.5)

        assert len(captured_kwargs) == 2
        assert captured_kwargs[1]["fill"] == "#2E3440"
        assert captured_kwargs[1]["outline"] == "#2E3440"
