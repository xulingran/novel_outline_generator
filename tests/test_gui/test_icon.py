"""
图标组件回归测试。
"""

from unittest.mock import MagicMock, patch

from gui.components.icon import (
    Icon,
    IconWeight,
    _cache_icon,
    _get_cached_icon,
    _icon_cache,
)


class TestIconDispatch:
    """测试图标分发与可用性。"""

    def test_hyphenated_icon_name_dispatches_to_underscore_method(self):
        """连字符名称应映射到下划线绘制方法。"""
        icon = object.__new__(Icon)
        icon._name = "file-text"
        icon._weight = IconWeight.REGULAR
        icon._color = "#000000"
        icon._size = 24

        called = {"drawn": False, "placeholder": False}

        class MockCanvas:
            def delete(self, *args, **kwargs):
                pass

            def create_polygon(self, *args, **kwargs):
                pass

            def create_rectangle(self, *args, **kwargs):
                pass

            def create_line(self, *args, **kwargs):
                called["drawn"] = True

            def create_oval(self, *args, **kwargs):
                pass

            def create_arc(self, *args, **kwargs):
                pass

            def create_text(self, *args, **kwargs):
                pass

        icon._canvas = MockCanvas()
        icon._get_color = lambda: "#000000"
        icon._get_line_width = lambda: 1.5
        icon._draw_placeholder = lambda *args, **kwargs: called.__setitem__("placeholder", True)

        icon._draw_canvas_icon()

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

        class MockCanvas:
            def create_oval(self, *args, **kwargs):
                captured_kwargs.append(kwargs)

        icon._canvas = MockCanvas()
        icon._get_bg_color = lambda: "#2E3440"
        icon._resolve_color_value = lambda c: c

        monkeypatch.setattr(
            "gui.components.icon.ctk.get_appearance_mode", lambda: "Dark", raising=False
        )

        icon._draw_moon(pad=2, size=24, color="#ffffff", line_width=1.5)

        assert len(captured_kwargs) == 2
        assert captured_kwargs[1]["fill"] == "#2E3440"
        assert captured_kwargs[1]["outline"] == "#2E3440"


class TestIconConfigureAndCache:
    """测试图标配置刷新与缓存键。"""

    def test_configure_refreshes_icon_without_old_draw_path(self):
        """配置图标属性时应走新刷新路径，不依赖旧 _draw 实现。"""
        icon = object.__new__(Icon)
        icon._name = "file-text"
        icon._weight = IconWeight.REGULAR
        icon._color = "#000000"
        icon._size = 24
        icon._normalize_name = lambda name: name
        icon._reload_icon = MagicMock()

        with patch("gui.components.icon.ctk.CTkLabel.configure", return_value=None):
            Icon.configure(icon, color="#ffffff")

        assert icon._color == "#ffffff"
        icon._reload_icon.assert_called_once()

    def test_cache_key_distinguishes_color_and_appearance(self):
        """缓存应区分颜色与主题，避免串色。"""
        _icon_cache.clear()
        img_light = object()
        img_dark = object()

        _cache_icon("house", 24, "#111111", "Light", img_light)
        _cache_icon("house", 24, "#ffffff", "Dark", img_dark)

        assert _get_cached_icon("house", 24, "#111111", "Light") is img_light
        assert _get_cached_icon("house", 24, "#ffffff", "Dark") is img_dark
        assert _get_cached_icon("house", 24, "#111111", "Dark") is None


class TestIconPngPathResolution:
    """测试 PNG 图标路径解析兼容性。"""

    def test_resolve_png_path_supports_flat_naming(self, tmp_path, monkeypatch):
        """应优先匹配当前仓库使用的扁平命名 name_size.png。"""
        icon = object.__new__(Icon)
        icon._name = "file"
        icon._size = 24

        icon_file = tmp_path / "file_24.png"
        icon_file.write_bytes(b"")

        monkeypatch.setattr("gui.components.icon.ICONS_DIR", tmp_path)

        resolved = icon._resolve_png_icon_path()
        assert resolved == icon_file

    def test_resolve_png_path_supports_nested_naming(self, tmp_path, monkeypatch):
        """应兼容尺寸子目录结构 icons/<size>/<name>.png。"""
        icon = object.__new__(Icon)
        icon._name = "file"
        icon._size = 24

        nested = tmp_path / "24"
        nested.mkdir(parents=True, exist_ok=True)
        icon_file = nested / "file.png"
        icon_file.write_bytes(b"")

        monkeypatch.setattr("gui.components.icon.ICONS_DIR", tmp_path)

        resolved = icon._resolve_png_icon_path()
        assert resolved == icon_file
