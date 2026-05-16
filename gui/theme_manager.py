"""
主题管理模块

管理GUI主题系统，包括Nord配色方案、设计系统常量、主题切换和状态持久化。
"""

import json
import logging
import platform
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Nord配色方案
# 参考: https://www.nordtheme.com/
NORD_COLORS = {
    "dark": {
        # 背景色
        "bg_primary": "#2E3440",  # Nord0 - 最深背景
        "bg_secondary": "#3B4252",  # Nord1 - 次要背景
        "bg_tertiary": "#434C5E",  # Nord2 - 第三层背景
        # 前景色
        "fg_primary": "#ECEFF4",  # Nord6 - 主文本
        "fg_secondary": "#D8DEE9",  # Nord4 - 次要文本
        "fg_tertiary": "#4C566A",  # Nord3 - 第三层文本
        # 强调色
        "accent": "#88C0D0",  # Nord8 - 青色强调
        "accent_secondary": "#81A1C1",  # Nord9 - 次要强调
        # 状态色
        "success": "#A3BE8C",  # Nord14 - 成功
        "warning": "#EBCB8B",  # Nord13 - 警告
        "error": "#BF616A",  # Nord11 - 错误
        "info": "#5E81AC",  # Nord10 - 信息
        # 边框色
        "border": "#434C5E",  # Nord2
        "border_light": "#4C566A",  # Nord3
    },
    "light": {
        # 背景色
        "bg_primary": "#F3F5F9",  # 应用主背景
        "bg_secondary": "#FFFFFF",  # 卡片/浮层背景
        "bg_tertiary": "#E9EEF6",  # 侧栏与弱强调背景
        # 前景色
        "fg_primary": "#1F2937",  # 主文本
        "fg_secondary": "#4B5563",  # 次级文本
        "fg_tertiary": "#6B7280",  # 弱文本
        # 强调色
        "accent": "#2F6FEB",  # 主按钮/激活态
        "accent_secondary": "#5B8CFF",  # hover/次强调
        # 状态色
        "success": "#1F9D6A",
        "warning": "#C97A12",
        "error": "#C0392B",
        "info": "#2F6FEB",
        # 边框色
        "border": "#D7DFEA",
        "border_dark": "#9AA9BF",
    },
}

# 设计系统 - 间距
SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32,
    "2xl": 48,
}

# 设计系统 - 字体
# 平台特定字体回退
_system = platform.system()
if _system == "Darwin":  # macOS
    _DEFAULT_FONT_FAMILY = "SF Pro Display"
elif _system == "Windows":
    _DEFAULT_FONT_FAMILY = "Segoe UI"
else:  # Linux/其他
    _DEFAULT_FONT_FAMILY = "Inter"

FONTS = {
    "family": _DEFAULT_FONT_FAMILY,
    "sizes": {
        "xs": 11,
        "sm": 13,
        "md": 15,
        "lg": 18,
        "xl": 24,
        "2xl": 32,
    },
    "weights": {
        "normal": "normal",
        "medium": "bold",
        "semibold": "bold",
        "bold": "bold",
    },
}

# 设计系统 - 圆角
CORNER_RADIUS = {
    "sm": 4,
    "md": 8,
    "lg": 12,
    "xl": 16,
}


class ThemeManager:
    """
    主题管理器

    管理GUI主题状态，包括：
    - Nord配色方案
    - 主题切换（light/dark/system）
    - 状态持久化
    - 回调机制
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings_file: str = "settings.json"):
        """
        初始化主题管理器

        Args:
            settings_file: 设置文件路径，默认为 "settings.json"
        """
        # 避免重复初始化
        if ThemeManager._initialized:
            return

        self._settings_file = Path(settings_file)
        self._theme: str = "dark"  # 默认主题
        self._callbacks: list[Callable[[str], None]] = []

        # 加载保存的主题设置
        self._load_theme()

        ThemeManager._initialized = True
        logger.info(f"ThemeManager initialized with theme: {self._theme}")

    def _load_theme(self) -> None:
        """从settings.json加载主题设置"""
        try:
            if self._settings_file.exists():
                with open(self._settings_file, encoding="utf-8") as f:
                    settings = json.load(f)
                    saved_theme = settings.get("theme", "dark")
                    if saved_theme in ("light", "dark", "system"):
                        self._theme = saved_theme
                        logger.info(f"Loaded theme from settings: {self._theme}")
                    else:
                        logger.warning(f"Invalid theme in settings: {saved_theme}, using default")
            else:
                logger.info("No settings file found, using default dark theme")
        except Exception as e:
            logger.error(f"Failed to load theme settings: {e}")
            self._theme = "dark"

    def _save_theme(self) -> None:
        """保存主题设置到settings.json"""
        try:
            settings = {}
            if self._settings_file.exists():
                with open(self._settings_file, encoding="utf-8") as f:
                    settings = json.load(f)

            settings["theme"] = self._theme

            with open(self._settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved theme to settings: {self._theme}")
        except Exception as e:
            logger.error(f"Failed to save theme settings: {e}")

    def get_color(self, name: str, mode: str = "auto") -> str | tuple[str, str]:
        """
        获取颜色值

        Args:
            name: 颜色名称，如 "bg_primary", "accent", "error"
            mode: 颜色模式
                - "auto": 返回元组 (light_color, dark_color)，自动适配主题
                - "light": 返回亮色主题颜色
                - "dark": 返回暗色主题颜色

        Returns:
            如果 mode="auto"，返回元组 (light_hex, dark_hex)
            否则返回单个hex颜色字符串

        Raises:
            KeyError: 如果颜色名称不存在
        """
        if name not in NORD_COLORS["dark"]:
            raise KeyError(
                f"Unknown color name: {name}. Available: {list(NORD_COLORS['dark'].keys())}"
            )

        if mode == "auto":
            # 返回元组，CustomTkinter会自动根据当前主题选择
            light_color = NORD_COLORS["light"].get(name, NORD_COLORS["light"]["fg_primary"])
            dark_color = NORD_COLORS["dark"][name]
            return (light_color, dark_color)
        elif mode == "light":
            return NORD_COLORS["light"].get(name, NORD_COLORS["light"]["fg_primary"])
        elif mode == "dark":
            return NORD_COLORS["dark"][name]
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto', 'light', or 'dark'")

    def set_theme(self, theme: str) -> None:
        """
        设置主题

        Args:
            theme: 主题名称，可以是 "light", "dark", "system"

        Raises:
            ValueError: 如果主题名称无效
        """
        if theme not in ("light", "dark", "system"):
            raise ValueError(f"Invalid theme: {theme}. Must be 'light', 'dark', or 'system'")

        if theme == self._theme:
            return

        self._theme = theme
        self.apply_theme()
        self._save_theme()

        # 通知所有订阅者
        self._notify_callbacks(theme)

        logger.info(f"Theme changed to: {theme}")

    def get_current_theme(self) -> str:
        """
        获取当前主题设置

        Returns:
            当前主题名称: "light", "dark", 或 "system"
        """
        return self._theme

    def on_theme_change(self, callback: Callable[[str], None]) -> None:
        """
        订阅主题变化事件

        Args:
            callback: 回调函数，接收新主题名称作为参数
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered theme change callback: {callback}")

    def remove_callback(self, callback: Callable[[str], None]) -> None:
        """
        取消订阅主题变化事件

        Args:
            callback: 要移除的回调函数
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Removed theme change callback: {callback}")

    def _notify_callbacks(self, theme: str) -> None:
        """通知所有回调函数主题已变化"""
        for callback in self._callbacks:
            try:
                callback(theme)
            except Exception as e:
                logger.error(f"Error in theme change callback: {e}")

    def apply_theme(self) -> None:
        """
        应用当前主题到CustomTkinter

        注意: 这个方法必须在创建任何CTk窗口之前调用
        """
        try:
            import customtkinter as ctk

            # 设置外观模式
            if self._theme == "system":
                ctk.set_appearance_mode("system")
            else:
                ctk.set_appearance_mode(self._theme)

            logger.info(f"Applied theme to CustomTkinter: {self._theme}")
        except ImportError:
            logger.error("Failed to import customtkinter")
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")


# 便捷函数 - 获取单例实例
def get_theme_manager() -> ThemeManager:
    """获取ThemeManager单例实例"""
    return ThemeManager()


# 便捷函数 - 快速获取颜色
def get_color(name: str, mode: str = "auto") -> str | tuple[str, str]:
    """
    快速获取颜色值（使用单例）

    Args:
        name: 颜色名称
        mode: 颜色模式 ("auto", "light", "dark")

    Returns:
        颜色值
    """
    return get_theme_manager().get_color(name, mode)


# 向后兼容的导出
__all__ = [
    "ThemeManager",
    "get_theme_manager",
    "get_color",
    "NORD_COLORS",
    "SPACING",
    "FONTS",
    "CORNER_RADIUS",
]
