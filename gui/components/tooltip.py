"""
工具提示组件

为控件添加悬停提示功能。
"""

import logging
import tkinter as tk
from typing import Any

import customtkinter as ctk

logger = logging.getLogger(__name__)


class Tooltip:
    """
    工具提示

    当鼠标悬停在控件上时显示提示文字。

    Args:
        widget: 要绑定提示的控件
        text: 提示文字
        delay: 显示延迟（毫秒）
        bg_color: 背景颜色
        text_color: 文字颜色
    """

    def __init__(
        self,
        widget: Any,
        text: str,
        delay: int = 500,
        bg_color: str | tuple[str, str] | None = None,
        text_color: str | tuple[str, str] | None = None,
    ):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tooltip_window: tk.Toplevel | None = None
        self._after_id: str | None = None

        from gui.theme_manager import get_color

        self._bg_color = bg_color or get_color("bg_tertiary", mode="auto")
        self._text_color = text_color or get_color("fg_primary", mode="auto")

        self._bind_events()

    def _bind_events(self):
        """绑定鼠标事件"""
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        """鼠标进入"""
        self._schedule_show()

    def _on_leave(self, event=None):
        """鼠标离开"""
        self._cancel_show()
        self._hide()

    def _schedule_show(self):
        """安排显示"""
        self._cancel_show()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_show(self):
        """取消显示"""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        """显示提示"""
        if self._tooltip_window:
            return

        try:
            x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

            self._tooltip_window = tk.Toplevel(self.widget)
            self._tooltip_window.wm_overrideredirect(True)
            self._tooltip_window.wm_geometry(f"+{x}+{y}")
            self._tooltip_window.wm_attributes("-topmost", True)

            bg = self._resolve_color(self._bg_color)
            fg = self._resolve_color(self._text_color)

            label = tk.Label(
                self._tooltip_window,
                text=self.text,
                justify="center",
                background=bg,
                foreground=fg,
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=4,
                font=("Arial", 11),
            )
            label.pack()

        except Exception as e:
            logger.debug(f"Failed to show tooltip: {e}")
            self._tooltip_window = None

    def _hide(self):
        """隐藏提示"""
        if self._tooltip_window:
            try:
                self._tooltip_window.destroy()
            except Exception:
                pass
            self._tooltip_window = None

    def _resolve_color(self, color: str | tuple[str, str]) -> str:
        """解析颜色值"""
        if isinstance(color, tuple):
            appearance = ctk.get_appearance_mode()
            return color[0] if appearance == "Light" else color[1]
        return color

    def update_text(self, text: str):
        """更新提示文字"""
        self.text = text
        if self._tooltip_window:
            self._hide()
            self._show()


def add_tooltip(widget: Any, text: str, **kwargs) -> Tooltip:
    """
    为控件添加工具提示

    Args:
        widget: 要绑定提示的控件
        text: 提示文字
        **kwargs: 其他 Tooltip 参数

    Returns:
        Tooltip 实例
    """
    return Tooltip(widget, text, **kwargs)
