"""
侧边导航栏组件

提供左侧导航菜单，支持激活指示器、图标和折叠状态。
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class NavItem(Enum):
    """导航项标识"""

    PROCESS = "process"
    CONFIG = "config"
    LOG = "log"
    ABOUT = "about"


@dataclass
class NavItemSpec:
    """导航项规格"""

    id: NavItem
    label: str
    icon: str
    badge: str | None = None


NAV_ITEMS = [
    NavItemSpec(NavItem.PROCESS, "处理", "rocket"),
    NavItemSpec(NavItem.CONFIG, "配置", "sliders"),
    NavItemSpec(NavItem.LOG, "日志", "terminal"),
    NavItemSpec(NavItem.ABOUT, "关于", "info"),
]


class Sidebar(ctk.CTkFrame):
    """
    侧边导航栏

    提供左侧导航菜单，包含品牌区、导航项、激活指示器和底部说明。
    支持宽度调整和折叠功能。
    """

    MIN_WIDTH = 180
    MAX_WIDTH = 320
    COLLAPSED_WIDTH = 60

    def __init__(
        self,
        master,
        width: int = 240,
        active_item: NavItem = NavItem.PROCESS,
        on_navigation: Callable[..., None] | None = None,
        **kwargs,
    ):
        self._width = width
        self._active_item = active_item
        self._on_navigation = on_navigation
        self._nav_buttons: dict[NavItem, ctk.CTkFrame] = {}
        self._collapsed = False
        self._dragging = False
        self._drag_start_x = 0
        self._drag_start_width = 0
        self._pending_width = width
        self._resize_after_id: str | None = None

        super().__init__(
            master,
            width=width,
            fg_color=get_color("bg_tertiary", mode="auto"),
            corner_radius=0,
            **kwargs,
        )

        self._setup_ui()
        self._setup_resize_handle()
        self._update_active_state()

    def _setup_ui(self):
        """设置 UI"""
        self.pack_propagate(False)

        self._content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._content_frame.pack(fill="both", expand=True)

        self._setup_logo()
        self._setup_navigation()
        self._setup_bottom()

    def _setup_resize_handle(self):
        """设置拖拽调整宽度的手柄"""
        self._resize_handle = ctk.CTkFrame(
            self,
            width=4,
            fg_color="transparent",
            cursor="sb_h_double_arrow",
        )
        self._resize_handle.place(relx=1.0, rely=0, anchor="ne", relheight=1.0)

        self._resize_handle.bind("<Button-1>", self._start_resize)
        self._resize_handle.bind("<B1-Motion>", self._do_resize)
        self._resize_handle.bind("<ButtonRelease-1>", self._end_resize)
        self._resize_handle.bind("<Enter>", self._on_handle_enter)
        self._resize_handle.bind("<Leave>", self._on_handle_leave)

    def _on_handle_enter(self, event):
        """鼠标进入调整手柄"""
        self._resize_handle.configure(fg_color=get_color("accent", mode="auto"))

    def _on_handle_leave(self, event):
        """鼠标离开调整手柄"""
        if not self._dragging:
            self._resize_handle.configure(fg_color="transparent")

    def _start_resize(self, event):
        """开始调整宽度"""
        self._dragging = True
        self._drag_start_x = event.x_root
        self._drag_start_width = self._width
        self._pending_width = self._width
        self._resize_handle.configure(fg_color=get_color("accent", mode="auto"))

    def _do_resize(self, event):
        """调整宽度中"""
        if not self._dragging:
            return

        delta = event.x_root - self._drag_start_x
        new_width = self._drag_start_width + delta
        new_width = max(self.MIN_WIDTH, min(self.MAX_WIDTH, new_width))

        if not self._collapsed:
            self._width = new_width
            self._pending_width = new_width
            if self._resize_after_id is None:
                self._resize_after_id = self.after(16, self._apply_pending_width)

    def _apply_pending_width(self):
        """以较低频率应用侧边栏宽度，减少拖拽时的整窗重绘"""
        self._resize_after_id = None

        if self._collapsed:
            return

        current_width = int(self.cget("width"))
        if abs(current_width - self._pending_width) >= 1:
            self.configure(width=self._pending_width)

    def _end_resize(self, event):
        """结束调整宽度"""
        self._dragging = False
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
            self._resize_after_id = None
        self._apply_pending_width()
        self._resize_handle.configure(fg_color="transparent")

    def toggle_collapse(self):
        """切换折叠状态"""
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.configure(width=self.COLLAPSED_WIDTH)
            self._hide_labels()
        else:
            self.configure(width=self._width)
            self._show_labels()

    def _hide_labels(self):
        """隐藏标签（折叠模式）"""
        for widget_name in ("_brand_text_frame", "_footer_label"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.pack_forget()

        for button in self._nav_buttons.values():
            label = getattr(button, "_label", None)
            if label is not None:
                label.pack_forget()

    def _show_labels(self):
        """显示标签（展开模式）"""
        if hasattr(self, "_brand_text_frame") and self._brand_text_frame.winfo_exists():
            self._brand_text_frame.pack(fill="x", pady=(SPACING["md"], 0))

        if hasattr(self, "_footer_label") and self._footer_label.winfo_exists():
            self._footer_label.pack(fill="x")

        for button in self._nav_buttons.values():
            label = getattr(button, "_label", None)
            if label is not None and label.winfo_exists():
                label.pack(side="left", fill="x", expand=True)

    def _setup_logo(self):
        """设置顶部品牌区"""
        self._brand_wrapper = ctk.CTkFrame(self._content_frame, fg_color="transparent")
        self._brand_wrapper.pack(fill="x", padx=SPACING["md"], pady=(SPACING["xl"], SPACING["lg"]))

        brand_card = ctk.CTkFrame(
            self._brand_wrapper,
            fg_color=get_color("bg_secondary", mode="auto"),
            border_width=1,
            border_color=get_color("border", mode="auto"),
            corner_radius=14,
        )
        brand_card.pack(fill="x")

        self._brand_card = brand_card

        inner = ctk.CTkFrame(brand_card, fg_color="transparent")
        inner.pack(fill="x", padx=SPACING["md"], pady=SPACING["lg"])

        try:
            from gui.components.icon import Icon, IconSize

            self._logo_icon = Icon(inner, name="rocket", size=IconSize.LG)
            self._logo_icon.pack(anchor="center")
        except Exception:
            self._logo_icon = None

        self._brand_text_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self._brand_text_frame.pack(fill="x", pady=(SPACING["md"], 0))

        self._logo_label = ctk.CTkLabel(
            self._brand_text_frame,
            text="Novel\nOutline\nGenerator",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
            justify="center",
        )
        self._logo_label.pack(fill="x")

        self._brand_subtitle = ctk.CTkLabel(
            self._brand_text_frame,
            text="AI 驱动的小说大纲工作台",
            font=ctk.CTkFont(size=12),
            text_color=get_color("fg_secondary", mode="auto"),
            justify="center",
        )
        self._brand_subtitle.pack(fill="x", pady=(SPACING["xs"], 0))

    def _setup_navigation(self):
        """设置导航区域"""
        self._nav_frame = ctk.CTkFrame(self._content_frame, fg_color="transparent")
        self._nav_frame.pack(fill="x", expand=True, padx=SPACING["md"])

        for item_spec in NAV_ITEMS:
            button = self._create_nav_button(self._nav_frame, item_spec)
            button.pack(fill="x", pady=(0, SPACING["sm"]))
            self._nav_buttons[item_spec.id] = button

    def _create_nav_button(self, parent, item_spec: NavItemSpec) -> ctk.CTkFrame:
        """创建导航按钮"""
        button = ctk.CTkFrame(
            parent,
            fg_color="transparent",
            border_width=0,
            border_color=get_color("border", mode="auto"),
            corner_radius=12,
            height=46,
        )
        button.pack_propagate(False)

        indicator = ctk.CTkFrame(
            button,
            width=4,
            fg_color="transparent",
            corner_radius=SPACING["xs"],
        )
        indicator.pack(side="left", fill="y", padx=(0, SPACING["sm"]))
        button._indicator = indicator

        try:
            from gui.components.icon import Icon, IconSize

            icon = Icon(
                button,
                name=item_spec.icon,
                size=IconSize.SM,
                color=get_color("fg_secondary", mode="auto"),
            )
            icon.pack(side="left", padx=(SPACING["sm"], SPACING["sm"]))
            button._icon = icon
        except Exception:
            pass

        label = ctk.CTkLabel(
            button,
            text=item_spec.label,
            font=ctk.CTkFont(size=14),
            text_color=get_color("fg_secondary", mode="auto"),
            anchor="w",
        )
        label.pack(side="left", fill="x", expand=True)
        button._label = label

        if item_spec.badge:
            badge = ctk.CTkLabel(
                button,
                text=item_spec.badge,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=get_color("bg_primary", mode="auto"),
                fg_color=get_color("accent", mode="auto"),
                corner_radius=SPACING["xs"],
                padx=SPACING["xs"],
            )
            badge.pack(side="right", padx=(0, SPACING["sm"]))

        click_targets = [button, label]
        if hasattr(button, "_icon"):
            click_targets.append(button._icon)

        for target in click_targets:
            target.bind("<Button-1>", lambda e, item_id=item_spec.id: self._on_nav_click(item_id))

        button.bind("<Enter>", lambda e: self._on_nav_enter(button, item_spec.id))
        button.bind("<Leave>", lambda e: self._on_nav_leave(button, item_spec.id))

        button._item_id = item_spec.id
        return button

    def _on_nav_click(self, item_id: NavItem):
        """导航项点击事件"""
        if self._on_navigation:
            self._on_navigation(item_id)

    def _on_nav_enter(self, button: ctk.CTkFrame, item_id: NavItem):
        """导航项悬停进入"""
        if item_id != self._active_item:
            button.configure(
                fg_color=get_color("bg_secondary", mode="auto"),
                border_color=get_color("border", mode="auto"),
                border_width=1,
            )

    def _on_nav_leave(self, button: ctk.CTkFrame, item_id: NavItem):
        """导航项悬停离开"""
        if item_id != self._active_item:
            button.configure(
                fg_color="transparent",
                border_color=get_color("border", mode="auto"),
                border_width=0,
            )

    def _update_active_state(self):
        """更新激活状态"""
        for item_id, button in self._nav_buttons.items():
            is_active = item_id == self._active_item

            if is_active:
                button.configure(
                    fg_color=get_color("bg_secondary", mode="auto"),
                    border_color=get_color("border", mode="auto"),
                    border_width=1,
                )
                button._indicator.configure(fg_color=get_color("accent", mode="auto"))

                if hasattr(button, "_icon"):
                    button._icon.configure(color=get_color("accent", mode="auto"))
                if hasattr(button, "_label"):
                    button._label.configure(
                        text_color=get_color("fg_primary", mode="auto"),
                        font=ctk.CTkFont(size=14, weight="bold"),
                    )
            else:
                button.configure(
                    fg_color="transparent",
                    border_color=get_color("border", mode="auto"),
                    border_width=0,
                )
                button._indicator.configure(fg_color="transparent")

                if hasattr(button, "_icon"):
                    button._icon.configure(color=get_color("fg_secondary", mode="auto"))
                if hasattr(button, "_label"):
                    button._label.configure(
                        text_color=get_color("fg_secondary", mode="auto"),
                        font=ctk.CTkFont(size=14),
                    )

    def _setup_bottom(self):
        """设置底部区域"""
        self._bottom_frame = ctk.CTkFrame(self._content_frame, fg_color="transparent")
        self._bottom_frame.pack(fill="x", side="bottom", padx=SPACING["md"], pady=SPACING["lg"])

        separator = ctk.CTkFrame(
            self._bottom_frame,
            height=1,
            fg_color=get_color("border", mode="auto"),
        )
        separator.pack(fill="x", pady=(0, SPACING["md"]))
        self._footer_separator = separator

        self._footer_label = ctk.CTkLabel(
            self._bottom_frame,
            text="在“关于”页中切换外观模式",
            font=ctk.CTkFont(size=11),
            text_color=get_color("fg_tertiary", mode="auto"),
            justify="center",
        )
        self._footer_label.pack(fill="x")

    def refresh_theme(self):
        """刷新主题相关样式"""
        self.configure(fg_color=get_color("bg_tertiary", mode="auto"))

        if hasattr(self, "_brand_card"):
            self._brand_card.configure(
                fg_color=get_color("bg_secondary", mode="auto"),
                border_color=get_color("border", mode="auto"),
            )

        if hasattr(self, "_logo_label"):
            self._logo_label.configure(text_color=get_color("fg_primary", mode="auto"))

        if hasattr(self, "_brand_subtitle"):
            self._brand_subtitle.configure(text_color=get_color("fg_secondary", mode="auto"))

        if hasattr(self, "_footer_separator"):
            self._footer_separator.configure(fg_color=get_color("border", mode="auto"))

        if hasattr(self, "_footer_label"):
            self._footer_label.configure(text_color=get_color("fg_tertiary", mode="auto"))

        self._update_active_state()

    def set_active_item(self, item_id: NavItem):
        """设置激活的导航项"""
        self._active_item = item_id
        self._update_active_state()

    def get_active_item(self) -> NavItem:
        """获取当前激活的导航项"""
        return self._active_item
