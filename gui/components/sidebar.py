"""
侧边导航栏组件

提供左侧导航菜单，支持激活指示器、图标、主题切换器。
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


# 导航项定义
NAV_ITEMS = [
    NavItemSpec(NavItem.PROCESS, "处理", "rocket"),
    NavItemSpec(NavItem.CONFIG, "配置", "sliders"),
    NavItemSpec(NavItem.LOG, "日志", "terminal"),
    NavItemSpec(NavItem.ABOUT, "关于", "info"),
]


class Sidebar(ctk.CTkFrame):
    """
    侧边导航栏

    提供左侧导航菜单，包含 Logo、导航项、激活指示器和主题切换器。

    Args:
        master: 父容器
        width: 侧边栏宽度，默认 240px
        active_item: 当前激活的导航项
        on_navigation: 导航回调函数，接收 NavItem 参数
        **kwargs: 其他 Frame 参数
    """

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

        super().__init__(
            master,
            width=width,
            fg_color=get_color("bg_secondary", mode="auto"),
            corner_radius=0,
            **kwargs,
        )

        self._setup_ui()
        self._update_active_state()

    def _setup_ui(self):
        """设置 UI"""
        # 禁止自动收缩宽度
        self.pack_propagate(False)

        # Logo 区域
        self._setup_logo()

        # 导航区域
        self._setup_navigation()

        # 底部区域（主题切换）
        self._setup_bottom()

    def _setup_logo(self):
        """设置 Logo 区域"""
        logo_frame = ctk.CTkFrame(self, fg_color="transparent", height=64)
        logo_frame.pack(fill="x", pady=(SPACING["lg"], SPACING["lg"]))
        logo_frame.pack_propagate(False)

        # Logo 图标
        try:
            from gui.components.icon import Icon, IconSize

            logo_icon = Icon(logo_frame, name="rocket", size=IconSize.LG)
            logo_icon.pack(side="left", padx=(SPACING["lg"], SPACING["sm"]))
        except Exception:
            pass

        # Logo 文字
        logo_label = ctk.CTkLabel(
            logo_frame,
            text="Novel\nOutline",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
            anchor="w",
        )
        logo_label.pack(side="left", fill="x", expand=True)

    def _setup_navigation(self):
        """设置导航区域"""
        nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        nav_frame.pack(fill="x", expand=True)

        for item_spec in NAV_ITEMS:
            button = self._create_nav_button(nav_frame, item_spec)
            button.pack(fill="x", padx=SPACING["md"], pady=(0, SPACING["xs"]))
            self._nav_buttons[item_spec.id] = button

    def _create_nav_button(self, parent, item_spec: NavItemSpec) -> ctk.CTkFrame:
        """创建导航按钮"""
        button = ctk.CTkFrame(
            parent,
            fg_color="transparent",
            corner_radius=SPACING["sm"],
            height=40,
        )
        button.pack_propagate(False)

        # 激活指示器（左侧竖线）
        indicator = ctk.CTkFrame(
            button,
            width=3,
            fg_color="transparent",
            corner_radius=SPACING["xs"],
        )
        indicator.pack(side="left", fill="y")
        button._indicator = indicator

        # 图标
        try:
            from gui.components.icon import Icon, IconSize

            icon = Icon(
                button,
                name=item_spec.icon,
                size=IconSize.SM,
                color=get_color("fg_secondary", mode="auto"),
            )
            icon.pack(side="left", padx=(SPACING["md"], SPACING["sm"]))
            button._icon = icon
        except Exception:
            pass

        # 文字
        label = ctk.CTkLabel(
            button,
            text=item_spec.label,
            font=ctk.CTkFont(size=14),
            text_color=get_color("fg_secondary", mode="auto"),
            anchor="w",
        )
        label.pack(side="left", fill="x", expand=True)
        button._label = label

        # 徽章（可选）
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

        # 绑定事件
        button.bind("<Button-1>", lambda e: self._on_nav_click(item_spec.id))
        label.bind("<Button-1>", lambda e: self._on_nav_click(item_spec.id))

        # 悬停效果
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
            button.configure(fg_color=get_color("bg_tertiary", mode="auto"))

    def _on_nav_leave(self, button: ctk.CTkFrame, item_id: NavItem):
        """导航项悬停离开"""
        if item_id != self._active_item:
            button.configure(fg_color="transparent")

    def _update_active_state(self):
        """更新激活状态"""
        for item_id, button in self._nav_buttons.items():
            is_active = item_id == self._active_item

            if is_active:
                # 激活状态
                button.configure(fg_color=get_color("bg_tertiary", mode="auto"))
                button._indicator.configure(fg_color=get_color("accent", mode="auto"))

                if hasattr(button, "_icon"):
                    button._icon.configure(color=get_color("accent", mode="auto"))
                if hasattr(button, "_label"):
                    button._label.configure(
                        text_color=get_color("accent", mode="auto"),
                        font=ctk.CTkFont(size=14, weight="bold"),
                    )
            else:
                # 非激活状态
                button.configure(fg_color="transparent")
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
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", side="bottom", pady=SPACING["lg"])

        # 分隔线
        separator = ctk.CTkFrame(
            bottom_frame,
            height=1,
            fg_color=get_color("border", mode="auto"),
        )
        separator.pack(fill="x", padx=SPACING["md"], pady=(0, SPACING["md"]))

        # 主题切换器
        theme_switcher = self._create_theme_switcher(bottom_frame)
        theme_switcher.pack(fill="x", padx=SPACING["md"])

    def _create_theme_switcher(self, parent) -> ctk.CTkFrame:
        """创建主题切换器"""
        switcher = ctk.CTkFrame(
            parent,
            fg_color="transparent",
            height=40,
        )
        switcher.pack_propagate(False)

        # 当前主题显示
        try:
            from gui.theme_manager import get_theme_manager

            theme_manager = get_theme_manager()
            current_theme = theme_manager.get_current_theme()

            theme_label = ctk.CTkLabel(
                switcher,
                text=self._get_theme_label(current_theme),
                font=ctk.CTkFont(size=13),
                text_color=get_color("fg_secondary", mode="auto"),
                anchor="w",
            )
            theme_label.pack(side="left", fill="x", expand=True)
            switcher._theme_label = theme_label

        except Exception as e:
            logger.debug(f"Failed to get theme: {e}")

        # 切换按钮
        try:
            from gui.components.icon import Icon, IconSize

            toggle_button = ctk.CTkFrame(
                switcher,
                fg_color="transparent",
                width=32,
                height=32,
            )
            toggle_button.pack(side="right")
            toggle_button.pack_propagate(False)

            # 绑定点击事件
            toggle_button.bind("<Button-1>", lambda e: self._toggle_theme())
            toggle_button._bind_label = theme_label

            # 图标
            toggle_icon = Icon(
                toggle_button,
                name="moon",
                size=IconSize.SM,
                color=get_color("fg_primary", mode="auto"),
            )
            toggle_icon.place(relx=0.5, rely=0.5, anchor="center")
            switcher._toggle_icon = toggle_icon

            # 悬停效果
            toggle_button.bind(
                "<Enter>",
                lambda e: toggle_button.configure(fg_color=get_color("bg_tertiary", mode="auto")),
            )
            toggle_button.bind("<Leave>", lambda e: toggle_button.configure(fg_color="transparent"))

        except Exception as e:
            logger.debug(f"Failed to create theme toggle: {e}")

        return switcher

    def _get_theme_label(self, theme: str) -> str:
        """获取主题标签"""
        labels = {
            "light": "浅色",
            "dark": "深色",
            "system": "跟随系统",
        }
        return labels.get(theme, theme)

    def _toggle_theme(self):
        """切换主题"""
        try:
            from gui.theme_manager import get_theme_manager

            theme_manager = get_theme_manager()
            current = theme_manager.get_current_theme()

            # 循环切换: dark -> light -> system -> dark
            themes = ["dark", "light", "system"]
            current_index = themes.index(current) if current in themes else 0
            next_theme = themes[(current_index + 1) % len(themes)]

            theme_manager.set_theme(next_theme)

            # 更新显示
            if hasattr(self, "_theme_label"):
                self._theme_label.configure(text=self._get_theme_label(next_theme))

            # 更新图标
            if hasattr(self, "_toggle_icon"):
                icon_name = (
                    "sun"
                    if next_theme == "light"
                    else "moon" if next_theme == "dark" else "desktop"
                )
                self._toggle_icon.configure(name=icon_name)

        except Exception as e:
            logger.error(f"Failed to toggle theme: {e}")

    def set_active_item(self, item_id: NavItem):
        """
        设置激活的导航项

        Args:
            item_id: 导航项 ID
        """
        self._active_item = item_id
        self._update_active_state()

    def get_active_item(self) -> NavItem:
        """获取当前激活的导航项"""
        return self._active_item
