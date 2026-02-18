"""
状态徽章组件

用于显示处理状态的视觉指示器。
"""

import logging
from enum import Enum
from typing import Literal

import customtkinter as ctk

from gui.theme_manager import get_color

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """处理状态枚举"""

    IDLE = "idle"
    PROCESSING = "processing"
    MERGING = "merging"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


STATUS_CONFIG: dict[ProcessingStatus, dict] = {
    ProcessingStatus.IDLE: {
        "text": "等待开始",
        "bg_color": ("#E5E7EB", "#374151"),
        "text_color": ("#6B7280", "#9CA3AF"),
        "icon": "○",
    },
    ProcessingStatus.PROCESSING: {
        "text": "处理中",
        "bg_color": ("#DBEAFE", "#1E3A5F"),
        "text_color": ("#2563EB", "#60A5FA"),
        "icon": "◐",
    },
    ProcessingStatus.MERGING: {
        "text": "合并中",
        "bg_color": ("#FEF3C7", "#78350F"),
        "text_color": ("#D97706", "#FBBF24"),
        "icon": "◈",
    },
    ProcessingStatus.COMPLETED: {
        "text": "已完成",
        "bg_color": ("#D1FAE5", "#064E3B"),
        "text_color": ("#059669", "#34D399"),
        "icon": "✓",
    },
    ProcessingStatus.CANCELLED: {
        "text": "已取消",
        "bg_color": ("#FEF3C7", "#78350F"),
        "text_color": ("#D97706", "#FBBF24"),
        "icon": "○",
    },
    ProcessingStatus.FAILED: {
        "text": "失败",
        "bg_color": ("#FEE2E2", "#7F1D1D"),
        "text_color": ("#DC2626", "#F87171"),
        "icon": "✗",
    },
}


class StatusBadge(ctk.CTkFrame):
    """
    状态徽章组件

    显示当前处理状态的视觉指示器。

    Args:
        master: 父容器
        status: 初始状态
        size: 徽章大小 ("sm" | "md" | "lg")
    """

    def __init__(
        self,
        master,
        status: ProcessingStatus = ProcessingStatus.IDLE,
        size: Literal["sm", "md", "lg"] = "md",
    ):
        self._status = status
        self._size = size

        self._size_config = {
            "sm": {"height": 24, "padding": 8, "font_size": 11},
            "md": {"height": 28, "padding": 12, "font_size": 12},
            "lg": {"height": 32, "padding": 16, "font_size": 13},
        }

        super().__init__(master, fg_color="transparent")

        self._setup_ui()
        self._update_appearance()

    def _setup_ui(self):
        """设置 UI"""
        config = self._size_config[self._size]

        self._container = ctk.CTkFrame(
            self,
            fg_color=get_color("bg_tertiary", mode="auto"),
            corner_radius=config["height"] // 2,
            height=config["height"],
        )
        self._container.pack(fill="x")

        self._icon_label = ctk.CTkLabel(
            self._container,
            text="",
            font=ctk.CTkFont(size=config["font_size"]),
            width=0,
        )
        self._icon_label.pack(side="left", padx=(config["padding"], 2))

        self._text_label = ctk.CTkLabel(
            self._container,
            text="",
            font=ctk.CTkFont(size=config["font_size"], weight="bold"),
        )
        self._text_label.pack(side="left", padx=(0, config["padding"]))

    def _update_appearance(self):
        """更新外观"""
        config = STATUS_CONFIG[self._status]

        bg_color = config["bg_color"]
        text_color = config["text_color"]

        appearance = ctk.get_appearance_mode()
        bg = bg_color[0] if appearance == "Light" else bg_color[1]
        fg = text_color[0] if appearance == "Light" else text_color[1]

        self._container.configure(fg_color=bg)
        self._icon_label.configure(text=config["icon"], text_color=fg)
        self._text_label.configure(text=config["text"], text_color=fg)

    def set_status(self, status: ProcessingStatus):
        """
        设置状态

        Args:
            status: 新的处理状态
        """
        if status != self._status:
            self._status = status
            self._update_appearance()

    def get_status(self) -> ProcessingStatus:
        """获取当前状态"""
        return self._status

    def refresh_theme(self):
        """刷新主题"""
        self._update_appearance()
