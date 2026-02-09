"""
卡片组件

统一的卡片容器，用于包裹功能区块。
"""

import logging

import customtkinter as ctk

from gui.theme_manager import CORNER_RADIUS, SPACING, get_color

logger = logging.getLogger(__name__)


class Card(ctk.CTkFrame):
    """
    卡片组件

    提供统一的容器样式，用于分组相关内容。

    Args:
        master: 父容器
        title: 卡片标题（可选）
        subtitle: 卡片副标题（可选）
        variant: 卡片样式变体 ("default", "bordered", "elevated")
        padding: 内边距大小
        **kwargs: 其他 Frame 参数
    """

    def __init__(
        self,
        master,
        title: str | None = None,
        subtitle: str | None = None,
        variant: str = "default",
        padding: str | int = "md",
        **kwargs,
    ):
        # 处理内边距
        if isinstance(padding, str):
            padding_value = SPACING.get(padding, SPACING["md"])
        else:
            padding_value = int(padding)

        # 设置卡片样式
        match variant:
            case "bordered":
                fg_color = get_color("bg_secondary", mode="auto")
                border_color = get_color("border", mode="auto")
                border_width = 1
            case "elevated":
                fg_color = get_color("bg_secondary", mode="auto")
                border_color = "transparent"
                border_width = 0
            case _:
                fg_color = get_color("bg_secondary", mode="auto")
                border_color = "transparent"
                border_width = 0

        super().__init__(
            master,
            fg_color=fg_color,
            border_color=border_color,
            border_width=border_width,
            corner_radius=CORNER_RADIUS["lg"],
            **kwargs,
        )

        self._title = title
        self._subtitle = subtitle
        self._padding = padding_value
        self._variant = variant

        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        # 内容容器
        self._content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._content_frame.pack(fill="both", expand=True, padx=self._padding, pady=self._padding)

        # 标题区域
        if self._title:
            self._title_frame = ctk.CTkFrame(self._content_frame, fg_color="transparent")
            self._title_frame.pack(fill="x", pady=(0, SPACING["md"]))

            title_label = ctk.CTkLabel(
                self._title_frame,
                text=self._title,
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=get_color("fg_primary", mode="auto"),
                anchor="w",
            )
            title_label.pack(fill="x")

            if self._subtitle:
                subtitle_label = ctk.CTkLabel(
                    self._title_frame,
                    text=self._subtitle,
                    font=ctk.CTkFont(size=13),
                    text_color=get_color("fg_secondary", mode="auto"),
                    anchor="w",
                )
                subtitle_label.pack(fill="x", pady=(SPACING["xs"], 0))

        # 内容区域（子组件可以 pack 到这里）
        self.content = ctk.CTkFrame(self._content_frame, fg_color="transparent")
        self.content.pack(fill="both", expand=True)

    def add_widget(self, widget, **pack_kwargs):
        """
        添加子组件到内容区

        Args:
            widget: 要添加的组件
            **pack_kwargs: pack() 参数
        """
        default_kwargs = {"fill": "x", "pady": SPACING["sm"]}
        default_kwargs.update(pack_kwargs)
        widget.pack(self.content, **default_kwargs)

    def clear(self):
        """清空内容区"""
        for widget in self.content.winfo_children():
            widget.destroy()


class StatCard(Card):
    """
    统计卡片

    用于展示关键指标的大数字卡片。

    Args:
        master: 父容器
        title: 标题
        value: 数值
        unit: 单位
        icon: 图标名称
        trend: 趋势 (可选: "up", "down", "neutral")
        **kwargs: 其他 Card 参数
    """

    def __init__(
        self,
        master,
        title: str,
        value: str | int | float,
        unit: str = "",
        icon: str | None = None,
        trend: str | None = None,
        **kwargs,
    ):
        self._stat_value = str(value)
        self._stat_unit = unit
        self._stat_icon = icon
        self._stat_trend = trend

        super().__init__(master, title=title, padding="md", **kwargs)

        # 重新设置内容（统计卡片布局不同）
        self._setup_stat_content()

    def _setup_stat_content(self):
        """设置统计卡片内容"""
        # 清空默认内容
        for widget in self.content.winfo_children():
            widget.destroy()

        # 主内容区
        main_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        # 顶部：图标 + 趋势指示器
        top_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        top_frame.pack(fill="x")

        if self._stat_icon:
            try:
                from gui.components.icon import Icon

                icon_widget = Icon(main_frame, name=self._stat_icon, size=IconSize.MD)
                icon_widget.pack(side="left", padx=(0, SPACING["sm"]))
            except Exception:
                pass

        # 趋势指示器
        if self._stat_trend:
            trend_color = {
                "up": get_color("success", mode="auto"),
                "down": get_color("error", mode="auto"),
                "neutral": get_color("fg_secondary", mode="auto"),
            }.get(self._stat_trend, get_color("fg_secondary", mode="auto"))

            trend_symbols = {"up": "↑", "down": "↓", "neutral": "→"}
            trend_label = ctk.CTkLabel(
                top_frame,
                text=trend_symbols.get(self._stat_trend, ""),
                text_color=trend_color,
                font=ctk.CTkFont(size=14, weight="bold"),
            )
            trend_label.pack(side="right")

        # 数值显示
        value_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        value_frame.pack(fill="both", expand=True, pady=(SPACING["sm"], 0))

        self.value_label = ctk.CTkLabel(
            value_frame,
            text=f"{self._stat_value}{self._stat_unit}",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        self.value_label.pack(anchor="w")

    def update_value(self, value: str | int | float, unit: str = ""):
        """更新数值显示"""
        self.value_label.configure(text=f"{value}{unit}")


# 导入 IconSize（放在最后避免循环导入）
try:
    from gui.components.icon import IconSize
except ImportError:
    IconSize = None
