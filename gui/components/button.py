"""
按钮组件

三级按钮系统：主按钮、次按钮、文字按钮。
"""

import logging
from enum import Enum

import customtkinter as ctk

from gui.theme_manager import CORNER_RADIUS, get_color

logger = logging.getLogger(__name__)


class ButtonVariant(Enum):
    """按钮样式变体"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    DANGER = "danger"
    SUCCESS = "success"


class ButtonSize(Enum):
    """按钮尺寸"""

    SM = "small"
    MD = "medium"
    LG = "large"


class Button(ctk.CTkButton):
    """
    统一按钮组件

    支持多种样式变体和尺寸，符合设计系统规范。

    Args:
        master: 父容器
        text: 按钮文字
        variant: 样式变体
        size: 按钮尺寸
        icon: 图标名称 (可选)
        icon_position: 图标位置 ("left" 或 "right")
        loading: 是否显示加载状态
        **kwargs: 其他 CTkButton 参数
    """

    # 尺寸规范
    SIZE_SPECS = {
        ButtonSize.SM: {"height": 32, "padding": (16, 8), "font_size": 13},
        ButtonSize.MD: {"height": 40, "padding": (24, 12), "font_size": 14},
        ButtonSize.LG: {"height": 48, "padding": (32, 16), "font_size": 15},
    }

    def __init__(
        self,
        master,
        text: str = "",
        variant: ButtonVariant = ButtonVariant.PRIMARY,
        size: ButtonSize = ButtonSize.MD,
        icon: str | None = None,
        icon_position: str = "left",
        loading: bool = False,
        **kwargs,
    ):
        self._variant = variant
        self._size = size
        self._icon_name = icon
        self._icon_position = icon_position
        self._loading = loading
        self._original_text = text

        # 获取样式
        style = self._get_style()
        size_spec = self.SIZE_SPECS[size]

        # 设置尺寸
        height = size_spec["height"]
        if "height" not in kwargs:
            kwargs["height"] = height

        # 设置字体
        font = ctk.CTkFont(
            size=size_spec["font_size"],
            weight="bold" if variant == ButtonVariant.PRIMARY else "normal",
        )
        if "font" not in kwargs:
            kwargs["font"] = font

        # 设置圆角
        corner_radius = CORNER_RADIUS["md"]

        super().__init__(
            master,
            text=text,
            fg_color=style["fg_color"],
            hover_color=style["hover_color"],
            text_color=style["text_color"],
            border_color=style.get("border_color"),
            border_width=style.get("border_width", 0),
            corner_radius=corner_radius,
            **kwargs,
        )

        # 添加图标
        self._icon_widget = None
        if icon:
            self._add_icon()

        # 加载状态
        if loading:
            self.set_loading(True)

    def _get_style(self) -> dict:
        """获取按钮样式"""
        match self._variant:
            case ButtonVariant.PRIMARY:
                return {
                    "fg_color": get_color("accent", mode="auto"),
                    "hover_color": self._adjust_color(get_color("accent", mode="dark"), 10),
                    "text_color": get_color("bg_primary", mode="auto"),
                }
            case ButtonVariant.SECONDARY:
                return {
                    "fg_color": get_color("bg_tertiary", mode="auto"),
                    "hover_color": get_color("border", mode="auto"),
                    "text_color": get_color("fg_primary", mode="auto"),
                }
            case ButtonVariant.TERTIARY:
                return {
                    "fg_color": "transparent",
                    "hover_color": get_color("bg_tertiary", mode="auto"),
                    "text_color": get_color("fg_primary", mode="auto"),
                }
            case ButtonVariant.DANGER:
                return {
                    "fg_color": get_color("error", mode="auto"),
                    "hover_color": self._adjust_color(get_color("error", mode="dark"), 10),
                    "text_color": get_color("bg_primary", mode="auto"),
                }
            case ButtonVariant.SUCCESS:
                return {
                    "fg_color": get_color("success", mode="auto"),
                    "hover_color": self._adjust_color(get_color("success", mode="dark"), 10),
                    "text_color": get_color("bg_primary", mode="auto"),
                }
            case _:
                return {
                    "fg_color": get_color("accent", mode="auto"),
                    "hover_color": get_color("accent_secondary", mode="auto"),
                    "text_color": get_color("bg_primary", mode="auto"),
                }

    def _adjust_color(
        self, hex_color: str | tuple[str, str], percent: int
    ) -> str | tuple[str, str]:
        """调整颜色亮度（用于 hover 效果）"""
        # 简化实现：返回原色
        return hex_color

    def _add_icon(self):
        """添加图标到按钮"""
        try:
            from gui.components.icon import Icon, IconSize

            size = IconSize.SM if self._size == ButtonSize.SM else IconSize.MD

            self._icon_widget = Icon(
                self,
                name=self._icon_name,
                size=size,
                color=self.cget("text_color"),
            )

            # 重新排列组件
            self._rearrange_content()

        except Exception as e:
            logger.debug(f"Failed to add icon to button: {e}")

    def _rearrange_content(self):
        """重新排列按钮内容（图标 + 文字）"""
        # CustomTkinter 的按钮不直接支持图标，这里简化处理
        # 在实际实现中，可能需要自定义绘制或使用 Frame 包装
        pass

    def set_loading(self, loading: bool):
        """
        设置加载状态

        Args:
            loading: 是否显示加载状态
        """
        self._loading = loading

        if loading:
            self._original_text = self.cget("text")
            self.configure(text="...", state="disabled")
        else:
            self.configure(text=self._original_text, state="normal")

    def update_variant(self, variant: ButtonVariant):
        """更新按钮样式"""
        self._variant = variant
        style = self._get_style()
        self.configure(
            fg_color=style["fg_color"],
            hover_color=style["hover_color"],
            text_color=style["text_color"],
        )


class IconButton(ctk.CTkButton):
    """
    图标按钮

    只显示图标的按钮，常用于工具栏。

    Args:
        master: 父容器
        icon: 图标名称
        size: 按钮尺寸
        variant: 样式变体
        tooltip: 工具提示文字
        **kwargs: 其他参数
    """

    def __init__(
        self,
        master,
        icon: str,
        size: ButtonSize = ButtonSize.MD,
        variant: ButtonVariant = ButtonVariant.TERTIARY,
        tooltip: str | None = None,
        **kwargs,
    ):
        self._icon_name = icon
        self._tooltip = tooltip

        # 尺寸规范（图标按钮是正方形）
        size_map = {
            ButtonSize.SM: 32,
            ButtonSize.MD: 40,
            ButtonSize.LG: 48,
        }
        btn_size = size_map.get(size, 40)

        # 样式
        style = self._get_style(variant)

        super().__init__(
            master,
            width=btn_size,
            height=btn_size,
            fg_color=style["fg_color"],
            hover_color=style["hover_color"],
            text_color=style["text_color"],
            corner_radius=CORNER_RADIUS["md"],
            **kwargs,
        )

        # 添加图标
        self._add_icon()

        # 添加工具提示
        if tooltip:
            self._bind_tooltip()

    def _get_style(self, variant: ButtonVariant) -> dict:
        """获取样式"""
        match variant:
            case ButtonVariant.PRIMARY:
                return {
                    "fg_color": get_color("accent", mode="auto"),
                    "hover_color": get_color("accent_secondary", mode="auto"),
                    "text_color": get_color("bg_primary", mode="auto"),
                }
            case ButtonVariant.SECONDARY:
                return {
                    "fg_color": get_color("bg_tertiary", mode="auto"),
                    "hover_color": get_color("border", mode="auto"),
                    "text_color": get_color("fg_primary", mode="auto"),
                }
            case _:
                return {
                    "fg_color": "transparent",
                    "hover_color": get_color("bg_tertiary", mode="auto"),
                    "text_color": get_color("fg_primary", mode="auto"),
                }

    def _add_icon(self):
        """添加图标"""
        try:
            from gui.components.icon import Icon, IconSize

            icon_widget = Icon(
                self,
                name=self._icon_name,
                size=IconSize.MD,
                color=self.cget("text_color"),
            )
            icon_widget.place(relx=0.5, rely=0.5, anchor="center")

        except Exception as e:
            logger.debug(f"Failed to add icon to button: {e}")

    def _bind_tooltip(self):
        """绑定工具提示"""

        # 简化实现：使用鼠标悬停事件
        def on_enter(event):
            # 可以在这里显示 tooltip
            pass

        def on_leave(event):
            pass

        self.bind("<Enter>", on_enter)
        self.bind("<Leave>", on_leave)
