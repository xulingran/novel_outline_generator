"""
图标组件

集成 Phosphor Icons，提供统一的图标风格和样式。

Phosphor Icons: https://phosphoricons.com/
"""

import logging
from enum import Enum

import customtkinter as ctk

from gui.theme_manager import get_color

logger = logging.getLogger(__name__)


class IconSize(Enum):
    """图标尺寸"""

    XS = 16
    SM = 20
    MD = 24
    LG = 32
    XL = 40
    XXL = 48


class IconWeight(Enum):
    """图标权重 (Phosphor 风格)"""

    REGULAR = "regular"
    BOLD = "bold"
    FILL = "fill"
    DUOTONE = "duotone"
    THIN = "thin"
    LIGHT = "light"


# Phosphor Icons 路径数据 (精选常用图标)
# 来源: https://phosphoricons.com/
# 使用 SVG path 数据在 Canvas 上绘制
PHOSPHOR_ICONS = {
    # 导航图标
    "house": {
        "regular": "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
        "fill": "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M12 3L3 11v9h18v-9L12 3z",
        "bold": "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
    },
    "gear": {
        "regular": "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z",
        "fill": "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z",
        "bold": "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    },
    "file-text": {
        "regular": "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8",
        "fill": "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6",
        "bold": "M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14.5 2v6h6 M16 13H8 M16 17H8 M10 9H8",
    },
    "clock": {
        "regular": "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z M12 6v6l4 2",
        "fill": "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z",
        "bold": "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 6v6l4 2",
    },
    "check-circle": {
        "regular": "M22 11.08V12a10 10 0 1 1-5.93-9.14 M22 4L12 14.01l-3-3",
        "fill": "M22 11.08V12a10 10 0 1 1-5.93-9.14",
        "bold": "M22 11.08V12a10 10 0 1 1-5.93-9.14 M22 4L12 14.01l-3-3",
    },
    "warning": {
        "regular": "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
        "fill": "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
        "bold": "M12 9v4 M12 17h.01 M3.5 18l7-12 7 12H3.5z",
    },
    "warning-circle": {
        "regular": "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z M12 8v4 M12 16h.01",
        "fill": "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z",
        "bold": "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 8v4 M12 16h.01",
    },
    "x-circle": {
        "regular": "M22 11.08V12a10 10 0 1 1-5.93-9.14 M8.5 9.5l7 7 M15.5 9.5l-7 7",
        "fill": "M22 11.08V12a10 10 0 1 1-5.93-9.14",
        "bold": "M22 11.08V12a10 10 0 1 1-5.93-9.14 M8.5 9.5l7 7 M15.5 9.5l-7 7",
    },
    "info": {
        "regular": "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z M12 16v-4 M12 8h.01",
        "fill": "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z",
        "bold": "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 16v-4 M12 8h.01",
    },
    "play": {
        "regular": "M5 3l14 9-14 9V3z",
        "fill": "M5 3l14 9-14 9V3z",
        "bold": "M7 5v14l11-7-11-7z",
    },
    "pause": {
        "regular": "M6 4h4v16H6zM14 4h4v16h-4z",
        "fill": "M6 4h4v16H6zM14 4h4v16h-4z",
        "bold": "M5 4h6v16H5zM13 4h6v16h-6z",
    },
    "stop": {
        "regular": "M6 6h12v12H6z",
        "fill": "M6 6h12v12H6z",
        "bold": "M5 5h14v14H5z",
    },
    "x": {
        "regular": "M6 6l12 12M18 6L6 18",
        "fill": "M6 6l12 12M18 6L6 18",
        "bold": "M6 6l12 12M18 6L6 18",
    },
    "plus": {
        "regular": "M12 5v14M5 12h14",
        "fill": "M12 5v14M5 12h14",
        "bold": "M12 5v14M5 12h14",
    },
    "folder": {
        "regular": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
        "fill": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
        "bold": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
    },
    "folder-open": {
        "regular": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z M2 14h20",
        "fill": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
        "bold": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z M2 14h20",
    },
    "moon": {
        "regular": "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z",
        "fill": "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z",
        "bold": "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z",
    },
    "sun": {
        "regular": "M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41 M12 6a6 6 0 1 0 0 12 6 6 0 0 0 0-12z",
        "fill": "M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41 M12 6a6 6 0 1 0 0 12 6 6 0 0 0 0-12z",
        "bold": "M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41 M12 6a6 6 0 1 0 0 12 6 6 0 0 0 0-12z",
    },
    "desktop": {
        "regular": "M21 4H3a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7l-2 3v1h8v-1l-2-3h7a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z",
        "fill": "M21 4H3a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7l-2 3v1h8v-1l-2-3h7a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z",
        "bold": "M21 4H3a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7l-2 3v1h8v-1l-2-3h7a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z",
    },
    "terminal": {
        "regular": "M4 17l6-6-6-6M12 19h8",
        "fill": "M4 17l6-6-6-6M12 19h8",
        "bold": "M4 17l6-6-6-6M12 19h8",
    },
    "activity": {
        "regular": "M22 12h-4l-3 9L9 3l-3 9H2",
        "fill": "M22 12h-4l-3 9L9 3l-3 9H2",
        "bold": "M22 12h-4l-3 9L9 3l-3 9H2",
    },
    "chart-line": {
        "regular": "M3 3v18h18 M18.7 8l-5.1 5.2-2.8-2.7L7 14.3",
        "fill": "M3 3v18h18",
        "bold": "M3 3v18h18 M18.7 8l-5.1 5.2-2.8-2.7L7 14.3",
    },
    "list": {
        "regular": "M8 6h13 M8 12h13 M8 18h13 M3 6h.01 M3 12h.01 M3 18h.01",
        "fill": "M8 6h13 M8 12h13 M8 18h13",
        "bold": "M8 6h13 M8 12h13 M8 18h13 M3 6h.01 M3 12h.01 M3 18h.01",
    },
    "sliders": {
        "regular": "M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6z M12 12v-3 M8 15v-6 M16 15v-3",
        "fill": "M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6z",
        "bold": "M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6z M12 12v-3 M8 15v-6 M16 15v-3",
    },
    "funnel": {
        "regular": "M22 3H2l8 9.46V19l4 2v-8.54L22 3z",
        "fill": "M22 3H2l8 9.46V19l4 2v-8.54L22 3z",
        "bold": "M22 3H2l8 9.46V19l4 2v-8.54L22 3z",
    },
    "trash": {
        "regular": "M3 6h18 M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2",
        "fill": "M3 6h18 M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2",
        "bold": "M3 6h18 M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2",
    },
    "download": {
        "regular": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
        "fill": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
        "bold": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
    },
    "upload": {
        "regular": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
        "fill": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
        "bold": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
    },
    "rocket": {
        "regular": "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0 M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5 M15 12l5-5",
        "fill": "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z M15 12l5-5",
        "bold": "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0 M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5 M15 12l5-5",
    },
    "books": {
        "regular": "M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20 M4 10h16 M8 2v18",
        "fill": "M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20",
        "bold": "M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20",
    },
}


class Icon(ctk.CTkCanvas):
    """
    图标组件

    基于 Canvas 绘制 Phosphor 风格图标。

    Args:
        master: 父容器
        name: 图标名称
        size: 图标尺寸，默认 MD (24px)
        weight: 图标权重，默认 REGULAR
        color: 图标颜色，默认自适应
        **kwargs: 其他 Canvas 参数
    """

    # 24x24 的标准 viewBox (模拟 SVG)
    VIEWBOX_SIZE = 24

    def __init__(
        self,
        master,
        name: str,
        size: IconSize | int = IconSize.MD,
        weight: IconWeight = IconWeight.REGULAR,
        color: str | tuple[str, str] | None = None,
        **kwargs,
    ):
        # 处理尺寸
        if isinstance(size, IconSize):
            pixel_size = size.value
        else:
            pixel_size = int(size)

        # 设置画布大小
        super().__init__(master, width=pixel_size, height=pixel_size, **kwargs)

        self._name = name
        self._weight = weight
        self._color = color or get_color("fg_primary", mode="auto")
        self._size = pixel_size

        # 绘制图标
        self._draw()

    def _get_path_data(self) -> str | None:
        """获取图标路径数据"""
        weight_key = self._weight.value

        # 如果请求的权重不存在，回退到 regular
        if self._name in PHOSPHOR_ICONS:
            if weight_key in PHOSPHOR_ICONS[self._name]:
                return PHOSPHOR_ICONS[self._name][weight_key]
            elif "regular" in PHOSPHOR_ICONS[self._name]:
                return PHOSPHOR_ICONS[self._name]["regular"]

        # 如果图标不存在，返回占位符
        logger.warning(f"Icon '{self._name}' not found, using placeholder")
        return "M12 2L2 22h20L12 2z"

    def _draw(self) -> None:
        """绘制图标"""
        self.delete("all")

        path_data = self._get_path_data()
        if not path_data:
            return

        # 计算缩放比例
        scale = self._size / self.VIEWBOX_SIZE

        # 解析路径数据并绘制
        self._render_path(path_data, scale)

    def _render_path(self, path_data: str, scale: float) -> None:
        """
        渲染 SVG 路径数据

        简化的 SVG 路径解析器，支持 M, L, H, V, A 命令
        """
        # 获取颜色
        color = self._color
        if isinstance(color, tuple):
            # 自适应颜色：根据当前主题选择
            appearance = ctk.get_appearance_mode()
            color = color[0] if appearance == "Light" else color[1]

        # 简化处理：将路径转换为多边形
        # 这里做一个简化的实现，将 SVG 路径转换为线条
        commands = self._parse_path(path_data)

        for cmd in commands:
            if cmd["type"] == "M":
                # 移动命令
                self._start_point = self._scale_point(cmd["x"], cmd["y"], scale)
            elif cmd["type"] == "L":
                # 直线命令
                if hasattr(self, "_start_point"):
                    end = self._scale_point(cmd["x"], cmd["y"], scale)
                    self.create_line(
                        self._start_point[0],
                        self._start_point[1],
                        end[0],
                        end[1],
                        fill=color,
                        width=2,
                        capstyle=ctk.ROUND,
                    )
                    self._start_point = end
            elif cmd["type"] == "H":
                # 水平线
                if hasattr(self, "_start_point"):
                    end = self._scale_point(cmd["x"], self._start_point[1] / scale, scale)
                    self.create_line(
                        self._start_point[0],
                        self._start_point[1],
                        end[0],
                        end[1],
                        fill=color,
                        width=2,
                        capstyle=ctk.ROUND,
                    )
                    self._start_point = end
            elif cmd["type"] == "V":
                # 垂直线
                if hasattr(self, "_start_point"):
                    end = self._scale_point(self._start_point[0] / scale, cmd["y"], scale)
                    self.create_line(
                        self._start_point[0],
                        self._start_point[1],
                        end[0],
                        end[1],
                        fill=color,
                        width=2,
                        capstyle=ctk.ROUND,
                    )
                    self._start_point = end
            elif cmd["type"] == "A":
                # 圆弧命令 - 简化为直线
                if hasattr(self, "_start_point"):
                    end = self._scale_point(cmd["x"], cmd["y"], scale)
                    self.create_line(
                        self._start_point[0],
                        self._start_point[1],
                        end[0],
                        end[1],
                        fill=color,
                        width=2,
                        capstyle=ctk.ROUND,
                    )
                    self._start_point = end

    def _parse_path(self, path_data: str) -> list[dict]:
        """解析 SVG 路径数据"""
        commands = []
        parts = path_data.split()

        i = 0
        while i < len(parts):
            part = parts[i]

            if part in "MLHVA":
                cmd_type = part
                i += 1

                if cmd_type in "ML":
                    # M x y 或 L x y
                    x = float(parts[i])
                    y = float(parts[i + 1])
                    commands.append({"type": cmd_type, "x": x, "y": y})
                    i += 2
                elif cmd_type == "H":
                    # H x
                    x = float(parts[i])
                    commands.append({"type": "H", "x": x})
                    i += 1
                elif cmd_type == "V":
                    # V y
                    y = float(parts[i])
                    commands.append({"type": "V", "y": y})
                    i += 1
                elif cmd_type == "A":
                    # A rx ry rotation large-arc sweep x y
                    x = float(parts[i + 5])
                    y = float(parts[i + 6])
                    commands.append({"type": "A", "x": x, "y": y})
                    i += 7

        return commands

    def _scale_point(self, x: float, y: float, scale: float) -> tuple[int, int]:
        """缩放坐标点"""
        return (int(x * scale), int(y * scale))

    def configure(self, **kwargs) -> None:
        """
        配置图标属性

        支持的参数:
        - name: 图标名称
        - size: 图标尺寸
        - weight: 图标权重
        - color: 图标颜色
        """
        if "name" in kwargs:
            self._name = kwargs.pop("name")
        if "weight" in kwargs:
            self._weight = kwargs.pop("weight")
        if "color" in kwargs:
            self._color = kwargs.pop("color")
        if "size" in kwargs:
            size = kwargs.pop("size")
            if isinstance(size, IconSize):
                self._size = size.value
            else:
                self._size = int(size)
            # 更新画布大小
            self.configure(width=self._size, height=self._size)

        super().configure(**kwargs)
        self._draw()

    @classmethod
    def register_icon(cls, name: str, paths: dict[str, str]) -> None:
        """
        注册自定义图标

        Args:
            name: 图标名称
            paths: 路径数据字典，格式: {"regular": "path", "bold": "path", ...}
        """
        PHOSPHOR_ICONS[name] = paths
