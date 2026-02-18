"""
图标组件

支持 PNG 图标资源和 Canvas 绘制后备方案。
设计原则：简洁、清晰、易于识别。
"""

import logging
from enum import Enum
from pathlib import Path

import customtkinter as ctk

from gui.theme_manager import get_color

logger = logging.getLogger(__name__)

ICONS_DIR = Path(__file__).parent.parent / "assets" / "icons"


class IconSize(Enum):
    """图标尺寸"""

    XS = 16
    SM = 20
    MD = 24
    LG = 32
    XL = 40
    XXL = 48


class IconWeight(Enum):
    """图标权重"""

    REGULAR = "regular"
    BOLD = "bold"
    LIGHT = "light"


_icon_cache: dict = {}


def _make_cache_key(name: str, size: int, color: str, appearance: str) -> str:
    """生成图标缓存键"""
    return f"{name}_{size}_{color}_{appearance}"


def _get_cached_icon(name: str, size: int, color: str, appearance: str):
    """获取缓存的图标"""
    cache_key = _make_cache_key(name, size, color, appearance)
    return _icon_cache.get(cache_key)


def _cache_icon(name: str, size: int, color: str, appearance: str, image) -> None:
    """缓存图标"""
    cache_key = _make_cache_key(name, size, color, appearance)
    _icon_cache[cache_key] = image


class Icon(ctk.CTkLabel):
    """
    图标组件

    优先使用 PNG 图标资源，后备使用 Canvas 绘制。

    Args:
        master: 父容器
        name: 图标名称
        size: 图标尺寸，默认 MD (24px)
        weight: 图标权重，默认 REGULAR
        color: 图标颜色，默认自适应
        **kwargs: 其他参数
    """

    AVAILABLE_ICONS = {
        "house",
        "gear",
        "settings",
        "file-text",
        "file",
        "clock",
        "check-circle",
        "check",
        "warning",
        "warning-circle",
        "x-circle",
        "x",
        "info",
        "play",
        "pause",
        "stop",
        "plus",
        "minus",
        "rocket",
        "folder",
        "folder-open",
        "moon",
        "sun",
        "desktop",
        "terminal",
        "activity",
        "list",
        "sliders",
        "funnel",
        "trash",
        "download",
        "upload",
        "book",
        "books",
        "document",
        "home",
    }

    def __init__(
        self,
        master,
        name: str,
        size: IconSize | int = IconSize.MD,
        weight: IconWeight = IconWeight.REGULAR,
        color: str | tuple[str, str] | None = None,
        **kwargs,
    ):
        if isinstance(size, IconSize):
            pixel_size = size.value
        else:
            pixel_size = int(size)

        self._name = self._normalize_name(name)
        self._weight = weight
        self._color = color or get_color("fg_primary", mode="auto")
        self._size = pixel_size

        super().__init__(master, text="", **kwargs)

        self._load_or_draw_icon()

    def _normalize_name(self, name: str) -> str:
        """标准化图标名称"""
        aliases = {
            "home": "house",
            "settings": "gear",
            "config": "gear",
        }
        return aliases.get(name.lower(), name.lower())

    def _get_color(self) -> str:
        """获取当前颜色"""
        color = self._color
        if isinstance(color, tuple):
            appearance = ctk.get_appearance_mode()
            color = color[0] if appearance == "Light" else color[1]
        return color

    def _load_or_draw_icon(self):
        """加载 PNG 图标或使用 Canvas 绘制"""
        if self._try_load_png_icon():
            return

        self._use_canvas_icon()

    def _try_load_png_icon(self) -> bool:
        """尝试加载 PNG 图标"""
        try:
            icon_path = self._resolve_png_icon_path()
            if icon_path is None:
                return False

            appearance = ctk.get_appearance_mode()
            color = self._get_color()
            cached = _get_cached_icon(self._name, self._size, color, appearance)
            if cached:
                self.configure(image=cached)
                return True

            from PIL import Image

            pil_image = Image.open(icon_path)

            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")  # type: ignore[assignment]

            pil_image = self._tint_image(pil_image, color)

            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(self._size, self._size),
            )

            _cache_icon(self._name, self._size, color, appearance, ctk_image)
            self.configure(image=ctk_image)
            return True

        except Exception as e:
            logger.debug(f"Failed to load PNG icon: {e}")
            return False

    def _resolve_png_icon_path(self) -> Path | None:
        """解析可用的 PNG 图标路径（兼容新旧资源结构）"""
        size_dir = self._get_size_directory()
        candidates = [
            ICONS_DIR / f"{self._name}_{size_dir}.png",
            ICONS_DIR / str(size_dir) / f"{self._name}.png",
            ICONS_DIR / f"{self._name.replace('-', '_')}_{size_dir}.png",
            ICONS_DIR / str(size_dir) / f"{self._name.replace('-', '_')}.png",
        ]

        for path in candidates:
            if path.exists():
                return path
        return None

    def _get_size_directory(self) -> int:
        """获取最接近的图标尺寸目录"""
        available_sizes = [16, 24, 32, 48]
        closest = min(available_sizes, key=lambda x: abs(x - self._size))
        return closest

    def _tint_image(self, image, color: str):
        """给图像着色"""
        try:

            if color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                r, g, b = 128, 128, 128

            pixels = image.load()
            width, height = image.size

            for x in range(width):
                for y in range(height):
                    if pixels[x, y][3] > 0:
                        alpha = pixels[x, y][3]
                        pixels[x, y] = (r, g, b, alpha)

            return image
        except Exception:
            return image

    def _use_canvas_icon(self):
        """使用 Canvas 绘制图标作为后备"""
        self._canvas = ctk.CTkCanvas(
            self,
            width=self._size + 4,
            height=self._size + 4,
            highlightthickness=0,
            bg=self._get_bg_color(),
        )
        self._canvas.pack(fill="both", expand=True)
        self._draw_canvas_icon()

    def _get_bg_color(self) -> str:
        """获取背景颜色"""
        bg = self.cget("fg_color")
        if isinstance(bg, tuple):
            appearance = ctk.get_appearance_mode()
            return str(bg[0]) if appearance == "Light" else str(bg[1])
        return str(bg) if bg else "transparent"

    def _resolve_color_value(self, color: str | tuple[str, str] | list[str]) -> str:
        """将颜色值统一解析为单一颜色字符串"""
        if isinstance(color, (tuple, list)) and len(color) >= 2:
            appearance = ctk.get_appearance_mode()
            return str(color[0]) if appearance == "Light" else str(color[1])
        if isinstance(color, str):
            return color
        return str(get_color("bg_primary", mode="dark"))

    def _draw_canvas_icon(self):
        """在 Canvas 上绘制图标"""
        if not hasattr(self, "_canvas"):
            return

        self._canvas.delete("all")

        if self._name not in self.AVAILABLE_ICONS:
            self._draw_placeholder()
            return

        color = self._get_color()
        line_width = self._get_line_width()
        size = self._size
        pad = 2

        method_name = self._name.replace("-", "_")
        draw_method = getattr(self, f"_draw_{method_name}", None)
        if draw_method:
            draw_method(pad, size, color, line_width)
        else:
            self._draw_placeholder()

    def _get_line_width(self) -> float:
        """获取线条宽度"""
        base_width = 1.5
        if self._weight == IconWeight.BOLD:
            return base_width * 1.5
        elif self._weight == IconWeight.LIGHT:
            return base_width * 0.7
        return base_width

    def _draw_placeholder(
        self, pad: int = 2, size: int = 24, color: str = "", line_width: float = 1.5
    ) -> None:
        """绘制占位符（问号）"""
        if not hasattr(self, "_canvas"):
            return
        cx, cy = size / 2 + pad, size / 2 + pad
        r = size / 3
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        self._canvas.create_text(cx, cy, text="?", fill=color, font=("Arial", int(size * 0.6)))

    # ============ 导航图标 ============

    def _draw_house(self, pad: int, size: int, color: str, line_width: float) -> None:
        """房屋图标"""
        s = size
        p = pad
        # 屋顶
        self._canvas.create_polygon(
            p,
            s * 0.7 + p,  # 左下
            s * 0.5 + p,
            p,  # 顶
            s + p,
            s * 0.7 + p,  # 右下
            outline=color,
            width=line_width,
            fill="",
            joinstyle=ctk.ROUND,
        )
        # 房身
        self._canvas.create_rectangle(
            s * 0.25 + p, s * 0.7 + p, s * 0.75 + p, s + p, outline=color, width=line_width, fill=""
        )
        # 门
        self._canvas.create_rectangle(
            s * 0.42 + p,
            s * 0.82 + p,
            s * 0.58 + p,
            s + p,
            outline=color,
            width=line_width,
            fill="",
        )

    def _draw_gear(self, pad: int, size: int, color: str, line_width: float) -> None:
        """齿轮图标"""
        import math

        cx, cy = size / 2 + pad, size / 2 + pad
        outer_r = size * 0.45
        inner_r = size * 0.2

        # 绘制齿轮齿
        num_teeth = 8
        for i in range(num_teeth):
            angle = (2 * math.pi / num_teeth) * i
            # 齿的外端点
            x1 = cx + outer_r * 1.15 * math.cos(angle)
            y1 = cy + outer_r * 1.15 * math.sin(angle)
            # 齿的根点
            x2 = cx + outer_r * 0.85 * math.cos(angle)
            y2 = cy + outer_r * 0.85 * math.sin(angle)

            # 绘制齿（用粗线表示）
            self._canvas.create_line(
                x1, y1, x2, y2, fill=color, width=line_width * 2, capstyle=ctk.ROUND
            )

        # 外圆
        self._canvas.create_oval(
            cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r, outline=color, width=line_width
        )
        # 内圆（孔）
        self._canvas.create_oval(
            cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline=color, width=line_width
        )

    # ============ 文件图标 ============

    def _draw_file(self, pad: int, size: int, color: str, line_width: float) -> None:
        """简单文件图标"""
        s = size
        p = pad
        # 文件主体
        self._canvas.create_rectangle(
            s * 0.25 + p,
            s * 0.1 + p,
            s * 0.85 + p,
            s * 0.9 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 折角
        self._canvas.create_line(
            s * 0.6 + p, s * 0.1 + p, s * 0.85 + p, s * 0.35 + p, fill=color, width=line_width
        )
        self._canvas.create_line(
            s * 0.85 + p, s * 0.35 + p, s * 0.6 + p, s * 0.35 + p, fill=color, width=line_width
        )

    def _draw_file_text(self, pad: int, size: int, color: str, line_width: float) -> None:
        """带文本的文件图标"""
        self._draw_file(pad, size, color, line_width)
        s = size
        p = pad
        # 文本行
        y_start = s * 0.45 + p
        line_height = s * 0.12
        for i in range(3):
            y = y_start + i * line_height
            self._canvas.create_line(
                s * 0.35 + p, y, s * 0.75 + p, y, fill=color, width=line_width, capstyle=ctk.ROUND
            )

    # ============ 状态图标 ============

    def _draw_clock(self, pad: int, size: int, color: str, line_width: float) -> None:
        """时钟图标"""
        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.42

        # 表盘
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        # 时针
        self._canvas.create_line(
            cx, cy, cx, cy - r * 0.5, fill=color, width=line_width * 1.2, capstyle=ctk.ROUND
        )
        # 分针
        self._canvas.create_line(
            cx, cy, cx + r * 0.7, cy, fill=color, width=line_width, capstyle=ctk.ROUND
        )
        # 中心点
        dot_r = max(1, line_width)
        self._canvas.create_oval(
            cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r, fill=color, outline=""
        )

    def _draw_check(self, pad: int, size: int, color: str, line_width: float) -> None:
        """勾选图标"""
        s = size
        p = pad
        # 绘制勾
        self._canvas.create_line(
            s * 0.25 + p,
            s * 0.55 + p,
            s * 0.42 + p,
            s * 0.72 + p,
            s * 0.75 + p,
            s * 0.28 + p,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
            joinstyle=ctk.ROUND,
        )

    def _draw_check_circle(self, pad: int, size: int, color: str, line_width: float) -> None:
        """圆形勾选图标"""
        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.42

        # 圆圈
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        # 勾
        self._draw_check(pad, size, color, line_width)

    def _draw_x(self, pad: int, size: int, color: str, line_width: float) -> None:
        """X 图标"""
        s = size
        p = pad
        self._canvas.create_line(
            s * 0.3 + p,
            s * 0.3 + p,
            s * 0.7 + p,
            s * 0.7 + p,
            fill=color,
            width=line_width * 1.3,
            capstyle=ctk.ROUND,
        )
        self._canvas.create_line(
            s * 0.7 + p,
            s * 0.3 + p,
            s * 0.3 + p,
            s * 0.7 + p,
            fill=color,
            width=line_width * 1.3,
            capstyle=ctk.ROUND,
        )

    def _draw_x_circle(self, pad: int, size: int, color: str, line_width: float) -> None:
        """圆形 X 图标"""
        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.42

        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        self._draw_x(pad, size, color, line_width)

    def _draw_warning(self, pad: int, size: int, color: str, line_width: float) -> None:
        """警告图标（三角形）"""
        s = size
        p = pad
        # 三角形
        self._canvas.create_polygon(
            s * 0.5 + p,
            s * 0.15 + p,  # 顶
            s * 0.15 + p,
            s * 0.82 + p,  # 左下
            s * 0.85 + p,
            s * 0.82 + p,  # 右下
            outline=color,
            width=line_width,
            fill="",
            joinstyle=ctk.ROUND,
        )
        # 感叹号
        self._canvas.create_line(
            s * 0.5 + p,
            s * 0.35 + p,
            s * 0.5 + p,
            s * 0.6 + p,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )
        self._canvas.create_oval(
            s * 0.47 + p, s * 0.7 + p, s * 0.53 + p, s * 0.76 + p, fill=color, outline=""
        )

    def _draw_warning_circle(self, pad: int, size: int, color: str, line_width: float) -> None:
        """圆形警告图标"""
        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.42

        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        # 感叹号
        self._canvas.create_line(
            cx,
            cy - r * 0.3,
            cx,
            cy + r * 0.1,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )
        dot_r = max(1.5, line_width)
        self._canvas.create_oval(
            cx - dot_r,
            cy + r * 0.3 - dot_r,
            cx + dot_r,
            cy + r * 0.3 + dot_r,
            fill=color,
            outline="",
        )

    def _draw_info(self, pad: int, size: int, color: str, line_width: float) -> None:
        """信息图标"""
        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.42

        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        # i
        self._canvas.create_line(
            cx,
            cy - r * 0.2,
            cx,
            cy + r * 0.35,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )
        dot_r = max(1.5, line_width)
        self._canvas.create_oval(
            cx - dot_r,
            cy - r * 0.5 - dot_r,
            cx + dot_r,
            cy - r * 0.5 + dot_r,
            fill=color,
            outline="",
        )

    # ============ 控制图标 ============

    def _draw_play(self, pad: int, size: int, color: str, line_width: float) -> None:
        """播放图标"""
        s = size
        p = pad
        cx = s / 2 + p
        cy = s / 2 + p
        h = s * 0.5
        w = s * 0.4

        self._canvas.create_polygon(
            cx - w / 2, cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2, cy, fill=color, outline=""
        )

    def _draw_pause(self, pad: int, size: int, color: str, line_width: float) -> None:
        """暂停图标"""
        s = size
        bar_w = s * 0.15
        bar_h = s * 0.5

        cx = s / 2 + pad
        cy = s / 2 + pad
        gap = s * 0.1

        # 左条
        self._canvas.create_rectangle(
            cx - gap - bar_w, cy - bar_h / 2, cx - gap, cy + bar_h / 2, fill=color, outline=""
        )
        # 右条
        self._canvas.create_rectangle(
            cx + gap, cy - bar_h / 2, cx + gap + bar_w, cy + bar_h / 2, fill=color, outline=""
        )

    def _draw_stop(self, pad: int, size: int, color: str, line_width: float) -> None:
        """停止图标"""
        s = size
        box_s = s * 0.55
        cx = s / 2 + pad
        cy = s / 2 + pad

        self._canvas.create_rectangle(
            cx - box_s / 2, cy - box_s / 2, cx + box_s / 2, cy + box_s / 2, fill=color, outline=""
        )

    def _draw_plus(self, pad: int, size: int, color: str, line_width: float) -> None:
        """加号图标"""
        s = size
        cx = s / 2 + pad
        cy = s / 2 + pad
        length = s * 0.5

        self._canvas.create_line(
            cx - length / 2,
            cy,
            cx + length / 2,
            cy,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )
        self._canvas.create_line(
            cx,
            cy - length / 2,
            cx,
            cy + length / 2,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )

    def _draw_minus(self, pad: int, size: int, color: str, line_width: float) -> None:
        """减号图标"""
        s = size
        cx = s / 2 + pad
        cy = s / 2 + pad
        length = s * 0.5

        self._canvas.create_line(
            cx - length / 2,
            cy,
            cx + length / 2,
            cy,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )

    def _draw_rocket(self, pad: int, size: int, color: str, line_width: float) -> None:
        """火箭图标"""
        s = size
        p = pad
        # 机身
        self._canvas.create_polygon(
            s * 0.5 + p,
            s * 0.1 + p,
            s * 0.7 + p,
            s * 0.45 + p,
            s * 0.5 + p,
            s * 0.9 + p,
            s * 0.3 + p,
            s * 0.45 + p,
            outline=color,
            width=line_width,
            fill="",
            joinstyle=ctk.ROUND,
        )
        # 窗口
        self._canvas.create_oval(
            s * 0.43 + p,
            s * 0.32 + p,
            s * 0.57 + p,
            s * 0.46 + p,
            outline=color,
            width=line_width,
        )
        # 尾焰
        self._canvas.create_polygon(
            s * 0.5 + p,
            s * 0.9 + p,
            s * 0.58 + p,
            s + p,
            s * 0.42 + p,
            s + p,
            outline=color,
            width=line_width,
            fill="",
            joinstyle=ctk.ROUND,
        )

    # ============ 文件夹图标 ============

    def _draw_folder(self, pad: int, size: int, color: str, line_width: float) -> None:
        """文件夹图标"""
        s = size
        p = pad
        # 后层
        self._canvas.create_rectangle(
            s * 0.15 + p,
            s * 0.35 + p,
            s * 0.85 + p,
            s * 0.8 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 前层
        self._canvas.create_rectangle(
            p, s * 0.45 + p, s + p, s * 0.8 + p, outline=color, width=line_width, fill=""
        )
        # 标签
        self._canvas.create_line(
            p, s * 0.45 + p, s * 0.35 + p, s * 0.45 + p, fill=color, width=line_width
        )
        self._canvas.create_line(
            s * 0.35 + p, s * 0.45 + p, s * 0.35 + p, s * 0.35 + p, fill=color, width=line_width
        )

    def _draw_folder_open(self, pad: int, size: int, color: str, line_width: float) -> None:
        """打开的文件夹图标"""
        s = size
        p = pad
        # 后盖（打开状态）
        self._canvas.create_polygon(
            p,
            s * 0.4 + p,
            s * 0.4 + p,
            s * 0.15 + p,
            s + p,
            s * 0.3 + p,
            s + p,
            s * 0.5 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 前盖
        self._canvas.create_rectangle(
            p, s * 0.5 + p, s + p, s * 0.8 + p, outline=color, width=line_width, fill=""
        )
        # 标签
        self._canvas.create_line(
            p, s * 0.5 + p, s * 0.35 + p, s * 0.5 + p, fill=color, width=line_width
        )
        self._canvas.create_line(
            s * 0.35 + p, s * 0.5 + p, s * 0.35 + p, s * 0.4 + p, fill=color, width=line_width
        )

    # ============ 主题图标 ============

    def _draw_moon(self, pad: int, size: int, color: str, line_width: float) -> None:
        """月亮图标"""

        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.4

        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=line_width)
        bg_color = self._get_bg_color()
        bg_color = self._resolve_color_value(bg_color)

        offset_x = r * 0.5
        self._canvas.create_oval(
            cx + offset_x - r * 0.85,
            cy - r * 0.85,
            cx + offset_x + r * 0.85,
            cy + r * 0.85,
            fill=bg_color,
            outline=bg_color,
        )

    def _draw_sun(self, pad: int, size: int, color: str, line_width: float) -> None:
        """太阳图标"""
        import math

        cx, cy = size / 2 + pad, size / 2 + pad
        r = size * 0.2

        # 太阳圆
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")

        # 光芒
        num_rays = 8
        ray_length = size * 0.18
        for i in range(num_rays):
            angle = (2 * math.pi / num_rays) * i
            if i % 2 == 0:
                # 主方向（水平垂直）
                x1 = cx + r * 1.3 * math.cos(angle)
                y1 = cy + r * 1.3 * math.sin(angle)
                x2 = cx + (r + ray_length) * math.cos(angle)
                y2 = cy + (r + ray_length) * math.sin(angle)
            else:
                # 对角方向（稍短）
                x1 = cx + r * 1.2 * math.cos(angle)
                y1 = cy + r * 1.2 * math.sin(angle)
                x2 = cx + (r + ray_length * 0.8) * math.cos(angle)
                y2 = cy + (r + ray_length * 0.8) * math.sin(angle)

            self._canvas.create_line(
                x1, y1, x2, y2, fill=color, width=line_width, capstyle=ctk.ROUND
            )

    # ============ 其他图标 ============

    def _draw_desktop(self, pad: int, size: int, color: str, line_width: float) -> None:
        """桌面显示器图标"""
        s = size
        p = pad
        # 屏幕
        self._canvas.create_rectangle(
            s * 0.15 + p,
            s * 0.15 + p,
            s * 0.85 + p,
            s * 0.65 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 底座
        self._canvas.create_line(
            s * 0.35 + p, s * 0.65 + p, s * 0.35 + p, s * 0.8 + p, fill=color, width=line_width
        )
        self._canvas.create_line(
            s * 0.65 + p, s * 0.65 + p, s * 0.65 + p, s * 0.8 + p, fill=color, width=line_width
        )
        self._canvas.create_line(
            s * 0.25 + p, s * 0.8 + p, s * 0.75 + p, s * 0.8 + p, fill=color, width=line_width
        )

    def _draw_terminal(self, pad: int, size: int, color: str, line_width: float) -> None:
        """终端图标"""
        s = size
        p = pad
        # 外框
        self._canvas.create_rectangle(
            s * 0.15 + p,
            s * 0.2 + p,
            s * 0.85 + p,
            s * 0.8 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 提示符
        self._canvas.create_line(
            s * 0.25 + p,
            s * 0.45 + p,
            s * 0.35 + p,
            s * 0.45 + p,
            fill=color,
            width=line_width,
            capstyle=ctk.ROUND,
        )
        self._canvas.create_line(
            s * 0.25 + p,
            s * 0.38 + p,
            s * 0.25 + p,
            s * 0.52 + p,
            fill=color,
            width=line_width,
            capstyle=ctk.ROUND,
        )
        # 光标
        self._canvas.create_line(
            s * 0.4 + p,
            s * 0.42 + p,
            s * 0.4 + p,
            s * 0.52 + p,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )

    def _draw_activity(self, pad: int, size: int, color: str, line_width: float) -> None:
        """活动图表图标"""
        s = size
        p = pad
        # 折线图
        points = [
            s * 0.2 + p,
            s * 0.7 + p,
            s * 0.35 + p,
            s * 0.5 + p,
            s * 0.5 + p,
            s * 0.6 + p,
            s * 0.65 + p,
            s * 0.35 + p,
            s * 0.8 + p,
            s * 0.45 + p,
        ]
        self._canvas.create_line(
            *points, fill=color, width=line_width, capstyle=ctk.ROUND, joinstyle=ctk.ROUND
        )
        # 坐标轴
        self._canvas.create_line(
            s * 0.15 + p,
            s * 0.25 + p,
            s * 0.15 + p,
            s * 0.75 + p,
            fill=color,
            width=line_width,
            capstyle=ctk.ROUND,
        )
        self._canvas.create_line(
            s * 0.15 + p,
            s * 0.75 + p,
            s * 0.85 + p,
            s * 0.75 + p,
            fill=color,
            width=line_width,
            capstyle=ctk.ROUND,
        )

    def _draw_list(self, pad: int, size: int, color: str, line_width: float) -> None:
        """列表图标"""
        s = size
        p = pad
        # 项目符号
        for i in range(3):
            y = s * 0.3 + p + i * s * 0.18
            dot_r = max(1, line_width)
            self._canvas.create_oval(
                s * 0.2 + p - dot_r,
                y - dot_r,
                s * 0.2 + p + dot_r,
                y + dot_r,
                fill=color,
                outline="",
            )
            # 线条
            self._canvas.create_line(
                s * 0.32 + p, y, s * 0.8 + p, y, fill=color, width=line_width, capstyle=ctk.ROUND
            )

    def _draw_sliders(self, pad: int, size: int, color: str, line_width: float) -> None:
        """滑块图标"""
        s = size
        p = pad
        # 三条水平线
        for i in range(3):
            y = s * 0.28 + p + i * s * 0.22
            self._canvas.create_line(
                s * 0.2 + p, y, s * 0.8 + p, y, fill=color, width=line_width, capstyle=ctk.ROUND
            )
            # 滑块
            cx = s * 0.4 + p + i * s * 0.1
            self._canvas.create_oval(
                cx - line_width * 1.5,
                y - line_width * 1.5,
                cx + line_width * 1.5,
                y + line_width * 1.5,
                fill=color,
                outline="",
            )

    def _draw_funnel(self, pad: int, size: int, color: str, line_width: float) -> None:
        """漏斗图标"""
        s = size
        p = pad
        # 漏斗形状
        self._canvas.create_polygon(
            s * 0.15 + p,
            s * 0.2 + p,
            s * 0.85 + p,
            s * 0.2 + p,
            s * 0.6 + p,
            s * 0.5 + p,
            s * 0.6 + p,
            s * 0.8 + p,
            s * 0.4 + p,
            s * 0.8 + p,
            s * 0.4 + p,
            s * 0.5 + p,
            outline=color,
            width=line_width,
            fill="",
            joinstyle=ctk.ROUND,
        )

    def _draw_trash(self, pad: int, size: int, color: str, line_width: float) -> None:
        """垃圾桶图标"""
        s = size
        p = pad
        # 盖子
        self._canvas.create_line(
            s * 0.2 + p,
            s * 0.2 + p,
            s * 0.8 + p,
            s * 0.2 + p,
            fill=color,
            width=line_width * 1.3,
            capstyle=ctk.ROUND,
        )
        self._canvas.create_line(
            s * 0.35 + p,
            s * 0.2 + p,
            s * 0.35 + p,
            s * 0.15 + p,
            s * 0.65 + p,
            s * 0.15 + p,
            s * 0.65 + p,
            s * 0.2 + p,
            fill=color,
            width=line_width * 1.3,
            capstyle=ctk.ROUND,
        )
        # 箱体
        self._canvas.create_rectangle(
            s * 0.28 + p,
            s * 0.25 + p,
            s * 0.72 + p,
            s * 0.8 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 竖线（纹理）
        self._canvas.create_line(
            s * 0.4 + p, s * 0.25 + p, s * 0.4 + p, s * 0.8 + p, fill=color, width=line_width * 0.8
        )
        self._canvas.create_line(
            s * 0.6 + p, s * 0.25 + p, s * 0.6 + p, s * 0.8 + p, fill=color, width=line_width * 0.8
        )

    def _draw_download(self, pad: int, size: int, color: str, line_width: float) -> None:
        """下载图标"""
        s = size
        p = pad
        cx = s / 2 + p
        # 箭头
        self._canvas.create_line(
            cx, s * 0.2 + p, cx, s * 0.6 + p, fill=color, width=line_width, capstyle=ctk.ROUND
        )
        # 箭头头部
        self._canvas.create_line(
            s * 0.35 + p,
            s * 0.5 + p,
            cx,
            s * 0.65 + p,
            s * 0.65 + p,
            s * 0.5 + p,
            fill=color,
            width=line_width,
            capstyle=ctk.ROUND,
            joinstyle=ctk.ROUND,
        )
        # 底线
        self._canvas.create_line(
            s * 0.25 + p,
            s * 0.75 + p,
            s * 0.75 + p,
            s * 0.75 + p,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )

    def _draw_upload(self, pad: int, size: int, color: str, line_width: float) -> None:
        """上传图标"""
        s = size
        p = pad
        cx = s / 2 + p
        # 箭头
        self._canvas.create_line(
            cx, s * 0.65 + p, cx, s * 0.25 + p, fill=color, width=line_width, capstyle=ctk.ROUND
        )
        # 箭头头部
        self._canvas.create_line(
            s * 0.35 + p,
            s * 0.4 + p,
            cx,
            s * 0.25 + p,
            s * 0.65 + p,
            s * 0.4 + p,
            fill=color,
            width=line_width,
            capstyle=ctk.ROUND,
            joinstyle=ctk.ROUND,
        )
        # 底线
        self._canvas.create_line(
            s * 0.25 + p,
            s * 0.75 + p,
            s * 0.75 + p,
            s * 0.75 + p,
            fill=color,
            width=line_width * 1.5,
            capstyle=ctk.ROUND,
        )

    def _draw_book(self, pad: int, size: int, color: str, line_width: float) -> None:
        """书籍图标"""
        s = size
        p = pad
        # 书的主体
        self._canvas.create_rectangle(
            s * 0.25 + p,
            s * 0.2 + p,
            s * 0.75 + p,
            s * 0.8 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 书脊
        self._canvas.create_line(
            s * 0.35 + p, s * 0.2 + p, s * 0.35 + p, s * 0.8 + p, fill=color, width=line_width
        )
        # 装饰线
        self._canvas.create_line(
            s * 0.4 + p, s * 0.35 + p, s * 0.7 + p, s * 0.35 + p, fill=color, width=line_width * 0.7
        )

    def _draw_books(self, pad: int, size: int, color: str, line_width: float) -> None:
        """多本书籍图标"""
        s = size
        p = pad
        # 后面的书（倾斜）
        self._canvas.create_polygon(
            s * 0.55 + p,
            s * 0.15 + p,
            s * 0.75 + p,
            s * 0.15 + p,
            s * 0.8 + p,
            s * 0.85 + p,
            s * 0.6 + p,
            s * 0.85 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 前面的书
        self._canvas.create_polygon(
            s * 0.2 + p,
            s * 0.2 + p,
            s * 0.5 + p,
            s * 0.2 + p,
            s * 0.55 + p,
            s * 0.8 + p,
            s * 0.25 + p,
            s * 0.8 + p,
            outline=color,
            width=line_width,
            fill="",
        )
        # 装饰线
        self._canvas.create_line(
            s * 0.28 + p,
            s * 0.35 + p,
            s * 0.47 + p,
            s * 0.35 + p,
            fill=color,
            width=line_width * 0.7,
        )

    def _draw_document(self, pad: int, size: int, color: str, line_width: float) -> None:
        """文档图标（file_text 的别名）"""
        self._draw_file_text(pad, size, color, line_width)

    def configure(self, **kwargs) -> None:
        """
        配置图标属性

        支持的参数:
        - name: 图标名称
        - size: 图标尺寸
        - weight: 图标权重
        - color: 图标颜色
        """
        icon_updated = False

        if "name" in kwargs:
            self._name = self._normalize_name(kwargs.pop("name"))
            icon_updated = True
        if "weight" in kwargs:
            self._weight = kwargs.pop("weight")
            icon_updated = True
        if "color" in kwargs:
            self._color = kwargs.pop("color")
            icon_updated = True
        if "size" in kwargs:
            size = kwargs.pop("size")
            if isinstance(size, IconSize):
                self._size = size.value
            else:
                self._size = int(size)
            icon_updated = True

        super().configure(**kwargs)
        if icon_updated:
            self._reload_icon()

    def _reload_icon(self) -> None:
        """重新加载图标（用于属性变更后刷新）"""
        if hasattr(self, "_canvas"):
            self._canvas.destroy()
            delattr(self, "_canvas")
        super().configure(image=None, width=self._size, height=self._size)
        self._load_or_draw_icon()
