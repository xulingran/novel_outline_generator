"""
圆形进度条组件

双环设计的圆形进度指示器，支持 determinate/indeterminate 模式。
"""

import logging
import math

import customtkinter as ctk

from gui.theme_manager import get_color

logger = logging.getLogger(__name__)


class ProgressRing(ctk.CTkCanvas):
    """
    圆形进度条

    双环设计：外环轨道显示总体进度，内环显示当前阶段。

    Args:
        master: 父容器
        size: 直径大小（像素）
        track_color: 轨道颜色
        progress_color: 进度颜色
        stroke_width: 线条宽度
        show_percentage: 是否显示百分比文字
        **kwargs: 其他 Canvas 参数
    """

    def __init__(
        self,
        master,
        size: int = 200,
        track_color: str | tuple[str, str] | None = None,
        progress_color: str | tuple[str, str] | None = None,
        stroke_width: int = 12,
        show_percentage: bool = True,
        **kwargs,
    ):
        self._size = size
        self._stroke_width = stroke_width
        self._show_percentage = show_percentage
        self._progress = 0.0
        self._is_indeterminate = False
        self._animation_offset = 0

        # 颜色
        self._track_color = track_color or get_color("bg_tertiary", mode="auto")
        self._progress_color = progress_color or get_color("accent", mode="auto")

        # 解析颜色（处理自适应颜色）
        self._resolved_track = self._resolve_color(self._track_color)
        self._resolved_progress = self._resolve_color(self._progress_color)

        super().__init__(
            master,
            width=size,
            height=size,
            highlightthickness=0,
            **kwargs,
        )

        self._draw()

    def _resolve_color(self, color: str | tuple[str, str]) -> str:
        """解析自适应颜色"""
        if isinstance(color, tuple):
            appearance = ctk.get_appearance_mode()
            return color[0] if appearance == "Light" else color[1]
        return color

    def _draw(self):
        """绘制进度环"""
        self.delete("all")

        center = self._size / 2
        radius = (self._size - self._stroke_width) / 2

        # 绘制轨道（底环）
        self._create_arc(
            center,
            radius,
            0,
            360,
            self._resolved_track,
            tags="track",
        )

        if self._is_indeterminate:
            # 不确定模式：旋转动画
            self._draw_indeterminate(center, radius)
        else:
            # 确定模式：显示进度
            self._draw_progress(center, radius)

        # 绘制百分比文字
        if self._show_percentage and not self._is_indeterminate:
            self._draw_text(center)

    def _create_arc(
        self,
        center: float,
        radius: float,
        start_angle: float,
        extent: float,
        color: str,
        tags: str | tuple = (),
        style: str = "arc",
    ):
        """创建圆弧"""
        x0 = center - radius
        y0 = center - radius
        x1 = center + radius
        y1 = center + radius

        self.create_arc(
            x0,
            y0,
            x1,
            y1,
            start=start_angle,
            extent=extent,
            style=style,
            outline=color,
            width=self._stroke_width,
            tags=tags,
        )

    def _draw_progress(self, center: float, radius: float):
        """绘制进度"""
        if self._progress <= 0:
            return

        # 计算进度角度（从顶部开始，-90度）
        extent = 360 * self._progress
        start_angle = 90  # CustomTkinter Canvas 中 0 度在右侧，90 度在顶部

        # 绘制进度弧
        self._create_arc(
            center,
            radius,
            start_angle,
            -extent,  # 逆时针方向
            self._resolved_progress,
            tags="progress",
        )

    def _draw_indeterminate(self, center: float, radius: float):
        """绘制不确定状态动画"""
        # 绘制旋转的弧段
        arc_length = 90
        start_angle = 90 + self._animation_offset

        self._create_arc(
            center,
            radius,
            start_angle,
            -arc_length,
            self._resolved_progress,
            tags="indeterminate",
        )

    def _draw_text(self, center: float):
        """绘制百分比文字"""
        percentage = int(self._progress * 100)
        text = f"{percentage}%"

        font_size = int(self._size * 0.2)  # 动态字体大小

        self.create_text(
            center,
            center,
            text=text,
            fill=self._resolve_color(get_color("fg_primary", mode="auto")),
            font=ctk.CTkFont(size=font_size, weight="bold"),
            tags="text",
        )

    def set_progress(self, value: float, animate: bool = True):
        """
        设置进度值

        Args:
            value: 进度值 (0.0 - 1.0)
            animate: 是否启用动画过渡
        """
        self._progress = max(0.0, min(1.0, value))
        self._is_indeterminate = False

        if animate:
            self._animate_to_progress()
        else:
            self._draw()

    def _animate_to_progress(self):
        """动画过渡到目标进度"""
        # 简化实现：直接绘制
        # 完整实现需要使用 AnimationManager
        self._draw()

    def set_indeterminate(self, indeterminate: bool = True):
        """
        设置不确定模式

        Args:
            indeterminate: 是否启用不确定模式
        """
        self._is_indeterminate = indeterminate

        if indeterminate:
            self._start_animation()
        else:
            self._stop_animation()

        self._draw()

    def _start_animation(self):
        """启动动画"""
        self._animate()

    def _animate(self):
        """动画帧"""
        if not self._is_indeterminate:
            return

        self._animation_offset = (self._animation_offset + 5) % 360
        self._draw()

        # 约 30fps
        self.after(33, self._animate)

    def _stop_animation(self):
        """停止动画"""
        self._animation_offset = 0

    def configure(self, **kwargs):
        """配置进度环属性"""
        if "progress_color" in kwargs:
            self._progress_color = kwargs.pop("progress_color")
            self._resolved_progress = self._resolve_color(self._progress_color)
        if "track_color" in kwargs:
            self._track_color = kwargs.pop("track_color")
            self._resolved_track = self._resolve_color(self._track_color)
        if "size" in kwargs:
            self._size = kwargs.pop("size")
            self.configure(width=self._size, height=self._size)

        super().configure(**kwargs)
        self._draw()

    def get_progress(self) -> float:
        """获取当前进度值"""
        return self._progress


class PhaseProgressRing(ProgressRing):
    """
    阶段进度环

    在进度环外显示阶段标签，用于处理流程的可视化。

    Args:
        master: 父容器
        phases: 阶段列表，如 ["分析", "处理", "合并"]
        **kwargs: ProgressRing 参数
    """

    def __init__(self, master, phases: list[str] | None = None, **kwargs):
        self._phases = phases or []
        self._current_phase = 0

        # 增加尺寸以容纳阶段标签
        kwargs.setdefault("size", 200)

        super().__init__(master, **kwargs)

        self._draw_phases()

    def set_phase(self, index: int):
        """设置当前阶段"""
        self._current_phase = max(0, min(len(self._phases) - 1, index))
        self._draw_phases()

    def _draw_phases(self):
        """绘制阶段标签"""
        # 清除旧的阶段标签
        self.delete("phase")

        if not self._phases:
            return

        # 计算标签位置（环绕圆环）
        center = self._size / 2
        outer_radius = self._size / 2 + 20

        num_phases = len(self._phases)

        for i, phase in enumerate(self._phases):
            # 计算角度
            angle = -90 + (360 / num_phases) * i

            # 转换为弧度
            rad = math.radians(angle)

            # 计算位置
            x = center + outer_radius * math.cos(rad)
            y = center + outer_radius * math.sin(rad)

            # 判断是否为当前或已完成阶段
            is_current = i == self._current_phase
            is_completed = i < self._current_phase

            # 颜色
            if is_current:
                color = self._resolve_color(get_color("accent", mode="auto"))
                font_weight = "bold"
            elif is_completed:
                color = self._resolve_color(get_color("success", mode="auto"))
                font_weight = "normal"
            else:
                color = self._resolve_color(get_color("fg_tertiary", mode="auto"))
                font_weight = "normal"

            # 绘制标签
            self.create_text(
                x,
                y,
                text=phase,
                fill=color,
                font=ctk.CTkFont(size=11, weight=font_weight),
                tags="phase",
                anchor="center",
            )

            # 绘制连接线
            if is_current or is_completed:
                inner_x = center + (self._size / 2 - 10) * math.cos(rad)
                inner_y = center + (self._size / 2 - 10) * math.sin(rad)

                self.create_line(
                    inner_x,
                    inner_y,
                    x,
                    y,
                    fill=color,
                    width=1,
                    tags="phase",
                )

    def _draw(self):
        """重写绘制方法，确保阶段标签在最上层"""
        super()._draw()
        self._draw_phases()

    def set_progress(self, value: float, phase: int | None = None):
        """
        设置进度和阶段

        Args:
            value: 进度值
            phase: 阶段索引（可选）
        """
        super().set_progress(value)

        if phase is not None:
            self.set_phase(phase)
