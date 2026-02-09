"""
动画管理器模块

提供流畅的 UI 动画支持，包括属性动画、颜色过渡、缓动函数等。
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:
    from typing import Any

    CTkWidget = Any
else:
    CTkWidget = object

logger = logging.getLogger(__name__)


class Easing(Enum):
    """缓动函数枚举"""

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    CUBIC_BEZIER = "cubic_bezier"


@dataclass
class AnimationConfig:
    """动画配置"""

    duration: int = 300  # 毫秒
    easing: Easing = Easing.EASE_OUT
    delay: int = 0  # 毫秒
    on_start: Callable[[], None] | None = None
    on_complete: Callable[[], None] | None = None
    on_update: Callable[[float], None] | None = None  # progress: 0.0-1.0


def lerp(start: float, end: float, t: float) -> float:
    """线性插值"""
    return start + (end - start) * t


def ease_in(t: float) -> float:
    """缓入"""
    return t * t


def ease_out(t: float) -> float:
    """缓出"""
    return 1 - (1 - t) * (1 - t)


def ease_in_out(t: float) -> float:
    """缓入缓出"""
    return t * t * (3 - 2 * t)


def cubic_bezier(
    t: float, p1: float = 0.4, p2: float = 0.0, p3: float = 0.2, p4: float = 1.0
) -> float:
    """三次贝塞尔曲线 (简化的 cubic-bezier(0.4, 0, 0.2, 1))"""
    # 使用简化的三次贝塞尔实现
    u = 1 - t
    return 3 * u * u * t * p1 + 3 * u * t * t * p3 + t * t * t


def apply_easing(t: float, easing: Easing) -> float:
    """应用缓动函数"""
    t = max(0.0, min(1.0, t))  # 限制在 0-1 之间

    match easing:
        case Easing.LINEAR:
            return t
        case Easing.EASE_IN:
            return ease_in(t)
        case Easing.EASE_OUT:
            return ease_out(t)
        case Easing.EASE_IN_OUT:
            return ease_in_out(t)
        case Easing.CUBIC_BEZIER:
            return cubic_bezier(t)
        case _:
            return t


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """将十六进制颜色转换为 RGB"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """将 RGB 转换为十六进制颜色"""
    return f"#{r:02x}{g:02x}{b:02x}"


def lerp_color(color1: str, color2: str, t: float) -> str:
    """在两种颜色之间插值"""
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)

    r = int(lerp(r1, r2, t))
    g = int(lerp(g1, g2, t))
    b = int(lerp(b1, b2, t))

    return rgb_to_hex(r, g, b)


class Animation:
    """
    动画实例

    管理单个动画的生命周期，包括更新循环和回调触发。
    """

    def __init__(self, config: AnimationConfig, widget: CTkWidget | None = None):
        self.config = config
        self.widget = widget
        self._start_time: float | None = None
        self._is_running = False
        self._is_complete = False
        self._after_id: str | None = None

    def start(self, master: ctk.CTk) -> None:
        """启动动画"""
        if self._is_running:
            return

        self._is_running = True
        self._is_complete = False
        self._start_time = time.time() * 1000  # 转换为毫秒

        if self.config.on_start:
            self.config.on_start()

        if self.config.delay > 0:
            master.after(self.config.delay, lambda: self._tick(master))
        else:
            self._tick(master)

    def _tick(self, master: ctk.CTk) -> None:
        """动画帧更新"""
        if not self._is_running:
            return

        current_time = time.time() * 1000
        elapsed = current_time - self._start_time - self.config.delay

        if elapsed >= self.config.duration:
            # 动画完成
            self._complete()
            return

        # 计算进度
        raw_progress = elapsed / self.config.duration
        eased_progress = apply_easing(raw_progress, self.config.easing)

        # 触发更新回调
        if self.config.on_update:
            self.config.on_update(eased_progress)

        # 调度下一帧 (约 60fps)
        self._after_id = master.after(16, lambda: self._tick(master))

    def _complete(self) -> None:
        """完成动画"""
        self._is_running = False
        self._is_complete = True

        # 确保最终值为 1.0
        if self.config.on_update:
            self.config.on_update(1.0)

        if self.config.on_complete:
            self.config.on_complete()

    def cancel(self) -> None:
        """取消动画"""
        self._is_running = False
        if self._after_id:
            # after_id 是字符串，需要通过 master 取消
            pass

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_complete(self) -> bool:
        return self._is_complete


class AnimationManager:
    """
    动画管理器

    管理所有动画实例，提供便捷的动画创建和控制方法。
    """

    def __init__(self):
        self._animations: list[Animation] = []
        self._counter = 0

    def animate(
        self,
        master: ctk.CTk,
        config: AnimationConfig,
        widget: CTkWidget | None = None,
    ) -> Animation:
        """
        创建并启动一个新动画

        Args:
            master: 主窗口
            config: 动画配置
            widget: 关联的 widget

        Returns:
            动画实例
        """
        animation = Animation(config, widget)
        self._animations.append(animation)
        animation.start(master)
        return animation

    def cancel_all(self) -> None:
        """取消所有动画"""
        for animation in self._animations:
            if animation.is_running:
                animation.cancel()
        self._animations.clear()

    def remove_completed(self) -> None:
        """移除已完成的动画"""
        self._animations = [a for a in self._animations if not a.is_complete]

    @property
    def active_count(self) -> int:
        """活跃动画数量"""
        return sum(1 for a in self._animations if a.is_running)

    # 便捷方法

    def fade_in(
        self,
        master: ctk.CTk,
        widget: CTkWidget,
        duration: int = 300,
        on_complete: Callable[[], None] | None = None,
    ) -> Animation:
        """
        淡入动画

        注意: CustomTkinter 不直接支持透明度，这是模拟实现
        """
        config = AnimationConfig(
            duration=duration,
            easing=Easing.EASE_OUT,
            on_complete=on_complete,
        )
        return self.animate(master, config, widget)

    def fade_out(
        self,
        master: ctk.CTk,
        widget: CTkWidget,
        duration: int = 300,
        on_complete: Callable[[], None] | None = None,
    ) -> Animation:
        """淡出动画"""
        config = AnimationConfig(
            duration=duration,
            easing=Easing.EASE_IN,
            on_complete=on_complete,
        )
        return self.animate(master, config, widget)

    def slide_in(
        self,
        master: ctk.CTk,
        widget: CTkWidget,
        from_y: int = 20,
        duration: int = 300,
        on_complete: Callable[[], None] | None = None,
    ) -> Animation:
        """滑入动画"""
        # CustomTkinter 不支持直接设置位置，这里只是示例实现
        # start_y = widget.winfo_y()

        def on_update(progress: float) -> None:
            # CustomTkinter 不支持直接设置位置，这里只是示例
            # current_y = int(lerp(start_y + from_y, target_y, progress))
            pass

        config = AnimationConfig(
            duration=duration,
            easing=Easing.CUBIC_BEZIER,
            on_update=on_update,
            on_complete=on_complete,
        )
        return self.animate(master, config, widget)

    def color_transition(
        self,
        master: ctk.CTk,
        widget: CTkWidget,
        attribute: str,
        from_color: str,
        to_color: str,
        duration: int = 200,
        on_complete: Callable[[], None] | None = None,
    ) -> Animation:
        """
        颜色过渡动画

        Args:
            master: 主窗口
            widget: 目标 widget
            attribute: 要改变的颜色属性，如 "fg_color", "text_color"
            from_color: 起始颜色 (十六进制)
            to_color: 目标颜色 (十六进制)
            duration: 动画时长 (毫秒)
            on_complete: 完成回调
        """

        def on_update(progress: float) -> None:
            current_color = lerp_color(from_color, to_color, progress)
            try:
                widget.configure(**{attribute: current_color})
            except Exception as e:
                logger.debug(f"Color transition update failed: {e}")

        config = AnimationConfig(
            duration=duration,
            easing=Easing.EASE_OUT,
            on_update=on_update,
            on_complete=on_complete,
        )
        return self.animate(master, config, widget)

    def count_up(
        self,
        master: ctk.CTk,
        widget: CTkWidget,
        from_value: float,
        to_value: float,
        duration: int = 500,
        format_string: str = "{:.0f}",
        on_complete: Callable[[], None] | None = None,
    ) -> Animation:
        """
        数字滚动动画

        Args:
            master: 主窗口
            widget: 目标 widget (通常是 Label)
            from_value: 起始值
            to_value: 目标值
            duration: 动画时长 (毫秒)
            format_string: 数字格式化字符串
            on_complete: 完成回调
        """

        def on_update(progress: float) -> None:
            current_value = lerp(from_value, to_value, progress)
            try:
                widget.configure(text=format_string.format(current_value))
            except Exception as e:
                logger.debug(f"Count up update failed: {e}")

        config = AnimationConfig(
            duration=duration,
            easing=Easing.CUBIC_BEZIER,
            on_update=on_update,
            on_complete=on_complete,
        )
        return self.animate(master, config, widget)


# 全局单例
_global_animation_manager: AnimationManager | None = None


def get_animation_manager() -> AnimationManager:
    """获取全局动画管理器单例"""
    global _global_animation_manager
    if _global_animation_manager is None:
        _global_animation_manager = AnimationManager()
    return _global_animation_manager
