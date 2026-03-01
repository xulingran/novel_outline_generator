"""
进度条组件

显示处理进度、统计信息、ETA 等信息。
"""

import logging

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class ProgressBar(ctk.CTkFrame):
    """
    进度条组件

    功能：
    - 总体进度条（0-100%）
    - 当前阶段指示器
    - 已完成/失败/部分完成块计数
    - ETA 显示（剩余时间 + 置信度）
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self._setup_ui()

        # 状态变量
        self.total_chunks = 0
        self.completed_chunks = 0
        self.failed_chunks = 0
        self.partial_chunks = 0
        self.current_phase = ""
        self.eta_seconds = 0
        self.eta_confidence = 0.0

    def _setup_ui(self):
        """设置 UI"""
        # 标题
        title_label = ctk.CTkLabel(
            self,
            text="处理进度",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        title_label.pack(pady=(SPACING["md"], SPACING["md"]))

        # 进度条
        self.progress_bar = ctk.CTkProgressBar(
            self,
            width=400,
            progress_color=get_color("accent", mode="auto"),
            fg_color=get_color("bg_secondary", mode="auto"),
        )
        self.progress_bar.pack(pady=SPACING["md"])
        self.progress_bar.set(0)

        # 统计信息框架
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(pady=SPACING["sm"])

        # 已完成
        self.completed_label = ctk.CTkLabel(
            stats_frame,
            text="已完成: 0/0",
            width=150,
            font=ctk.CTkFont(size=12),
            text_color=get_color("fg_primary", mode="auto"),
        )
        self.completed_label.pack(side="left", padx=SPACING["md"])

        # 失败
        self.failed_label = ctk.CTkLabel(
            stats_frame,
            text="失败: 0",
            width=100,
            font=ctk.CTkFont(size=12),
            text_color=get_color("error", mode="auto"),
        )
        self.failed_label.pack(side="left", padx=SPACING["md"])

        # 部分完成
        self.partial_label = ctk.CTkLabel(
            stats_frame,
            text="部分完成: 0",
            width=100,
            font=ctk.CTkFont(size=12),
            text_color=get_color("warning", mode="auto"),
        )
        self.partial_label.pack(side="left", padx=SPACING["md"])

        # 当前阶段
        self.phase_label = ctk.CTkLabel(
            self,
            text="等待开始...",
            font=ctk.CTkFont(size=12),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self.phase_label.pack(pady=SPACING["sm"])

        # ETA 显示
        self.eta_label = ctk.CTkLabel(
            self,
            text="预估剩余时间: --",
            font=ctk.CTkFont(size=12),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self.eta_label.pack(pady=(SPACING["sm"], SPACING["md"]))

    def update_progress(
        self,
        completed: int,
        total: int,
        failed: int = 0,
        partial: int = 0,
        phase: str = "",
        eta_seconds: int | float | str | None = None,
        eta_confidence: float | str = 0.0,
        progress: float | None = None,
    ):
        """
        更新进度

        Args:
            completed: 已完成块数
            total: 总块数
            failed: 失败块数
            partial: 部分完成块数
            phase: 当前阶段
            eta_seconds: 预估剩余时间（秒）
            eta_confidence: 置信度（0-1）或等级字符串（low/medium/high）
            progress: 进度值（0-1），优先于 completed/total 计算
        """
        self.completed_chunks = completed
        self.total_chunks = total
        self.failed_chunks = failed
        self.partial_chunks = partial
        self.current_phase = phase
        normalized_eta_seconds = self._normalize_eta_seconds(eta_seconds)
        normalized_confidence = self._normalize_confidence(eta_confidence)
        self.eta_seconds = normalized_eta_seconds
        self.eta_confidence = normalized_confidence

        # 更新进度条
        progress_value = progress
        if progress_value is None and total > 0:
            progress_value = completed / total
        if progress_value is not None:
            progress_value = max(0.0, min(1.0, progress_value))
            self.progress_bar.set(progress_value)

        # 更新统计标签
        self.completed_label.configure(text=f"已完成: {completed}/{total}")
        self.failed_label.configure(text=f"失败: {failed}")
        self.partial_label.configure(text=f"部分完成: {partial}")

        # 更新阶段
        if phase:
            self.phase_label.configure(text=f"当前阶段: {phase}")
        else:
            self.phase_label.configure(text="处理中...")

        # 更新 ETA
        if normalized_eta_seconds > 0:
            eta_text = self._format_eta(normalized_eta_seconds, normalized_confidence)
            self.eta_label.configure(text=eta_text)
        else:
            self.eta_label.configure(text="预估剩余时间: 计算中...")

    def _normalize_eta_seconds(self, seconds: int | float | str | None) -> int:
        """将 ETA 秒数归一化为非负整数。"""
        if seconds is None:
            return 0
        try:
            return max(0, int(float(seconds)))
        except (TypeError, ValueError):
            return 0

    def _normalize_confidence(self, confidence: float | str | None) -> float:
        """将置信度统一为 0-1 浮点值。"""
        if isinstance(confidence, str):
            mapping = {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.9,
            }
            key = confidence.strip().lower()
            if key in mapping:
                return mapping[key]
            try:
                value = float(key)
            except ValueError:
                return 0.0
            return max(0.0, min(1.0, value))
        try:
            value = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    def _format_eta(self, seconds: int, confidence: float) -> str:
        """
        格式化 ETA 显示

        Args:
            seconds: 剩余秒数
            confidence: 置信度（0-1）

        Returns:
            格式化的 ETA 字符串
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours}小时")
        if minutes > 0:
            time_parts.append(f"{minutes}分钟")
        if secs > 0 or not time_parts:
            time_parts.append(f"{secs}秒")

        time_str = "".join(time_parts)

        # 置信度指示
        if confidence >= 0.8:
            confidence_str = "高"
        elif confidence >= 0.5:
            confidence_str = "中"
        else:
            confidence_str = "低"

        return f"预估剩余时间: {time_str} (置信度: {confidence_str})"

    def reset(self):
        """重置进度"""
        self.update_progress(
            completed=0, total=0, failed=0, partial=0, phase="等待开始...", eta_seconds=0
        )
        self.progress_bar.set(0)

    def set_indeterminate(self):
        """设置为不确定状态（用于处理中但无法确定进度时）"""
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

    def set_determinate(self):
        """设置为确定状态"""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")

    def get_progress(self) -> float:
        """获取当前进度（0-1）"""
        return float(self.progress_bar.get())
