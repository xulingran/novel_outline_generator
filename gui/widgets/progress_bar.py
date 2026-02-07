"""
进度条组件

显示处理进度、统计信息、ETA 等信息。
"""

import logging

import customtkinter as ctk

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
        title_label = ctk.CTkLabel(self, text="处理进度", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(10, 10))

        # 进度条
        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        # 统计信息框架
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(pady=5)

        # 已完成
        self.completed_label = ctk.CTkLabel(
            stats_frame, text="已完成: 0/0", width=150, font=ctk.CTkFont(size=12)
        )
        self.completed_label.pack(side="left", padx=10)

        # 失败
        self.failed_label = ctk.CTkLabel(
            stats_frame, text="失败: 0", width=100, font=ctk.CTkFont(size=12), text_color="red"
        )
        self.failed_label.pack(side="left", padx=10)

        # 部分完成
        self.partial_label = ctk.CTkLabel(
            stats_frame,
            text="部分完成: 0",
            width=100,
            font=ctk.CTkFont(size=12),
            text_color="orange",
        )
        self.partial_label.pack(side="left", padx=10)

        # 当前阶段
        self.phase_label = ctk.CTkLabel(
            self, text="等待开始...", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.phase_label.pack(pady=5)

        # ETA 显示
        self.eta_label = ctk.CTkLabel(
            self, text="预估剩余时间: --", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.eta_label.pack(pady=(5, 10))

    def update_progress(
        self,
        completed: int,
        total: int,
        failed: int = 0,
        partial: int = 0,
        phase: str = "",
        eta_seconds: int | None = None,
        eta_confidence: float = 0.0,
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
            eta_confidence: 置信度（0-1）
        """
        self.completed_chunks = completed
        self.total_chunks = total
        self.failed_chunks = failed
        self.partial_chunks = partial
        self.current_phase = phase
        self.eta_seconds = eta_seconds or 0
        self.eta_confidence = eta_confidence

        # 更新进度条
        if total > 0:
            progress = completed / total
            self.progress_bar.set(progress)

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
        if eta_seconds and eta_seconds > 0:
            eta_text = self._format_eta(eta_seconds, eta_confidence)
            self.eta_label.configure(text=eta_text)
        else:
            self.eta_label.configure(text="预估剩余时间: 计算中...")

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
