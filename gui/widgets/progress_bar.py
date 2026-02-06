"""处理进度组件。"""

from typing import Any

import customtkinter as ctk


class ProgressBar(ctk.CTkFrame):
    """显示进度、阶段、失败数和 ETA。"""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.total_chunks = 0
        self.completed_chunks = 0
        self.failed_chunks = 0
        self.partial_chunks = 0
        self.current_phase = ""
        self.eta_seconds = 0
        self.eta_confidence = 0.0
        self._setup_ui()

    def _setup_ui(self) -> None:
        title_label = ctk.CTkLabel(self, text="处理进度", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(10, 10))

        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(pady=5)

        self.completed_label = ctk.CTkLabel(stats_frame, text="已完成: 0/0", width=150)
        self.completed_label.pack(side="left", padx=10)

        self.failed_label = ctk.CTkLabel(stats_frame, text="失败: 0", width=100, text_color="red")
        self.failed_label.pack(side="left", padx=10)

        self.partial_label = ctk.CTkLabel(
            stats_frame,
            text="部分完成: 0",
            width=100,
            text_color="orange",
        )
        self.partial_label.pack(side="left", padx=10)

        self.phase_label = ctk.CTkLabel(self, text="等待开始...", text_color="gray")
        self.phase_label.pack(pady=5)

        self.eta_label = ctk.CTkLabel(self, text="预估剩余时间: --", text_color="gray")
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
    ) -> None:
        """刷新展示状态。"""
        self.completed_chunks = completed
        self.total_chunks = total
        self.failed_chunks = failed
        self.partial_chunks = partial
        self.current_phase = phase
        self.eta_seconds = max(0, eta_seconds or 0)
        self.eta_confidence = eta_confidence

        if total > 0:
            self.progress_bar.set(max(0.0, min(1.0, completed / total)))

        self.completed_label.configure(text=f"已完成: {completed}/{total}")
        self.failed_label.configure(text=f"失败: {failed}")
        self.partial_label.configure(text=f"部分完成: {partial}")
        self.phase_label.configure(text=f"当前阶段: {phase}" if phase else "处理中...")

        if self.eta_seconds > 0:
            self.eta_label.configure(text=self._format_eta(self.eta_seconds, eta_confidence))
        else:
            self.eta_label.configure(text="预估剩余时间: 计算中...")

    def _format_eta(self, seconds: int, confidence: float) -> str:
        """格式化 ETA 文本。"""
        safe_seconds = max(seconds, 0)
        hours = safe_seconds // 3600
        minutes = (safe_seconds % 3600) // 60
        secs = safe_seconds % 60

        parts: list[str] = []
        if hours:
            parts.append(f"{hours}小时")
        if minutes:
            parts.append(f"{minutes}分钟")
        if secs or not parts:
            parts.append(f"{secs}秒")

        if confidence >= 0.8:
            level = "高"
        elif confidence >= 0.5:
            level = "中"
        else:
            level = "低"

        return f"预估剩余时间: {''.join(parts)} (置信度: {level})"

    def reset(self) -> None:
        """重置为初始状态。"""
        self.update_progress(0, 0, 0, 0, "等待开始...", 0, 0.0)
        self.progress_bar.set(0)

    def set_indeterminate(self) -> None:
        """切换为不确定进度模式。"""
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

    def set_determinate(self) -> None:
        """切换为确定进度模式。"""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")

    def get_progress(self) -> float:
        """获取当前 0-1 进度。"""
        return float(self.progress_bar.get())
