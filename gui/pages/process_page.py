"""
处理页面

核心功能页面，三栏布局：文件区、进度区、日志区。
"""

import logging
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class ProcessPage(ctk.CTkFrame):
    """
    处理页面

    固定两行布局：
    - 上行：左侧文件区 + 右侧进度区
    - 下行：日志区
    """

    def __init__(self, master, **kwargs):
        if "fg_color" not in kwargs:
            kwargs["fg_color"] = get_color("bg_primary", mode="auto")
        super().__init__(master, **kwargs)

        self._current_file: Path | None = None
        self._on_start_callback: Callable | None = None
        self._on_cancel_callback: Callable | None = None
        self._all_logs: list[str] = []

        # 合并进度状态管理
        self._last_phase: str = ""
        self._initial_outline_count: int = 0
        self._is_merge_phase: bool = False

        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        # 主容器（带内边距）
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=40, pady=40)

        main_container.grid_columnconfigure(0, weight=32, minsize=320)
        main_container.grid_columnconfigure(1, weight=68, minsize=520)
        main_container.grid_rowconfigure(0, weight=4)
        main_container.grid_rowconfigure(1, weight=3)
        main_container.grid_rowconfigure(2, weight=0)

        # 上行：文件区 + 进度区
        top_left = ctk.CTkFrame(main_container, fg_color="transparent")
        top_left.grid(
            row=0, column=0, sticky="nsew", padx=(0, SPACING["lg"]), pady=(0, SPACING["lg"])
        )

        top_right = ctk.CTkFrame(main_container, fg_color="transparent")
        top_right.grid(row=0, column=1, sticky="nsew", pady=(0, SPACING["lg"]))

        self._setup_file_section(top_left)
        self._setup_progress_section(top_right)

        # 下行：日志区整行
        bottom_log = ctk.CTkFrame(main_container, fg_color="transparent")
        bottom_log.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, SPACING["lg"]))
        self._setup_log_section(bottom_log)

        # 底部操作栏
        action_row = ctk.CTkFrame(main_container, fg_color="transparent")
        action_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._setup_action_bar(action_row)

    def refresh_layout(self):
        """固定布局，无需动态重排。"""
        return

    def _setup_file_section(self, parent):
        """设置文件选择区"""
        # 文件选择卡片
        from gui.components.card import Card

        file_card = Card(parent, title="文件选择", padding="md")
        file_card.pack(fill="x", pady=(0, SPACING["md"]))

        # 拖放区域
        drop_zone = ctk.CTkFrame(
            file_card.content,
            fg_color=get_color("bg_tertiary", mode="auto"),
            border_color=get_color("border", mode="auto"),
            border_width=2,
            corner_radius=8,
            height=120,
        )
        drop_zone.pack(fill="x", pady=(0, SPACING["sm"]))
        drop_zone.pack_propagate(False)

        drop_label = ctk.CTkLabel(
            drop_zone,
            text="拖放文件到此处\n或",
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        drop_label.place(relx=0.5, rely=0.4, anchor="center")

        # 选择文件按钮
        from gui.components.button import Button, ButtonSize, ButtonVariant

        select_button = Button(
            drop_zone,
            text="选择文件",
            variant=ButtonVariant.PRIMARY,
            size=ButtonSize.SM,
            command=self._on_select_file,
        )
        select_button.place(relx=0.5, rely=0.7, anchor="center")

        # 文件信息卡片
        info_card = Card(parent, title="文件信息", padding="md")
        info_card.pack(fill="x", pady=(0, SPACING["md"]))

        # 信息网格
        info_frame = ctk.CTkFrame(info_card.content, fg_color="transparent")
        info_frame.pack(fill="x")

        self._file_info_labels = {}

        for label, key in [
            ("大小", "size"),
            ("Tokens", "tokens"),
            ("预估块数", "chunks"),
            ("修改时间", "mtime"),
        ]:
            row = ctk.CTkFrame(info_frame, fg_color="transparent")
            row.pack(fill="x", pady=SPACING["xs"])

            key_label = ctk.CTkLabel(
                row,
                text=label,
                font=ctk.CTkFont(size=12),
                text_color=get_color("fg_secondary", mode="auto"),
                width=80,
                anchor="w",
            )
            key_label.pack(side="left")

            value_label = ctk.CTkLabel(
                row,
                text="--",
                font=ctk.CTkFont(size=12),
                text_color=get_color("fg_primary", mode="auto"),
                anchor="w",
            )
            value_label.pack(side="left", fill="x", expand=True)

            self._file_info_labels[key] = value_label

    def _setup_progress_section(self, parent):
        """设置进度区"""
        # 进度可视化卡片
        from gui.components.card import Card

        progress_card = Card(parent, title="处理进度", padding="lg")
        progress_card.pack(fill="both", expand=True, pady=(0, SPACING["md"]))

        self._progress_bar = ctk.CTkProgressBar(
            progress_card.content,
            height=16,
        )
        self._progress_bar.pack(fill="x", pady=(SPACING["md"], SPACING["sm"]))
        self._progress_bar.set(0)

        self._progress_text_label = ctk.CTkLabel(
            progress_card.content,
            text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self._progress_text_label.pack(anchor="e", pady=(0, SPACING["sm"]))

        # 统计卡片网格
        stats_frame = ctk.CTkFrame(progress_card.content, fg_color="transparent")
        stats_frame.pack(fill="x", pady=SPACING["md"])

        self._stat_labels = {}

        for key, label in [("completed", "已完成"), ("failed", "失败"), ("partial", "部分完成")]:
            stat_card = ctk.CTkFrame(
                stats_frame,
                fg_color=get_color("bg_secondary", mode="auto"),
                border_width=1,
                border_color=get_color("border", mode="auto"),
                corner_radius=8,
                height=80,
            )
            stat_card.pack(side="left", fill="both", expand=True, padx=(0, SPACING["sm"]))
            stat_card.pack_propagate(False)

            stat_label = ctk.CTkLabel(
                stat_card,
                text="0",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color=get_color("fg_primary", mode="auto"),
            )
            stat_label.pack(pady=(SPACING["sm"], SPACING["xs"]))

            desc_label = ctk.CTkLabel(
                stat_card,
                text=label,
                font=ctk.CTkFont(size=12),
                text_color=get_color("fg_secondary", mode="auto"),
            )
            desc_label.pack()

            self._stat_labels[key] = stat_label

        # 当前阶段
        self._phase_label = ctk.CTkLabel(
            progress_card.content,
            text="等待开始...",
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self._phase_label.pack(pady=SPACING["sm"])

        # ETA
        self._eta_label = ctk.CTkLabel(
            progress_card.content,
            text="预估剩余时间: --",
            font=ctk.CTkFont(size=12),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self._eta_label.pack()

    def _setup_log_section(self, parent):
        """设置日志区"""
        # 日志卡片
        from gui.components.card import Card

        log_card = Card(parent, title="实时日志", padding="sm")
        log_card.pack(fill="both", expand=True)

        # 日志级别过滤
        filter_frame = ctk.CTkFrame(log_card.content, fg_color="transparent")
        filter_frame.pack(fill="x", pady=(0, SPACING["sm"]))

        self._log_level_var = ctk.StringVar(value="ALL")

        for level in ["ALL", "INFO", "WARNING", "ERROR"]:
            rb = ctk.CTkRadioButton(
                filter_frame,
                text=level,
                variable=self._log_level_var,
                value=level,
                command=self._filter_logs,
                font=ctk.CTkFont(size=11),
            )
            rb.pack(side="left", padx=(0, SPACING["sm"]))

        # 日志文本框
        self._log_text = ctk.CTkTextbox(
            log_card.content,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=get_color("bg_primary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        self._log_text.pack(fill="both", expand=True)

    def _setup_action_bar(self, parent):
        """设置底部操作栏"""
        action_frame = ctk.CTkFrame(parent, fg_color="transparent")
        action_frame.pack(fill="x")

        from gui.components.button import Button, ButtonSize, ButtonVariant

        # 开始按钮
        self._start_button = Button(
            action_frame,
            text="开始处理",
            variant=ButtonVariant.PRIMARY,
            size=ButtonSize.MD,
            command=self._on_start,
            width=120,
        )
        self._start_button.pack(side="left")

        # 取消按钮
        self._cancel_button = Button(
            action_frame,
            text="取消",
            variant=ButtonVariant.DANGER,
            size=ButtonSize.MD,
            command=self._on_cancel,
            width=100,
            state="disabled",
        )
        self._cancel_button.pack(side="left", padx=(SPACING["sm"], 0))

        # 打开输出目录按钮（右侧）
        output_button = Button(
            action_frame,
            text="打开输出",
            variant=ButtonVariant.SECONDARY,
            size=ButtonSize.MD,
            command=self._on_open_output,
            width=120,
        )
        output_button.pack(side="right")

    def _on_select_file(self):
        """选择文件事件"""
        from tkinter import filedialog

        filepath = filedialog.askopenfilename(
            filetypes=[
                ("文本文件", "*.txt"),
                ("Markdown 文件", "*.md"),
                ("所有文件", "*.*"),
            ],
            title="选择要处理的文件",
        )

        if filepath:
            self.set_file(Path(filepath))

    def set_file(self, filepath: Path):
        """设置当前文件"""
        self._current_file = filepath

        # 更新显示
        from config import get_processing_config
        from tokenizer import count_tokens

        try:
            stat = filepath.stat()
            size_mb = stat.st_size / (1024 * 1024)

            # 更新文件信息
            self._file_info_labels["size"].configure(text=f"{size_mb:.2f} MB")

            content = filepath.read_text(encoding="utf-8")
            token_count = count_tokens(content)
            self._file_info_labels["tokens"].configure(text=f"{token_count:,}")

            proc_config = get_processing_config()
            chunk_count = (token_count // proc_config.target_tokens_per_chunk) + 1
            self._file_info_labels["chunks"].configure(text=str(chunk_count))

            from datetime import datetime

            mtime = datetime.fromtimestamp(stat.st_mtime)
            self._file_info_labels["mtime"].configure(text=mtime.strftime("%Y-%m-%d %H:%M"))

            # 启用开始按钮
            self._start_button.configure(state="normal")

            logger.info(f"Selected file: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load file: {e}")

    def _on_start(self):
        """开始处理"""
        if self._on_start_callback:
            self._on_start_callback()

        # 更新 UI 状态
        self._start_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")

    def _on_cancel(self):
        """取消处理"""
        if self._on_cancel_callback:
            self._on_cancel_callback()

        # 更新 UI 状态
        self._start_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")

    def _on_open_output(self):
        """打开输出目录"""
        import platform
        import subprocess

        from config import get_processing_config

        output_dir = Path(get_processing_config().output_dir)
        if not output_dir.exists():
            output_dir = Path.cwd()

        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            logger.error(f"Failed to open output directory: {e}")

    def _filter_logs(self):
        """过滤日志"""
        if not hasattr(self, "_log_text"):
            return

        selected_level = self._log_level_var.get()
        self._log_text.delete("1.0", "end")

        for message in self._all_logs:
            if selected_level != "ALL" and f" - {selected_level} - " not in message:
                continue
            self._log_text.insert("end", message + "\n")
        self._log_text.see("end")

    def update_progress(
        self,
        completed: int,
        total: int,
        failed: int = 0,
        partial: int = 0,
        phase: str = "",
        eta_seconds: int = 0,
        # 新增合并相关参数
        merge_level: int = 0,
        merge_batch_current: int = 0,
        merge_batch_total: int = 0,
        merge_outlines_count: int = 0,
    ):
        """更新进度（支持生成和合并阶段）"""

        # 进入合并阶段（从任何阶段或空阶段）
        phase_changed = phase != self._last_phase
        if phase == "merging" and not self._is_merge_phase:
            self._is_merge_phase = True
            # 如果从生成阶段切换，保存初始大纲数量
            if self._last_phase == "processing":
                self._initial_outline_count = completed
                # 重置进度条
                if hasattr(self, "_progress_bar"):
                    self._progress_bar.set(0)
                if hasattr(self, "_progress_text_label"):
                    self._progress_text_label.configure(text="0%")
            # 从其他阶段直接进入合并阶段（使用当前大纲数量作为初始值）
            elif phase_changed:
                # 如果没有初始大纲数量，使用当前大纲数量作为初始值
                if self._initial_outline_count == 0 and merge_outlines_count > 0:
                    self._initial_outline_count = merge_outlines_count
                # 重置进度条
                if hasattr(self, "_progress_bar"):
                    self._progress_bar.set(0)
                if hasattr(self, "_progress_text_label"):
                    self._progress_text_label.configure(text="0%")

        # 离开合并阶段
        elif phase != "merging" and self._is_merge_phase:
            self._is_merge_phase = False

        # 更新最后阶段
        self._last_phase = phase

        # 根据阶段更新进度
        if phase == "merging" and self._is_merge_phase:
            # 合并阶段：使用合并进度计算
            progress = self._calculate_merge_progress(
                merge_level=merge_level,
                merge_batch_current=merge_batch_current,
                merge_batch_total=merge_batch_total,
                merge_outlines_count=merge_outlines_count,
            )

            # 更新进度条
            if hasattr(self, "_progress_bar"):
                self._progress_bar.set(progress)
            if hasattr(self, "_progress_text_label"):
                self._progress_text_label.configure(text=f"{int(progress * 100)}%")

            # 更新阶段文本（显示合并详情）
            if hasattr(self, "_phase_label"):
                self._phase_label.configure(
                    text=f"当前阶段: 正在合并大纲 (层级 {merge_level}, 批次 {merge_batch_current}/{merge_batch_total})"
                )

            # 合并阶段不显示 ETA
            if hasattr(self, "_eta_label"):
                self._eta_label.configure(text="合并中...")

        elif phase == "processing":
            # 生成阶段：使用原有逻辑
            progress = completed / total if total > 0 else 0
            if hasattr(self, "_progress_bar"):
                self._progress_bar.set(progress)
            if hasattr(self, "_progress_text_label"):
                self._progress_text_label.configure(text=f"{int(progress * 100)}%")

            # 更新统计
            if hasattr(self, "_stat_labels"):
                self._stat_labels["completed"].configure(text=str(completed))
                self._stat_labels["failed"].configure(text=str(failed))
                self._stat_labels["partial"].configure(text=str(partial))

            # 更新阶段
            if hasattr(self, "_phase_label"):
                self._phase_label.configure(text=f"当前阶段: 正在生成大纲 ({completed}/{total})")

            # 更新 ETA
            if eta_seconds > 0 and hasattr(self, "_eta_label"):
                hours = eta_seconds // 3600
                minutes = (eta_seconds % 3600) // 60
                secs = eta_seconds % 60

                time_parts = []
                if hours > 0:
                    time_parts.append(f"{hours}小时")
                if minutes > 0:
                    time_parts.append(f"{minutes}分钟")
                if secs > 0 or not time_parts:
                    time_parts.append(f"{secs}秒")

                self._eta_label.configure(text=f"预估剩余时间: {''.join(time_parts)}")

        elif phase == "saving":
            # 保存阶段：显示完成状态
            if hasattr(self, "_progress_bar"):
                self._progress_bar.set(1.0)
            if hasattr(self, "_progress_text_label"):
                self._progress_text_label.configure(text="100%")
            if hasattr(self, "_phase_label"):
                self._phase_label.configure(text="当前阶段: 正在保存结果...")
        else:
            # 未知阶段，记录警告但保持静默（不影响现有流程）
            logger.debug(f"Unknown phase: {phase}")

    def append_log(self, message: str):
        """追加日志"""
        self._all_logs.append(message)
        self._filter_logs()

    def set_callbacks(self, on_start: Callable | None = None, on_cancel: Callable | None = None):
        """设置回调函数"""
        self._on_start_callback = on_start
        self._on_cancel_callback = on_cancel

    def reset(self):
        """重置页面状态"""
        if hasattr(self, "_progress_bar"):
            self._progress_bar.set(0)
        if hasattr(self, "_progress_text_label"):
            self._progress_text_label.configure(text="0%")

        for label in self._stat_labels.values():
            label.configure(text="0")

        self._phase_label.configure(text="等待开始...")
        self._eta_label.configure(text="预估剩余时间: --")

        self._start_button.configure(state="disabled")
        self._cancel_button.configure(state="disabled")

    def _calculate_merge_progress(
        self,
        merge_level: int,
        merge_batch_current: int,
        merge_batch_total: int,
        merge_outlines_count: int,
    ) -> float:
        """
        计算合并阶段进度

        公式: 进度 = 层级进度 * 0.4 + 批次进度 * 0.4 + 大纲缩减进度 * 0.2

        Args:
            merge_level: 合并层级（当前递归深度）
            merge_batch_current: 当前批次索引
            merge_batch_total: 总批次数
            merge_outlines_count: 当前正在合并的大纲数量

        Returns:
            0-1 之间的进度值
        """
        # 层级进度：越接近顶层（level=1），进度越高
        level_progress = 1.0 - (merge_level / (merge_level + 5)) if merge_level > 0 else 0.8

        # 批次进度
        if merge_batch_total > 0:
            batch_progress = merge_batch_current / merge_batch_total
        else:
            batch_progress = 0.0

        # 大纲缩减进度
        if self._initial_outline_count > 0:
            reduction_progress = 1.0 - (merge_outlines_count / self._initial_outline_count)
            # 根据是否有缩减进度分配权重
            total = level_progress * 0.4 + batch_progress * 0.4 + reduction_progress * 0.2
        else:
            # 没有初始大纲数量时，重新分配权重给层级和批次
            total = level_progress * 0.5 + batch_progress * 0.5

        # 限制在 [0, 1] 范围
        return max(0.0, min(1.0, total))
