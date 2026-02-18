"""
处理页面

核心功能页面，三栏布局：文件区、进度区、日志区。
"""

import dataclasses
import logging
import re
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)

BREAKPOINTS = {
    "compact": 900,
    "normal": 1200,
}


@dataclasses.dataclass
class MergeProgressState:
    """合并进度状态管理

    封装合并阶段的状态变量，避免状态不一致。
    """

    last_phase: str = ""
    initial_outline_count: int = 0
    is_merge_phase: bool = False

    def reset(self) -> None:
        """重置状态到初始值"""
        self.last_phase = ""
        self.initial_outline_count = 0
        self.is_merge_phase = False

    def on_phase_transition(
        self, new_phase: str, current_outline_count: int, fallback_completed: int = 0
    ) -> bool:
        """处理阶段切换

        Args:
            new_phase: 新阶段名称
            current_outline_count: 当前大纲数量
            fallback_completed: 后备完成数（当首次进入合并阶段时使用）

        Returns:
            bool: 是否发生了从 processing 到 merging 的切换
        """
        is_transition = self.last_phase == "processing" and new_phase == "merging"
        self.last_phase = new_phase

        if is_transition:
            self.is_merge_phase = True
            # 优先使用 fallback_completed（即生成阶段已完成的大纲数）
            if fallback_completed > 0:
                self.initial_outline_count = fallback_completed
            elif current_outline_count > 0:
                self.initial_outline_count = current_outline_count

        # 如果首次进入合并阶段且 initial_outline_count 未设置，使用后备值
        if (
            new_phase == "merging"
            and self.initial_outline_count == 0
            and (fallback_completed > 0 or current_outline_count > 0)
        ):
            self.initial_outline_count = (
                fallback_completed if fallback_completed > 0 else current_outline_count
            )

        return is_transition


class ProcessPage(ctk.CTkFrame):
    """
    处理页面

    响应式布局：
    - 宽度 >= 1200px：双列布局（文件区 + 进度区）
    - 宽度 900-1200px：双列布局（调整比例）
    - 宽度 < 900px：单列布局（堆叠）
    """

    def __init__(self, master, **kwargs):
        if "fg_color" not in kwargs:
            kwargs["fg_color"] = get_color("bg_primary", mode="auto")
        super().__init__(master, **kwargs)

        self._current_file: Path | None = None
        self._on_start_callback: Callable | None = None
        self._on_cancel_callback: Callable | None = None
        self._all_logs: list[str] = []
        self._current_layout_mode: str = "normal"

        self._merge_state = MergeProgressState()

        self._setup_ui()
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        """窗口大小变化时调整布局"""
        if not hasattr(self, "_main_container"):
            return
        width = event.width
        new_mode = self._get_layout_mode(width)
        if new_mode != self._current_layout_mode:
            self._current_layout_mode = new_mode
            self._apply_layout_mode(new_mode)

    def _get_layout_mode(self, width: int) -> str:
        """根据宽度获取布局模式"""
        if width < BREAKPOINTS["compact"]:
            return "compact"
        elif width < BREAKPOINTS["normal"]:
            return "medium"
        return "normal"

    def _apply_layout_mode(self, mode: str):
        """应用布局模式"""
        if not hasattr(self, "_top_left") or not hasattr(self, "_top_right"):
            return

        match mode:
            case "compact":
                self._top_left.grid(
                    row=0, column=0, columnspan=2, sticky="nsew", pady=(0, SPACING["md"])
                )
                self._top_right.grid(
                    row=1, column=0, columnspan=2, sticky="nsew", pady=(0, SPACING["lg"])
                )
                self._bottom_log.grid(row=2, column=0, columnspan=2, sticky="nsew")
                self._action_row.grid(row=3, column=0, columnspan=2, sticky="ew")
                self._main_container.grid_columnconfigure(0, weight=1, minsize=0)
                self._main_container.grid_columnconfigure(1, weight=0, minsize=0)
                self._main_container.grid_rowconfigure(0, weight=0)
                self._main_container.grid_rowconfigure(1, weight=2)
                self._main_container.grid_rowconfigure(2, weight=3)
                self._main_container.grid_rowconfigure(3, weight=0)
            case "medium":
                self._top_left.grid(
                    row=0, column=0, sticky="nsew", padx=(0, SPACING["md"]), pady=(0, SPACING["lg"])
                )
                self._top_right.grid(row=0, column=1, sticky="nsew", pady=(0, SPACING["lg"]))
                self._bottom_log.grid(row=1, column=0, columnspan=2, sticky="nsew")
                self._action_row.grid(row=2, column=0, columnspan=2, sticky="ew")
                self._main_container.grid_columnconfigure(0, weight=40, minsize=280)
                self._main_container.grid_columnconfigure(1, weight=60, minsize=400)
                self._main_container.grid_rowconfigure(0, weight=4)
                self._main_container.grid_rowconfigure(1, weight=3)
                self._main_container.grid_rowconfigure(2, weight=0)
                self._main_container.grid_rowconfigure(3, weight=0)
            case _:
                self._top_left.grid(
                    row=0, column=0, sticky="nsew", padx=(0, SPACING["lg"]), pady=(0, SPACING["lg"])
                )
                self._top_right.grid(row=0, column=1, sticky="nsew", pady=(0, SPACING["lg"]))
                self._bottom_log.grid(row=1, column=0, columnspan=2, sticky="nsew")
                self._action_row.grid(row=2, column=0, columnspan=2, sticky="ew")
                self._main_container.grid_columnconfigure(0, weight=32, minsize=320)
                self._main_container.grid_columnconfigure(1, weight=68, minsize=520)
                self._main_container.grid_rowconfigure(0, weight=4)
                self._main_container.grid_rowconfigure(1, weight=3)
                self._main_container.grid_rowconfigure(2, weight=0)
                self._main_container.grid_rowconfigure(3, weight=0)

    def _setup_ui(self):
        """设置 UI"""
        self._main_container = ctk.CTkFrame(self, fg_color="transparent")
        self._main_container.pack(fill="both", expand=True, padx=40, pady=40)

        self._main_container.grid_columnconfigure(0, weight=32, minsize=320)
        self._main_container.grid_columnconfigure(1, weight=68, minsize=520)
        self._main_container.grid_rowconfigure(0, weight=4)
        self._main_container.grid_rowconfigure(1, weight=3)
        self._main_container.grid_rowconfigure(2, weight=0)

        self._top_left = ctk.CTkFrame(self._main_container, fg_color="transparent")
        self._top_left.grid(
            row=0, column=0, sticky="nsew", padx=(0, SPACING["lg"]), pady=(0, SPACING["lg"])
        )

        self._top_right = ctk.CTkFrame(self._main_container, fg_color="transparent")
        self._top_right.grid(row=0, column=1, sticky="nsew", pady=(0, SPACING["lg"]))

        self._setup_file_section(self._top_left)
        self._setup_progress_section(self._top_right)

        self._bottom_log = ctk.CTkFrame(self._main_container, fg_color="transparent")
        self._bottom_log.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, SPACING["lg"]))
        self._setup_log_section(self._bottom_log)

        self._action_row = ctk.CTkFrame(self._main_container, fg_color="transparent")
        self._action_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._setup_action_bar(self._action_row)

    def refresh_layout(self):
        """刷新布局"""
        self.update_idletasks()
        width = self.winfo_width()
        self._current_layout_mode = self._get_layout_mode(width)
        self._apply_layout_mode(self._current_layout_mode)

    def _safe_update_progress_bar(self, value: float) -> None:
        """安全更新进度条（避免重复 hasattr 检查）"""
        if hasattr(self, "_progress_bar"):
            self._progress_bar.set(value)

    def _safe_update_progress_text(self, text: str) -> None:
        """安全更新进度文本（避免重复 hasattr 检查）"""
        if hasattr(self, "_progress_text_label"):
            self._progress_text_label.configure(text=text)

    def _safe_update_phase_label(self, text: str) -> None:
        """安全更新阶段标签"""
        if hasattr(self, "_phase_label"):
            self._phase_label.configure(text=text)

    def _safe_update_eta_label(self, text: str) -> None:
        """安全更新 ETA 标签"""
        if hasattr(self, "_eta_label"):
            self._eta_label.configure(text=text)

    def _update_status_badge(self, status) -> None:
        """更新状态徽章"""
        if hasattr(self, "_status_badge"):
            self._status_badge.set_status(status)

    def _setup_file_section(self, parent):
        """设置文件选择区"""
        from gui.components.card import Card

        file_card = Card(parent, title="文件选择", padding="md")
        file_card.pack(fill="x", pady=(0, SPACING["md"]))

        file_select_zone = ctk.CTkFrame(
            file_card.content,
            fg_color=get_color("bg_tertiary", mode="auto"),
            border_color=get_color("border", mode="auto"),
            border_width=2,
            corner_radius=8,
            height=140,
        )
        file_select_zone.pack(fill="x", pady=(0, SPACING["sm"]))
        file_select_zone.pack_propagate(False)

        try:
            from gui.components.icon import Icon, IconSize

            file_icon = Icon(
                file_select_zone,
                name="file-text",
                size=IconSize.LG,
                color=get_color("fg_secondary", mode="auto"),
            )
            file_icon.place(relx=0.5, rely=0.35, anchor="center")
        except Exception:
            pass

        select_hint = ctk.CTkLabel(
            file_select_zone,
            text="选择要处理的文本文件",
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        select_hint.place(relx=0.5, rely=0.55, anchor="center")

        from gui.components.button import Button, ButtonSize, ButtonVariant

        select_button = Button(
            file_select_zone,
            text="选择文件",
            variant=ButtonVariant.PRIMARY,
            size=ButtonSize.SM,
            command=self._on_select_file,
        )
        select_button.place(relx=0.5, rely=0.78, anchor="center")

        self._selected_file_label = ctk.CTkLabel(
            file_card.content,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=get_color("accent", mode="auto"),
            wraplength=280,
        )
        self._selected_file_label.pack(fill="x", pady=(SPACING["xs"], 0))

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
        from gui.components.card import Card
        from gui.components.status_badge import ProcessingStatus, StatusBadge

        progress_card = Card(parent, title="处理进度", padding="lg")
        progress_card.pack(fill="both", expand=True, pady=(0, SPACING["md"]))

        header_frame = ctk.CTkFrame(progress_card.content, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, SPACING["sm"]))

        self._status_badge = StatusBadge(header_frame, status=ProcessingStatus.IDLE, size="sm")
        self._status_badge.pack(side="right")

        self._progress_bar = ctk.CTkProgressBar(
            progress_card.content,
            height=16,
        )
        self._progress_bar.pack(fill="x", pady=(SPACING["sm"], SPACING["sm"]))
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

        from config import get_processing_config
        from tokenizer import count_tokens

        try:
            stat = filepath.stat()
            size_mb = stat.st_size / (1024 * 1024)

            if hasattr(self, "_selected_file_label"):
                self._selected_file_label.configure(text=f"✓ {filepath.name}")

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

            self._start_button.configure(state="normal")

            logger.info(f"Selected file: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load file: {e}")

    def _on_start(self):
        """开始处理"""
        from gui.components.status_badge import ProcessingStatus

        if self._on_start_callback:
            self._on_start_callback()

        self._start_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")
        self._update_status_badge(ProcessingStatus.PROCESSING)

    def _on_cancel(self):
        """取消处理"""
        from gui.components.status_badge import ProcessingStatus

        if self._on_cancel_callback:
            self._on_cancel_callback()
        self._update_status_badge(ProcessingStatus.CANCELLED)

    def set_final_status(self, success: bool, cancelled: bool = False):
        """
        设置最终状态

        Args:
            success: 是否成功完成
            cancelled: 是否被取消
        """
        from gui.components.status_badge import ProcessingStatus

        if cancelled:
            self._update_status_badge(ProcessingStatus.CANCELLED)
        elif success:
            self._update_status_badge(ProcessingStatus.COMPLETED)
        else:
            self._update_status_badge(ProcessingStatus.FAILED)

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
            if selected_level != "ALL":
                message_level = self._extract_log_level(message)
                if message_level != selected_level:
                    continue
            self._log_text.insert("end", message + "\n")
        self._log_text.see("end")

    def _extract_log_level(self, message: str) -> str | None:
        """从日志头部提取日志级别，避免正文关键字误判。"""
        levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        prefix = message.strip()[:120].upper()

        # 直接前缀格式，如: "INFO: ...", "[ERROR] ...", "(WARNING) ..."
        direct_patterns = [
            r"^\[(DEBUG|INFO|WARNING|ERROR)\]",
            r"^\((DEBUG|INFO|WARNING|ERROR)\)",
            r"^(DEBUG|INFO|WARNING|ERROR)\s*[:\-]",
        ]
        for pattern in direct_patterns:
            match = re.match(pattern, prefix)
            if match:
                return match.group(1)

        # 常见 logging 格式，如: "time - name - LEVEL - message"
        for part in prefix.split(" - ")[:4]:
            normalized = part.strip(" []()")
            if normalized in levels:
                return normalized

        return None

    def update_progress(
        self,
        completed: int,
        total: int,
        failed: int = 0,
        partial: int = 0,
        phase: str = "",
        eta_seconds: int = 0,
        merge_level: int = 0,
        merge_batch_current: int = 0,
        merge_batch_total: int = 0,
        merge_outlines_count: int = 0,
    ):
        """更新进度（支持生成和合并阶段）"""
        from gui.components.status_badge import ProcessingStatus

        is_transition = self._merge_state.on_phase_transition(
            phase, merge_outlines_count, completed
        )

        if is_transition or (phase == "merging" and not self._merge_state.is_merge_phase):
            self._merge_state.is_merge_phase = True
            if self._merge_state.initial_outline_count == 0 and completed > 0:
                self._merge_state.initial_outline_count = completed
            self._safe_update_progress_bar(0)
            self._safe_update_progress_text("0%")
            self._update_status_badge(ProcessingStatus.MERGING)

        if phase != "merging" and self._merge_state.is_merge_phase:
            self._merge_state.is_merge_phase = False

        if phase == "merging" and self._merge_state.is_merge_phase:
            progress = self._calculate_merge_progress(
                merge_level=merge_level,
                merge_batch_current=merge_batch_current,
                merge_batch_total=merge_batch_total,
                merge_outlines_count=merge_outlines_count,
            )

            self._safe_update_progress_bar(progress)
            self._safe_update_progress_text(f"{int(progress * 100)}%")

            self._safe_update_phase_label(
                f"当前阶段: 正在合并大纲 (层级 {merge_level}, 批次 {merge_batch_current}/{merge_batch_total})"
            )

            self._safe_update_eta_label("合并中...")

        elif phase == "processing":
            progress = completed / total if total > 0 else 0
            self._safe_update_progress_bar(progress)
            self._safe_update_progress_text(f"{int(progress * 100)}%")

            self._update_status_badge(ProcessingStatus.PROCESSING)

            if hasattr(self, "_stat_labels"):
                self._stat_labels["completed"].configure(text=str(completed))
                self._stat_labels["failed"].configure(text=str(failed))
                self._stat_labels["partial"].configure(text=str(partial))

            self._safe_update_phase_label(f"当前阶段: 正在生成大纲 ({completed}/{total})")

            # 更新 ETA
            if eta_seconds > 0:
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

                self._safe_update_eta_label(f"预估剩余时间: {''.join(time_parts)}")

        elif phase == "saving":
            # 保存阶段：显示完成状态
            self._safe_update_progress_bar(1.0)
            self._safe_update_progress_text("100%")
            self._safe_update_phase_label("当前阶段: 正在保存结果...")
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
        self._safe_update_progress_bar(0)
        self._safe_update_progress_text("0%")

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
        level_progress: float = 1.0 - (merge_level / (merge_level + 5)) if merge_level > 0 else 0.8

        # 批次进度
        batch_progress: float
        if merge_batch_total > 0:
            batch_progress = merge_batch_current / merge_batch_total
        else:
            batch_progress = 0.0

        # 大纲缩减进度
        total: float
        if self._merge_state.initial_outline_count > 0:
            # 确保大纲数量非负
            safe_outlines_count = max(0, merge_outlines_count)
            reduction_progress = 1.0 - (
                safe_outlines_count / self._merge_state.initial_outline_count
            )
            # 根据是否有缩减进度分配权重
            total = level_progress * 0.4 + batch_progress * 0.4 + reduction_progress * 0.2
        else:
            # 没有初始大纲数量时，重新分配权重给层级和批次
            total = level_progress * 0.5 + batch_progress * 0.5

        # 限制在 [0, 1] 范围
        return max(0.0, min(1.0, total))
