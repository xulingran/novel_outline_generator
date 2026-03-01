"""
文件选择器组件

提供文件浏览、文件信息显示、token 预估等功能。
"""

import logging
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from config import get_processing_config
from gui.theme_manager import SPACING, get_color
from tokenizer import count_tokens

logger = logging.getLogger(__name__)


class FileSelector(ctk.CTkFrame):
    """
    文件选择器组件

    功能：
    - 文件浏览按钮
    - 文件信息显示（大小、修改时间、token 数量）
    - 预估块数显示
    """

    def __init__(
        self,
        master,
        title: str = "文件选择",
        file_types: list[tuple[str, str]] | None = None,
        on_file_selected: Callable[[Path], None] | None = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.title = title
        self.file_types = file_types or [
            ("文本文件", "*.txt"),
            ("Markdown 文件", "*.md"),
            ("所有文件", "*.*"),
        ]
        self.on_file_selected = on_file_selected
        self.current_file: Path | None = None

        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        # 标题
        title_label = ctk.CTkLabel(
            self,
            text=self.title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        title_label.pack(pady=(SPACING["md"], SPACING["sm"]))

        # 文件路径显示
        self.file_path_label = ctk.CTkLabel(
            self,
            text="未选择文件",
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self.file_path_label.pack(pady=SPACING["sm"])

        # 文件信息显示
        self.file_info_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.file_info_frame.pack(pady=SPACING["sm"], fill="x")

        # 大小
        self.size_label = ctk.CTkLabel(
            self.file_info_frame,
            text="大小: --",
            width=200,
            text_color=get_color("fg_primary", mode="auto"),
        )
        self.size_label.pack(side="left", padx=SPACING["sm"])

        # Tokens
        self.tokens_label = ctk.CTkLabel(
            self.file_info_frame,
            text="Tokens: --",
            width=200,
            text_color=get_color("fg_primary", mode="auto"),
        )
        self.tokens_label.pack(side="left", padx=SPACING["sm"])

        # 预估块数
        self.chunks_label = ctk.CTkLabel(
            self.file_info_frame,
            text="预估块数: --",
            width=150,
            text_color=get_color("fg_primary", mode="auto"),
        )
        self.chunks_label.pack(side="left", padx=SPACING["sm"])

        # 修改时间
        self.mtime_label = ctk.CTkLabel(
            self.file_info_frame,
            text="",
            text_color=get_color("fg_secondary", mode="auto"),
            width=200,
        )
        self.mtime_label.pack(side="left", padx=SPACING["sm"])

        # 选择按钮
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=(SPACING["md"], SPACING["md"]))

        self.select_button = ctk.CTkButton(
            button_frame,
            text="选择文件",
            command=self._on_select_file,
            width=120,
            fg_color=get_color("accent", mode="auto"),
            hover_color=get_color("accent_secondary", mode="auto"),
            text_color=get_color("bg_primary", mode="auto"),
        )
        self.select_button.pack()

    def _on_select_file(self):
        """选择文件事件处理"""
        from tkinter import filedialog

        filepath = filedialog.askopenfilename(filetypes=self.file_types, title="选择要处理的文件")

        if filepath:
            self.set_file(Path(filepath))

    def set_file(self, filepath: Path):
        """
        设置当前文件

        Args:
            filepath: 文件路径
        """
        if not filepath.exists():
            logger.error(f"文件不存在: {filepath}")
            return

        self.current_file = filepath
        self.file_path_label.configure(text=f"文件: {filepath.name}")
        self._update_file_info()

        # 回调
        if self.on_file_selected:
            self.on_file_selected(filepath)

        logger.info(f"选择文件: {filepath}")

    def _update_file_info(self):
        """更新文件信息显示"""
        if not self.current_file or not self.current_file.exists():
            return

        try:
            stat = self.current_file.stat()
            size = stat.st_size
            mtime = stat.st_mtime

            # 大小
            size_mb = size / (1024 * 1024)
            self.size_label.configure(text=f"大小: {size_mb:.2f} MB")

            # Token 数量
            try:
                content = self.current_file.read_text(encoding="utf-8")
                token_count = count_tokens(content)
                self.tokens_label.configure(text=f"Tokens: {token_count:,}")

                # 预估块数
                proc_config = get_processing_config()
                chunk_count = (token_count // proc_config.target_tokens_per_chunk) + 1
                self.chunks_label.configure(text=f"预估块数: {chunk_count}")
            except Exception as e:
                logger.error(f"计算 token 数量失败: {e}")
                self.tokens_label.configure(text="Tokens: --")
                self.chunks_label.configure(text="预估块数: --")

            # 修改时间
            from datetime import datetime

            mtime_dt = datetime.fromtimestamp(mtime)
            mtime_str = mtime_dt.strftime("%Y-%m-%d %H:%M")
            self.mtime_label.configure(text=mtime_str)

        except Exception as e:
            logger.error(f"更新文件信息失败: {e}")

    def get_file(self) -> Path | None:
        """获取当前选择的文件"""
        return self.current_file

    def clear(self):
        """清空选择"""
        self.current_file = None
        self.file_path_label.configure(text="未选择文件")
        self.size_label.configure(text="大小: --")
        self.tokens_label.configure(text="Tokens: --")
        self.chunks_label.configure(text="预估块数: --")
        self.mtime_label.configure(text="")
