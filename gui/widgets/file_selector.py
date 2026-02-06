"""文件选择组件。"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import customtkinter as ctk

from config import get_processing_config
from tokenizer import count_tokens

logger = logging.getLogger(__name__)


class FileSelector(ctk.CTkFrame):
    """提供文件选择与基础信息展示。"""

    def __init__(
        self,
        master: Any,
        title: str = "文件选择",
        file_types: list[tuple[str, str]] | None = None,
        on_file_selected: Callable[[Path], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.title = title
        self.file_types = file_types or [
            ("文本文件", "*.txt"),
            ("Markdown 文件", "*.md"),
            ("所有文件", "*.*"),
        ]
        if not self.file_types:
            self.file_types = [
                ("文本文件", "*.txt"),
                ("Markdown 文件", "*.md"),
                ("所有文件", "*.*"),
            ]
        self.on_file_selected = on_file_selected
        self.current_file: Path | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        title_label = ctk.CTkLabel(self, text=self.title, font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(10, 5))

        self.file_path_label = ctk.CTkLabel(self, text="未选择文件", text_color="gray")
        self.file_path_label.pack(pady=5)

        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(pady=5, fill="x")

        self.size_label = ctk.CTkLabel(info_frame, text="大小: --", width=200)
        self.size_label.pack(side="left", padx=5)

        self.tokens_label = ctk.CTkLabel(info_frame, text="Tokens: --", width=200)
        self.tokens_label.pack(side="left", padx=5)

        self.chunks_label = ctk.CTkLabel(info_frame, text="预估块数: --", width=150)
        self.chunks_label.pack(side="left", padx=5)

        self.mtime_label = ctk.CTkLabel(info_frame, text="", text_color="gray", width=200)
        self.mtime_label.pack(side="left", padx=5)

        self.select_button = ctk.CTkButton(
            self, text="选择文件", command=self._on_select_file, width=120
        )
        self.select_button.pack(pady=(10, 10))

    def _on_select_file(self) -> None:
        from tkinter import filedialog

        filepath = filedialog.askopenfilename(filetypes=self.file_types, title="选择要处理的文件")
        if filepath:
            self.set_file(Path(filepath))

    def set_file(self, filepath: Path) -> None:
        """设置当前文件并更新展示。"""
        if not filepath.exists():
            logger.error(f"文件不存在: {filepath}")
            return

        self.current_file = filepath
        self.file_path_label.configure(text=f"文件: {filepath.name}")
        self._update_file_info()

        if self.on_file_selected:
            self.on_file_selected(filepath)

    def _update_file_info(self) -> None:
        if not self.current_file or not self.current_file.exists():
            return

        stat = self.current_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        self.size_label.configure(text=f"大小: {size_mb:.2f} MB")

        try:
            content = self.current_file.read_text(encoding="utf-8")
            token_count = count_tokens(content)
            self.tokens_label.configure(text=f"Tokens: {token_count:,}")
            chunk_target = max(get_processing_config().target_tokens_per_chunk, 1)
            chunk_count = max(1, (token_count + chunk_target - 1) // chunk_target)
            self.chunks_label.configure(text=f"预估块数: {chunk_count}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"文件统计失败: {exc}")
            self.tokens_label.configure(text="Tokens: --")
            self.chunks_label.configure(text="预估块数: --")

        from datetime import datetime

        mtime_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        self.mtime_label.configure(text=mtime_str)

    def get_file(self) -> Path | None:
        """获取当前已选择文件。"""
        return self.current_file

    def clear(self) -> None:
        """清空当前选择。"""
        self.current_file = None
        self.file_path_label.configure(text="未选择文件")
        self.size_label.configure(text="大小: --")
        self.tokens_label.configure(text="Tokens: --")
        self.chunks_label.configure(text="预估块数: --")
        self.mtime_label.configure(text="")
