"""日志查看组件。"""

import logging
from pathlib import Path
from typing import Any

import customtkinter as ctk

logger = logging.getLogger(__name__)


class LogViewer(ctk.CTkFrame):
    """实时显示并过滤日志文件。"""

    def __init__(
        self,
        master: Any,
        log_file: Path | None = None,
        auto_refresh: bool = True,
        refresh_interval: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.log_file = log_file
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self._refresh_job: str | None = None
        self.log_levels = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"]
        self.current_level = "ALL"

        self._setup_ui()
        if self.log_file and self.log_file.exists():
            self.refresh_log()
        if self.auto_refresh:
            self._schedule_refresh()

    def _setup_ui(self) -> None:
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", pady=(10, 5))

        ctk.CTkLabel(toolbar, text="系统日志", font=ctk.CTkFont(size=16, weight="bold")).pack(
            side="left", padx=10
        )

        ctk.CTkLabel(toolbar, text="级别:").pack(side="left", padx=(20, 5))
        self.level_menu = ctk.CTkOptionMenu(
            toolbar,
            values=self.log_levels,
            command=self._on_level_change,
        )
        self.level_menu.set("ALL")
        self.level_menu.pack(side="left", padx=5)

        ctk.CTkButton(toolbar, text="刷新", command=self.refresh_log, width=80).pack(
            side="right", padx=5
        )
        ctk.CTkButton(toolbar, text="清空", command=self.clear_log, width=80).pack(
            side="right", padx=5
        )

        self.auto_scroll_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(toolbar, text="自动滚动", variable=self.auto_scroll_var).pack(
            side="right", padx=10
        )

        self.log_text = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(5, 10))

    def _schedule_refresh(self) -> None:
        if self.auto_refresh:
            self._refresh_job = self.after(self.refresh_interval, self._auto_refresh)

    def _auto_refresh(self) -> None:
        self.refresh_log()
        if self.auto_scroll_var.get():
            self.log_text.see("end")
        self._schedule_refresh()

    def _on_level_change(self, choice: str) -> None:
        self.current_level = choice
        self.refresh_log()

    def set_log_file(self, log_file: Path) -> None:
        """更新日志文件路径。"""
        self.log_file = log_file
        self.refresh_log()

    def refresh_log(self) -> None:
        """读取并刷新日志内容。"""
        if not self.log_file:
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", "未指定日志文件")
            return

        if not self.log_file.exists():
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", f"日志文件不存在: {self.log_file}")
            return

        try:
            lines = self.log_file.read_text(encoding="utf-8").splitlines(keepends=True)
            if self.current_level != "ALL":
                lines = self._filter_by_level(lines, self.current_level)
            lines = lines[-1000:]
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", "".join(lines))
            if self.auto_scroll_var.get():
                self.log_text.see("end")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"读取日志文件失败: {exc}")
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", f"读取日志失败: {exc}")

    def _filter_by_level(self, lines: list[str], level: str) -> list[str]:
        """按级别过滤日志行。"""
        priorities = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        target = priorities.get(level, 0)
        filtered: list[str] = []
        for line in lines:
            found_level = None
            for lv, priority in priorities.items():
                if f" - {lv} - " in line:
                    found_level = priority
                    break
            if found_level is None or found_level >= target:
                filtered.append(line)
        return filtered

    def clear_log(self) -> None:
        """清空日志文件并刷新文本框。"""
        from tkinter import messagebox

        if not self.log_file:
            messagebox.showwarning("警告", "未指定日志文件")
            return

        if not messagebox.askyesno("确认", "确定要清空日志文件吗？"):
            return

        try:
            self.log_file.write_text("", encoding="utf-8")
            self.log_text.delete("0.0", "end")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"清空日志失败: {exc}")
            messagebox.showerror("错误", f"清空日志失败: {exc}")

    def stop_auto_refresh(self) -> None:
        """停止自动刷新。"""
        self.auto_refresh = False
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
            self._refresh_job = None

    def start_auto_refresh(self) -> None:
        """启动自动刷新。"""
        self.auto_refresh = True
        if self._refresh_job is None:
            self._schedule_refresh()

    def append_log(self, message: str) -> None:
        """追加一行日志到显示区。"""
        self.log_text.insert("end", message)
        if self.auto_scroll_var.get():
            self.log_text.see("end")

    def get_text(self) -> str:
        """获取当前显示文本。"""
        return str(self.log_text.get("0.0", "end"))

    def search(self, keyword: str) -> int:
        """返回关键字出现次数。"""
        return self.get_text().count(keyword)
