"""
日志页面

全宽表格视图，支持级别过滤和搜索。
"""

import logging
from pathlib import Path

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class LogPage(ctk.CTkFrame):
    """
    日志页面

    全宽表格视图，展示系统日志。
    """

    # 日志级别颜色
    LEVEL_COLORS = {
        "DEBUG": get_color("fg_tertiary", mode="auto"),
        "INFO": get_color("info", mode="auto"),
        "WARNING": get_color("warning", mode="auto"),
        "ERROR": get_color("error", mode="auto"),
    }

    def __init__(self, master, log_file: Path | None = None, **kwargs):
        super().__init__(master, **kwargs)

        self._log_file = log_file
        self._current_level = "ALL"
        self._search_keyword = ""

        self._setup_ui()

        # 加载日志
        if self._log_file and self._log_file.exists():
            self._load_logs()

    def _setup_ui(self):
        """设置 UI"""
        # 主容器
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=40, pady=40)

        # 工具栏
        toolbar = ctk.CTkFrame(main_container, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, SPACING["lg"]))

        # 标题
        title_label = ctk.CTkLabel(
            toolbar,
            text="系统日志",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        title_label.pack(side="left")

        # 右侧工具
        tools_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        tools_frame.pack(side="right")

        # 搜索框
        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", self._on_search_change)

        search_entry = ctk.CTkEntry(
            tools_frame,
            textvariable=self._search_var,
            placeholder_text="搜索日志...",
            width=200,
        )
        search_entry.pack(side="left", padx=(0, SPACING["sm"]))

        # 级别过滤
        self._level_var = ctk.StringVar(value="ALL")

        level_menu = ctk.CTkOptionMenu(
            tools_frame,
            values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            variable=self._level_var,
            command=self._on_level_change,
            width=100,
        )
        level_menu.pack(side="left", padx=(0, SPACING["sm"]))

        # 刷新按钮
        from gui.components.button import Button, ButtonSize, ButtonVariant

        refresh_button = Button(
            tools_frame,
            text="刷新",
            variant=ButtonVariant.SECONDARY,
            size=ButtonSize.SM,
            command=self._refresh_logs,
            width=80,
        )
        refresh_button.pack(side="left", padx=(0, SPACING["sm"]))

        # 清空按钮
        clear_button = Button(
            tools_frame,
            text="清空",
            variant=ButtonVariant.TERTIARY,
            size=ButtonSize.SM,
            command=self._clear_logs,
            width=80,
        )
        clear_button.pack(side="left")

        # 日志内容区
        content_frame = ctk.CTkFrame(
            main_container, fg_color=get_color("bg_secondary", mode="auto"), corner_radius=8
        )
        content_frame.pack(fill="both", expand=True)

        # 表头
        header = ctk.CTkFrame(
            content_frame, fg_color=get_color("bg_tertiary", mode="auto"), height=40
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        for text, width in [("时间", 160), ("级别", 80), ("消息", 0)]:
            label = ctk.CTkLabel(
                header,
                text=text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=get_color("fg_secondary", mode="auto"),
                anchor="w",
            )
            label.pack(side="left", padx=SPACING["sm"], pady=SPACING["xs"])
            if width > 0:
                label.pack(side="left", width=width, padx=SPACING["sm"], pady=SPACING["xs"])

        # 日志显示（使用 Textbox）
        self._log_text = ctk.CTkTextbox(
            content_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=get_color("bg_primary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        self._log_text.pack(fill="both", expand=True, padx=SPACING["sm"], pady=(0, SPACING["sm"]))

        # 配置标签样式
        self._log_text.tag_config("DEBUG", foreground=self.LEVEL_COLORS["DEBUG"])
        self._log_text.tag_config("INFO", foreground=self.LEVEL_COLORS["INFO"])
        self._log_text.tag_config("WARNING", foreground=self.LEVEL_COLORS["WARNING"])
        self._log_text.tag_config("ERROR", foreground=self.LEVEL_COLORS["ERROR"])

    def _load_logs(self):
        """加载日志文件"""
        if not self._log_file or not self._log_file.exists():
            self._log_text.insert("end", "未指定日志文件或文件不存在\n")
            return

        try:
            with open(self._log_file, encoding="utf-8") as f:
                lines = f.readlines()

            # 应用过滤
            filtered = self._filter_logs(lines)

            # 限制行数
            if len(filtered) > 1000:
                filtered = filtered[-1000:]

            # 显示
            self._log_text.delete("1.0", "end")

            for line in filtered:
                self._insert_log_line(line.strip())

        except Exception as e:
            logger.error(f"Failed to load logs: {e}")
            self._log_text.delete("1.0", "end")
            self._log_text.insert("end", f"读取日志失败: {e}\n")

    def _filter_logs(self, lines: list[str]) -> list[str]:
        """过滤日志"""
        filtered = []

        for line in lines:
            # 级别过滤
            if self._current_level != "ALL":
                if f" - {self._current_level} - " not in line:
                    # 保留无级别信息的行
                    if any(f" - {lvl} - " in line for lvl in ["DEBUG", "INFO", "WARNING", "ERROR"]):
                        continue

            # 搜索过滤
            if self._search_keyword and self._search_keyword.lower() not in line.lower():
                continue

            filtered.append(line)

        return filtered

    def _insert_log_line(self, line: str):
        """插入单行日志（带格式）"""
        # 解析日志级别
        level = "INFO"  # 默认
        for lvl in ["ERROR", "WARNING", "DEBUG", "INFO"]:
            if f" - {lvl} - " in line:
                level = lvl
                break

        # 解析时间（假设格式: 2025-01-15 10:30:45）
        parts = line.split(" - ", 2)
        if len(parts) >= 3:
            timestamp = parts[0]
            msg_level = parts[1]
            message = parts[2]

            # 格式化输出
            self._log_text.insert("end", f"{timestamp:<20} ", "")
            self._log_text.insert("end", f"{msg_level:<8} ", level)
            self._log_text.insert("end", f"{message}\n", level)
        else:
            # 无法解析，直接输出
            self._log_text.insert("end", f"{line}\n", level)

    def _on_level_change(self, choice: str):
        """级别变更"""
        self._current_level = choice
        self._load_logs()

    def _on_search_change(self, *args):
        """搜索变更"""
        self._search_keyword = self._search_var.get()
        self._load_logs()

    def _refresh_logs(self):
        """刷新日志"""
        self._load_logs()

    def _clear_logs(self):
        """清空日志文件"""
        from tkinter import messagebox

        if not self._log_file:
            messagebox.showwarning("警告", "未指定日志文件")
            return

        if messagebox.askyesno("确认", "确定要清空日志文件吗？"):
            try:
                self._log_file.write_text("", encoding="utf-8")
                self._log_text.delete("1.0", "end")
                logger.info("Logs cleared")
            except Exception as e:
                logger.error(f"Failed to clear logs: {e}")
                messagebox.showerror("错误", f"清空日志失败: {e}")

    def set_log_file(self, log_file: Path):
        """设置日志文件"""
        self._log_file = log_file
        self._load_logs()
