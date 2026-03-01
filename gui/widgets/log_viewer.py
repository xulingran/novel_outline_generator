"""
日志查看器组件

实时读取和显示日志文件，支持自动滚动、日志级别过滤、清空等功能。
"""

import logging
from pathlib import Path

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class LogViewer(ctk.CTkFrame):
    """
    日志查看器组件

    功能：
    - 实时读取日志文件
    - 自动滚动到最新日志
    - 支持日志级别过滤
    - 支持清空日志
    """

    def __init__(
        self,
        master,
        log_file: Path | None = None,
        auto_refresh: bool = True,
        refresh_interval: int = 1000,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.log_file = log_file
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self._refresh_job = None

        # 日志级别过滤
        self.log_levels = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"]
        self.current_level = "ALL"

        self._setup_ui()

        # 初始加载
        if self.log_file and self.log_file.exists():
            self.refresh_log()

        # 启动自动刷新
        if self.auto_refresh:
            self._start_auto_refresh()

    def _setup_ui(self):
        """设置 UI"""
        # 工具栏
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", pady=(SPACING["md"], SPACING["sm"]))

        # 标题
        title_label = ctk.CTkLabel(
            toolbar,
            text="系统日志",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        title_label.pack(side="left", padx=SPACING["md"])

        # 日志级别选择
        level_label = ctk.CTkLabel(
            toolbar,
            text="级别:",
            text_color=get_color("fg_secondary", mode="auto"),
        )
        level_label.pack(side="left", padx=(SPACING["lg"], SPACING["sm"]))

        self.level_menu = ctk.CTkOptionMenu(
            toolbar,
            values=self.log_levels,
            command=self._on_level_change,
            fg_color=get_color("bg_secondary", mode="auto"),
            button_color=get_color("accent", mode="auto"),
            button_hover_color=get_color("accent_secondary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        self.level_menu.set("ALL")
        self.level_menu.pack(side="left", padx=SPACING["sm"])

        # 刷新按钮
        refresh_button = ctk.CTkButton(
            toolbar,
            text="刷新",
            command=self.refresh_log,
            width=80,
            fg_color=get_color("accent", mode="auto"),
            hover_color=get_color("accent_secondary", mode="auto"),
            text_color=get_color("bg_primary", mode="auto"),
        )
        refresh_button.pack(side="right", padx=SPACING["sm"])

        # 清空按钮
        clear_button = ctk.CTkButton(
            toolbar,
            text="清空",
            command=self.clear_log,
            width=80,
            fg_color=get_color("bg_secondary", mode="auto"),
            hover_color=get_color("bg_tertiary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        clear_button.pack(side="right", padx=SPACING["sm"])

        # 自动滚动开关
        self.auto_scroll_var = ctk.BooleanVar(value=True)
        auto_scroll_checkbox = ctk.CTkCheckBox(
            toolbar,
            text="自动滚动",
            variable=self.auto_scroll_var,
            text_color=get_color("fg_secondary", mode="auto"),
            fg_color=get_color("accent", mode="auto"),
            hover_color=get_color("accent_secondary", mode="auto"),
        )
        auto_scroll_checkbox.pack(side="right", padx=SPACING["md"])

        # 日志显示区域
        self.log_text = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=get_color("bg_primary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
            border_color=get_color("border", mode="auto"),
        )
        self.log_text.pack(
            fill="both", expand=True, padx=SPACING["md"], pady=(SPACING["sm"], SPACING["md"])
        )

    def _start_auto_refresh(self):
        """启动自动刷新"""
        if self._refresh_job is None:
            self._schedule_refresh()

    def _schedule_refresh(self):
        """调度下次刷新"""
        if self.auto_refresh:
            self._refresh_job = self.after(self.refresh_interval, self._auto_refresh)

    def _auto_refresh(self):
        """自动刷新回调"""
        if self.log_file and self.log_file.exists():
            # 刷新日志
            self.refresh_log()

            # 如果启用了自动滚动，滚动到底部
            if self.auto_scroll_var.get():
                self.log_text.see("end")

        # 继续调度
        self._schedule_refresh()

    def _on_level_change(self, choice: str):
        """日志级别改变事件"""
        self.current_level = choice
        self.refresh_log()

    def set_log_file(self, log_file: Path):
        """
        设置日志文件

        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        self.refresh_log()

    def refresh_log(self):
        """刷新日志显示"""
        if not self.log_file:
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", "未指定日志文件")
            return

        if not self.log_file.exists():
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", f"日志文件不存在: {self.log_file}")
            return

        try:
            # 读取日志文件
            with open(self.log_file, encoding="utf-8") as f:
                lines = f.readlines()

            # 应用级别过滤
            if self.current_level != "ALL":
                lines = self._filter_by_level(lines, self.current_level)

            # 限制显示行数（最多 1000 行）
            lines = lines[-1000:] if len(lines) > 1000 else lines

            # 更新显示
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", "".join(lines))

            # 如果启用了自动滚动，滚动到底部
            if self.auto_scroll_var.get():
                self.log_text.see("end")

        except Exception as e:
            logger.error(f"读取日志文件失败: {e}")
            self.log_text.delete("0.0", "end")
            self.log_text.insert("0.0", f"读取日志失败: {e}")

    def _filter_by_level(self, lines: list[str], level: str) -> list[str]:
        """
        按日志级别过滤

        Args:
            lines: 日志行列表
            level: 日志级别

        Returns:
            过滤后的日志行列表
        """
        level_priorities = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        target_priority = level_priorities.get(level, 0)

        filtered = []
        for line in lines:
            # 检查是否包含日志级别
            has_level = False
            line_priority = 0

            for lvl, priority in level_priorities.items():
                if f" - {lvl} - " in line:
                    has_level = True
                    line_priority = priority
                    break

            if has_level and line_priority >= target_priority:
                filtered.append(line)
            elif not has_level:
                # 没有级别信息的行保留
                filtered.append(line)

        return filtered

    def clear_log(self):
        """清空日志文件"""
        from tkinter import messagebox

        if not self.log_file:
            messagebox.showwarning("警告", "未指定日志文件")
            return

        if messagebox.askyesno("确认", "确定要清空日志文件吗？"):
            try:
                if self.log_file.exists():
                    self.log_file.write_text("", encoding="utf-8")
                    self.log_text.delete("0.0", "end")
                    logger.info("日志已清空")
            except Exception as e:
                logger.error(f"清空日志失败: {e}")
                messagebox.showerror("错误", f"清空日志失败: {e}")

    def stop_auto_refresh(self):
        """停止自动刷新"""
        self.auto_refresh = False
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
            self._refresh_job = None

    def start_auto_refresh(self):
        """启动自动刷新"""
        self.auto_refresh = True
        if self._refresh_job is None:
            self._start_auto_refresh()

    def append_log(self, message: str):
        """
        追加日志消息

        Args:
            message: 日志消息
        """
        self.log_text.insert("end", message)
        if self.auto_scroll_var.get():
            self.log_text.see("end")

    def get_text(self) -> str:
        """获取当前显示的日志文本"""
        return str(self.log_text.get("0.0", "end"))

    def search(self, keyword: str) -> int:
        """
        搜索日志内容并高亮显示

        Args:
            keyword: 搜索关键字

        Returns:
            匹配数量
        """
        # 清除之前的高亮
        self._clear_highlight()

        if not keyword:
            return 0

        # 优先使用底层 Tk Text 的 search（CTkTextbox 本身不一定暴露 search）
        count = self._count_matches(keyword)

        if count > 0:
            self._highlight_text(keyword)

        return count

    def _resolve_text_widget(self):
        """获取可用于 search/tag_* 的文本组件。"""
        backend = getattr(self.log_text, "_textbox", None)
        if backend is not None:
            return backend
        return self.log_text

    def _count_matches(self, keyword: str) -> int:
        """统计关键字匹配数量。"""
        text_widget = self._resolve_text_widget()

        if hasattr(text_widget, "search"):
            count = 0
            start = "1.0"
            while True:
                pos = text_widget.search(keyword, start, stopindex="end")
                if not pos:
                    break
                count += 1
                start = f"{pos}+{len(keyword)}c"
            return count

        # 降级方案：无法使用 Tk search 时，退回纯文本统计
        return self.get_text().count(keyword)

    def _highlight_text(self, keyword: str) -> None:
        """高亮显示指定文本

        Args:
            keyword: 要高亮的关键字
        """
        text_widget = self._resolve_text_widget()

        if not hasattr(text_widget, "search") or not hasattr(text_widget, "tag_config"):
            return

        # 配置高亮标签样式
        try:
            text_widget.tag_config(
                "search_highlight",
                background=get_color("accent", mode="auto"),
                foreground=get_color("bg_primary", mode="auto"),
            )
        except Exception:
            return

        # 查找并标记所有匹配项
        start = "1.0"
        while True:
            pos = text_widget.search(keyword, start, stopindex="end")
            if not pos:
                break
            end_pos = f"{pos}+{len(keyword)}c"
            text_widget.tag_add("search_highlight", pos, end_pos)
            start = end_pos

    def _clear_highlight(self) -> None:
        """清除所有高亮标记"""
        text_widget = self._resolve_text_widget()
        if hasattr(text_widget, "tag_remove"):
            try:
                text_widget.tag_remove("search_highlight", "1.0", "end")
            except Exception:
                pass
