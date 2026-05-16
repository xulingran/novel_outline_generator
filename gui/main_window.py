"""
主窗口模块（新版本）

侧边导航 + 主内容区布局，支持页面切换动画。
"""

import asyncio
import logging
import weakref
from pathlib import Path

import customtkinter as ctk

from gui.components.sidebar import NavItem, Sidebar
from gui.theme_manager import get_color, get_theme_manager

logger = logging.getLogger(__name__)


class MainWindow(ctk.CTk):
    """
    主窗口

    侧边导航 + 主内容区布局，支持：
    - 处理页面：文件选择、进度可视化、实时日志
    - 配置页面：API 和处理参数配置
    - 日志页面：系统日志查看
    - 关于页面：应用信息
    """

    WINDOW_MIN_SIZE = (800, 600)
    WINDOW_INITIAL_SIZE = (1200, 800)
    COMPACT_MODE_THRESHOLD = 900

    def __init__(self):
        super().__init__()
        self._async_worker = None
        self._cancel_event = None
        self._cancel_requested = False

        self.title("Novel Outline Generator")
        self.geometry(f"{self.WINDOW_INITIAL_SIZE[0]}x{self.WINDOW_INITIAL_SIZE[1]}")
        self.minsize(*self.WINDOW_MIN_SIZE)

        # 获取主题管理器
        self._theme_manager = get_theme_manager()
        self._theme_manager.apply_theme()
        self.configure(fg_color=get_color("bg_primary", mode="auto"))

        # 订阅主题变化
        self._theme_manager.on_theme_change(self._on_theme_changed)

        self._setup_ui()
        self._setup_logging()
        self._compact_mode = False
        self.bind("<Configure>", self._on_resize)

        logger.info("Main window initialized")

    def _setup_ui(self):
        """设置 UI"""
        # 配置网格布局
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 侧边导航栏
        self._sidebar = Sidebar(
            self,
            active_item=NavItem.PROCESS,
            on_navigation=self._on_navigation,
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")

        # 主内容区
        self._content_frame = ctk.CTkFrame(self, fg_color=get_color("bg_primary", mode="auto"))
        self._content_frame.grid(row=0, column=1, sticky="nsew")

        # 页面容器
        self._pages = {}
        self._current_page = None

        # 加载所有页面
        self._load_pages()

        # 显示默认页面
        self._show_page(NavItem.PROCESS)

    def _load_pages(self):
        """加载所有页面"""
        from gui.pages.about_page import AboutPage
        from gui.pages.config_page import ConfigPage
        from gui.pages.log_page import LogPage
        from gui.pages.process_page import ProcessPage

        # 处理页面
        process_page = ProcessPage(self._content_frame)
        process_page.set_callbacks(
            on_start=self._on_start_processing,
            on_cancel=self._on_cancel_processing,
        )
        self._pages[NavItem.PROCESS] = process_page

        # 配置页面
        self._pages[NavItem.CONFIG] = ConfigPage(self._content_frame)

        # 日志页面
        log_file = Path("logs/app.log")
        self._pages[NavItem.LOG] = LogPage(self._content_frame, log_file=log_file)

        # 关于页面
        self._pages[NavItem.ABOUT] = AboutPage(self._content_frame)

    def _show_page(self, nav_item: NavItem):
        """显示指定页面（带淡入动画效果）"""
        # 隐藏当前页面
        if self._current_page:
            self._pages[self._current_page].pack_forget()

        # 显示新页面
        self._current_page = nav_item
        page = self._pages[nav_item]
        page.pack(fill="both", expand=True)
        if hasattr(page, "refresh_layout"):
            try:
                page.refresh_layout()
            except Exception:
                logger.debug("Failed to refresh page layout", exc_info=True)

        # 更新侧边栏激活状态
        self._sidebar.set_active_item(nav_item)

        # 页面切换淡入动画
        self._animate_page_transition(page)

        logger.debug(f"Showing page: {nav_item.value}")

    def _animate_page_transition(self, page, duration_ms: int = 150) -> None:
        """执行页面切换淡入动画

        Args:
            page: 要动画显示的页面
            duration_ms: 动画持续时间（毫秒）
        """
        try:
            # 保存原始透明度
            original_alpha = page.cget("fg_color")

            # 设置初始透明度（通过降低亮度模拟）
            page.configure(fg_color=get_color("bg_secondary", mode="auto"))

            # 渐变动画
            def fade_in(step: int = 0, steps: int = 5) -> None:
                if step >= steps:
                    # 恢复原始颜色
                    page.configure(fg_color=original_alpha)
                    return

                # 计算中间颜色（简化处理：直接使用背景色）
                progress = step / steps
                if progress > 0.5:
                    page.configure(fg_color=get_color("bg_primary", mode="auto"))

                # 调度下一帧
                delay = duration_ms // steps
                self.after(delay, lambda: fade_in(step + 1, steps))

            # 启动动画
            fade_in()

        except Exception:
            # 动画失败不影响功能
            logger.debug("Page transition animation failed", exc_info=True)

    def _on_navigation(self, nav_item: NavItem):
        """导航回调"""
        self._show_page(nav_item)

    def _on_theme_changed(self, theme: str):
        """主题变化回调"""
        self.configure(fg_color=get_color("bg_primary", mode="auto"))
        self._content_frame.configure(fg_color=get_color("bg_primary", mode="auto"))
        if hasattr(self, "_sidebar"):
            self._sidebar.refresh_theme()

        for page in self._pages.values():
            try:
                page.configure(fg_color=get_color("bg_primary", mode="auto"))
            except Exception:
                logger.debug("Failed to refresh page fg color", exc_info=True)

            if hasattr(page, "refresh_layout"):
                try:
                    page.refresh_layout()
                except Exception:
                    logger.debug("Failed to refresh page layout after theme change", exc_info=True)

        self.update_idletasks()
        logger.info(f"Theme changed to: {theme}")

    def _on_resize(self, event):
        """窗口大小变化回调"""
        if event.widget != self:
            return
        width = self._normalize_width(event.width)
        should_be_compact = width < self.COMPACT_MODE_THRESHOLD

        if should_be_compact != self._compact_mode:
            self._compact_mode = should_be_compact
            self._apply_compact_mode(should_be_compact)

    def _normalize_width(self, width: int) -> int:
        """将窗口物理像素宽度转换为逻辑宽度，适配高 DPI 缩放"""
        try:
            scaling = ctk.ScalingTracker.get_window_scaling(self)
        except Exception:
            scaling = 1.0

        if scaling <= 0:
            return width

        return int(width / scaling)

    def _apply_compact_mode(self, compact: bool):
        """应用紧凑模式"""
        if hasattr(self._sidebar, "toggle_collapse"):
            if compact and not self._sidebar._collapsed:
                self._sidebar.toggle_collapse()
            elif not compact and self._sidebar._collapsed:
                self._sidebar.toggle_collapse()

    def _setup_logging(self):
        """设置日志捕获"""
        # 创建日志处理器，将日志发送到处理页面
        if NavItem.PROCESS in self._pages:
            handler = GUILogHandler(self._pages[NavItem.PROCESS])
            logging.getLogger().addHandler(handler)

    def get_process_page(self):
        """获取处理页面"""
        return self._pages.get(NavItem.PROCESS)

    def _on_start_processing(self):
        """开始处理回调"""
        process_page = self.get_process_page()
        if process_page is None:
            return

        file_path = getattr(process_page, "_current_file", None)
        if file_path is None:
            process_page.append_log("未选择文件，无法开始处理")
            return

        if self._async_worker and self._async_worker.is_alive():
            process_page.append_log("已有任务在运行，请稍候")
            return

        self._cancel_requested = False
        process_page.append_log(f"开始处理文件: {file_path}")

        from gui.async_worker import AsyncWorker

        self._async_worker = AsyncWorker(
            coro=self._build_processing_coro(str(file_path)),
            progress_callback=self._on_progress_update,
            completion_callback=self._on_processing_complete,
            error_callback=self._on_processing_error,
        )
        self._async_worker.start()

    def _build_processing_coro(self, file_path: str):
        """构建异步处理协程"""
        import asyncio

        async def _process():
            from services.novel_processing_service import NovelProcessingService

            self._cancel_event = asyncio.Event()
            if self._cancel_requested:
                self._cancel_event.set()
            service = NovelProcessingService(
                progress_callback=self._on_progress_update,
                cancel_event=self._cancel_event,
            )
            return await service.process_novel(file_path=file_path, resume=False)

        return _process()

    def _on_progress_update(self, progress_data: dict):
        """进度回调（来自工作线程）"""
        self.after(0, lambda: self._apply_progress_update(progress_data))

    def _apply_progress_update(self, progress_data: dict):
        """在主线程更新进度 UI"""
        process_page = self.get_process_page()
        if process_page is None:
            return

        # 使用服务层统一的标准键名
        completed = progress_data.get("completed_chunks", 0)
        total = progress_data.get("total_chunks", 0)
        failed = progress_data.get("failed_chunks", 0)
        partial = progress_data.get("partial_chunks", 0)
        phase = progress_data.get("phase", "")
        eta_seconds = progress_data.get("eta_seconds", 0)

        # 合并相关参数
        merge_level = progress_data.get("merge_level", 0)
        merge_batch_current = progress_data.get("merge_batch_current", 0)
        merge_batch_total = progress_data.get("merge_batch_total", 0)
        merge_outlines_count = progress_data.get("merge_outlines_count", 0)

        process_page.update_progress(
            completed=completed,
            total=total,
            failed=failed,
            partial=partial,
            phase=phase,
            eta_seconds=eta_seconds,
            merge_level=merge_level,
            merge_batch_current=merge_batch_current,
            merge_batch_total=merge_batch_total,
            merge_outlines_count=merge_outlines_count,
        )

    def _on_processing_complete(self, result: dict):
        """处理完成回调（来自工作线程）"""
        self.after(0, lambda: self._finish_processing(result=result, error=None))

    def _on_processing_error(self, error: Exception):
        """处理失败回调（来自工作线程）"""
        self.after(0, lambda: self._finish_processing(result=None, error=error))

    def _finish_processing(self, result: dict | None, error: Exception | None):
        """收尾并恢复按钮状态"""
        from tkinter import messagebox

        from gui.utils import format_error_dialog

        process_page = self.get_process_page()
        if process_page is None:
            return

        self._async_worker = None
        self._cancel_event = None
        cancelled = self._cancel_requested or isinstance(error, asyncio.CancelledError)
        self._cancel_requested = False

        if cancelled:
            process_page.append_log("处理已取消")
            process_page.set_final_status(success=False, cancelled=True)
            cancelled_error = asyncio.CancelledError("处理任务已被用户取消")
            title, detail = format_error_dialog(cancelled_error)
            messagebox.showinfo(title, detail)
            return

        if error is not None:
            logger.error(f"处理失败: {error}")
            process_page.append_log(f"处理失败: {error}")
            process_page.set_final_status(success=False, cancelled=False)
            title, detail = format_error_dialog(error)
            messagebox.showerror(title, detail)
            return

        output_dir = result.get("output_dir", "") if result else ""
        process_page.append_log("处理完成")
        process_page.set_final_status(success=True, cancelled=False)
        if output_dir:
            messagebox.showinfo("完成", f"大纲生成完成\n输出目录: {output_dir}")
        else:
            messagebox.showinfo("完成", "大纲生成完成")

    def _on_cancel_processing(self):
        """取消处理回调"""
        from tkinter import messagebox

        process_page = self.get_process_page()
        if process_page is None:
            return

        if not messagebox.askyesno(
            "确认取消",
            "确定要取消当前处理任务吗？\n\n已完成的进度将被保存，可稍后继续处理。",
        ):
            return

        self._cancel_requested = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        process_page.append_log("正在取消处理任务...")
        process_page._cancel_button.configure(state="disabled", text="取消中...")

    def run(self):
        """启动应用"""
        logger.info("Starting main window")
        self.mainloop()


class GUILogHandler(logging.Handler):
    """
    GUI 日志处理器

    将日志消息发送到 GUI 的处理页面。

    使用弱引用持有 process_page，避免内存泄漏。
    当页面被销毁时，弱引用会自动变为 None。
    """

    def __init__(self, process_page):
        super().__init__()
        # 使用弱引用持有 process_page，避免内存泄漏
        self._process_page_ref = weakref.ref(process_page)

    def emit(self, record):
        """发送日志记录"""
        try:
            # 获取弱引用指向的对象，如果对象已被销毁则返回 None
            process_page = self._process_page_ref()
            if process_page is None:
                # 页面已被销毁，移除此 handler
                logging.getLogger().removeHandler(self)
                return

            msg = self.format(record)
            # 在主线程中安全地更新 GUI
            process_page.after(0, lambda: process_page.append_log(msg))
        except Exception:
            self.handleError(record)
