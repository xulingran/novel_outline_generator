"""
主窗口模块（新版本）

侧边导航 + 主内容区布局，支持页面切换动画。
"""

import asyncio
import logging
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

    WINDOW_MIN_SIZE = (1000, 700)
    WINDOW_INITIAL_SIZE = (1200, 800)

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
        """显示指定页面"""
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

        # TODO: 添加页面切换动画
        logger.debug(f"Showing page: {nav_item.value}")

    def _on_navigation(self, nav_item: NavItem):
        """导航回调"""
        self._show_page(nav_item)

    def _on_theme_changed(self, theme: str):
        """主题变化回调"""
        logger.info(f"Theme changed to: {theme}")

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

        completed = progress_data.get("completed_chunks", progress_data.get("completed", 0))
        total = progress_data.get("total_chunks", progress_data.get("total", 0))
        failed = progress_data.get("failed_chunks", progress_data.get("failed", 0))
        partial = progress_data.get("partial_chunks", progress_data.get("partial", 0))
        phase = progress_data.get("phase", "")
        eta_seconds = progress_data.get("eta_seconds", 0)

        process_page.update_progress(
            completed=completed,
            total=total,
            failed=failed,
            partial=partial,
            phase=phase,
            eta_seconds=eta_seconds,
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

        process_page = self.get_process_page()
        if process_page is None:
            return

        self._async_worker = None
        self._cancel_event = None
        cancelled = self._cancel_requested or isinstance(error, asyncio.CancelledError)
        self._cancel_requested = False

        if getattr(process_page, "_current_file", None) is not None:
            process_page._start_button.configure(state="normal")
        process_page._cancel_button.configure(state="disabled")

        if cancelled:
            process_page.append_log("处理已取消")
            return

        if error is not None:
            logger.error(f"处理失败: {error}")
            process_page.append_log(f"处理失败: {error}")
            messagebox.showerror("错误", f"处理失败: {error}")
            return

        output_dir = result.get("output_dir", "") if result else ""
        process_page.append_log("处理完成")
        if output_dir:
            messagebox.showinfo("完成", f"大纲生成完成\n输出目录: {output_dir}")
        else:
            messagebox.showinfo("完成", "大纲生成完成")

    def _on_cancel_processing(self):
        """取消处理回调"""
        process_page = self.get_process_page()
        if process_page is None:
            return

        self._cancel_requested = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        process_page.append_log("正在取消处理任务...")
        process_page._cancel_button.configure(state="disabled")

    def run(self):
        """启动应用"""
        logger.info("Starting main window")
        self.mainloop()


class GUILogHandler(logging.Handler):
    """
    GUI 日志处理器

    将日志消息发送到 GUI 的处理页面。
    """

    def __init__(self, process_page):
        super().__init__()
        self._process_page = process_page

    def emit(self, record):
        """发送日志记录"""
        try:
            msg = self.format(record)
            # 在主线程中安全地更新 GUI
            self._process_page.after(0, lambda: self._process_page.append_log(msg))
        except Exception:
            self.handleError(record)
