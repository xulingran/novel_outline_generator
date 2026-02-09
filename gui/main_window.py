"""
主窗口模块（新版本）

侧边导航 + 主内容区布局，支持页面切换动画。
"""

import logging
from pathlib import Path

import customtkinter as ctk

from gui.components.sidebar import NavItem, Sidebar
from gui.theme_manager import get_theme_manager

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

        self.title("Novel Outline Generator")
        self.geometry(f"{self.WINDOW_INITIAL_SIZE[0]}x{self.WINDOW_INITIAL_SIZE[1]}")
        self.minsize(*self.WINDOW_MIN_SIZE)

        # 获取主题管理器
        self._theme_manager = get_theme_manager()
        self._theme_manager.apply_theme()

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
        self._content_frame = ctk.CTkFrame(self, fg_color="transparent")
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
        self._pages[NavItem.PROCESS] = ProcessPage(self._content_frame)

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
