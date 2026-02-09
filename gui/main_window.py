"""
主窗口模块

基于 CustomTkinter 的桌面应用主窗口，包含处理、配置、日志、关于四个标签页。
"""

import logging
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

import customtkinter as ctk

from config import get_api_config, get_processing_config, init_config
from gui.async_worker import AsyncWorker
from gui.config_dialog import ConfigDialog
from gui.theme_manager import SPACING, ThemeManager, get_color
from gui.widgets.file_selector import FileSelector
from gui.widgets.log_viewer import LogViewer
from gui.widgets.progress_bar import ProgressBar

logger = logging.getLogger(__name__)


class MainWindow(ctk.CTk):
    """
    主窗口类

    提供 tabbed interface，包含处理、配置、日志、关于四个功能标签页。
    """

    def __init__(self):
        super().__init__()

        # 初始化主题管理器
        self.theme_manager = ThemeManager()

        # 窗口基本设置
        self.title("小说大纲生成器 v2.0")
        self.geometry("1000x700")

        # 初始化配置
        init_config()
        self.api_config = get_api_config()
        self.processing_config = get_processing_config()

        # 当前处理的文件
        self.current_file: Path | None = None

        # 异步任务工作线程
        self.async_worker: AsyncWorker | None = None

        # 跨线程 UI 事件队列（仅主线程消费）
        self._ui_event_queue: SimpleQueue[tuple[str, Any]] = SimpleQueue()
        self._ui_poll_interval_ms = 50

        # 应用主题
        self.theme_manager.apply_theme()

        # 添加主题切换UI
        self._setup_theme_switcher()

        # 创建标签页
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill="both", expand=True, padx=SPACING["md"], pady=SPACING["md"])

        self.tab_view.add("处理")
        self.tab_view.add("配置")
        self.tab_view.add("日志")
        self.tab_view.add("关于")

        # 获取标签页引用
        self.tab_process = self.tab_view.tab("处理")
        self.tab_config = self.tab_view.tab("配置")
        self.tab_log = self.tab_view.tab("日志")
        self.tab_about = self.tab_view.tab("关于")

        # 设置各个标签页
        self.setup_process_tab()
        self.setup_config_tab()
        self.setup_log_tab()
        self.setup_about_tab()

        # 启动 UI 事件轮询（必须在主线程）
        self.after(self._ui_poll_interval_ms, self._drain_ui_events)

        logger.info("主窗口初始化完成")

    def setup_process_tab(self):
        """设置处理标签页"""
        # 主容器
        main_frame = ctk.CTkFrame(self.tab_process)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 文件选择器
        self.file_selector = FileSelector(main_frame)
        self.file_selector.pack(fill="x", padx=10, pady=10)

        # 进度条
        self.progress_bar_widget = ProgressBar(main_frame)
        self.progress_bar_widget.pack(fill="x", padx=10, pady=10)

        # 控制按钮区域
        control_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        control_frame.pack(fill="x", padx=10, pady=(10, 10))

        self.start_button = ctk.CTkButton(
            control_frame,
            text="开始处理",
            command=self.on_start_processing,
            width=120,
            state="disabled",
        )
        self.start_button.pack(side="left", padx=10)

        self.cancel_button = ctk.CTkButton(
            control_frame,
            text="取消处理",
            command=self.on_cancel_processing,
            width=120,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=10)

        self.open_output_button = ctk.CTkButton(
            control_frame, text="打开输出目录", command=self.on_open_output_dir, width=150
        )
        self.open_output_button.pack(side="left", padx=10)

        # 取消事件绑定
        self.file_selector.on_file_selected = self.on_file_selected

    def setup_config_tab(self):
        """设置配置标签页"""
        main_frame = ctk.CTkScrollableFrame(self.tab_config)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 标题
        title_label = ctk.CTkLabel(
            main_frame, text="配置管理", font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=10)

        # 说明
        info_label = ctk.CTkLabel(
            main_frame,
            text="点击下方按钮打开配置编辑器，修改 API 密钥、处理参数等配置。\n修改后需要重启应用才能生效。",
            text_color="gray",
        )
        info_label.pack(pady=10)

        # 按钮
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=20)

        config_button = ctk.CTkButton(
            button_frame,
            text="打开配置编辑器",
            command=self.on_open_config_dialog,
            width=180,
            height=40,
        )
        config_button.pack()

        # 当前配置简要显示
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=20, padx=10)

        config_label = ctk.CTkLabel(
            config_frame, text="当前配置", font=ctk.CTkFont(size=16, weight="bold")
        )
        config_label.pack(pady=10)

        # API 提供商
        provider_text = f"API 提供商: {self.api_config.provider}"
        provider_label = ctk.CTkLabel(config_frame, text=provider_text)
        provider_label.pack(pady=5, anchor="w", padx=20)

        # 目标分块大小
        chunk_text = f"目标分块大小: {self.processing_config.target_tokens_per_chunk} tokens"
        chunk_label = ctk.CTkLabel(config_frame, text=chunk_text)
        chunk_label.pack(pady=5, anchor="w", padx=20)

        # 并发限制
        parallel_text = f"并发限制: {self.processing_config.parallel_limit}"
        parallel_label = ctk.CTkLabel(config_frame, text=parallel_text)
        parallel_label.pack(pady=5, anchor="w", padx=20)

    def setup_log_tab(self):
        """设置日志标签页"""
        from config import get_processing_config

        config = get_processing_config()
        log_dir = Path(config.log_dir if hasattr(config, "log_dir") else "logs")
        log_file = log_dir / "novel_outline.log"

        # 使用 LogViewer 组件
        self.log_viewer = LogViewer(self.tab_log, log_file=log_file)
        self.log_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_about_tab(self):
        """设置关于标签页"""
        main_frame = ctk.CTkScrollableFrame(self.tab_about)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 标题
        title_label = ctk.CTkLabel(
            main_frame, text="小说大纲生成器", font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)

        # 版本
        version_label = ctk.CTkLabel(main_frame, text="版本 2.0.0", font=ctk.CTkFont(size=14))
        version_label.pack(pady=5)

        # 描述
        desc_frame = ctk.CTkFrame(main_frame)
        desc_frame.pack(fill="x", pady=20, padx=20)

        desc_text = ctk.CTkTextbox(desc_frame, height=200, width=700, fg_color="transparent")
        desc_text.pack(pady=10, padx=10)
        desc_text.insert(
            "0.0",
            "简介\n"
            "====\n"
            "小说大纲生成器是一款基于 AI 的小说大纲自动生成工具。\n"
            "它能够自动将长篇小说文本分割成合适的块，并行调用 LLM API\n"
            "生成每块的大纲，然后递归合并成完整的小说总纲。\n\n"
            "特性\n"
            "====\n"
            "• 支持多种 LLM 提供商（OpenAI, Gemini, 智谱, AiHubMix）\n"
            "• 智能文本分块，按句子边界切分\n"
            "• 并行处理，提高效率\n"
            "• 递归合并，生成完整大纲\n"
            "• 进度跟踪，实时显示处理状态\n"
            "• 断点恢复，支持从失败处继续\n"
            "• 失败重试，自动处理错误\n\n"
            "技术栈\n"
            "======\n"
            "• Python 3.14+\n"
            "• CustomTkinter (GUI)\n"
            "• asyncio (异步处理)\n"
            "• FastAPI (Web API)\n\n"
            "开源协议\n"
            "========\n"
            "MIT License\n",
        )
        desc_text.configure(state="disabled")

        # 按钮
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=20)

        # TODO: 添加 GitHub 链接按钮等
        info_label = ctk.CTkLabel(
            main_frame, text="© 2025 Novel Outline Generator", text_color="gray"
        )
        info_label.pack(pady=10)

    # ===== 事件处理函数 =====

    def on_file_selected(self, filepath: Path):
        """文件选择回调"""
        self.current_file = filepath
        self.start_button.configure(state="normal")
        logger.info(f"选择文件: {self.current_file}")

    def on_select_file(self):
        """选择文件事件处理（已弃用，使用 FileSelector 组件）"""
        # 此方法保留是为了兼容性，但实际功能已移至 FileSelector 组件
        pass

    def update_file_info(self):
        """更新文件信息显示（已弃用，使用 FileSelector 组件）"""
        # 此方法保留是为了兼容性，但实际功能已移至 FileSelector 组件
        pass

    def on_start_processing(self):
        """开始处理事件"""
        if not self.current_file:
            return
        if self.async_worker and self.async_worker.is_alive():
            logger.warning("已有处理任务正在运行，忽略重复启动")
            return

        logger.info(f"开始处理文件: {self.current_file}")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")

        # 重置进度
        self.progress_bar_widget.reset()
        resume = self._ask_resume_preference()
        process_file = self._build_processing_coro(resume=resume)

        # 启动异步工作线程
        self.async_worker = AsyncWorker(
            coro=process_file,
            progress_callback=self._on_progress_update,
            completion_callback=self._on_processing_complete,
            error_callback=self._on_processing_error,
        )
        self.async_worker.start()

    def _confirm_resume_dialog(self) -> bool:
        """询问用户是否从上次进度恢复。"""
        from tkinter import messagebox

        return bool(messagebox.askyesno("继续处理", "检测到上次进度，是否继续？"))

    def _ask_resume_preference(self) -> bool:
        """根据进度文件和当前文件判断是否恢复处理。"""
        if self.current_file is None:
            return False

        from services.progress_service import ProgressService

        progress = ProgressService().load_progress()
        if progress is None:
            return False

        progress_file_name = Path(progress.txt_file).name
        if progress_file_name != self.current_file.name:
            return False

        return self._confirm_resume_dialog()

    def _build_processing_coro(self, resume: bool):
        """构建处理协程，便于测试和回调注入。"""
        import asyncio

        async def process_file():
            from services.novel_processing_service import NovelProcessingService

            self.cancel_event = asyncio.Event()
            service = NovelProcessingService(
                progress_callback=self._on_progress_update,
                cancel_event=self.cancel_event,
            )
            return await service.process_novel(file_path=str(self.current_file), resume=resume)

        return process_file()

    def _on_progress_update(self, progress_data: dict):
        """进度更新回调"""
        # 此方法可能由异步线程调用，不能直接触碰 Tk
        self._post_ui_event("progress", progress_data)

    def _do_progress_update(self, progress_data: dict):
        """在主线程中执行进度更新"""
        self.progress_bar_widget.update_progress(
            completed=progress_data.get("completed_chunks", 0),
            total=progress_data.get("total_chunks", 0),
            failed=progress_data.get("failed_chunks", 0),
            partial=progress_data.get("partial_chunks", 0),
            phase=progress_data.get("phase", ""),
            eta_seconds=progress_data.get("eta_seconds"),
            eta_confidence=progress_data.get("eta_confidence", 0.0),
            progress=progress_data.get("progress"),
        )

    def _on_processing_complete(self, result: dict):
        """处理完成回调"""
        logger.info(f"处理完成: {result}")
        self._post_ui_event("complete", result)

    def _do_processing_complete(self, result: dict):
        """在主线程中处理完成"""
        from tkinter import messagebox

        self.reset_ui_state()

        output_path = result.get("output_path")
        if output_path:
            messagebox.showinfo("完成", f"大纲生成完成！\n\n输出文件: {output_path}")
        else:
            messagebox.showinfo("完成", "大纲生成完成！")

    def _on_processing_error(self, error: Exception):
        """处理错误回调"""
        logger.error(f"处理失败: {error}")
        self._post_ui_event("error", error)

    def _post_ui_event(self, event_type: str, payload: Any) -> None:
        """线程安全地投递 UI 事件，交由主线程处理。"""
        self._ui_event_queue.put((event_type, payload))

    def _drain_ui_events(self) -> None:
        """主线程轮询并消费 UI 事件队列。"""
        try:
            while True:
                event_type, payload = self._ui_event_queue.get_nowait()
                if event_type == "progress":
                    self._do_progress_update(payload)
                elif event_type == "complete":
                    self._do_processing_complete(payload)
                elif event_type == "error":
                    self._do_processing_error(payload)
        except Empty:
            pass
        finally:
            if self._is_window_alive():
                self.after(self._ui_poll_interval_ms, self._drain_ui_events)

    def _is_window_alive(self) -> bool:
        """判断窗口是否仍然可用（避免销毁后继续调度）。"""
        exists = getattr(self, "winfo_exists", None)
        if callable(exists):
            try:
                return bool(exists())
            except Exception:
                return False
        return True

    def _do_processing_error(self, error: Exception):
        """在主线程中处理错误"""
        from tkinter import messagebox

        self.reset_ui_state()
        messagebox.showerror("错误", f"处理失败: {error}")

    def on_cancel_processing(self):
        """取消处理事件"""
        if hasattr(self, "cancel_event") and self.cancel_event is not None:
            self.cancel_event.set()
        if self.async_worker:
            self.async_worker.stop()
            self.async_worker = None
            logger.info("用户取消处理")

        self.reset_ui_state()

    def reset_ui_state(self):
        """重置 UI 状态"""
        self.start_button.configure(state="normal" if self.current_file else "disabled")
        self.cancel_button.configure(state="disabled")
        if hasattr(self, "select_file_button"):
            self.select_file_button.configure(state="normal")
        if hasattr(self, "phase_label"):
            self.phase_label.configure(text="等待开始...")

    def on_open_output_dir(self):
        """打开输出目录"""
        import platform
        import subprocess

        from config import get_processing_config

        config = get_processing_config()
        output_dir = Path(config.output_dir)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        system = platform.system()
        try:
            if system == "Windows":
                subprocess.run(["explorer", str(output_dir)])
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(output_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(output_dir)])
        except Exception as e:
            logger.error(f"打开输出目录失败: {e}")

    def on_open_config_file(self):
        """打开配置文件（已弃用，使用 ConfigDialog）"""
        # 保留是为了兼容性，现在使用 on_open_config_dialog
        self.on_open_config_dialog()

    def on_open_config_dialog(self):
        """打开配置编辑器对话框"""
        ConfigDialog(self)

    def on_restart_app(self):
        """重启应用"""
        from tkinter import messagebox

        if messagebox.askyesno("确认重启", "重启应用以使配置更改生效。\n\n确定要重启吗？"):
            logger.info("准备重启应用...")
            # 记录重启标志
            restart_flag = Path(".restart_flag")
            restart_flag.write_text("1")
            # 关闭应用
            self.quit()

    def on_refresh_log(self):
        """刷新日志显示（已弃用，使用 LogViewer 组件）"""
        # 保留是为了兼容性，现在使用 LogViewer 组件
        if hasattr(self, "log_viewer"):
            self.log_viewer.refresh_log()

    def on_clear_log(self):
        """清空日志（已弃用，使用 LogViewer 组件）"""
        # 保留是为了兼容性，现在使用 LogViewer 组件
        if hasattr(self, "log_viewer"):
            self.log_viewer.clear_log()

    def _setup_theme_switcher(self):
        """设置主题切换UI"""
        # 主题切换框架（放在窗口顶部）
        theme_frame = ctk.CTkFrame(self, fg_color=get_color("bg_secondary", mode="auto"))
        theme_frame.pack(fill="x", side="top", padx=SPACING["md"], pady=(SPACING["md"], 0))

        # 标题标签
        title_label = ctk.CTkLabel(
            theme_frame,
            text="小说大纲生成器",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        title_label.pack(side="left", padx=SPACING["md"])

        # 主题切换按钮组
        theme_button_frame = ctk.CTkFrame(theme_frame, fg_color="transparent")
        theme_button_frame.pack(side="right", padx=SPACING["md"])

        # 亮色主题按钮
        light_button = ctk.CTkButton(
            theme_button_frame,
            text="☀️",
            width=40,
            command=lambda: self._switch_theme("light"),
            fg_color=get_color("bg_primary", mode="auto"),
            hover_color=get_color("bg_tertiary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        light_button.pack(side="left", padx=SPACING["xs"])

        # 暗色主题按钮
        dark_button = ctk.CTkButton(
            theme_button_frame,
            text="🌙",
            width=40,
            command=lambda: self._switch_theme("dark"),
            fg_color=get_color("bg_primary", mode="auto"),
            hover_color=get_color("bg_tertiary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        dark_button.pack(side="left", padx=SPACING["xs"])

        # 系统主题按钮
        system_button = ctk.CTkButton(
            theme_button_frame,
            text="💻",
            width=40,
            command=lambda: self._switch_theme("system"),
            fg_color=get_color("bg_primary", mode="auto"),
            hover_color=get_color("bg_tertiary", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        system_button.pack(side="left", padx=SPACING["xs"])

        # 保存主题按钮引用（用于更新高亮状态）
        self.theme_buttons = {
            "light": light_button,
            "dark": dark_button,
            "system": system_button,
        }

        # 更新按钮高亮状态
        self._update_theme_button_states()

    def _switch_theme(self, theme: str):
        """
        切换主题

        Args:
            theme: 主题名称 ("light", "dark", "system")
        """
        self.theme_manager.set_theme(theme)
        self.theme_manager.apply_theme()
        self._update_theme_button_states()
        logger.info(f"主题已切换到: {theme}")

    def _update_theme_button_states(self):
        """更新主题切换按钮的视觉状态（高亮当前主题）"""
        current_theme = self.theme_manager.get_current_theme()
        accent_color = get_color("accent", mode="auto")

        for theme_name, button in self.theme_buttons.items():
            if theme_name == current_theme:
                # 当前主题：使用强调色
                button.configure(fg_color=accent_color)
            else:
                # 其他主题：使用次要背景色
                button.configure(fg_color=get_color("bg_primary", mode="auto"))


def main():
    """启动 GUI 应用"""
    from config import init_config
    from utils import setup_logging

    # 初始化日志
    setup_logging()

    # 初始化配置
    init_config()

    # 创建并启动 GUI
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
