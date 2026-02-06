"""GUI 主窗口。"""

import asyncio
import logging
import platform
import subprocess
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import customtkinter as ctk

from config import get_api_config, get_processing_config, init_config
from gui.async_worker import AsyncWorker
from gui.config_dialog import ConfigDialog
from gui.widgets.file_selector import FileSelector
from gui.widgets.log_viewer import LogViewer
from gui.widgets.progress_bar import ProgressBar

logger = logging.getLogger(__name__)


class MainWindow(ctk.CTk):
    """小说大纲桌面应用主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("小说大纲生成器 v2.0")
        self.geometry("1000x700")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        init_config()
        self.api_config = get_api_config()
        self.processing_config = get_processing_config()

        self.current_file: Path | None = None
        self.cancel_event: asyncio.Event | None = None
        self.async_worker: AsyncWorker | None = None

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        for name in ["处理", "配置", "日志", "关于"]:
            self.tab_view.add(name)

        self.tab_process = self.tab_view.tab("处理")
        self.tab_config = self.tab_view.tab("配置")
        self.tab_log = self.tab_view.tab("日志")
        self.tab_about = self.tab_view.tab("关于")

        self.setup_process_tab()
        self.setup_config_tab()
        self.setup_log_tab()
        self.setup_about_tab()

    def setup_process_tab(self) -> None:
        """构建处理标签页。"""
        main_frame = ctk.CTkFrame(self.tab_process)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.file_selector = FileSelector(main_frame, on_file_selected=self.on_file_selected)
        self.file_selector.pack(fill="x", padx=10, pady=10)

        self.progress_bar_widget = ProgressBar(main_frame)
        self.progress_bar_widget.pack(fill="x", padx=10, pady=10)

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
            control_frame,
            text="打开输出目录",
            command=self.on_open_output_dir,
            width=150,
        )
        self.open_output_button.pack(side="left", padx=10)

    def setup_config_tab(self) -> None:
        """构建配置标签页。"""
        main_frame = ctk.CTkScrollableFrame(self.tab_config)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(main_frame, text="配置管理", font=ctk.CTkFont(size=18, weight="bold")).pack(
            pady=10
        )
        ctk.CTkLabel(
            main_frame,
            text="修改 API 或处理参数后需要重启应用生效。",
            text_color="gray",
        ).pack(pady=10)

        ctk.CTkButton(
            main_frame,
            text="打开配置编辑器",
            command=self.on_open_config_dialog,
            width=180,
            height=40,
        ).pack(pady=10)

        summary = ctk.CTkFrame(main_frame)
        summary.pack(fill="x", pady=20, padx=10)
        ctk.CTkLabel(summary, text="当前配置", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=10
        )
        ctk.CTkLabel(summary, text=f"API 提供商: {self.api_config.provider}").pack(
            pady=5,
            anchor="w",
            padx=20,
        )
        ctk.CTkLabel(
            summary,
            text=f"目标分块大小: {self.processing_config.target_tokens_per_chunk} tokens",
        ).pack(pady=5, anchor="w", padx=20)
        ctk.CTkLabel(summary, text=f"并发限制: {self.processing_config.parallel_limit}").pack(
            pady=5,
            anchor="w",
            padx=20,
        )

    def setup_log_tab(self) -> None:
        """构建日志标签页。"""
        log_file = Path("logs") / "novel_outline.log"
        self.log_viewer = LogViewer(self.tab_log, log_file=log_file)
        self.log_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_about_tab(self) -> None:
        """构建关于标签页。"""
        main_frame = ctk.CTkScrollableFrame(self.tab_about)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            main_frame, text="小说大纲生成器", font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=20)
        ctk.CTkLabel(main_frame, text="版本 2.0.0", font=ctk.CTkFont(size=14)).pack(pady=5)

        desc = ctk.CTkTextbox(main_frame, height=260, width=700)
        desc.pack(pady=10, padx=10)
        desc.insert(
            "0.0",
            "基于 LLM 的小说大纲生成工具。\n\n"
            "- 支持 OpenAI / Gemini / 智谱 / AiHubMix\n"
            "- 支持长文本自动分块并行处理\n"
            "- 支持进度跟踪与断点续跑\n"
            "- 提供 Web UI、CLI 和桌面 GUI 三种模式\n",
        )
        desc.configure(state="disabled")

    def on_file_selected(self, filepath: Path) -> None:
        """文件选择回调。"""
        self.current_file = filepath
        self.start_button.configure(state="normal")

    def on_start_processing(self) -> None:
        """开始处理文件。"""
        if not self.current_file:
            return
        if self.async_worker and self.async_worker.is_alive():
            logger.warning("已有处理任务正在运行，忽略重复启动请求")
            return

        resume = self._ask_resume_preference()

        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_bar_widget.reset()

        self.cancel_event = asyncio.Event()

        self.async_worker = AsyncWorker(
            coro=self._build_processing_coro(resume),
            completion_callback=self._on_processing_complete,
            error_callback=self._on_processing_error,
        )
        self.async_worker.start()

    def _ask_resume_preference(self) -> bool:
        """根据现有进度询问是否恢复。"""
        from services.progress_service import ProgressService

        if not self.current_file:
            return False

        progress_data = ProgressService().load_progress()
        if not progress_data:
            return False

        if progress_data.txt_file != str(self.current_file):
            return False

        return self._confirm_resume_dialog()

    @staticmethod
    def _confirm_resume_dialog() -> bool:
        """弹出恢复进度确认框。"""
        try:
            from tkinter import messagebox
        except ModuleNotFoundError:
            return False

        return messagebox.askyesno(
            "恢复进度",
            "检测到该文件存在未完成进度，是否从上次进度继续？",
        )

    def _build_processing_coro(self, resume: bool) -> Coroutine[Any, Any, dict[str, Any]]:
        """构建后台处理协程。"""

        async def process_file() -> dict[str, Any]:
            from services.novel_processing_service import NovelProcessingService

            service = NovelProcessingService(
                progress_callback=self._on_progress_update,
                cancel_event=self.cancel_event,
            )
            return await service.process_novel(file_path=str(self.current_file), resume=resume)

        return process_file()

    def _on_progress_update(self, progress_data: dict[str, Any]) -> None:
        """接收后台线程进度并切回 UI 线程更新。"""
        self.after(0, lambda: self._do_progress_update(progress_data))

    def _do_progress_update(self, progress_data: dict[str, Any]) -> None:
        self.progress_bar_widget.update_progress(
            completed=progress_data.get("completed_chunks", 0),
            total=progress_data.get("total_chunks", 0),
            failed=progress_data.get("failed_chunks", 0),
            partial=progress_data.get("partial_chunks", 0),
            phase=progress_data.get("phase", ""),
            eta_seconds=progress_data.get("eta_seconds"),
            eta_confidence=progress_data.get("eta_confidence", 0.0),
        )

    def _on_processing_complete(self, result: dict[str, Any]) -> None:
        """后台任务完成回调。"""
        self.after(0, lambda: self._do_processing_complete(result))

    def _do_processing_complete(self, result: dict[str, Any]) -> None:
        from tkinter import messagebox

        self.async_worker = None
        self.cancel_event = None
        self.reset_ui_state()
        messagebox.showinfo(
            "完成", f"大纲生成完成。\n输出目录: {result.get('output_dir', 'outputs')}"
        )

    def _on_processing_error(self, error: Exception) -> None:
        """后台任务失败回调。"""
        self.after(0, lambda: self._do_processing_error(error))

    def _do_processing_error(self, error: Exception) -> None:
        from tkinter import messagebox

        self.async_worker = None
        self.cancel_event = None
        self.reset_ui_state()
        messagebox.showerror("错误", f"处理失败: {error}")

    def on_cancel_processing(self) -> None:
        """取消当前处理任务。"""
        if self.cancel_event:
            self.cancel_event.set()
        if not self.async_worker:
            self.reset_ui_state()
            return

        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        self.async_worker.stop()
        self._wait_worker_shutdown()

    def _wait_worker_shutdown(self) -> None:
        """等待后台线程结束后恢复 UI 状态，避免并发启动任务。"""
        if self.async_worker and self.async_worker.is_alive():
            self.after(100, self._wait_worker_shutdown)
            return

        self.async_worker = None
        self.cancel_event = None
        self.reset_ui_state()

    def reset_ui_state(self) -> None:
        """恢复按钮状态。"""
        self.start_button.configure(state="normal" if self.current_file else "disabled")
        self.cancel_button.configure(state="disabled")

    def on_open_output_dir(self) -> None:
        """打开输出目录。"""
        output_dir = Path(self.processing_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            system = platform.system()
            if system == "Windows":
                subprocess.run(["explorer", str(output_dir)], check=False)
            elif system == "Darwin":
                subprocess.run(["open", str(output_dir)], check=False)
            else:
                subprocess.run(["xdg-open", str(output_dir)], check=False)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"打开输出目录失败: {exc}")

    def on_open_config_dialog(self) -> None:
        """打开配置编辑对话框。"""
        ConfigDialog(self)

    def on_close(self) -> None:
        """窗口关闭事件。"""
        self.on_cancel_processing()
        if hasattr(self, "log_viewer"):
            self.log_viewer.stop_auto_refresh()
        self.destroy()


def main() -> None:
    """启动 GUI 应用。"""
    from utils import setup_logging

    setup_logging()
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
