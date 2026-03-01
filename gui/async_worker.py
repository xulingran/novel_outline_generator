"""
异步任务桥接模块

在独立线程中运行 asyncio 任务，并通过回调与 GUI 主线程通信。
"""

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class AsyncWorker(threading.Thread):
    """
    桥接 asyncio 和 GUI 事件循环的工作线程。

    在独立线程中运行异步任务，通过队列或回调将结果和进度更新
    传递回 GUI 主线程。

    **重要线程安全说明：**
    所有回调函数（progress_callback、completion_callback、error_callback）
    都会在此 AsyncWorker 工作线程中被调用，而非 GUI 主线程。
    调用者必须确保回调函数内部使用线程安全方式与 GUI 交互，
    例如在 CustomTkinter 中使用 `after()` 方法调度到主线程：

        def on_progress(data):
            root.after(0, lambda: update_ui(data))

        worker = AsyncWorker(
            coro=my_coro,
            progress_callback=on_progress
        )

    Args:
        coro: 要执行的异步协程
        progress_callback: 进度更新回调函数，接收 dict 参数
        completion_callback: 完成回调函数，接收任务结果
        error_callback: 错误回调函数，接收 Exception
    """

    def __init__(
        self,
        coro: Coroutine,
        progress_callback: Callable[[dict], None] | None = None,
        completion_callback: Callable[[Any], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
    ):
        super().__init__(daemon=True)
        self.coro = coro
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._result: Any = None
        self._exception: Exception | None = None

    def run(self) -> None:
        """线程主函数：在新的事件循环中运行异步任务。"""
        # 创建新的事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            logger.debug("AsyncWorker: 开始执行异步任务")
            self._result = self.loop.run_until_complete(self.coro)
            logger.debug("AsyncWorker: 任务执行完成")

            # 在主线程中调用完成回调
            if self.completion_callback:
                self.completion_callback(self._result)

        except Exception as e:
            logger.exception("AsyncWorker: 任务执行失败")
            self._exception = e

            # 在主线程中调用错误回调
            if self.error_callback:
                self.error_callback(e)

        finally:
            # 清理事件循环
            try:
                self.loop.close()
            except Exception as e:
                logger.warning(f"AsyncWorker: 关闭事件循环时出错: {e}")

    def stop(self) -> None:
        """请求停止任务。"""
        logger.debug("AsyncWorker: 收到停止请求")
        self._stop_event.set()

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

    def is_stopped(self) -> bool:
        """检查是否已请求停止。"""
        return self._stop_event.is_set()

    @property
    def result(self) -> Any:
        """获取任务结果。"""
        return self._result

    @property
    def exception(self) -> Exception | None:
        """获取任务异常。"""
        return self._exception


def run_async_in_thread(
    coro: Coroutine,
    progress_callback: Callable[[dict], None] | None = None,
    completion_callback: Callable[[Any], None] | None = None,
    error_callback: Callable[[Exception], None] | None = None,
) -> AsyncWorker:
    """
    在独立线程中运行异步任务。

    Args:
        coro: 要执行的异步协程
        progress_callback: 进度更新回调函数
        completion_callback: 完成回调函数
        error_callback: 错误回调函数

    Returns:
        AsyncWorker 实例
    """
    worker = AsyncWorker(
        coro=coro,
        progress_callback=progress_callback,
        completion_callback=completion_callback,
        error_callback=error_callback,
    )
    worker.start()
    return worker
