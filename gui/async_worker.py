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

    **线程安全说明：**
    如果提供了 root 参数，所有回调会自动通过 root.after() 调度到 GUI 主线程。
    如果未提供 root，回调将在工作线程中直接调用——调用者需自行保证线程安全。

    Args:
        coro: 要执行的异步协程
        progress_callback: 进度更新回调函数，接收 dict 参数
        completion_callback: 完成回调函数，接收任务结果
        error_callback: 错误回调函数，接收 Exception
        root: 可选的 tkinter 根窗口，提供后回调自动调度到主线程
    """

    def __init__(
        self,
        coro: Coroutine,
        progress_callback: Callable[[dict], None] | None = None,
        completion_callback: Callable[[Any], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
        root: Any | None = None,
    ):
        super().__init__(daemon=True)
        self.coro = coro
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        self._root = root
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._result: Any = None
        self._exception: Exception | None = None

    def _invoke_callback(self, callback: Callable[..., None] | None, *args: Any) -> None:
        """安全调用回调：有 root 时调度到主线程，否则直接调用。"""
        if callback is None:
            return
        if self._root is not None:
            try:
                self._root.after(0, lambda: callback(*args))
            except Exception:
                callback(*args)
        else:
            callback(*args)

    def run(self) -> None:
        """线程主函数：在新的事件循环中运行异步任务。"""
        # 创建新的事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            logger.debug("AsyncWorker: 开始执行异步任务")
            self._result = self.loop.run_until_complete(self.coro)
            logger.debug("AsyncWorker: 任务执行完成")

            self._invoke_callback(self.completion_callback, self._result)

        except Exception as e:
            logger.exception("AsyncWorker: 任务执行失败")
            self._exception = e

            self._invoke_callback(self.error_callback, e)

        finally:
            # 清理事件循环
            try:
                self.loop.close()
            except Exception as e:
                logger.warning("AsyncWorker: 关闭事件循环时出错: %s", e)

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
