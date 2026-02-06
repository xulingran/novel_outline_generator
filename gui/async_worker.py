"""异步任务桥接模块。"""

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class AsyncWorker(threading.Thread):
    """在独立线程中运行协程任务并回调结果。"""

    def __init__(
        self,
        coro: Coroutine[Any, Any, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        completion_callback: Callable[[Any], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.coro = coro
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._task: asyncio.Task[Any] | None = None
        self._result: Any = None
        self._exception: Exception | None = None

    def run(self) -> None:
        """线程主函数，在私有事件循环中执行协程。"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self._task = self.loop.create_task(self.coro)
            self._result = self.loop.run_until_complete(self._task)
            if self.completion_callback and not self._stop_event.is_set():
                self.completion_callback(self._result)
        except asyncio.CancelledError:
            logger.info("AsyncWorker task cancelled")
        except Exception as exc:  # noqa: BLE001
            self._exception = exc
            logger.exception("AsyncWorker task failed")
            if self.error_callback and not self._stop_event.is_set():
                self.error_callback(exc)
        finally:
            try:
                if self._task and not self._task.done():
                    self._task.cancel()
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                logger.debug("AsyncWorker loop cleanup skipped", exc_info=True)
            finally:
                self.loop.close()

    def stop(self) -> None:
        """请求停止任务。"""
        self._stop_event.set()
        if not self.loop or not self.loop.is_running():
            return

        def _cancel_task() -> None:
            if self._task and not self._task.done():
                self._task.cancel()

        self.loop.call_soon_threadsafe(_cancel_task)

    def is_stopped(self) -> bool:
        """检查是否已经请求停止。"""
        return self._stop_event.is_set()

    @property
    def result(self) -> Any:
        """获取任务执行结果。"""
        return self._result

    @property
    def exception(self) -> Exception | None:
        """获取任务异常。"""
        return self._exception


def run_async_in_thread(
    coro: Coroutine[Any, Any, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    completion_callback: Callable[[Any], None] | None = None,
    error_callback: Callable[[Exception], None] | None = None,
) -> AsyncWorker:
    """在线程中运行协程并返回工作线程对象。"""
    worker = AsyncWorker(
        coro=coro,
        progress_callback=progress_callback,
        completion_callback=completion_callback,
        error_callback=error_callback,
    )
    worker.start()
    return worker
