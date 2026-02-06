"""测试异步任务桥接。"""

import asyncio
from unittest.mock import MagicMock

from gui.async_worker import AsyncWorker, run_async_in_thread


class TestAsyncWorker:
    """AsyncWorker 单元测试。"""

    def test_run_success(self) -> None:
        async def task():
            await asyncio.sleep(0.01)
            return {"ok": True}

        completion = MagicMock()
        worker = AsyncWorker(task(), completion_callback=completion)
        worker.start()
        worker.join(timeout=3)

        assert worker.result == {"ok": True}
        assert worker.exception is None
        completion.assert_called_once()

    def test_run_error(self) -> None:
        async def task():
            await asyncio.sleep(0.01)
            raise ValueError("boom")

        error_cb = MagicMock()
        worker = AsyncWorker(task(), error_callback=error_cb)
        worker.start()
        worker.join(timeout=3)

        assert isinstance(worker.exception, ValueError)
        error_cb.assert_called_once()

    def test_stop_sets_flag(self) -> None:
        async def task():
            await asyncio.sleep(0.2)
            return "done"

        worker = AsyncWorker(task())
        worker.start()
        worker.stop()
        worker.join(timeout=3)

        assert worker.is_stopped()


class TestHelper:
    """run_async_in_thread 辅助函数测试。"""

    def test_run_async_in_thread(self) -> None:
        async def task():
            return 42

        worker = run_async_in_thread(task())
        worker.join(timeout=3)
        assert worker.result == 42
