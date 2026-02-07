"""
测试 async_worker 模块

测试异步任务桥接功能，包括线程启动、进度回调、完成回调和错误处理。
"""

import asyncio
from unittest.mock import MagicMock

from gui.async_worker import AsyncWorker, run_async_in_thread


class TestAsyncWorker:
    """测试 AsyncWorker 类"""

    def test_init(self):
        """测试 AsyncWorker 初始化"""

        async def dummy_task():
            return "result"

        progress_callback = MagicMock()
        completion_callback = MagicMock()
        error_callback = MagicMock()

        worker = AsyncWorker(
            dummy_task(),
            progress_callback=progress_callback,
            completion_callback=completion_callback,
            error_callback=error_callback,
        )

        # 协程对象不能直接比较，所以检查它不是 None
        assert worker.coro is not None
        assert worker.progress_callback == progress_callback
        assert worker.completion_callback == completion_callback
        assert worker.error_callback == error_callback
        assert worker.loop is None
        assert worker.result is None
        assert worker.exception is None
        assert worker.daemon is True

    def test_run_success(self, event_loop: asyncio.AbstractEventLoop):
        """测试成功运行任务"""

        async def success_task():
            await asyncio.sleep(0.01)
            return {"status": "success", "data": "test_data"}

        completion_callback = MagicMock()
        error_callback = MagicMock()

        worker = AsyncWorker(
            success_task(),
            completion_callback=completion_callback,
            error_callback=error_callback,
        )

        # 启动工作线程
        worker.start()
        worker.join(timeout=5)

        # 验证结果
        assert worker.result == {"status": "success", "data": "test_data"}
        assert worker.exception is None
        assert completion_callback.called

    def test_run_error(self, event_loop: asyncio.AbstractEventLoop):
        """测试任务出错"""

        async def error_task():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        completion_callback = MagicMock()
        error_callback = MagicMock()

        worker = AsyncWorker(
            error_task(),
            completion_callback=completion_callback,
            error_callback=error_callback,
        )

        # 启动工作线程
        worker.start()
        worker.join(timeout=5)

        # 验证错误处理
        assert worker.result is None
        assert isinstance(worker.exception, ValueError)
        assert str(worker.exception) == "Test error"
        assert error_callback.called
        assert not completion_callback.called

    def test_stop(self, event_loop: asyncio.AbstractEventLoop):
        """测试停止任务"""

        async def long_running_task():
            await asyncio.sleep(10)
            return "should not complete"

        worker = AsyncWorker(long_running_task())

        # 启动线程
        worker.start()

        # 立即请求停止
        worker.stop()
        worker.join(timeout=2)

        # 验证停止标志被设置
        assert worker.is_stopped()

    def test_is_stopped(self, event_loop: asyncio.AbstractEventLoop):
        """测试停止状态检查"""

        async def dummy_task():
            return "result"

        worker = AsyncWorker(dummy_task())

        # 初始状态
        assert not worker.is_stopped()

        # 停止后
        worker._stop_event.set()
        assert worker.is_stopped()

    def test_result_property(self, event_loop: asyncio.AbstractEventLoop):
        """测试结果属性"""

        async def return_value():
            await asyncio.sleep(0.01)
            return 42

        worker = AsyncWorker(return_value())
        worker.start()
        worker.join(timeout=5)

        assert worker.result == 42

    def test_exception_property(self, event_loop: asyncio.AbstractEventLoop):
        """测试异常属性"""

        async def raise_error():
            await asyncio.sleep(0.01)
            raise RuntimeError("Test runtime error")

        worker = AsyncWorker(raise_error())
        worker.start()
        worker.join(timeout=5)

        assert isinstance(worker.exception, RuntimeError)
        assert str(worker.exception) == "Test runtime error"

    def test_progress_callback_integration(self, event_loop: asyncio.AbstractEventLoop):
        """测试进度回调集成"""

        async def task_with_progress():
            # 在实际使用中，进度回调会被任务内部调用
            await asyncio.sleep(0.01)
            return "done"

        progress_callback = MagicMock()

        worker = AsyncWorker(
            task_with_progress(),
            progress_callback=progress_callback,
        )

        worker.start()
        worker.join(timeout=5)

        # 注意：在真实场景中，进度回调会在任务执行期间被调用
        # 这里只是验证 worker 能够接受和存储回调
        assert worker.progress_callback == progress_callback


class TestRunAsyncInThread:
    """测试 run_async_in_thread 辅助函数"""

    def test_run_async_in_thread(self, event_loop: asyncio.AbstractEventLoop):
        """测试通过辅助函数运行异步任务"""

        async def simple_task():
            await asyncio.sleep(0.01)
            return "async_result"

        completion_callback = MagicMock()

        worker = run_async_in_thread(
            simple_task(),
            completion_callback=completion_callback,
        )
        worker.join(timeout=5)

        assert worker.result == "async_result"
        assert completion_callback.called

    def test_run_async_in_thread_with_all_callbacks(self, event_loop: asyncio.AbstractEventLoop):
        """测试带有所有回调的辅助函数"""

        async def task():
            await asyncio.sleep(0.01)
            return "complete"

        progress_callback = MagicMock()
        completion_callback = MagicMock()
        error_callback = MagicMock()

        worker = run_async_in_thread(
            task(),
            progress_callback=progress_callback,
            completion_callback=completion_callback,
            error_callback=error_callback,
        )
        worker.join(timeout=5)

        assert worker.progress_callback == progress_callback
        assert completion_callback.called

    def test_run_async_in_thread_error_handling(self, event_loop: asyncio.AbstractEventLoop):
        """测试辅助函数的错误处理"""

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Helper test error")

        error_callback = MagicMock()

        worker = run_async_in_thread(
            failing_task(),
            error_callback=error_callback,
        )
        worker.join(timeout=5)

        assert isinstance(worker.exception, ValueError)
        assert error_callback.called


class TestAsyncWorkerEdgeCases:
    """测试 AsyncWorker 边界情况"""

    def test_task_with_none_result(self, event_loop: asyncio.AbstractEventLoop):
        """测试返回 None 的任务"""

        async def none_task():
            await asyncio.sleep(0.01)
            return None

        worker = AsyncWorker(none_task())

        worker.start()
        worker.join(timeout=5)

        assert worker.result is None
        assert worker.exception is None

    def test_task_with_complex_result(self, event_loop: asyncio.AbstractEventLoop):
        """测试返回复杂结构的任务"""

        async def complex_task():
            await asyncio.sleep(0.01)
            return {
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "tuple": (4, 5, 6),
            }

        worker = AsyncWorker(complex_task())

        worker.start()
        worker.join(timeout=5)

        assert worker.result["list"] == [1, 2, 3]
        assert worker.result["dict"]["nested"] == "value"
        assert worker.result["tuple"] == (4, 5, 6)

    def test_task_with_multiple_errors(self, event_loop: asyncio.AbstractEventLoop):
        """测试多个错误只记录第一个"""

        async def multi_error_task():
            try:
                raise ValueError("First error")
            except ValueError:
                pass
            raise RuntimeError("Second error")

        worker = AsyncWorker(multi_error_task())

        worker.start()
        worker.join(timeout=5)

        # 应该记录第二个错误（最终抛出的）
        assert isinstance(worker.exception, RuntimeError)

    def test_daemon_thread(self, event_loop: asyncio.AbstractEventLoop):
        """验证工作线程是守护线程"""

        async def dummy():
            return "result"

        worker = AsyncWorker(dummy())
        assert worker.daemon is True

    def test_callbacks_can_be_none(self, event_loop: asyncio.AbstractEventLoop):
        """测试回调可以为 None"""

        async def dummy():
            return "result"

        worker = AsyncWorker(
            dummy(),
            progress_callback=None,
            completion_callback=None,
            error_callback=None,
        )

        # 应该正常运行，只是不会调用任何回调
        worker.start()
        worker.join(timeout=5)

        assert worker.result == "result"
