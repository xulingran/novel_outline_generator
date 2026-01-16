"""单元测试：任务队列管理模块"""

import pytest

from services.task_queue import QueueTask, TaskQueue, get_global_queue


class TestQueueTask:
    """测试 QueueTask 类"""

    def test_initialization(self):
        """测试任务初始化"""
        task = QueueTask(id="test-123", file_path="/test/file.txt")
        assert task.id == "test-123"
        assert task.file_path == "/test/file.txt"
        assert task.status == "pending"
        assert task.progress == 0.0
        assert task.logs == []
        assert not task.token_logged
        assert not task.is_cancelled()

    def test_log(self):
        """测试日志记录"""
        task = QueueTask(id="test", file_path="/test.txt")
        task.log("第一条日志")
        task.log("第二条日志")
        assert len(task.logs) == 2
        assert task.logs[0] == "第一条日志"
        assert task.logs[1] == "第二条日志"

    def test_log_truncation(self):
        """测试日志截断"""
        task = QueueTask(id="test", file_path="/test.txt")
        # 添加超过200条日志
        for i in range(250):
            task.log(f"日志 {i}")
        assert len(task.logs) == 200
        assert task.log_offset == 50  # 250 - 200 = 50
        # 检查最后一条是"日志 249"
        assert task.logs[-1] == "日志 249"

    def test_cancel(self):
        """测试取消任务"""
        task = QueueTask(id="test", file_path="/test.txt")
        assert not task.is_cancelled()
        task.cancel()
        assert task.status == "cancelled"
        assert "取消" in task.message  # 检查包含"取消"而不是精确匹配
        assert task.is_cancelled()


class TestTaskQueue:
    """测试 TaskQueue 类"""

    @pytest.fixture
    def queue(self):
        """创建测试队列"""
        return TaskQueue(max_concurrent=1)

    @pytest.mark.asyncio
    async def test_add_task(self, queue):
        """测试添加任务到队列"""
        task_id = await queue.add_task("/test/file1.txt")
        assert task_id is not None
        assert isinstance(task_id, str)

    @pytest.mark.asyncio
    async def test_get_pending_task(self, queue):
        """测试获取等待中的任务"""
        task_id = await queue.add_task("/test/file.txt")
        task = await queue.get_task(task_id)
        assert task is not None
        assert task.id == task_id
        assert task.status == "pending"
        assert task.file_path == "/test/file.txt"

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, queue):
        """测试获取不存在的任务"""
        task = await queue.get_task("nonexistent-id")
        assert task is None

    @pytest.mark.asyncio
    async def test_list_tasks(self, queue):
        """测试列出所有任务"""
        await queue.add_task("/test/file1.txt")
        await queue.add_task("/test/file2.txt")
        await queue.add_task("/test/file3.txt")

        tasks = await queue.list_tasks()
        assert len(tasks) == 3
        assert all("id" in t for t in tasks)
        assert all("status" in t for t in tasks)
        assert all("file_path" in t for t in tasks)

    @pytest.mark.asyncio
    async def test_cancel_pending_task(self, queue):
        """测试取消等待中的任务"""
        task_id = await queue.add_task("/test/file.txt")

        success = await queue.cancel_task(task_id)
        assert success is True

        task = await queue.get_task(task_id)
        # 任务应该从队列中移除
        assert task is None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, queue):
        """测试取消不存在的任务"""
        success = await queue.cancel_task("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_clear_queue(self, queue):
        """测试清空队列"""
        await queue.add_task("/test/file1.txt")
        await queue.add_task("/test/file2.txt")
        await queue.add_task("/test/file3.txt")

        count = await queue.clear_queue()
        assert count == 3

        # 验证队列已清空
        tasks = await queue.list_tasks()
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, queue):
        """测试获取队列统计"""
        await queue.add_task("/test/file1.txt")
        await queue.add_task("/test/file2.txt")

        stats = await queue.get_stats()
        assert stats["pending"] == 2
        assert stats["running"] == 0
        assert stats["total"] == 2

    @pytest.mark.asyncio
    async def test_task_to_dict(self, queue):
        """测试任务转换为字典"""
        task = QueueTask(id="test-123", file_path="/test/file.txt")
        task.log("测试日志")
        task.progress = 0.5

        task_dict = TaskQueue._task_to_dict(task)
        assert task_dict["id"] == "test-123"
        assert task_dict["file_path"] == "/test/file.txt"
        assert task_dict["status"] == "pending"
        assert task_dict["progress"] == 0.5
        assert task_dict["logs"] == ["测试日志"]
        assert "created_at" in task_dict

    @pytest.mark.asyncio
    async def test_stop(self, queue):
        """测试停止队列处理器"""
        # 添加一个任务
        await queue.add_task("/test/file.txt")

        # 停止队列
        await queue.stop()

        # 再次添加任务应该可以
        new_task_id = await queue.add_task("/test/file2.txt")
        assert new_task_id is not None


class TestGlobalQueue:
    """测试全局队列实例"""

    def test_get_global_queue_singleton(self):
        """测试获取全局队列（单例模式）"""
        queue1 = get_global_queue()
        queue2 = get_global_queue()
        assert queue1 is queue2

    @pytest.mark.asyncio
    async def test_global_queue_add_task(self):
        """测试全局队列添加任务"""
        queue = get_global_queue()
        task_id = await queue.add_task("/test/global.txt")
        assert task_id is not None

        # 清理
        await queue.clear_queue()
