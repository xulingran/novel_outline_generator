"""
任务队列管理模块
支持多文件按顺序处理，支持队列管理和取消功能
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueueTask:
    """队列任务数据模型"""

    id: str
    file_path: str
    status: str = "pending"  # pending|running|success|error|cancelled
    message: str = ""
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    should_force_complete: bool = False  # 是否强制完成（忽略未完成的块）
    started_at: float | None = None
    completed_at: float | None = None
    log_offset: int = 0
    token_logged: bool = False
    created_at: float = field(default_factory=time.time)
    merge_progress_logged: bool = False
    token_logged_displayed: bool = False

    def log(self, text: str) -> None:
        """添加日志"""
        self.logs.append(text)
        if len(self.logs) > 200:
            overflow = len(self.logs) - 200
            del self.logs[:overflow]
            self.log_offset += overflow

    def cancel(self) -> None:
        """设置取消标志"""
        self.status = "cancelled"
        self.message = "用户取消"
        self.cancel_event.set()

    def force_complete(self) -> None:
        """强制完成任务（忽略未完成的块）"""
        self.should_force_complete = True
        self.message = "强制完成中"

    def is_cancelled(self) -> bool:
        """检查是否被取消"""
        return self.cancel_event.is_set()


class TaskQueue:
    """任务队列管理器"""

    def __init__(self, max_concurrent: int = 1, run_queue_task_callback=None):
        """
        Args:
            max_concurrent: 最大并发处理数（默认为1，按顺序处理）
        """
        self.max_concurrent = max_concurrent
        self._queue: deque[QueueTask] = deque()
        self._running_tasks: dict[str, QueueTask] = {}
        self._lock = asyncio.Lock()
        self._processor_task: asyncio.Task | None = None
        self._run_queue_task_callback = run_queue_task_callback
        self._completed_tasks: deque[QueueTask] = deque()  # 保留已完成的任务

    async def add_task(self, file_path: str) -> str:
        """
        添加任务到队列

        Args:
            file_path: 文件路径

        Returns:
            任务ID
        """
        async with self._lock:
            task = QueueTask(id=str(uuid.uuid4()), file_path=file_path)
            self._queue.append(task)
            logger.info(f"任务 {task.id} 已添加到队列: {file_path}")

            # 如果处理器未运行，启动它
            if self._processor_task is None or self._processor_task.done():
                self._processor_task = asyncio.create_task(self._process_queue())

            return task.id

    async def get_task(self, task_id: str) -> QueueTask | None:
        """获取任务"""
        async with self._lock:
            # 先在运行任务中查找
            if task_id in self._running_tasks:
                return self._running_tasks[task_id]

            # 在队列中查找
            for task in self._queue:
                if task.id == task_id:
                    return task

            return None

    async def list_tasks(self) -> list[dict[str, Any]]:
        """
        列出所有任务

        Returns:
            任务列表
        """
        async with self._lock:
            tasks = []

            # 队列中的任务
            for task in self._queue:
                tasks.append(self._task_to_dict(task))

            # 正在运行的任务
            for task in self._running_tasks.values():
                tasks.append(self._task_to_dict(task))

            # 已完成的任务（保留最近10个）
            for task in list(self._completed_tasks)[-10:]:
                tasks.append(self._task_to_dict(task))

            return tasks

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功取消
        """
        async with self._lock:
            # 检查正在运行的任务
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.cancel()
                logger.info(f"取消运行中的任务: {task_id}")
                return True

            # 检查队列中的任务
            for i, task in enumerate(self._queue):
                if task.id == task_id and task.status == "pending":
                    task.cancel()
                    # 从队列中移除
                    del self._queue[i]
                    logger.info(f"取消队列中的任务: {task_id}")
                    return True

        return False

    async def force_complete_task(self, task_id: str) -> bool:
        """
        强制完成任务（忽略未完成的块，直接合并已有结果）

        Args:
            task_id: 任务ID

        Returns:
            是否成功触发强制完成
        """
        async with self._lock:
            # 只对运行中的任务支持强制完成
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.force_complete()  # 设置标志并触发取消事件
                logger.info(f"强制完成任务: {task_id}（将合并已有部分结果）")
                return True

        return False

    async def clear_queue(self) -> int:
        """
        清空队列（仅取消未开始的任务）

        Returns:
            取消的任务数量
        """
        async with self._lock:
            count = 0
            for task in self._queue:
                if task.status == "pending":
                    task.cancel()
                    count += 1

            self._queue.clear()
            logger.info(f"清空队列，取消了 {count} 个任务")
            return count

    async def get_stats(self) -> dict[str, Any]:
        """获取队列统计"""
        async with self._lock:
            pending_count = sum(1 for t in self._queue if t.status == "pending")
            running_count = len(self._running_tasks)

            return {
                "pending": pending_count,
                "running": running_count,
                "total": pending_count + running_count,
            }

    def set_callback(self, callback) -> None:
        """设置任务处理回调函数"""
        self._run_queue_task_callback = callback

    async def stop(self) -> None:
        """停止队列处理器"""
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def _process_queue(self) -> None:
        """队列处理器（后台运行）"""
        logger.info("队列处理器已启动")

        try:
            while True:
                # 检查是否可以启动新任务
                if len(self._running_tasks) < self.max_concurrent and self._queue:
                    # 取出下一个任务
                    task = self._queue.popleft()

                    # 跳过已取消的任务
                    if task.is_cancelled():
                        logger.info(f"任务 {task.id} 已取消，跳过")
                        continue

                    # 启动任务
                    task.status = "running"
                    task.started_at = time.time()
                    self._running_tasks[task.id] = task

                    # 创建任务处理协程
                    asyncio.create_task(self._run_task(task))

                # 等待一段时间再检查
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info("队列处理器已停止")
            raise

    async def _run_task(self, task: QueueTask) -> None:
        """运行单个任务"""
        logger.info(f"开始处理任务: {task.id}")

        try:
            # 使用依赖注入的回调处理任务
            if self._run_queue_task_callback is None:
                raise RuntimeError("未设置 run_queue_task 回调")
            await self._run_queue_task_callback(task)

        except asyncio.CancelledError:
            task.status = "cancelled"
            task.message = "任务被取消"
            logger.info(f"任务 {task.id} 被取消")
        except Exception as e:
            task.status = "error"
            task.message = str(e)
            task.log(f"错误: {e}")
            logger.error(f"任务 {task.id} 处理失败: {e}")
        finally:
            task.completed_at = time.time()

            # 从运行任务列表中移除
            if task.id in self._running_tasks:
                del self._running_tasks[task.id]

            # 添加到已完成列表（success/error/cancelled）
            if task.status in ("success", "error", "cancelled"):
                self._completed_tasks.append(task)
                # 只保留最近的20个已完成任务
                if len(self._completed_tasks) > 20:
                    self._completed_tasks.popleft()

            logger.info(f"任务 {task.id} 处理完成，状态: {task.status}")

    @staticmethod
    def _task_to_dict(task: QueueTask) -> dict[str, Any]:
        """将任务转换为字典（用于 API 响应）"""
        return {
            "id": task.id,
            "file_path": task.file_path,
            "status": task.status,
            "message": task.message,
            "progress": task.progress,
            "result": task.result,
            "logs": task.logs,
            "log_offset": task.log_offset,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
        }


# 全局任务队列实例
_global_queue: TaskQueue | None = None


def get_global_queue() -> TaskQueue:
    """获取全局任务队列实例"""
    global _global_queue
    if _global_queue is None:
        _global_queue = TaskQueue(max_concurrent=1)
    return _global_queue
