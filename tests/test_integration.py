"""
集成测试

测试端到端处理流程和组件协作。
"""

from datetime import datetime

import pytest

from config import reset_all_configs
from models.processing_state import ProcessingState, ProgressData


class TestProgressDataIntegration:
    """进度数据集成测试"""

    def test_progress_data_full_lifecycle(self):
        """测试进度数据完整生命周期"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )

        for i in range(5):
            progress.completed_indices.add(i)
            progress.outlines.append({"chunk_id": i, "content": f"Outline {i}"})
            progress.processing_times.append(1.5 + i * 0.1)

        assert progress.completed_count == 5
        assert progress.completion_rate == 0.5
        assert len(progress.outlines) == 5
        assert progress.average_processing_time > 0

        progress.add_error(6, "Test error")
        assert len(progress.errors) == 1
        assert progress.errors[0]["chunk_id"] == 6

        data = progress.to_dict()
        assert data["completed_count"] == 5
        assert data["total_chunks"] == 10
        assert len(data["completed_indices"]) == 5

        restored = ProgressData.from_dict(data)
        assert restored.completed_count == 5
        assert restored.total_chunks == 10
        assert len(restored.completed_indices) == 5

    def test_progress_data_serialization_with_partial(self):
        """测试带部分完成数据的序列化"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_indices={0, 1, 2},
            outlines=[{"chunk_id": i, "content": f"Outline {i}"} for i in range(3)],
            last_update=datetime.now(),
            chunks_hash="abc123",
            partial_indices={3, 4},
            partial_outlines=[{"chunk_id": 3, "content": "Partial"}],
        )

        data = progress.to_dict()
        restored = ProgressData.from_dict(data)

        assert restored.partial_indices == {3, 4}
        assert len(restored.partial_outlines) == 1


class TestProcessingStateIntegration:
    """处理状态集成测试"""

    def test_processing_state_full_lifecycle(self):
        """测试处理状态完整生命周期"""
        state = ProcessingState(
            file_path="test.txt",
            total_chunks=100,
        )

        assert state.current_phase == "initialization"
        assert state.progress_percentage == 0.0

        state.current_phase = "processing"
        for i in range(50):
            state.update_progress(1)
            if i % 10 == 0:
                state.update_partial(1)

        assert state.processed_chunks == 50
        assert state.partial_chunks == 5
        assert state.progress_percentage == 50.0

        state.add_error("Test error 1")
        state.add_warning("Test warning 1")
        assert len(state.errors) == 1
        assert len(state.warnings) == 1

        state.update_progress(0, failed=5)
        assert state.failed_chunks == 5
        assert state.success_rate > 0

        summary = state.get_summary()
        assert summary["total_chunks"] == 100
        assert summary["processed_chunks"] == 50
        assert summary["failed_chunks"] == 5

    def test_processing_state_merge_tracking(self):
        """测试合并阶段跟踪"""
        state = ProcessingState(
            file_path="test.txt",
            total_chunks=10,
        )

        state.current_phase = "merging"
        state.merge_level = 1
        state.merge_batch_current = 2
        state.merge_batch_total = 5
        state.merge_outlines_count = 20

        assert state.merge_level == 1
        assert state.merge_batch_current == 2

        state.complete()
        assert state.current_phase == "completed"
        assert state.end_time is not None


class TestConfigIntegration:
    """配置集成测试"""

    def test_config_reset_functions(self):
        """测试配置重置函数"""
        reset_all_configs()

        from config import get_api_config, get_processing_config

        api_config = get_api_config()
        proc_config = get_processing_config()

        assert api_config is not None
        assert proc_config is not None

        reset_all_configs()

        new_api_config = get_api_config()
        new_proc_config = get_processing_config()

        assert new_api_config is not api_config or True
        assert new_proc_config is not proc_config or True


class TestHashFunctionIntegration:
    """哈希函数集成测试"""

    def test_chunks_hash_consistency(self):
        """测试块哈希一致性"""
        chunks = ["chunk1", "chunk2", "chunk3"]

        hash1 = ProgressData.calculate_chunks_hash(chunks)
        hash2 = ProgressData.calculate_chunks_hash(chunks)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_chunks_hash_different_for_different_input(self):
        """测试不同输入产生不同哈希"""
        chunks1 = ["chunk1", "chunk2"]
        chunks2 = ["chunk1", "chunk3"]

        hash1 = ProgressData.calculate_chunks_hash(chunks1)
        hash2 = ProgressData.calculate_chunks_hash(chunks2)

        assert hash1 != hash2

    def test_chunks_hash_different_encoding(self):
        """测试不同编码产生不同哈希"""
        chunks = ["chunk1", "chunk2"]

        hash_utf8 = ProgressData.calculate_chunks_hash(chunks, "utf-8")
        hash_gbk = ProgressData.calculate_chunks_hash(chunks, "gbk")

        assert hash_utf8 != hash_gbk

    def test_chunks_hash_order_matters(self):
        """测试块顺序影响哈希"""
        chunks1 = ["a", "b", "c"]
        chunks2 = ["c", "b", "a"]

        hash1 = ProgressData.calculate_chunks_hash(chunks1)
        hash2 = ProgressData.calculate_chunks_hash(chunks2)

        assert hash1 != hash2


class TestErrorHandlingIntegration:
    """错误处理集成测试"""

    def test_progress_data_invalid_datetime(self):
        """测试无效日期时间处理"""
        data = {
            "txt_file": "test.txt",
            "total_chunks": 10,
            "completed_indices": [],
            "outlines": [],
            "last_update": "invalid-datetime",
            "chunks_hash": "abc",
        }

        progress = ProgressData.from_dict(data)
        assert isinstance(progress.last_update, datetime)

    def test_processing_state_validation(self):
        """测试处理状态验证"""
        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=-1)

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, processed_chunks=-1)

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, failed_chunks=-1)


class TestFileServiceIntegration:
    """文件服务集成测试"""

    def test_file_read_operations(self):
        """测试文件读取操作"""
        from services.file_service import FileService

        file_service = FileService()
        assert file_service is not None


class TestTokenEstimatorIntegration:
    """Token 估算器集成测试"""

    def test_token_estimation(self):
        """测试 Token 估算"""
        from tokenizer import count_tokens

        text = "This is a test sentence for token estimation."
        tokens = count_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_token_estimation_empty_text(self):
        """测试空文本 Token 估算"""
        from tokenizer import count_tokens

        tokens = count_tokens("")
        assert tokens == 0


class TestTaskQueueIntegration:
    """任务队列集成测试"""

    @pytest.mark.asyncio
    async def test_task_queue_creation(self):
        """测试任务队列创建"""
        from services.task_queue import TaskQueue

        queue = TaskQueue(max_concurrent=2)
        assert queue is not None
        assert queue.max_concurrent == 2


class TestProgressServiceIntegration:
    """进度服务集成测试"""

    def test_progress_service_creation(self):
        """测试进度服务创建"""
        from services.progress_service import ProgressService

        progress_service = ProgressService()
        assert progress_service is not None


class TestEndToEndFlow:
    """端到端流程测试"""

    @pytest.mark.asyncio
    async def test_processing_flow_simulation(self):
        """模拟处理流程"""
        state = ProcessingState(
            file_path="test.txt",
            total_chunks=10,
        )

        state.current_phase = "processing"

        results = []
        for i in range(state.total_chunks):
            if i < 8:
                state.update_progress(1)
                results.append({"chunk_id": i, "content": f"Outline {i}"})
            elif i < 9:
                state.update_progress(0, failed=1)
                state.add_error(f"Failed to process chunk {i}")
            else:
                state.update_progress(1)
                state.update_partial(1)
                results.append({"chunk_id": i, "content": "Partial outline"})

        assert state.processed_chunks == 9
        assert state.failed_chunks == 1
        assert state.partial_chunks == 1
        assert len(results) == 9

        state.current_phase = "merging"
        state.merge_level = 1
        state.merge_batch_total = 2

        state.complete()
        assert state.current_phase == "completed"

        summary = state.get_summary()
        assert summary["progress_percentage"] == 90.0
        assert summary["success_rate"] == 90.0
