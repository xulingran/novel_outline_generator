"""测试处理状态模型"""

from datetime import datetime

from models.processing_state import ProcessingState, ProgressData


class TestProgressData:
    """ProgressData 测试类"""

    def test_initialization(self):
        """测试初始化"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=5,
            completed_indices={0, 1, 2, 3, 4},
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )
        assert progress.total_chunks == 10
        assert progress.completed_count == 5
        assert len(progress.completed_indices) == 5

    def test_completion_rate(self):
        """测试完成率计算"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=5,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )
        assert progress.completion_rate == 0.5

        progress.total_chunks = 0
        assert progress.completion_rate == 0.0

    def test_average_processing_time(self):
        """测试平均处理时间计算"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
            processing_times=[1.0, 2.0, 3.0],
        )
        assert progress.average_processing_time == 2.0

        progress.processing_times = []
        assert progress.average_processing_time == 0.0

    def test_add_error(self):
        """测试添加错误"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )
        progress.add_error(1, "Test error")
        assert len(progress.errors) == 1
        assert progress.errors[0]["chunk_id"] == 1
        assert progress.errors[0]["error"] == "Test error"

    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime.now()
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=5,
            completed_indices={0, 1},
            outlines=[{"id": 1}],
            last_update=now,
            chunks_hash="abc123",
            processing_times=[1.0, 2.0],
        )
        data = progress.to_dict()
        assert data["txt_file"] == "test.txt"
        assert data["total_chunks"] == 10
        assert data["completed_indices"] == [0, 1]
        assert data["processing_times"] == [1.0, 2.0]

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "txt_file": "test.txt",
            "total_chunks": 10,
            "completed_count": 5,
            "completed_indices": [0, 1, 2],
            "outlines": [{"id": 1}],
            "last_update": datetime.now().isoformat(),
            "chunks_hash": "abc123",
            "processing_times": [1.0, 2.0],
            "errors": [],
        }
        progress = ProgressData.from_dict(data)
        assert progress.txt_file == "test.txt"
        assert progress.total_chunks == 10
        assert 0 in progress.completed_indices

    def test_calculate_chunks_hash(self):
        """测试计算块哈希"""
        chunks = ["chunk1", "chunk2", "chunk3"]
        hash1 = ProgressData.calculate_chunks_hash(chunks)
        hash2 = ProgressData.calculate_chunks_hash(chunks)
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest

    def test_calculate_chunks_hash_order_sensitive(self):
        """测试块哈希对顺序敏感"""
        chunks1 = ["chunk1", "chunk2", "chunk3"]
        chunks2 = ["chunk3", "chunk2", "chunk1"]
        hash1 = ProgressData.calculate_chunks_hash(chunks1)
        hash2 = ProgressData.calculate_chunks_hash(chunks2)
        assert hash1 != hash2  # 不同顺序应产生不同哈希


class TestProcessingState:
    """ProcessingState 测试类"""

    def test_initialization(self):
        """测试初始化"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.file_path == "test.txt"
        assert state.total_chunks == 10
        assert state.processed_chunks == 0
        assert state.failed_chunks == 0
        assert state.current_phase == "initialization"
        assert state.merge_level == 0
        assert state.merge_batch_current == 0
        assert state.merge_batch_total == 0
        assert state.merge_outlines_count == 0

    def test_elapsed_time(self):
        """测试已用时间计算"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        import time

        time.sleep(0.01)
        elapsed = state.elapsed_time
        assert elapsed >= 0.01

        state.end_time = datetime.now()
        elapsed = state.elapsed_time
        assert elapsed >= 0

    def test_progress_percentage(self):
        """测试进度百分比"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.progress_percentage == 0.0
        state.processed_chunks = 5
        assert state.progress_percentage == 50.0
        state.total_chunks = 0
        assert state.progress_percentage == 0.0

    def test_success_rate(self):
        """测试成功率"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.success_rate == 0.0
        state.processed_chunks = 8
        state.failed_chunks = 2
        assert state.success_rate == 80.0

        state.processed_chunks = 0
        state.failed_chunks = 0
        assert state.success_rate == 0.0

    def test_update_progress(self):
        """测试更新进度"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        state.update_progress(processed=3)
        assert state.processed_chunks == 3
        state.update_progress(processed=2, failed=1)
        assert state.processed_chunks == 5
        assert state.failed_chunks == 1

    def test_add_error(self):
        """测试添加错误"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        state.add_error("Test error")
        assert len(state.errors) == 1
        assert "Test error" in state.errors[0]

    def test_add_warning(self):
        """测试添加警告"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        state.add_warning("Test warning")
        assert len(state.warnings) == 1
        assert "Test warning" in state.warnings[0]

    def test_complete(self):
        """测试完成标记"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.end_time is None
        state.complete()
        assert state.end_time is not None
        assert state.current_phase == "completed"

    def test_fail(self):
        """测试失败标记"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        state.fail("Error message")
        assert state.end_time is not None
        assert state.current_phase == "failed"
        assert len(state.errors) == 1

    def test_get_summary(self):
        """测试获取摘要"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        state.processed_chunks = 5
        state.failed_chunks = 1
        state.add_error("Error1")
        state.add_warning("Warning1")

        summary = state.get_summary()
        assert summary["file_path"] == "test.txt"
        assert summary["total_chunks"] == 10
        assert summary["processed_chunks"] == 5
        assert summary["failed_chunks"] == 1
        assert summary["errors_count"] == 1
        assert summary["warnings_count"] == 1
        assert "progress_percentage" in summary
        assert "success_rate" in summary
        assert "elapsed_time" in summary

    def test_merge_level_tracking(self):
        """测试合并层级跟踪"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.merge_level == 0
        state.merge_level = 1
        assert state.merge_level == 1
        state.merge_outlines_count = 5
        assert state.merge_outlines_count == 5
        state.merge_batch_current = 2
        state.merge_batch_total = 4
        assert state.merge_batch_current == 2
        assert state.merge_batch_total == 4
