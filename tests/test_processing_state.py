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
            completed_indices={0, 1, 2, 3, 4},
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )
        assert progress.total_chunks == 10
        assert progress.completed_count == 5  # 从 completed_indices 计算
        assert len(progress.completed_indices) == 5

    def test_completion_rate(self):
        """测试完成率计算"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_indices={0, 1, 2, 3, 4},
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
            completed_indices={0, 1},
            outlines=[{"id": 1}],
            last_update=now,
            chunks_hash="abc123",
            processing_times=[1.0, 2.0],
        )
        data = progress.to_dict()
        assert data["txt_file"] == "test.txt"
        assert data["total_chunks"] == 10
        assert data["completed_count"] == 2  # 从 completed_indices 计算
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

    def test_progress_data_with_partial_fields_serialization(self):
        """测试进度数据包含部分完成字段的序列化与反序列化"""
        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=5,
            completed_indices={0, 1, 2},
            outlines=[{"chunk_id": i, "plot": ["event1"]} for i in range(3)],
            last_update=datetime.now(),
            chunks_hash="abc123",
            encoding="utf-8",
            partial_indices={3, 4},
            partial_outlines=[{"chunk_id": 3, "is_partial": True, "plot": ["partial_event"]}],
        )

        # 序列化
        data_dict = progress_data.to_dict()
        assert "partial_indices" in data_dict
        assert data_dict["partial_indices"] == [3, 4]
        assert "partial_outlines" in data_dict
        assert len(data_dict["partial_outlines"]) == 1

        # 反序列化
        restored = ProgressData.from_dict(data_dict)
        assert restored.partial_indices == {3, 4}
        assert len(restored.partial_outlines) == 1
        assert restored.partial_outlines[0]["is_partial"] is True

    def test_calculate_chunks_hash(self):
        """测试计算块哈希"""
        chunks = ["chunk1", "chunk2", "chunk3"]
        hash1 = ProgressData.calculate_chunks_hash(chunks)
        hash2 = ProgressData.calculate_chunks_hash(chunks)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_calculate_chunks_hash_order_sensitive(self):
        """测试块哈希对顺序敏感"""
        chunks1 = ["chunk1", "chunk2", "chunk3"]
        chunks2 = ["chunk3", "chunk2", "chunk1"]
        hash1 = ProgressData.calculate_chunks_hash(chunks1)
        hash2 = ProgressData.calculate_chunks_hash(chunks2)
        assert hash1 != hash2

    def test_calculate_chunks_hash_encoding_sensitive(self):
        """测试块哈希对编码敏感"""
        chunks = ["chunk1", "chunk2"]
        hash_utf8 = ProgressData.calculate_chunks_hash(chunks, "utf-8")
        hash_gbk = ProgressData.calculate_chunks_hash(chunks, "gbk")
        assert hash_utf8 != hash_gbk

    def test_calculate_chunks_hash_empty_list(self):
        """测试空列表哈希"""
        hash_result = ProgressData.calculate_chunks_hash([])
        assert len(hash_result) == 64

    def test_calculate_chunks_hash_special_characters(self):
        """测试特殊字符哈希"""
        chunks = ["中文内容", "emoji 🎉", "special\n\tchars"]
        hash_result = ProgressData.calculate_chunks_hash(chunks)
        assert len(hash_result) == 64


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

    def test_update_partial(self):
        """测试更新部分完成计数"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.partial_chunks == 0
        state.update_partial()
        assert state.partial_chunks == 1
        state.update_partial(2)
        assert state.partial_chunks == 3

    def test_partial_chunks_initialization(self):
        """测试部分完成计数器初始化"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        assert state.partial_chunks == 0

    def test_get_summary_includes_partial_chunks(self):
        """测试摘要包含部分完成计数"""
        state = ProcessingState(file_path="test.txt", total_chunks=10)
        state.processed_chunks = 5
        state.failed_chunks = 1
        state.partial_chunks = 2

        summary = state.get_summary()
        assert summary["processed_chunks"] == 5
        assert summary["failed_chunks"] == 1
        assert summary["partial_chunks"] == 2

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

    def test_validation_negative_total_chunks(self):
        """测试验证负数总块数"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=-1)

    def test_validation_negative_processed_chunks(self):
        """测试验证负数已处理块数"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, processed_chunks=-1)

    def test_validation_negative_failed_chunks(self):
        """测试验证负数失败块数"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, failed_chunks=-1)

    def test_validation_negative_partial_chunks(self):
        """测试验证负数部分完成块数"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, partial_chunks=-1)

    def test_validation_negative_merge_level(self):
        """测试验证负数合并层级"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, merge_level=-1)

    def test_validation_negative_merge_batch_current(self):
        """测试验证负数当前批次"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, merge_batch_current=-1)

    def test_validation_negative_merge_batch_total(self):
        """测试验证负数总批次"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, merge_batch_total=-1)

    def test_validation_negative_merge_outlines_count(self):
        """测试验证负数大纲数量"""
        import pytest

        with pytest.raises(ValueError):
            ProcessingState(file_path="test.txt", total_chunks=10, merge_outlines_count=-1)


class TestProgressDataEdgeCases:
    """ProgressData 边界情况测试"""

    def test_from_dict_missing_fields(self):
        """测试从字典创建时缺少字段"""
        data = {"txt_file": "test.txt"}
        progress = ProgressData.from_dict(data)
        assert progress.txt_file == "test.txt"
        assert progress.total_chunks == 0
        assert progress.completed_indices == set()

    def test_from_dict_invalid_datetime(self):
        """测试从字典创建时无效日期时间"""
        data = {
            "txt_file": "test.txt",
            "total_chunks": 10,
            "completed_indices": [],
            "outlines": [],
            "last_update": "invalid-datetime-format",
            "chunks_hash": "abc",
        }
        progress = ProgressData.from_dict(data)
        assert isinstance(progress.last_update, datetime)

    def test_to_dict_sorted_completed_indices(self):
        """测试转换字典时排序已完成索引"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_indices={5, 2, 8, 1},
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc",
        )
        data = progress.to_dict()
        assert data["completed_indices"] == [1, 2, 5, 8]

    def test_to_dict_sorted_partial_indices(self):
        """测试转换字典时排序部分完成索引"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc",
            partial_indices={3, 1, 2},
        )
        data = progress.to_dict()
        assert data["partial_indices"] == [1, 2, 3]

    def test_completion_rate_zero_total(self):
        """测试总块数为零时的完成率"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc",
        )
        assert progress.completion_rate == 0.0

    def test_average_processing_time_empty(self):
        """测试空处理时间列表的平均值"""
        progress = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc",
            processing_times=[],
        )
        assert progress.average_processing_time == 0.0
