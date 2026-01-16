"""测试部分完成功能"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import ProcessingState, ProgressData, TextChunk


class TestSplitChunkIntoFive:
    """测试将块拆分为5个小块"""

    def test_split_chunk_into_five_equal_division(self):
        """测试平均拆分为5个小块"""
        from services.novel_processing_service import NovelProcessingService

        # 创建一个500字符的块
        text = "A" * 500
        chunk = TextChunk(id=1, content=text, token_count=500, start_position=0, end_position=500)

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            sub_chunks = service._split_chunk_into_sub_chunks(chunk)

            assert len(sub_chunks) == 5

            # 检查每个小块的基本属性
            for i, sub_chunk in enumerate(sub_chunks):
                assert sub_chunk.id == 1  # 保持原始chunk_id
                assert sub_chunk.start_position == i * 100
                assert sub_chunk.end_position == (i + 1) * 100 if i < 4 else 500

            # 检查内容（前4个小块各100字符，最后一个小块100字符）
            assert len(sub_chunks[0].content) == 100
            assert len(sub_chunks[1].content) == 100
            assert len(sub_chunks[2].content) == 100
            assert len(sub_chunks[3].content) == 100
            assert len(sub_chunks[4].content) == 100

    def test_split_chunk_into_five_rounding(self):
        """测试非均匀分割时的四舍五入处理"""
        from services.novel_processing_service import NovelProcessingService

        # 创建一个503字符的块（不能被5整除）
        text = "B" * 503
        chunk = TextChunk(id=2, content=text, token_count=503, start_position=0, end_position=503)

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            sub_chunks = service._split_chunk_into_sub_chunks(chunk)

            assert len(sub_chunks) == 5

            # 前4个小块各100字符，最后一个小块103字符
            assert len(sub_chunks[0].content) == 100
            assert len(sub_chunks[1].content) == 100
            assert len(sub_chunks[2].content) == 100
            assert len(sub_chunks[3].content) == 100
            assert len(sub_chunks[4].content) == 103

    def test_split_chunk_into_five_preserves_token_count(self):
        """测试拆分后token计数正确"""
        from services.novel_processing_service import NovelProcessingService
        from tokenizer import count_tokens

        text = "Hello world. " * 50
        chunk = TextChunk(
            id=3,
            content=text,
            token_count=count_tokens(text),
            start_position=0,
            end_position=len(text),
        )

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            sub_chunks = service._split_chunk_into_sub_chunks(chunk)

            total_sub_tokens = sum(sc.token_count for sc in sub_chunks)
            # 允许一些误差，因为token计数可能有边界效应
            assert abs(total_sub_tokens - chunk.token_count) < 10

    def test_split_chunk_into_five_last_chunk_gets_remainder(self):
        """测试最后一个小块包含剩余内容"""
        from services.novel_processing_service import NovelProcessingService

        text = "C" * 21
        chunk = TextChunk(id=4, content=text, token_count=21, start_position=0, end_position=21)

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            sub_chunks = service._split_chunk_into_sub_chunks(chunk)

            assert len(sub_chunks) == 5
            # 21 / 5 = 4.2，前4个各4字符，最后1个5字符
            assert len(sub_chunks[0].content) == 4
            assert len(sub_chunks[1].content) == 4
            assert len(sub_chunks[2].content) == 4
            assert len(sub_chunks[3].content) == 4
            assert len(sub_chunks[4].content) == 5


class TestMergePartialOutlines:
    """测试合并部分完成的小块大纲"""

    def test_merge_partial_outlines_basic(self):
        """测试基本合并"""
        from services.novel_processing_service import NovelProcessingService

        outline1 = {
            "plot": ["event1", "event2"],
            "characters": ["Alice", "Bob"],
            "relationships": [["Alice", "Bob", "friend"]],
        }

        outline2 = {
            "plot": ["event3", "event4"],
            "characters": ["Charlie", "Bob"],
            "relationships": [["Bob", "Charlie", "colleague"]],
        }

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            merged = service._merge_partial_outlines([outline1, outline2], 1)

            # 检查合并后的事件
            assert set(merged["plot"]) == {"event1", "event2", "event3", "event4"}

            # 检查合并后的人物（去重）
            assert set(merged["characters"]) == {"Alice", "Bob", "Charlie"}

            # 检查合并后的关系
            assert len(merged["relationships"]) == 2
            assert ["Alice", "Bob", "friend"] in merged["relationships"]
            assert ["Bob", "Charlie", "colleague"] in merged["relationships"]

    def test_merge_partial_outlines_with_partial_flag(self):
        """测试合并后设置is_partial标记"""
        from services.novel_processing_service import NovelProcessingService

        outlines = [
            {"plot": ["event1"], "characters": [], "relationships": []},
            {"plot": ["event2"], "characters": [], "relationships": []},
        ]

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            merged = service._merge_partial_outlines(outlines, 1)

            assert merged["is_partial"] is True
            assert merged["chunk_id"] == 1
            assert merged["sub_chunk_count"] == 2

    def test_merge_partial_outlines_preserves_partial_outlines(self):
        """测试合并后保留原始小块大纲"""
        from services.novel_processing_service import NovelProcessingService

        outline1 = {"plot": ["event1"], "characters": [], "relationships": []}
        outline2 = {"plot": ["event2"], "characters": [], "relationships": []}

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            merged = service._merge_partial_outlines([outline1, outline2], 1)

            assert "partial_outlines" in merged
            assert len(merged["partial_outlines"]) == 2
            assert merged["partial_outlines"][0] == outline1
            assert merged["partial_outlines"][1] == outline2

    def test_merge_partial_outlines_merges_raw_responses(self):
        """测试合并原始响应"""
        from services.novel_processing_service import NovelProcessingService

        outline1 = {
            "plot": ["event1"],
            "raw_response": "Response 1",
            "characters": [],
            "relationships": [],
        }
        outline2 = {
            "plot": ["event2"],
            "raw_response": "Response 2",
            "characters": [],
            "relationships": [],
        }

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            merged = service._merge_partial_outlines([outline1, outline2], 1)

            assert "raw_response" in merged
            assert "Response 1" in merged["raw_response"]
            assert "Response 2" in merged["raw_response"]

    def test_merge_partial_outlines_averages_processing_time(self):
        """测试平均处理时间"""
        from services.novel_processing_service import NovelProcessingService

        outline1 = {
            "plot": ["event1"],
            "processing_time": 1.0,
            "characters": [],
            "relationships": [],
        }
        outline2 = {
            "plot": ["event2"],
            "processing_time": 3.0,
            "characters": [],
            "relationships": [],
        }

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            merged = service._merge_partial_outlines([outline1, outline2], 1)

            assert "processing_time" in merged
            assert merged["processing_time"] == 2.0  # (1.0 + 3.0) / 2

    def test_merge_partial_outlines_handles_missing_fields(self):
        """测试处理缺失字段"""
        from services.novel_processing_service import NovelProcessingService

        outline1 = {"plot": ["event1"], "characters": [], "relationships": []}
        # outline2 缺少 processing_time
        outline2 = {"plot": ["event2"], "characters": [], "relationships": []}

        # Mock the llm_service to avoid API key validation
        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            merged = service._merge_partial_outlines([outline1, outline2], 1)

            # 仍应正常合并
            assert set(merged["plot"]) == {"event1", "event2"}
            # processing_time 可能不存在或为某个默认值
            # 不抛异常即通过


class TestProcessFailingChunkAsPartial:
    """测试部分完成处理逻辑"""

    def create_mock_service(self):
        """Helper to create a mock service without API key issues"""
        from services.novel_processing_service import NovelProcessingService

        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )

            # 模拟必要的依赖
            mock_llm = MagicMock()
            mock_llm.call = AsyncMock()
            service.llm_service = mock_llm

            # 模拟进度服务
            mock_progress = MagicMock()
            mock_progress.update_chunk_completed = MagicMock()
            mock_progress.add_progress_error = MagicMock()
            service.progress_service = mock_progress

            # 模拟ETA估算器
            service.eta_estimator = MagicMock()
            service.eta_estimator.add_completion = MagicMock()

            # 模拟回调
            service.progress_callback = MagicMock()
            service.cancel_event = MagicMock()
            service.cancel_event.is_set.return_value = False

            # 模拟处理状态
            service.processing_state = ProcessingState(file_path="test.txt", total_chunks=10)
            service.total_prompt_tokens = 0
            service.total_completion_tokens = 0
            service.total_tokens = 0

            return service

    @pytest.mark.asyncio
    async def test_process_failing_chunk_all_sub_chunks_success(self):
        """测试所有小块都成功"""
        chunk = TextChunk(
            id=1, content="Test content" * 100, token_count=100, start_position=0, end_position=1000
        )

        service = self.create_mock_service()

        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.content = '{"plot": ["test"], "characters": [], "relationships": []}'
        mock_response.token_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        service.llm_service.call.return_value = mock_response

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )

        result = await service._process_failing_chunk_as_partial(
            chunk, MagicMock(), progress_data, service.processing_state
        )

        # 应该返回5个成功的小块大纲
        assert len(result) == 5

        # 部分完成状态应该被更新
        assert 1 in progress_data.partial_indices
        assert len(progress_data.partial_outlines) == 5

        # 部分完成计数器应该增加
        assert service.processing_state.partial_chunks == 1

        # 进度不应该被更新（部分完成不计入processed_chunks）
        assert service.processing_state.processed_chunks == 0

    @pytest.mark.asyncio
    async def test_process_failing_chunk_some_sub_chunks_fail(self):
        """测试部分小块失败"""
        chunk = TextChunk(
            id=2, content="Test content" * 100, token_count=100, start_position=0, end_position=1000
        )

        service = self.create_mock_service()

        # 前3次成功，后2次失败
        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                mock_response = MagicMock()
                mock_response.content = '{"plot": ["test"], "characters": [], "relationships": []}'
                mock_response.token_usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
                return mock_response
            else:
                from exceptions import APIError

                raise APIError("Simulated API error")

        service.llm_service.call = AsyncMock(side_effect=mock_call)

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )

        result = await service._process_failing_chunk_as_partial(
            chunk, MagicMock(), progress_data, service.processing_state
        )

        # 应该返回3个成功的小块大纲
        assert len(result) == 3

        # 部分完成状态应该被更新
        assert 2 in progress_data.partial_indices
        assert len(progress_data.partial_outlines) == 3

    @pytest.mark.asyncio
    async def test_process_failing_chunk_all_sub_chunks_fail(self):
        """测试所有小块都失败"""
        chunk = TextChunk(
            id=3, content="Test content" * 100, token_count=100, start_position=0, end_position=1000
        )

        service = self.create_mock_service()

        # 模拟所有调用都失败
        from exceptions import APIError

        service.llm_service.call = AsyncMock(side_effect=APIError("All failed"))

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )

        result = await service._process_failing_chunk_as_partial(
            chunk, MagicMock(), progress_data, service.processing_state
        )

        # 应该返回空列表
        assert len(result) == 0

        # 不应该有部分完成记录
        assert 3 not in progress_data.partial_indices

    @pytest.mark.asyncio
    async def test_process_failing_chunk_updates_token_usage(self):
        """测试token使用统计"""
        chunk = TextChunk(
            id=4, content="Test content" * 100, token_count=100, start_position=0, end_position=1000
        )

        service = self.create_mock_service()

        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.content = '{"plot": ["test"], "characters": [], "relationships": []}'
        mock_response.token_usage = {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }
        service.llm_service.call.return_value = mock_response

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )

        await service._process_failing_chunk_as_partial(
            chunk, MagicMock(), progress_data, service.processing_state
        )

        # Token使用应该被累计（5个小块 * 300 tokens）
        assert service.total_prompt_tokens == 1000
        assert service.total_completion_tokens == 500
        assert service.total_tokens == 1500

    @pytest.mark.asyncio
    async def test_process_failing_chunk_emits_progress(self):
        """测试进度回调"""
        chunk = TextChunk(
            id=5, content="Test content" * 100, token_count=100, start_position=0, end_position=1000
        )

        service = self.create_mock_service()

        # 模拟LLM响应
        mock_response = MagicMock()
        mock_response.content = '{"plot": ["test"], "characters": [], "relationships": []}'
        mock_response.token_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        service.llm_service.call.return_value = mock_response

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=10,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="abc123",
        )

        await service._process_failing_chunk_as_partial(
            chunk, MagicMock(), progress_data, service.processing_state
        )

        # 进度回调应该被调用
        service.progress_callback.assert_called()
        # 检查最后一次调用包含部分完成信息
        last_call_args = service.progress_callback.call_args[0][0]
        assert "partial_info" in last_call_args
        assert "5块部分完成" in last_call_args["partial_info"]

    def test_emit_progress_partial_weight(self):
        """测试部分完成权重参与进度计算"""
        from services.novel_processing_service import NovelProcessingService

        with patch("services.llm_service.create_llm_service") as mock_create:
            mock_create.return_value = MagicMock()

            service = NovelProcessingService(
                progress_callback=MagicMock(), cancel_event=MagicMock()
            )
            service.processing_state = ProcessingState(file_path="test.txt", total_chunks=10)
            service.processing_state.processed_chunks = 2
            service.processing_state.partial_chunks = 1

            service.eta_estimator = MagicMock()
            service.eta_estimator.estimate.return_value = {
                "eta_seconds": None,
                "confidence": None,
                "method": None,
            }

            progress_data = ProgressData(
                txt_file="test.txt",
                total_chunks=10,
                completed_count=2,
                completed_indices={1, 2},
                outlines=[],
                last_update=datetime.now(),
                chunks_hash="abc123",
            )
            progress_data.partial_indices.add(5)
            progress_data.partial_outlines = [
                {"original_chunk_id": 5},
                {"original_chunk_id": 5},
                {"original_chunk_id": 5},
            ]
            service.current_progress_data = progress_data

            service._emit_progress()

            last_call_args = service.progress_callback.call_args[0][0]
            assert last_call_args["progress"] == pytest.approx((2 + 3 / 5) / 10)
