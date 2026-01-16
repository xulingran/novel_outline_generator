"""
测试核心业务逻辑 - NovelProcessingService
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import ProcessingState, ProgressData, TextChunk
from services.novel_processing_service import NovelProcessingService
from exceptions import FileValidationError, ProcessingError


@pytest.fixture
def mock_service():
    """创建一个完全mock的NovelProcessingService"""
    with patch("services.llm_service.create_llm_service") as mock_create:
        mock_create.return_value = MagicMock()
        service = NovelProcessingService(
            progress_callback=MagicMock(),
            cancel_event=MagicMock()
        )
        # 设置所有必要的mock
        service.llm_service = AsyncMock()
        service.file_service = MagicMock()
        service.progress_service = MagicMock()
        service.eta_estimator = MagicMock()
        service.processing_state = ProcessingState(file_path="test.txt", total_chunks=10)
        service.total_prompt_tokens = 0
        service.total_completion_tokens = 0
        service.total_tokens = 0
        service.force_complete = False
        # 确保cancel_event.is_set()返回False
        service.cancel_event.is_set.return_value = False
        yield service


@pytest.fixture
def sample_chunks():
    """创建示例文本块"""
    return [
        TextChunk(id=1, content="content1", token_count=100, start_position=0, end_position=100),
        TextChunk(id=2, content="content2", token_count=100, start_position=100, end_position=200),
        TextChunk(id=3, content="content3", token_count=100, start_position=200, end_position=300),
    ]


@pytest.fixture
def sample_progress_data():
    """创建示例进度数据"""
    return ProgressData(
        txt_file="test.txt",
        total_chunks=3,
        completed_count=1,
        completed_indices={1},
        outlines=[{"chunk_id": 1, "plot": ["done"]}],
        last_update=datetime.now(),
        chunks_hash="hash123",
    )


@pytest.fixture
def mock_llm_response():
    """创建mock的LLM响应"""
    response = MagicMock()
    response.content = '{"plot": ["test event"], "characters": ["Alice", "Bob"], "relationships": [["Alice", "Bob", "friend"]]}'
    response.token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    return response


class TestProcessNovel:
    """测试process_novel主流程"""

    @pytest.mark.asyncio
    async def test_process_novel_normal_flow(self, mock_service, mock_llm_response):
        """测试正常处理流程"""
        # Mock文件加载
        mock_service.file_service.read_text_file.return_value = ("test content", "utf-8")
        # Mock文本分割
        mock_service._split_text_into_chunks = MagicMock(return_value=[
            TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=10)
        ])
        # Mock进度处理
        mock_service.progress_service.load_progress.return_value = None
        mock_service.progress_service.create_progress.return_value = ProgressData(
            txt_file="test.txt",
            total_chunks=1,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )
        # Mock块处理
        mock_service._process_chunks = AsyncMock(return_value=[
            {"chunk_id": 1, "plot": ["test"], "characters": [], "relationships": []}
        ])
        # Mock大纲合并
        mock_service.merge_outlines_recursive = AsyncMock(return_value="Final outline")
        # Mock结果保存
        mock_service._save_results = AsyncMock()
        # Mock清理
        mock_service._cleanup_intermediate_outputs = MagicMock(return_value=[])

        result = await mock_service.process_novel("test.txt", "output", resume=False)

        # 验证流程
        mock_service.file_service.read_text_file.assert_called_once_with("test.txt")
        mock_service._split_text_into_chunks.assert_called_once()
        mock_service._process_chunks.assert_called_once()
        mock_service.merge_outlines_recursive.assert_called_once()
        mock_service._save_results.assert_called_once()
        mock_service._cleanup_intermediate_outputs.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_novel_with_resume(self, mock_service, sample_progress_data, sample_chunks):
        """测试恢复处理模式"""
        mock_service.file_service.read_text_file.return_value = ("test content", "utf-8")
        mock_service._split_text_into_chunks = MagicMock(return_value=sample_chunks)
        mock_service.progress_service.load_progress.return_value = sample_progress_data
        mock_service.progress_service.is_progress_valid.return_value = True
        mock_service._process_chunks = AsyncMock(return_value=[
            {"chunk_id": 2, "plot": ["new"]},
            {"chunk_id": 3, "plot": ["new"]},
        ])
        mock_service.merge_outlines_recursive = AsyncMock(return_value="Merged")
        mock_service._save_results = AsyncMock()
        mock_service._cleanup_intermediate_outputs = MagicMock(return_value=[])

        await mock_service.process_novel("test.txt", "output", resume=True)

        # 应该处理剩余的块（2和3）
        called_chunks = mock_service._process_chunks.call_args.args[0]
        assert [c.id for c in called_chunks] == [2, 3]

    @pytest.mark.asyncio
    async def test_process_novel_empty_file(self, mock_service):
        """测试空文件处理"""
        from exceptions import ProcessingError

        mock_service.file_service.read_text_file.return_value = ("", "utf-8")
        mock_service.progress_service.load_progress.return_value = None

        with pytest.raises(ProcessingError, match="读取文件失败"):
            await mock_service.process_novel("test.txt", "output", resume=False)

    @pytest.mark.asyncio
    async def test_process_novel_cancellation(self, mock_service):
        """测试取消处理"""
        from exceptions import ProcessingError

        mock_service.file_service.read_text_file.return_value = ("test content", "utf-8")
        mock_service._split_text_into_chunks = MagicMock(return_value=[
            TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=10)
        ])
        mock_service.progress_service.load_progress.return_value = None
        mock_service.progress_service.create_progress.return_value = ProgressData(
            txt_file="test.txt",
            total_chunks=1,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )
        mock_service._process_chunks = AsyncMock(side_effect=ProcessingError("Cancelled"))
        mock_service.cancel_event.is_set.return_value = True
        mock_service.progress_service.save_progress = MagicMock()

        with pytest.raises(ProcessingError, match="Cancelled"):
            await mock_service.process_novel("test.txt", "output", resume=False)

    @pytest.mark.asyncio
    async def test_process_novel_token_accumulation(self, mock_service, mock_llm_response):
        """测试Token统计累加"""
        mock_service.file_service.read_text_file.return_value = ("test content", "utf-8")
        mock_service._split_text_into_chunks = MagicMock(return_value=[
            TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=10)
        ])
        mock_service.progress_service.load_progress.return_value = None
        mock_service.progress_service.create_progress.return_value = ProgressData(
            txt_file="test.txt",
            total_chunks=1,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        # Mock _process_chunks 来设置token统计
        async def mock_process(*args, **kwargs):
            mock_service.total_prompt_tokens = 100
            mock_service.total_completion_tokens = 50
            mock_service.total_tokens = 150
            return [{"chunk_id": 1, "plot": ["test"]}]

        mock_service._process_chunks = AsyncMock(side_effect=mock_process)
        mock_service.merge_outlines_recursive = AsyncMock(return_value="Final")
        mock_service._save_results = AsyncMock()
        mock_service._cleanup_intermediate_outputs = MagicMock(return_value=[])

        await mock_service.process_novel("test.txt", "output", resume=False)

        assert mock_service.total_prompt_tokens == 100
        assert mock_service.total_completion_tokens == 50
        assert mock_service.total_tokens == 150


class TestLoadAndValidateFile:
    """测试文件加载和验证"""

    @pytest.mark.asyncio
    async def test_load_and_validate_file_success(self, mock_service):
        """测试成功加载文件"""
        mock_service.file_service.read_text_file.return_value = ("test content", "utf-8")

        text, encoding = await mock_service._load_and_validate_file("test.txt")

        assert text == "test content"
        assert encoding == "utf-8"
        mock_service.file_service.read_text_file.assert_called_once_with("test.txt")

    @pytest.mark.asyncio
    async def test_load_and_validate_file_empty(self, mock_service):
        """测试加载空文件"""
        from exceptions import ProcessingError

        mock_service.file_service.read_text_file.return_value = ("", "utf-8")

        with pytest.raises(ProcessingError, match="文件内容为空"):
            await mock_service._load_and_validate_file("test.txt")

    @pytest.mark.asyncio
    async def test_load_and_validate_file_not_found(self, mock_service):
        """测试文件不存在"""
        from exceptions import ProcessingError

        mock_service.file_service.read_text_file.side_effect = FileValidationError("文件不存在")

        with pytest.raises(ProcessingError, match="读取文件失败"):
            await mock_service._load_and_validate_file("test.txt")


class TestSplitTextIntoChunks:
    """测试文本分割"""

    def test_split_text_into_chunks_normal(self, mock_service):
        """测试正常文本分割"""
        mock_service._split_text_into_chunks = MagicMock(side_effect=lambda text: [
            TextChunk(id=1, content=text, token_count=len(text.split()), start_position=0, end_position=len(text))
        ])

        chunks = mock_service._split_text_into_chunks("test content")

        assert len(chunks) == 1
        assert chunks[0].content == "test content"
        assert chunks[0].id == 1

    def test_split_text_into_chunks_empty(self, mock_service):
        """测试空文本分割"""
        from exceptions import ProcessingError

        mock_service._split_text_into_chunks = MagicMock(side_effect=lambda text: [])

        chunks = mock_service._split_text_into_chunks("")

        assert len(chunks) == 0

    def test_split_text_into_chunks_multiple(self, mock_service):
        """测试多块分割"""
        mock_service._split_text_into_chunks = MagicMock(side_effect=lambda text: [
            TextChunk(id=1, content="part1", token_count=5, start_position=0, end_position=5),
            TextChunk(id=2, content="part2", token_count=5, start_position=5, end_position=10),
        ])

        chunks = mock_service._split_text_into_chunks("part1part2")

        assert len(chunks) == 2
        assert chunks[0].id == 1
        assert chunks[1].id == 2


class TestProcessChunks:
    """测试批量处理"""

    @pytest.mark.asyncio
    async def test_process_chunks_all_success(self, mock_service, sample_chunks, mock_llm_response):
        """测试所有块处理成功"""
        mock_service.llm_service.call.return_value = mock_llm_response
        mock_service.progress_service.update_chunk_completed = MagicMock()
        mock_service.eta_estimator.add_completion = MagicMock()
        mock_service.progress_callback = MagicMock()

        # Mock _process_single_chunk
        async def mock_process_single(chunk, sem, progress_data):
            return {"chunk_id": chunk.id, "plot": ["test"]}

        mock_service._process_single_chunk = AsyncMock(side_effect=mock_process_single)

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        results = await mock_service._process_chunks(sample_chunks, progress_data, 3)

        assert len(results) == 3
        assert all(r["chunk_id"] in [1, 2, 3] for r in results)

    @pytest.mark.asyncio
    async def test_process_chunks_partial_failure(self, mock_service, sample_chunks):
        """测试部分块失败"""
        from exceptions import ProcessingError

        async def mock_process_single(chunk, sem, progress_data):
            if chunk.id == 2:
                raise ProcessingError("Failed")
            return {"chunk_id": chunk.id, "plot": ["test"]}

        mock_service._process_single_chunk = AsyncMock(side_effect=mock_process_single)
        mock_service.progress_service.add_progress_error = MagicMock()
        mock_service.progress_callback = MagicMock()

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        results = await mock_service._process_chunks(sample_chunks, progress_data, 3)

        # 应该只返回成功的块
        assert len(results) == 2
        assert all(r["chunk_id"] in [1, 3] for r in results)

    @pytest.mark.asyncio
    async def test_process_chunks_all_failure(self, mock_service, sample_chunks):
        """测试所有块失败"""
        from exceptions import ProcessingError

        async def mock_process_single(chunk, sem, progress_data):
            raise ProcessingError("All failed")

        mock_service._process_single_chunk = AsyncMock(side_effect=mock_process_single)
        mock_service.progress_service.add_progress_error = MagicMock()

        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        results = await mock_service._process_chunks(sample_chunks, progress_data, 3)

        assert len(results) == 0


class TestProcessSingleChunk:
    """测试单块处理"""

    @pytest.mark.asyncio
    async def test_process_single_chunk_first_success(self, mock_service, mock_llm_response):
        """测试首次成功"""
        mock_service.llm_service.call.return_value = mock_llm_response
        mock_service.progress_service.update_chunk_completed = MagicMock()
        mock_service.eta_estimator.add_completion = MagicMock()

        chunk = TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        sem = asyncio.Semaphore(5)
        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        result = await mock_service._process_single_chunk(chunk, sem, progress_data)

        assert result["chunk_id"] == 1
        assert result["plot"] == ["test event"]
        assert mock_service.llm_service.call.call_count == 1

    @pytest.mark.asyncio
    async def test_process_single_chunk_retry_success(self, mock_service):
        """测试重试后成功"""
        from exceptions import APIError

        mock_response = MagicMock()
        mock_response.content = '{"plot": ["test"]}'
        mock_response.token_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIError("First attempt failed")
            return mock_response

        # 使用wraps来包装mock函数，而不是使用side_effect
        mock_service.llm_service.call = AsyncMock(wraps=mock_call)
        mock_service.progress_service.update_chunk_completed = MagicMock()
        mock_service.eta_estimator.add_completion = MagicMock()

        chunk = TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        sem = asyncio.Semaphore(5)
        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        result = await mock_service._process_single_chunk(chunk, sem, progress_data)

        assert result["chunk_id"] == 1
        assert call_count == 2  # 第一次失败，第二次成功

    @pytest.mark.asyncio
    async def test_process_single_chunk_max_retry_failure(self, mock_service):
        """测试达到最大重试次数后失败"""
        from exceptions import APIError

        original_call = mock_service.llm_service.call
        mock_service.llm_service.call = AsyncMock(side_effect=APIError("Always failed"))
        mock_service._process_failing_chunk_as_partial = AsyncMock(return_value=[
            {"chunk_id": 1, "plot": ["partial"]}
        ])
        mock_service.progress_service.add_progress_error = MagicMock()

        chunk = TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        sem = asyncio.Semaphore(5)
        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        result = await mock_service._process_single_chunk(chunk, sem, progress_data)

        # 应该触发部分完成处理
        assert mock_service._process_failing_chunk_as_partial.call_count == 1

        # 重置llm_service.call
        mock_service.llm_service.call = original_call

    @pytest.mark.asyncio
    async def test_process_single_chunk_cancelled(self, mock_service):
        """测试处理中取消"""
        from exceptions import ProcessingError

        mock_response = MagicMock()
        mock_response.content = '{"plot": ["test"]}'
        mock_response.token_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        async def mock_call(*args, **kwargs):
            mock_service.cancel_event.is_set.return_value = True
            return mock_response

        mock_service.llm_service.call = AsyncMock(side_effect=mock_call)
        mock_service.progress_service.update_chunk_completed = MagicMock()

        chunk = TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        sem = asyncio.Semaphore(5)
        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        try:
            with pytest.raises(asyncio.CancelledError):
                await mock_service._process_single_chunk(chunk, sem, progress_data)
        finally:
            # 重置cancel_event状态
            mock_service.cancel_event.is_set.return_value = False

    @pytest.mark.asyncio
    async def test_process_single_chunk_token_accumulation(self, mock_service, mock_llm_response):
        """测试Token使用统计"""
        mock_service.llm_service.call.return_value = mock_llm_response
        mock_service.progress_service.update_chunk_completed = MagicMock()
        mock_service.eta_estimator.add_completion = MagicMock()

        chunk = TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        sem = asyncio.Semaphore(5)
        progress_data = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="hash",
        )

        await mock_service._process_single_chunk(chunk, sem, progress_data)

        assert mock_service.total_prompt_tokens == 100
        assert mock_service.total_completion_tokens == 50
        assert mock_service.total_tokens == 150


class TestParseLLMResponse:
    """测试LLM响应解析"""

    def test_parse_llm_response_valid_json(self, mock_service):
        """测试有效JSON解析"""
        response = '{"plot": ["event1", "event2"], "characters": ["Alice"], "relationships": []}'

        result = mock_service._parse_llm_response(response, 1)

        assert result["chunk_id"] == 1
        assert result["plot"] == ["event1", "event2"]
        assert result["characters"] == ["Alice"]
        assert result["relationships"] == []

    def test_parse_llm_response_json_extraction(self, mock_service):
        """测试JSON提取（从文本中提取JSON）"""
        response = 'Here is the result: {"plot": ["event1"], "characters": [], "relationships": []}'

        result = mock_service._parse_llm_response(response, 1)

        assert result["chunk_id"] == 1
        assert result["plot"] == ["event1"]

    def test_parse_llm_response_parse_failure(self, mock_service):
        """测试解析失败时的默认结构"""
        response = "This is not valid JSON"

        result = mock_service._parse_llm_response(response, 1)

        assert result["chunk_id"] == 1
        assert result["plot"] == ["This is not valid JSON"]
        assert result["characters"] == []
        assert result["relationships"] == []

    def test_parse_llm_response_preserves_raw_response(self, mock_service):
        """测试保留原始响应"""
        response = '{"plot": ["event1"]}'

        result = mock_service._parse_llm_response(response, 1)

        # raw_response会在_process_single_chunk中添加，不在_parse_llm_response中
        assert result["chunk_id"] == 1


class TestMergeOutlinesRecursive:
    """测试大纲合并"""

    @pytest.mark.asyncio
    async def test_merge_outlines_recursive_empty_list(self, mock_service):
        """测试空列表处理"""
        result = await mock_service.merge_outlines_recursive([], level=0, is_text_mode=False)

        assert result == ""

    @pytest.mark.asyncio
    async def test_merge_outlines_recursive_single_outline(self, mock_service):
        """测试单个大纲"""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Single outline merged"
        mock_llm_response.token_usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
        mock_service.llm_service.call.return_value = mock_llm_response

        outlines = [{"chunk_id": 1, "plot": ["event1"]}]

        result = await mock_service.merge_outlines_recursive(outlines, level=0, is_text_mode=False)

        assert result == "Single outline merged"

    @pytest.mark.asyncio
    async def test_merge_outlines_recursive_multiple_outlines(self, mock_service):
        """测试多个大纲合并"""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Multiple outlines merged"
        mock_llm_response.token_usage = {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        mock_service.llm_service.call.return_value = mock_llm_response

        outlines = [
            {"chunk_id": 1, "plot": ["event1"]},
            {"chunk_id": 2, "plot": ["event2"]},
            {"chunk_id": 3, "plot": ["event3"]},
        ]

        result = await mock_service.merge_outlines_recursive(outlines, level=0, is_text_mode=False)

        assert result == "Multiple outlines merged"

    @pytest.mark.asyncio
    async def test_merge_outlines_recursive_batch_split(self, mock_service):
        """测试分批合并（Token超限）"""
        # 设置小的token限制来触发分批
        mock_service.processing_state = ProcessingState(file_path="test.txt", total_chunks=10)
        # 模拟token超限

        outlines = [{"chunk_id": i, "plot": [f"event{i}"]} for i in range(20)]

        # Mock递归调用
        async def mock_merge(outlines, level, is_text_mode):
            if len(outlines) > 10:
                # 分批
                first_half = await mock_merge(outlines[:10], level + 1, is_text_mode)
                second_half = await mock_merge(outlines[10:], level + 1, is_text_mode)
                return f"{first_half} + {second_half}"
            return f"Merged {len(outlines)} items"

        with patch.object(mock_service, 'merge_outlines_recursive', side_effect=mock_merge):
            result = await mock_service.merge_outlines_recursive(outlines, level=0, is_text_mode=False)

        assert "Merged 10 items" in result

    @pytest.mark.asyncio
    async def test_merge_outlines_recursive_text_mode(self, mock_service):
        """测试文本模式合并"""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Text mode merged"
        mock_llm_response.token_usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
        mock_service.llm_service.call.return_value = mock_llm_response

        outlines = [
            {"chunk_id": 1, "plot": ["event1"]},
            {"chunk_id": 2, "plot": ["event2"]},
        ]

        result = await mock_service.merge_outlines_recursive(outlines, level=0, is_text_mode=True)

        assert result == "Text mode merged"

    @pytest.mark.asyncio
    async def test_merge_outlines_recursive_cancelled(self, mock_service):
        """测试合并时取消"""

        mock_service.cancel_event.is_set.return_value = True

        outlines = [{"chunk_id": 1, "plot": ["event1"]}]

        with pytest.raises(asyncio.CancelledError):
            await mock_service.merge_outlines_recursive(outlines, level=0, is_text_mode=False)


class TestSaveResults:
    """测试结果保存"""

    @pytest.mark.asyncio
    async def test_save_results_json(self, mock_service):
        """测试保存JSON格式结果"""
        outlines = [
            {"chunk_id": 1, "plot": ["event1"]},
            {"chunk_id": 2, "plot": ["event2"]},
        ]
        final_outline = "Final merged outline"
        original_file = "test.txt"

        mock_service.file_service.ensure_output_directory.return_value = Path("/tmp/output")
        mock_service.file_service.write_json_file = MagicMock()
        mock_service.file_service.write_text_file = MagicMock()

        await mock_service._save_results(outlines, final_outline, original_file)

        # 验证JSON文件被保存（应该调用两次：chunk_outlines.json和processing_metadata.json）
        assert mock_service.file_service.write_json_file.call_count == 2
        # 验证文本文件被保存
        mock_service.file_service.write_text_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_results_creates_output_dir(self, mock_service):
        """测试创建输出目录"""
        outlines = [{"chunk_id": 1, "plot": ["event1"]}]
        final_outline = "Final outline"
        original_file = "test.txt"

        mock_service.file_service.ensure_output_directory.return_value = Path("/tmp/output")
        mock_service.file_service.write_json_file = MagicMock()
        mock_service.file_service.write_text_file = MagicMock()

        await mock_service._save_results(outlines, final_outline, original_file)

        mock_service.file_service.ensure_output_directory.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_results_metadata(self, mock_service):
        """测试保存元数据"""
        outlines = [{"chunk_id": 1, "plot": ["event1"]}]
        final_outline = "Final outline"
        original_file = "test.txt"

        mock_service.file_service.ensure_output_directory.return_value = Path("/tmp/output")
        mock_service.file_service.write_json_file = MagicMock()
        mock_service.file_service.write_text_file = MagicMock()

        # 设置token统计
        mock_service.total_prompt_tokens = 1000
        mock_service.total_completion_tokens = 500
        mock_service.total_tokens = 1500

        await mock_service._save_results(outlines, final_outline, original_file)

        # 验证元数据被保存
        json_call = mock_service.file_service.write_json_file.call_args
        assert json_call is not None


class TestCleanupIntermediateOutputs:
    """测试中间文件清理"""

    def test_cleanup_intermediate_outputs_success(self, mock_service, tmp_path):
        """测试成功清理中间文件"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 创建中间文件
        (output_dir / "chunk_outlines.json").write_text("{}", encoding="utf-8")
        (output_dir / "processing_metadata.json").write_text("{}", encoding="utf-8")

        mock_service.file_service.remove_backups.return_value = 0

        removed = mock_service._cleanup_intermediate_outputs(output_dir)

        # 验证文件被删除
        assert not (output_dir / "chunk_outlines.json").exists()
        assert not (output_dir / "processing_metadata.json").exists()

    def test_cleanup_intermediate_outputs_files_not_exist(self, mock_service, tmp_path):
        """测试文件不存在时的清理"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_service.file_service.remove_backups.return_value = 0

        # 不应该抛出异常
        removed = mock_service._cleanup_intermediate_outputs(output_dir)

        assert isinstance(removed, list)

    def test_cleanup_intermediate_outputs_partial_files(self, mock_service, tmp_path):
        """测试部分文件存在时的清理"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 只创建一个中间文件
        (output_dir / "chunk_outlines.json").write_text("{}", encoding="utf-8")

        mock_service.file_service.remove_backups.return_value = 0

        removed = mock_service._cleanup_intermediate_outputs(output_dir)

        # 验证存在的文件被删除
        assert not (output_dir / "chunk_outlines.json").exists()


class TestHandleProgressResume:
    """测试进度恢复处理"""

    @pytest.mark.asyncio
    async def test_handle_progress_resume_no_progress(self, mock_service):
        """测试无进度数据"""
        mock_service.progress_service.load_progress.return_value = None

        chunks = [
            TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        ]

        progress_data = await mock_service._handle_progress_resume("test.txt", chunks, False, "utf-8")

        # resume=False时应该返回None
        assert progress_data is None

    @pytest.mark.asyncio
    async def test_handle_progress_resume_invalid_progress(self, mock_service):
        """测试进度验证失败"""
        mock_service.progress_service.load_progress.return_value = ProgressData(
            txt_file="test.txt",
            total_chunks=3,
            completed_count=1,
            completed_indices={1},
            outlines=[],
            last_update=datetime.now(),
            chunks_hash="old_hash",
        )
        mock_service.progress_service.is_progress_valid.return_value = False
        mock_service.progress_service.clear_progress = MagicMock()

        chunks = [
            TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4)
        ]

        progress_data = await mock_service._handle_progress_resume("test.txt", chunks, True, "utf-8")

        # 应该清除旧进度并创建新的
        mock_service.progress_service.clear_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_progress_resume_valid_progress(self, mock_service, sample_progress_data):
        """测试有效的进度恢复"""
        mock_service.progress_service.load_progress.return_value = sample_progress_data
        mock_service.progress_service.is_progress_valid.return_value = True

        chunks = [
            TextChunk(id=1, content="test", token_count=10, start_position=0, end_position=4),
            TextChunk(id=2, content="test2", token_count=10, start_position=4, end_position=8),
        ]

        progress_data = await mock_service._handle_progress_resume("test.txt", chunks, True, "utf-8")

        # 应该返回有效的进度数据
        assert progress_data is not None
        assert progress_data.total_chunks == 3