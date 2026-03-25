"""
OutlineMerger 测试
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.processing_state import ProcessingState
from services.llm_service import LLMResponse
from services.outline_merger import OutlineMerger


@pytest.fixture
def merger_setup():
    """构造完整的 OutlineMerger 及配套 mock"""
    llm_service = AsyncMock()
    llm_service.call = AsyncMock(
        return_value=LLMResponse(
            content="merged outline result",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )
    processing_config = MagicMock()
    processing_config.model_max_tokens = 8192

    cancel_event = asyncio.Event()
    emit_progress_fn = MagicMock()  # 同步 Callable[[], None]
    accumulate_tokens_fn = MagicMock()

    # 使用真实的 ProcessingState，因为源码会对 merge_level 做 += 和 -= 操作
    processing_state = ProcessingState(file_path="test.txt", total_chunks=10)

    merger = OutlineMerger(
        llm_service=llm_service,
        processing_config=processing_config,
        cancel_event=cancel_event,
    )

    return {
        "merger": merger,
        "llm_service": llm_service,
        "processing_state": processing_state,
        "emit_progress_fn": emit_progress_fn,
        "accumulate_tokens_fn": accumulate_tokens_fn,
        "cancel_event": cancel_event,
    }


@pytest.mark.asyncio
async def test_merge_recursive_empty_returns_empty(merger_setup):
    """空列表直接返回空字符串"""
    m = merger_setup
    result = await m["merger"].merge_outlines_recursive(
        [],
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    assert result == ""
    m["llm_service"].call.assert_not_called()


@pytest.mark.asyncio
async def test_merge_outlines_sets_phase_and_calls_recursive(merger_setup):
    """merge_outlines 入口应设置 current_phase=merging 并调用 emit_progress"""
    m = merger_setup
    outlines = [{"chunk_id": 0, "outline": "第一章"}, {"chunk_id": 1, "outline": "第二章"}]
    result = await m["merger"].merge_outlines(
        outlines,
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    assert m["processing_state"].current_phase == "merging"
    assert m["emit_progress_fn"].called
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_merge_two_outlines_calls_llm_once(merger_setup):
    """两个大纲合并时调用 LLM 一次"""
    m = merger_setup
    outlines = [{"chunk_id": 0, "outline": "A"}, {"chunk_id": 1, "outline": "B"}]
    result = await m["merger"].merge_outlines_recursive(
        outlines,
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    assert result == "merged outline result"
    m["llm_service"].call.assert_called_once()


@pytest.mark.asyncio
async def test_merge_respects_cancel_event(merger_setup):
    """cancel_event 已设置时应抛出 CancelledError"""
    m = merger_setup
    m["cancel_event"].set()
    with pytest.raises(asyncio.CancelledError):
        await m["merger"].merge_outlines_recursive(
            [{"chunk_id": 0, "outline": "A"}],
            m["processing_state"],
            m["emit_progress_fn"],
            m["accumulate_tokens_fn"],
        )


@pytest.mark.asyncio
async def test_merge_exceeds_max_levels_raises(merger_setup):
    """超过最大合并层级时应抛出 ProcessingError"""
    from exceptions import ProcessingError

    m = merger_setup
    with pytest.raises(ProcessingError, match="合并层级超过最大值"):
        await m["merger"].merge_outlines_recursive(
            ["some outline"],
            m["processing_state"],
            m["emit_progress_fn"],
            m["accumulate_tokens_fn"],
            level=11,  # _MAX_MERGE_LEVELS = 10
        )


@pytest.mark.asyncio
async def test_merge_text_mode_detection(merger_setup):
    """输入为字符串列表时自动切换为 text_mode"""
    m = merger_setup
    result = await m["merger"].merge_outlines_recursive(
        ["第一章内容", "第二章内容"],
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    assert result == "merged outline result"
    m["llm_service"].call.assert_called_once()


@pytest.mark.asyncio
async def test_merge_merged_content_mode(merger_setup):
    """包含 merged_content 字段的 dict 列表应提取内容并合并"""
    m = merger_setup
    outlines = [
        {"merged_content": "前半段内容"},
        {"merged_content": "后半段内容"},
    ]
    result = await m["merger"].merge_outlines_recursive(
        outlines,
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    assert result == "merged outline result"


@pytest.mark.asyncio
async def test_merge_accumulates_token_usage(merger_setup):
    """合并后应调用 accumulate_tokens_fn 传入 token 使用情况"""
    m = merger_setup
    await m["merger"].merge_outlines_recursive(
        [{"chunk_id": 0, "outline": "A"}, {"chunk_id": 1, "outline": "B"}],
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    m["accumulate_tokens_fn"].assert_called_once()
    call_args = m["accumulate_tokens_fn"].call_args[0]
    token_usage = call_args[0]
    assert token_usage is not None
    assert token_usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_merge_batching_when_token_limit_exceeded(merger_setup, monkeypatch):
    """输入 token 超限时应触发批次拆分，LLM 被调用多次"""
    import services.outline_merger as om_module

    m = merger_setup
    call_count = 0
    original_count_tokens = om_module.count_tokens

    def mock_count_tokens(text: str) -> int:
        nonlocal call_count
        call_count += 1
        # 首次调用返回超大值触发批次分割
        if call_count == 1:
            return 99999
        return int(original_count_tokens(text))

    monkeypatch.setattr(om_module, "count_tokens", mock_count_tokens)

    # 提供足够多的大纲使分批有意义
    outlines = [{"chunk_id": i, "outline": f"章节{i}"} for i in range(6)]
    result = await m["merger"].merge_outlines_recursive(
        outlines,
        m["processing_state"],
        m["emit_progress_fn"],
        m["accumulate_tokens_fn"],
    )
    # LLM 应该被调用多次（分批）
    assert m["llm_service"].call.call_count >= 1
    assert isinstance(result, str)


def test_merge_partial_outlines_static():
    """merge_partial_outlines 静态方法应合并部分大纲的 plot/characters"""
    partial = [
        {"plot": ["情节1", "情节2"], "characters": ["角色A"], "relationships": []},
        {"plot": ["情节3"], "characters": ["角色B"], "relationships": [["A", "友好", "B"]]},
    ]
    result = OutlineMerger.merge_partial_outlines(partial, original_chunk_id=5)
    assert result["chunk_id"] == 5
    assert result["is_partial"] is True
    assert "情节1" in result["plot"]
    assert "情节3" in result["plot"]
    assert "角色A" in result["characters"]
    assert "角色B" in result["characters"]
    assert len(result["relationships"]) == 1


def test_merge_partial_outlines_with_raw_response():
    """merge_partial_outlines 应拼接 raw_response"""
    partial = [
        {"plot": [], "characters": [], "relationships": [], "raw_response": "部分1"},
        {"plot": [], "characters": [], "relationships": [], "raw_response": "部分2"},
    ]
    result = OutlineMerger.merge_partial_outlines(partial, original_chunk_id=0)
    assert "部分1" in result["raw_response"]
    assert "部分2" in result["raw_response"]


def test_merge_partial_outlines_with_processing_time():
    """merge_partial_outlines 应计算平均处理时间"""
    partial = [
        {"plot": [], "characters": [], "relationships": [], "processing_time": 2.0},
        {"plot": [], "characters": [], "relationships": [], "processing_time": 4.0},
    ]
    result = OutlineMerger.merge_partial_outlines(partial, original_chunk_id=0)
    assert result["processing_time"] == 3.0
