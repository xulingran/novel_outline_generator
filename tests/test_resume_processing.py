"""测试恢复处理逻辑"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import ProgressData, TextChunk


@pytest.mark.asyncio
async def test_resume_processes_remaining_chunks():
    """恢复时应继续处理剩余块"""
    from services.novel_processing_service import NovelProcessingService

    with patch("services.llm_service.create_llm_service") as mock_create:
        mock_create.return_value = MagicMock()
        service = NovelProcessingService(progress_callback=MagicMock(), cancel_event=MagicMock())

    chunks = [
        TextChunk(id=1, content="A", token_count=1, start_position=0, end_position=1),
        TextChunk(id=2, content="B", token_count=1, start_position=1, end_position=2),
        TextChunk(id=3, content="C", token_count=1, start_position=2, end_position=3),
    ]

    progress_data = ProgressData(
        txt_file="test.txt",
        total_chunks=3,
        completed_count=1,
        completed_indices={1},
        outlines=[{"chunk_id": 1, "plot": ["done"]}],
        last_update=datetime.now(),
        chunks_hash="hash",
    )

    service.progress_service = MagicMock()
    service.progress_service.load_progress.return_value = progress_data
    service.progress_service.is_progress_valid.return_value = True
    service.progress_service.clear_progress = MagicMock()
    service.progress_service.finalize_progress = MagicMock()

    service._load_and_validate_file = AsyncMock(return_value=("content", "utf-8"))
    service._split_text_into_chunks = MagicMock(return_value=chunks)
    service._process_chunks = AsyncMock(
        return_value=[
            {"chunk_id": 2, "plot": ["new"]},
            {"chunk_id": 3, "plot": ["new"]},
        ]
    )
    service.merge_outlines_recursive = AsyncMock(return_value="merged")
    service._save_results = AsyncMock()
    service.file_service = MagicMock()
    service.file_service.remove_backups.return_value = 0
    service._cleanup_intermediate_outputs = MagicMock(return_value=[])

    await service.process_novel("test.txt", resume=True)

    assert service._process_chunks.call_count == 1
    called_chunks = service._process_chunks.call_args.args[0]
    assert [chunk.id for chunk in called_chunks] == [2, 3]
    assert service._process_chunks.call_args.kwargs["progress_data"] is progress_data
    assert service._process_chunks.call_args.kwargs["total_chunks"] == 3

    merged_outlines = service.merge_outlines_recursive.call_args.args[0]
    assert {item.get("chunk_id") for item in merged_outlines} == {1, 2, 3}
