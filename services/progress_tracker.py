"""
进度追踪模块
负责处理进度报告、进度恢复和 ETA 估算
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from config import ProcessingConfig
from models.outline import TextChunk
from models.processing_state import ProcessingState, ProgressData
from services.eta_estimator import ETAEstimator
from services.outline_merger import OutlineMerger
from services.progress_service import ProgressService

logger = logging.getLogger(__name__)


class ProgressTracker:
    """进度追踪器：负责进度上报、ETA 估算与进度恢复"""

    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        eta_estimator: ETAEstimator,
        progress_service: ProgressService,
        processing_config: ProcessingConfig,
    ) -> None:
        self.progress_callback = progress_callback
        self.eta_estimator = eta_estimator
        self.progress_service = progress_service
        self.processing_config = processing_config

    def emit_progress(
        self,
        processing_state: ProcessingState | None,
        current_progress_data: ProgressData | None,
        chunk_id: int | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        partial_info: str | None = None,
    ) -> None:
        """向外部回调当前进度，便于 Web UI 实时显示。
        设计为尽量轻量、容错，不影响主流程。
        """
        if not self.progress_callback or not processing_state:
            return

        total = processing_state.total_chunks or 0
        completed = processing_state.processed_chunks
        failed = processing_state.failed_chunks
        partial = processing_state.partial_chunks

        partial_weight = self._calculate_partial_weight(partial, current_progress_data)
        effective_completed = completed + partial_weight
        progress = (effective_completed / total) if total > 0 else 0.0

        eta_result = self.eta_estimator.estimate(
            total_chunks=total,
            completed_chunks=completed,
            failed_chunks=failed,
        )

        payload = self._build_progress_payload(
            progress=progress,
            completed=completed,
            failed=failed,
            partial=partial,
            total=total,
            eta_result=eta_result,
            processing_state=processing_state,
            chunk_id=chunk_id,
            error=error,
            token_usage=token_usage,
            partial_info=partial_info,
        )
        self._safe_emit_progress(payload)

    def _safe_emit_progress(self, payload: dict[str, Any]) -> None:
        """安全地调用进度回调，避免中断主流程"""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception as e:
            logger.debug(f"Progress callback failed: {e}")

    def _build_progress_payload(
        self,
        progress: float,
        completed: int,
        failed: int,
        partial: int,
        total: int,
        eta_result: dict[str, Any],
        processing_state: ProcessingState,
        chunk_id: int | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        partial_info: str | None = None,
    ) -> dict[str, Any]:
        """构建进度回调的数据载荷"""
        payload: dict[str, Any] = {
            "progress": progress,
            "completed_chunks": completed,
            "failed_chunks": failed,
            "partial_chunks": partial,
            "total_chunks": total,
            "phase": processing_state.current_phase,
            "merge_level": processing_state.merge_level,
            "merge_batch_current": processing_state.merge_batch_current,
            "merge_batch_total": processing_state.merge_batch_total,
            "merge_outlines_count": processing_state.merge_outlines_count,
        }

        if eta_result.get("eta_seconds") is not None:
            payload["eta_seconds"] = eta_result["eta_seconds"]
            payload["eta_confidence"] = eta_result.get("confidence")
            payload["eta_method"] = eta_result.get("method")
        if chunk_id is not None:
            payload["last_chunk_id"] = chunk_id
        if error is not None:
            payload["last_error"] = error
        if token_usage is not None:
            payload["token_usage"] = token_usage
        if partial_info is not None:
            payload["partial_info"] = partial_info

        return payload

    def _calculate_partial_weight(
        self,
        partial_count: int,
        current_progress_data: ProgressData | None,
    ) -> float:
        """计算部分完成块的权重贡献"""
        if not current_progress_data or partial_count <= 0:
            return 0.0

        chunk_sub_counts: dict[int, int] = defaultdict(int)
        chunk_total_sub_counts: dict[int, int] = {}

        for outline in current_progress_data.partial_outlines:
            original_chunk_id = outline.get("original_chunk_id") or outline.get("chunk_id")
            if original_chunk_id in current_progress_data.partial_indices:
                chunk_sub_counts[original_chunk_id] += 1
                total_sub_chunks = outline.get("total_sub_chunks")
                if isinstance(total_sub_chunks, int) and total_sub_chunks > 0:
                    existing = chunk_total_sub_counts.get(original_chunk_id)
                    if existing is None or total_sub_chunks > existing:
                        chunk_total_sub_counts[original_chunk_id] = total_sub_chunks

        partial_weight = 0.0
        for chunk_id_for_weight, sub_count in chunk_sub_counts.items():
            total_sub_chunks = chunk_total_sub_counts.get(
                chunk_id_for_weight, self.processing_config.sub_chunk_count
            )
            partial_weight += sub_count / total_sub_chunks

        return partial_weight

    async def handle_progress_resume(
        self,
        file_path: str,
        chunks: list[TextChunk],
        resume: bool,
        encoding: str,
        processing_state: ProcessingState | None,
    ) -> ProgressData | None:
        """处理进度恢复或初始化"""
        if not resume:
            return None

        progress_data = self.progress_service.load_progress()
        if not progress_data:
            return None

        chunks_hash = ProgressData.calculate_chunks_hash(
            [c.content for c in chunks], encoding=encoding
        )

        if not self.progress_service.is_progress_valid(
            progress_data, file_path, [c.content for c in chunks], chunks_hash
        ):
            logger.info("进度无效，将重新开始")
            self.progress_service.clear_progress()
            return None

        if progress_data.partial_outlines:
            partial_grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            existing_chunk_ids = {
                outline.get("chunk_id")
                for outline in progress_data.outlines
                if isinstance(outline, dict)
            }
            for outline in progress_data.partial_outlines:
                chunk_id = outline.get("original_chunk_id") or outline.get("chunk_id")
                if chunk_id in progress_data.partial_indices:
                    partial_grouped[chunk_id].append(outline)

            for chunk_id, sub_outlines in partial_grouped.items():
                if chunk_id in existing_chunk_ids:
                    continue
                merged = OutlineMerger.merge_partial_outlines(sub_outlines, chunk_id)
                progress_data.outlines.append(merged)
                existing_chunk_ids.add(chunk_id)
                logger.debug(f"恢复时合并块 {chunk_id} 的 {len(sub_outlines)} 个小块大纲")

        if processing_state:
            processing_state.processed_chunks = progress_data.completed_count
            processing_state.partial_chunks = len(progress_data.partial_indices)
            processing_state.failed_chunks = len(progress_data.errors)
            processing_state.total_chunks = progress_data.total_chunks

        logger.info(
            f"恢复进度: 完全完成 {progress_data.completed_count}/{progress_data.total_chunks}, "
            f"部分完成 {len(progress_data.partial_indices)} 个块"
        )
        return progress_data
