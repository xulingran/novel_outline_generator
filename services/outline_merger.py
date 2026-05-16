"""
大纲合并模块
负责将多个块大纲合并为最终大纲
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, cast

from config import MAX_INPUT_TOKEN_RATIO, ProcessingConfig
from exceptions import ProcessingError
from models.processing_state import ProcessingState
from prompts import merge_prompt, merge_text_prompt
from services.llm_service import LLMService
from tokenizer import count_tokens

logger = logging.getLogger(__name__)


class OutlineMerger:
    """大纲合并器：负责将多个块大纲递归合并为最终大纲"""

    _MAX_MERGE_LEVELS: int = 10
    _BATCH_SIZE_RATIO: float = 0.8

    def __init__(
        self,
        llm_service: LLMService,
        processing_config: ProcessingConfig,
        cancel_event: asyncio.Event,
    ) -> None:
        self.llm_service = llm_service
        self.processing_config = processing_config
        self.cancel_event = cancel_event

    def _check_cancelled(self) -> None:
        """检查是否已取消"""
        if self.cancel_event.is_set():
            raise asyncio.CancelledError("任务已取消")

    async def merge_outlines(
        self,
        outlines: list[dict[str, Any]],
        processing_state: ProcessingState,
        emit_progress_fn: Callable[[], None],
        accumulate_tokens_fn: Callable[[dict[str, int] | None, str], None],
    ) -> str:
        """合并大纲入口：更新阶段为 merging 后递归合并"""
        processing_state.current_phase = "merging"
        emit_progress_fn()
        final_outline = await self.merge_outlines_recursive(
            outlines,
            processing_state,
            emit_progress_fn,
            accumulate_tokens_fn,
        )
        return final_outline

    def _normalize_outlines(
        self,
        outlines: list[dict[str, Any]] | list[str],
        is_text_mode: bool,
    ) -> tuple[list[dict[str, Any]] | list[str], bool]:
        """检测并标准化大纲格式，返回 (标准化后的大纲列表, 是否文本模式)"""
        if is_text_mode or not outlines:
            return outlines, is_text_mode

        first = outlines[0]
        if isinstance(first, str):
            return outlines, True
        if isinstance(first, dict) and "merged_content" in first:
            outlines_dicts = cast(list[dict[str, Any]], outlines)
            return [item["merged_content"] for item in outlines_dicts], True
        return outlines, False

    def _build_merge_prompt(
        self,
        outlines: list[dict[str, Any]] | list[str],
        is_text_mode: bool,
    ) -> str:
        """根据模式生成合并提示词"""
        if is_text_mode:
            return merge_text_prompt(cast(list[str], outlines))
        outlines_json = json.dumps(outlines, ensure_ascii=False)
        return merge_prompt(outlines_json)

    async def merge_outlines_recursive(
        self,
        outlines: list[dict[str, Any]] | list[str],
        processing_state: ProcessingState,
        emit_progress_fn: Callable[[], None],
        accumulate_tokens_fn: Callable[[dict[str, int] | None, str], None],
        level: int = 1,
        is_text_mode: bool = False,
    ) -> str:
        """递归合并大纲"""
        if level > self._MAX_MERGE_LEVELS:
            raise ProcessingError(
                f"合并层级超过最大值 {self._MAX_MERGE_LEVELS}，可能存在数据异常或递归循环"
            )

        if not outlines:
            return ""

        self._check_cancelled()

        processing_state.merge_level += 1
        processing_state.merge_outlines_count = len(outlines)
        emit_progress_fn()

        outlines, is_text_mode = self._normalize_outlines(outlines, is_text_mode)
        merge_prompt_text = self._build_merge_prompt(outlines, is_text_mode)

        input_tokens = count_tokens(merge_prompt_text)
        max_input_tokens = int(self.processing_config.model_max_tokens * MAX_INPUT_TOKEN_RATIO)

        if input_tokens <= max_input_tokens:
            logger.debug(f"层级 {level}: 合并 {len(outlines)} 个大纲块")
            llm_response = await self.llm_service.call(merge_prompt_text)

            self._check_cancelled()
            accumulate_tokens_fn(llm_response.token_usage, "合并")

            processing_state.merge_level -= 1
            emit_progress_fn()
            return llm_response.content

        return await self._merge_in_batches(
            outlines,
            processing_state,
            emit_progress_fn,
            accumulate_tokens_fn,
            level,
            is_text_mode,
            max_input_tokens,
            input_tokens,
        )

    async def _merge_in_batches(
        self,
        outlines: list[dict[str, Any]] | list[str],
        processing_state: ProcessingState,
        emit_progress_fn: Callable[[], None],
        accumulate_tokens_fn: Callable[[dict[str, int] | None, str], None],
        level: int,
        is_text_mode: bool,
        max_input_tokens: int,
        input_tokens: int,
    ) -> str:
        """将过大输入拆分为批次，逐批合并后递归合并批次结果"""
        logger.warning(f"层级 {level}: 输入过大，拆分为多个批次")
        batch_size = max(
            1, int(max_input_tokens / (input_tokens / len(outlines)) * self._BATCH_SIZE_RATIO)
        )
        batches = [outlines[i : i + batch_size] for i in range(0, len(outlines), batch_size)]

        processing_state.merge_batch_total = len(batches)
        processing_state.merge_batch_current = 0
        emit_progress_fn()

        merged_batches: list[str] = []
        for idx, batch in enumerate(batches, 1):
            self._check_cancelled()
            processing_state.merge_batch_current = idx
            emit_progress_fn()
            logger.debug(f"处理批次 {idx}/{len(batches)}")
            merged = await self.merge_outlines_recursive(
                batch,
                processing_state,
                emit_progress_fn,
                accumulate_tokens_fn,
                level + 1,
                is_text_mode,
            )
            merged_batches.append(merged)

        if len(merged_batches) == 1:
            processing_state.merge_level -= 1
            emit_progress_fn()
            return merged_batches[0]

        logger.debug(f"合并 {len(merged_batches)} 个批次的结果")
        result = await self.merge_outlines_recursive(
            merged_batches,
            processing_state,
            emit_progress_fn,
            accumulate_tokens_fn,
            level + 1,
            is_text_mode=True,
        )
        processing_state.merge_level -= 1
        emit_progress_fn()
        return result

    @staticmethod
    def merge_partial_outlines(
        partial_outlines: list[dict[str, Any]], original_chunk_id: int
    ) -> dict[str, Any]:
        """将部分完成的小块大纲合并为一个完整大纲"""
        all_plot: list[str] = []
        all_characters: set[str] = set()
        all_relationships: set[tuple[str, str, str]] = set()

        for outline in partial_outlines:
            plot = outline.get("plot", [])
            if isinstance(plot, list):
                all_plot.extend([p for p in plot if isinstance(p, str)])

            characters = outline.get("characters", [])
            if isinstance(characters, list):
                all_characters.update(c for c in characters if isinstance(c, str))

            relationships = outline.get("relationships", [])
            if isinstance(relationships, list):
                for rel in relationships:
                    if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                        all_relationships.add((str(rel[0]), str(rel[1]), str(rel[2])))

        merged_outline: dict[str, Any] = {
            "chunk_id": original_chunk_id,
            "is_partial": True,
            "sub_chunk_count": len(partial_outlines),
            "plot": all_plot,
            "characters": sorted(all_characters),
            "relationships": [list(rel) for rel in sorted(all_relationships)],
            "partial_outlines": partial_outlines,
        }

        if all("raw_response" in outline for outline in partial_outlines):
            merged_outline["raw_response"] = "\n\n".join(
                [outline["raw_response"] for outline in partial_outlines]
            )

        processing_times = [
            outline.get("processing_time", 0)
            for outline in partial_outlines
            if "processing_time" in outline
        ]
        if processing_times:
            merged_outline["processing_time"] = sum(processing_times) / len(processing_times)

        return merged_outline
