"""
块处理模块
负责文本块的并行处理、重试机制和拆分降级
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from config import ProcessingConfig
from exceptions import APIError, ProcessingError
from models.outline import TextChunk
from models.processing_state import ProcessingState, ProgressData
from prompts import chunk_prompt
from services.eta_estimator import ETAEstimator
from services.llm_service import LLMService
from services.outline_merger import OutlineMerger
from services.progress_service import ProgressService
from tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """块处理器：负责文本块的并行处理、重试和拆分降级"""

    def __init__(
        self,
        llm_service: LLMService,
        processing_config: ProcessingConfig,
        cancel_event: asyncio.Event,
        progress_service: ProgressService,
        eta_estimator: ETAEstimator,
        emit_progress_fn: Callable[..., None],
        accumulate_tokens_fn: Callable[[dict[str, int] | None, str], None],
    ) -> None:
        self.llm_service = llm_service
        self.processing_config = processing_config
        self.cancel_event = cancel_event
        self.progress_service = progress_service
        self.eta_estimator = eta_estimator
        self._emit_progress = emit_progress_fn
        self._accumulate_token_usage = accumulate_tokens_fn

    def _check_cancelled(self) -> None:
        """检查是否已取消"""
        if self.cancel_event.is_set():
            raise asyncio.CancelledError("任务已取消")

    async def process_chunks(
        self,
        chunks: list[TextChunk],
        processing_state: ProcessingState,
        progress_data: ProgressData | None = None,
        total_chunks: int | None = None,
        force_complete: bool = False,
    ) -> tuple[list[dict[str, Any]], ProgressData]:
        """处理所有文本块，返回 (outlines, progress_data)"""
        processing_state.current_phase = "processing"
        processing_state.total_chunks = total_chunks if total_chunks is not None else len(chunks)
        processing_state.processing_start_time = datetime.now()
        self.eta_estimator.start_processing()
        self._emit_progress()

        if progress_data is None:
            progress_data = self.progress_service.create_progress(
                processing_state.file_path,
                len(chunks),
                ProgressData.calculate_chunks_hash([c.content for c in chunks]),
            )

        sem = asyncio.Semaphore(self.processing_config.parallel_limit)
        tasks = [
            self.process_single_chunk(chunk, sem, processing_state, progress_data)
            for chunk in chunks
        ]

        completed_successfully = False
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_outlines: list[dict[str, Any]] = []
            has_cancelled = False
            for idx, result in enumerate(results, 1):
                if isinstance(result, asyncio.CancelledError):
                    has_cancelled = True
                    continue
                if isinstance(result, Exception):
                    logger.error(f"块 {idx} 处理失败: {result}")
                    processing_state.add_error(f"块 {idx}: {str(result)}")
                    processing_state.update_progress(processed=0, failed=1)
                    self.progress_service.add_progress_error(progress_data, idx, str(result))
                    self._emit_progress(chunk_id=idx, error=str(result))
                else:
                    successful_outlines.append(cast(dict[str, Any], result))

            if has_cancelled and not force_complete:
                raise asyncio.CancelledError()
            if has_cancelled and force_complete:
                logger.info("强制完成模式：忽略未完成的块，继续合并已有结果")

            completed_successfully = True

        except asyncio.CancelledError:
            logger.info("处理被取消")
            raise
        except Exception as e:
            logger.exception("处理文本块时发生错误")
            raise ProcessingError(f"处理文本块失败: {str(e)}") from e
        finally:
            try:
                if completed_successfully:
                    self.progress_service.finalize_progress(progress_data)
                else:
                    self.progress_service.save_progress(progress_data)
            except Exception as save_err:
                logger.exception("保存进度失败: %s", save_err)

        successful_outlines.sort(key=lambda x: x.get("chunk_id", 0))
        logger.info(f"成功处理 {len(successful_outlines)}/{len(chunks)} 个块")
        return successful_outlines, progress_data

    async def process_single_chunk(
        self,
        chunk: TextChunk,
        sem: asyncio.Semaphore,
        processing_state: ProcessingState,
        progress_data: Any,
    ) -> dict[str, Any]:
        """处理单个文本块，支持重试和部分完成"""
        self._check_cancelled()

        async with sem:
            chunk_id = chunk.id
            logger.debug(f"开始处理块 {chunk_id}")

            last_error: Exception | None = None
            for attempt in range(1, self.processing_config.max_retry + 1):
                try:
                    return await self._attempt_single_call(
                        chunk, chunk_id, processing_state, progress_data
                    )
                except asyncio.CancelledError:
                    logger.info(f"块 {chunk_id} 处理被取消")
                    raise
                except (APIError, ProcessingError) as e:
                    last_error = e
                    if attempt < self.processing_config.max_retry:
                        logger.warning(
                            f"块 {chunk_id} 第 {attempt}/{self.processing_config.max_retry} 次尝试失败: "
                            f"{type(e).__name__}: {e}，将重试"
                        )
                        await asyncio.sleep(self.processing_config.retry_backoff_base * attempt)
                    else:
                        logger.error(
                            f"块 {chunk_id} 经过 {self.processing_config.max_retry} 次重试后仍然失败: "
                            f"{type(e).__name__}: {e}"
                        )
                except Exception as e:
                    last_error = e
                    logger.error(
                        f"块 {chunk_id} 遇到未预期的错误: {type(e).__name__}: {e}", exc_info=True
                    )
                    if attempt < self.processing_config.max_retry:
                        logger.warning(
                            f"块 {chunk_id} 将在 "
                            f"{self.processing_config.retry_backoff_base * attempt} 秒后重试"
                        )
                        await asyncio.sleep(self.processing_config.retry_backoff_base * attempt)
                    else:
                        logger.error(f"块 {chunk_id} 已达到最大重试次数，放弃处理")

            return await self._handle_failed_chunk(
                chunk, sem, processing_state, progress_data, last_error
            )

    async def _attempt_single_call(
        self,
        chunk: TextChunk,
        chunk_id: int,
        processing_state: ProcessingState,
        progress_data: Any,
    ) -> dict[str, Any]:
        """对单个块执行一次 LLM 调用，成功时返回大纲数据"""
        self._check_cancelled()

        start_time = datetime.now()
        prompt = chunk_prompt(chunk.content, chunk_id)
        llm_response = await self.llm_service.call(prompt, chunk_id)
        response = llm_response.content

        self._check_cancelled()
        self._accumulate_token_usage(llm_response.token_usage, f"块 {chunk_id}")

        outline_data = self.parse_llm_response(response, chunk_id)
        if "chunk_id" not in outline_data:
            outline_data["chunk_id"] = chunk_id

        processing_time = (datetime.now() - start_time).total_seconds()
        outline_data["raw_response"] = response
        outline_data["processing_time"] = processing_time

        self.progress_service.update_chunk_completed(
            progress_data, chunk_id, outline_data, processing_time
        )
        processing_state.update_progress(processed=1)
        self.eta_estimator.add_completion(processing_time, progress_data.completed_count)
        self._emit_progress(chunk_id=chunk_id)

        logger.debug(f"块 {chunk_id} 处理完成，耗时: {processing_time:.2f}秒")
        return outline_data

    async def _handle_failed_chunk(
        self,
        chunk: TextChunk,
        sem: asyncio.Semaphore,
        processing_state: ProcessingState,
        progress_data: Any,
        last_error: Exception | None,
    ) -> dict[str, Any]:
        """所有重试失败后，尝试拆分小块处理，最终失败则抛出异常"""
        chunk_id = chunk.id
        logger.info(
            f"块 {chunk_id} 重试失败，尝试拆分为{self.processing_config.sub_chunk_count}个小块重新处理"
        )
        try:
            partial_outlines = await self.process_failing_chunk_as_partial(
                chunk, sem, processing_state, progress_data
            )
            if partial_outlines:
                return OutlineMerger.merge_partial_outlines(partial_outlines, chunk_id)
            raise ProcessingError(f"块 {chunk_id} 拆分后所有小块都失败")
        except Exception as split_error:
            logger.error(f"块 {chunk_id} 拆分重试也失败: {split_error}")
            processing_state.update_progress(processed=0, failed=1)
            self.progress_service.add_progress_error(
                progress_data, chunk_id, str(last_error or split_error)
            )
            self._emit_progress(chunk_id=chunk_id, error=str(last_error or split_error))
            raise ProcessingError(
                f"块 {chunk_id} 处理失败: {str(last_error or split_error)}"
            ) from (last_error or split_error)

    async def process_failing_chunk_as_partial(
        self,
        chunk: TextChunk,
        sem: asyncio.Semaphore,
        processing_state: ProcessingState,
        progress_data: Any,
    ) -> list[dict[str, Any]]:
        """将失败的块拆分为多个小块，逐个处理，返回成功的小块大纲列表"""
        chunk_id = chunk.id
        sub_chunks = self.split_chunk_into_sub_chunks(chunk)
        logger.info(f"块 {chunk_id} 已拆分为 {len(sub_chunks)} 个小块")

        successful_sub_outlines: list[dict[str, Any]] = []
        failed_sub_chunks = 0

        for sub_idx, sub_chunk in enumerate(sub_chunks, 1):
            sub_chunk_id = f"{chunk_id}_sub_{sub_idx}"
            try:
                logger.debug(f"处理块 {chunk_id} 的小块 {sub_idx}/{len(sub_chunks)}")
                start_time = datetime.now()
                prompt = chunk_prompt(sub_chunk.content, sub_chunk_id)
                llm_response = await self.llm_service.call(prompt, sub_chunk_id)
                response = llm_response.content

                self._accumulate_token_usage(llm_response.token_usage, f"子块 {sub_chunk_id}")
                sub_outline = self.parse_llm_response(response, sub_chunk_id)

                processing_time = (datetime.now() - start_time).total_seconds()
                sub_outline["raw_response"] = response
                sub_outline["processing_time"] = processing_time
                sub_outline["sub_chunk_index"] = sub_idx
                sub_outline["sub_chunk_id"] = sub_chunk_id
                sub_outline["original_chunk_id"] = chunk_id
                sub_outline["total_sub_chunks"] = len(sub_chunks)

                successful_sub_outlines.append(sub_outline)
                logger.debug(f"子块 {sub_chunk_id} 处理成功，耗时: {processing_time:.2f}秒")

            except asyncio.CancelledError:
                logger.info(f"子块 {sub_chunk_id} 处理被取消")
                raise
            except (APIError, ProcessingError) as e:
                logger.warning(f"子块 {sub_chunk_id} API/处理错误: {type(e).__name__}: {e}，将丢弃")
                failed_sub_chunks += 1
            except Exception as e:
                logger.error(
                    f"子块 {sub_chunk_id} 遇到未预期的错误: {type(e).__name__}: {e}，将丢弃",
                    exc_info=True,
                )
                failed_sub_chunks += 1

        if not successful_sub_outlines:
            logger.warning(f"块 {chunk_id} 所有小块都处理失败")
            return []

        progress_data.partial_indices.add(chunk_id)
        progress_data.partial_outlines.extend(successful_sub_outlines)
        processing_state.update_partial(1)

        logger.info(
            f"块 {chunk_id} 部分完成: 成功 {len(successful_sub_outlines)}/{len(sub_chunks)} 个小块，"
            f"失败 {failed_sub_chunks} 个小块"
        )
        self._emit_progress(chunk_id=chunk_id, partial_info=f"{chunk_id}块部分完成")
        return successful_sub_outlines

    def split_chunk_into_sub_chunks(self, chunk: TextChunk) -> list[TextChunk]:
        """将一个块拆分为多个小块（数量由配置决定）"""
        text = chunk.content
        total_length = len(text)
        if total_length == 0:
            return []

        sub_chunk_count = min(self.processing_config.sub_chunk_count, total_length)
        chunk_size = total_length // sub_chunk_count

        sub_chunks: list[TextChunk] = []
        start_position = chunk.start_position

        for idx in range(sub_chunk_count):
            start = idx * chunk_size
            end = total_length if idx == sub_chunk_count - 1 else start + chunk_size
            sub_content = text[start:end]
            sub_chunks.append(
                TextChunk(
                    id=chunk.id,
                    content=sub_content,
                    token_count=count_tokens(sub_content),
                    start_position=start_position + start,
                    end_position=start_position + end,
                )
            )

        return sub_chunks

    @staticmethod
    def parse_llm_response(response: str, chunk_id: int | str) -> dict[str, Any]:
        """解析 LLM 响应为结构化大纲数据"""
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                if "chunk_id" not in data:
                    data["chunk_id"] = chunk_id
                return cast(dict[str, Any], data)
            raise ValueError("LLM响应不是JSON对象")
        except (json.JSONDecodeError, ValueError, TypeError):
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if isinstance(data, dict):
                        if "chunk_id" not in data:
                            data["chunk_id"] = chunk_id
                        return cast(dict[str, Any], data)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            logger.warning(f"块 {chunk_id} 响应无法解析为JSON，使用原始文本")
            return {
                "chunk_id": chunk_id,
                "plot": [response],
                "characters": [],
                "relationships": [],
            }
