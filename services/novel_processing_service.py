"""
小说处理服务模块
核心业务逻辑，处理小说文本并生成大纲
"""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from config import get_processing_config
from exceptions import APIError, ProcessingError
from models.outline import TextChunk
from models.processing_state import ProcessingState, ProgressData
from prompts import chunk_prompt, merge_prompt, merge_text_prompt
from services.eta_estimator import ETAEstimator
from services.file_service import FileService
from services.llm_service import create_llm_service
from services.progress_service import ProgressService
from splitter import split_text
from tokenizer import count_tokens

logger = logging.getLogger(__name__)

# 常量定义
SUB_CHUNK_COUNT = 5  # 失败块拆分为的小块数量
RETRY_BACKOFF_BASE = 1  # 重试退避基数（秒）


class NovelProcessingService:
    """小说处理服务类"""

    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ):
        self.processing_config = get_processing_config()
        self.llm_service = create_llm_service()
        self.progress_service = ProgressService()
        self.file_service = FileService()
        self.processing_state: ProcessingState | None = None
        self.progress_callback = progress_callback
        self.current_progress_data: ProgressData | None = None
        self.cancel_event = cancel_event or asyncio.Event()
        # Token统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        # 强制完成标志
        self.force_complete: bool = False
        # ETA 估算器
        self.eta_estimator = ETAEstimator(
            window_size=20,
            outlier_threshold=2.5,
            min_samples=3,
        )
        self.eta_estimator.set_parallel_limit(self.processing_config.parallel_limit)

    async def process_novel(
        self, file_path: str, output_dir: str | None = None, resume: bool = True
    ) -> dict[str, Any]:
        """
        处理小说文件，生成大纲

        Args:
            file_path: 小说文件路径
            output_dir: 输出目录（可选）
            resume: 是否恢复进度

        Returns:
            Dict[str, Any]: 处理结果
        """
        # 重置token统计（确保每次处理都是新的统计）
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        # 初始化处理状态，块数后续拆分时再更新
        self.processing_state = ProcessingState(file_path=file_path, total_chunks=0)
        self.processing_state.current_phase = "loading"
        self._emit_progress()

        try:
            # 1. 读取和分割文本
            text, encoding = await self._load_and_validate_file(file_path)
            chunks = self._split_text_into_chunks(text)
            self.processing_state.total_chunks = len(chunks)
            if not chunks:
                raise ProcessingError("未检测到可处理的内容")
            self._emit_progress()

            # 2. 处理或恢复进度
            outlines = await self._handle_progress_resume(file_path, chunks, resume, encoding)

            # 3. 处理文本块
            if outlines is None:
                outlines = await self._process_chunks(chunks)

            # 4. 合并大纲
            self.processing_state.current_phase = "merging"
            self._emit_progress()
            final_outline = await self.merge_outlines_recursive(outlines)
            self._emit_progress(
                token_usage={
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                    "total_tokens": self.total_tokens,
                }
            )

            # 5. 保存结果
            if output_dir:
                self.processing_config.output_dir = output_dir

            await self._save_results(outlines, final_outline, file_path)
            # 5.1 清理备份文件（成功完成后删除 outputs 下的 .bak）
            try:
                removed = self.file_service.remove_backups(
                    self.processing_config.output_dir, "*.bak"
                )
                logger.debug(f"已清理备份文件: {removed} 个")
            except Exception as cleanup_err:
                logger.warning(f"清理备份文件失败: {cleanup_err}")

            # 5.2 清理中间结果文件
            try:
                cleaned = self._cleanup_intermediate_outputs(
                    Path(self.processing_config.output_dir)
                )
                if cleaned:
                    logger.info(f"已清理中间结果文件: {', '.join(cleaned)}")
            except Exception as cleanup_err:
                logger.warning(f"清理中间结果文件失败: {cleanup_err}")

            # 6. 完成处理
            self.processing_state.complete()
            self._emit_progress()

            return {
                "success": True,
                "final_outline": final_outline,
                "chunk_count": len(chunks),
                "processing_time": self.processing_state.elapsed_time,
                "output_dir": self.processing_config.output_dir,
                "token_usage": {
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                    "total_tokens": self.total_tokens,
                },
            }

        except asyncio.CancelledError:
            logger.info("Novel processing cancelled")
            raise
        except Exception as e:
            if self.processing_state:
                self.processing_state.fail(str(e))
                self._emit_progress()
            logger.error(f"处理小说失败: {e}")
            raise ProcessingError(f"处理小说失败: {str(e)}") from e

    async def _load_and_validate_file(self, file_path: str) -> tuple[str, str]:
        """加载并验证文件"""
        logger.info(f"正在读取文件: {file_path}")
        try:
            text, encoding = self.file_service.read_text_file(file_path)
            if not text.strip():
                raise ProcessingError("文件内容为空")
            return text, encoding
        except Exception as e:
            raise ProcessingError(f"读取文件失败: {str(e)}") from e

    def _split_text_into_chunks(self, text: str) -> list[TextChunk]:
        """分割文本为块"""
        logger.info("正在分割文本...")
        try:
            raw_chunks = split_text(text)

            # 转换为TextChunk对象
            chunks = []
            position = 0
            for idx, chunk_content in enumerate(raw_chunks, 1):
                token_count = count_tokens(chunk_content)
                chunks.append(
                    TextChunk(
                        id=idx,
                        content=chunk_content,
                        token_count=token_count,
                        start_position=position,
                        end_position=position + len(chunk_content),
                    )
                )
                position += len(chunk_content)

            logger.info(f"文本已分割为 {len(chunks)} 个块")
            return chunks

        except Exception as e:
            raise ProcessingError(f"分割文本失败: {str(e)}") from e

    async def _handle_progress_resume(
        self,
        file_path: str,
        chunks: list[TextChunk],
        resume: bool,
        encoding: str,
    ) -> list[dict[str, Any]] | None:
        """Handle progress resume or initialization"""
        if not resume:
            return None

        # 加载进度
        progress_data = self.progress_service.load_progress()
        if not progress_data:
            return None

        # 计算当前哈希（考虑编码）
        chunks_hash = ProgressData.calculate_chunks_hash(
            [c.content for c in chunks], encoding=encoding
        )

        # 验证进度是否有效
        if not self.progress_service.is_progress_valid(
            progress_data, file_path, [c.content for c in chunks], chunks_hash
        ):
            logger.info("进度无效，将重新开始")
            self.progress_service.clear_progress()
            return None

        # 合并部分完成的小块为完整大纲
        # partial_outlines 存储的是小块级别的大纲，需要按 original_chunk_id 分组合并
        if progress_data.partial_outlines:
            from collections import defaultdict

            partial_grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for outline in progress_data.partial_outlines:
                chunk_id = outline.get("original_chunk_id") or outline.get("chunk_id")
                if chunk_id in progress_data.partial_indices:
                    partial_grouped[chunk_id].append(outline)

            # 合并每个块的小块大纲
            for chunk_id, sub_outlines in partial_grouped.items():
                merged = self._merge_partial_outlines(sub_outlines, chunk_id)
                progress_data.outlines.append(merged)
                logger.debug(f"恢复时合并块 {chunk_id} 的 {len(sub_outlines)} 个小块大纲")

        logger.info(
            f"恢复进度: 完全完成 {progress_data.completed_count}/{progress_data.total_chunks}, "
            f"部分完成 {len(progress_data.partial_indices)} 个块"
        )
        return progress_data.outlines

    async def _process_chunks(self, chunks: list[TextChunk]) -> list[dict[str, Any]]:
        """处理所有文本块"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")

        processing_state = self.processing_state
        processing_state.current_phase = "processing"
        processing_state.total_chunks = len(chunks)
        processing_state.processing_start_time = datetime.now()
        self.eta_estimator.start_processing()
        self._emit_progress()

        # 创建进度数据
        progress_data = self.progress_service.create_progress(
            processing_state.file_path,
            len(chunks),
            ProgressData.calculate_chunks_hash([c.content for c in chunks]),
        )
        self.current_progress_data = progress_data

        # 使用信号量控制并发
        sem = asyncio.Semaphore(self.processing_config.parallel_limit)

        # 创建任务
        tasks = []
        for chunk in chunks:
            task = self._process_single_chunk(chunk, sem, progress_data)
            tasks.append(task)

        # 等待所有任务完成
        try:
            outlines = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            successful_outlines: list[dict[str, Any]] = []
            cancelled_error: asyncio.CancelledError | None = None
            for idx, result in enumerate(outlines, 1):
                if isinstance(result, asyncio.CancelledError):
                    cancelled_error = result
                    continue
                if isinstance(result, Exception):
                    logger.error(f"块 {idx} 处理失败: {result}")
                    processing_state.add_error(f"块 {idx}: {str(result)}")
                    processing_state.update_progress(processed=0, failed=1)
                    self.progress_service.add_progress_error(progress_data, idx, str(result))
                    self._emit_progress(chunk_id=idx, error=str(result))
                else:
                    successful_outlines.append(cast(dict[str, Any], result))

            # 检查是否强制完成（忽略未完成的块）
            if cancelled_error is not None:
                if self.force_complete:
                    logger.info("强制完成模式：忽略未完成的块，继续合并已有结果")
                else:
                    raise cancelled_error

        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise ProcessingError(f"处理文本块失败: {str(e)}") from e
        finally:
            # 保存最终进度
            self.progress_service.finalize_progress(progress_data)

        # 按chunk_id排序
        successful_outlines.sort(key=lambda x: x.get("chunk_id", 0))

        logger.info(f"成功处理 {len(successful_outlines)}/{len(chunks)} 个块")
        return successful_outlines

    async def _process_single_chunk(
        self, chunk: TextChunk, sem: asyncio.Semaphore, progress_data: Any
    ) -> dict[str, Any]:
        """处理单个文本块，支持重试和部分完成"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")

        # 检查是否被取消
        if self.cancel_event.is_set():
            logger.info("任务已被取消")
            raise asyncio.CancelledError()

        processing_state = self.processing_state
        async with sem:
            chunk_id = chunk.id
            logger.debug(f"开始处理块 {chunk_id}")

            # 重试机制：最多重试 MAX_RETRY 次
            last_error: Exception | None = None
            for attempt in range(1, self.processing_config.max_retry + 1):
                try:
                    # 检查是否被取消
                    if self.cancel_event.is_set():
                        logger.info("任务已被取消")
                        raise asyncio.CancelledError()

                    start_time = datetime.now()

                    # 生成提示
                    prompt = chunk_prompt(chunk.content, chunk_id)

                    # 调用LLM
                    llm_response = await self.llm_service.call(prompt, chunk_id)
                    response = llm_response.content

                    # 检查是否被取消（在LLM调用后）
                    if self.cancel_event.is_set():
                        logger.info("任务已被取消")
                        raise asyncio.CancelledError()

                    # 累计token使用情况
                    if llm_response.token_usage:
                        prompt_tokens = llm_response.token_usage.get("prompt_tokens", 0) or 0
                        completion_tokens = (
                            llm_response.token_usage.get("completion_tokens", 0) or 0
                        )
                        total_tokens = llm_response.token_usage.get("total_tokens", 0) or 0

                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens
                        self.total_tokens += total_tokens

                        logger.debug(
                            f"块 {chunk_id} Token使用: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}"
                        )

                    # 尝试解析JSON响应
                    outline_data = self._parse_llm_response(response, chunk_id)

                    # 记录处理时间
                    processing_time = (datetime.now() - start_time).total_seconds()

                    # 保存原始响应
                    outline_data["raw_response"] = response
                    outline_data["processing_time"] = processing_time

                    # 更新进度
                    self.progress_service.update_chunk_completed(
                        progress_data, chunk_id, outline_data, processing_time
                    )
                    processing_state.update_progress(processed=1)

                    # 添加到 ETA 估算器
                    self.eta_estimator.add_completion(
                        processing_time, progress_data.completed_count
                    )

                    self._emit_progress(chunk_id=chunk_id)

                    logger.debug(f"块 {chunk_id} 处理完成，耗时: {processing_time:.2f}秒")
                    return outline_data

                except asyncio.CancelledError:
                    logger.info(f"块 {chunk_id} 处理被取消")
                    raise
                except (APIError, ProcessingError) as e:
                    # API错误和处理错误应该重试
                    last_error = e
                    if attempt < self.processing_config.max_retry:
                        logger.warning(
                            f"块 {chunk_id} 第 {attempt}/{self.processing_config.max_retry} 次尝试失败: {type(e).__name__}: {e}，将重试"
                        )
                        await asyncio.sleep(RETRY_BACKOFF_BASE * attempt)  # 指数退避
                    else:
                        logger.error(
                            f"块 {chunk_id} 经过 {self.processing_config.max_retry} 次重试后仍然失败: {type(e).__name__}: {e}"
                        )
                except Exception as e:
                    # 其他未预期的异常也记录并重试
                    last_error = e
                    logger.error(
                        f"块 {chunk_id} 遇到未预期的错误: {type(e).__name__}: {e}", exc_info=True
                    )
                    if attempt < self.processing_config.max_retry:
                        logger.warning(
                            f"块 {chunk_id} 将在 {RETRY_BACKOFF_BASE * attempt} 秒后重试"
                        )
                        await asyncio.sleep(RETRY_BACKOFF_BASE * attempt)
                    else:
                        logger.error(f"块 {chunk_id} 已达到最大重试次数，放弃处理")

            # 所有重试都失败后，尝试拆分为多个小块重新处理
            logger.info(f"块 {chunk_id} 重试失败，尝试拆分为{SUB_CHUNK_COUNT}个小块重新处理")
            try:
                partial_outlines = await self._process_failing_chunk_as_partial(
                    chunk, sem, progress_data, processing_state
                )
                # 返回合并后的部分完成大纲
                if partial_outlines:
                    # 将部分完成的小块合并为一个大纲
                    merged_outline = self._merge_partial_outlines(partial_outlines, chunk_id)
                    # 注意: _process_failing_chunk_as_partial 已经将小块添加到 partial_outlines
                    # 这里只返回合并结果，不再添加到 progress_data.outlines
                    # 避免重复添加（在进度恢复时会自动合并）
                    return merged_outline
                else:
                    raise ProcessingError(f"块 {chunk_id} 拆分后所有小块都失败")
            except Exception as split_error:
                logger.error(f"块 {chunk_id} 拆分重试也失败: {split_error}")
                # 拆分失败，按原逻辑处理
                processing_state.update_progress(processed=0, failed=1)
                self.progress_service.add_progress_error(
                    progress_data, chunk_id, str(last_error or split_error)
                )
                self._emit_progress(chunk_id=chunk_id, error=str(last_error or split_error))
                raise ProcessingError(
                    f"块 {chunk_id} 处理失败: {str(last_error or split_error)}"
                ) from (last_error or split_error)

    async def _process_failing_chunk_as_partial(
        self,
        chunk: TextChunk,
        sem: asyncio.Semaphore,
        progress_data: Any,
        processing_state: ProcessingState,
    ) -> list[dict[str, Any]]:
        """将失败的分块拆分为多个小块，逐个处理，返回成功的小块大纲列表"""
        chunk_id = chunk.id

        # 拆分为多个小块
        sub_chunks = self._split_chunk_into_five(chunk)
        logger.info(f"块 {chunk_id} 已拆分为 {len(sub_chunks)} 个小块")

        # 处理每个小块
        successful_sub_outlines: list[dict[str, Any]] = []
        failed_sub_chunks = 0

        for sub_idx, sub_chunk in enumerate(sub_chunks, 1):
            try:
                # 为子块创建唯一标识符
                sub_chunk_id = f"{chunk_id}_sub_{sub_idx}"
                logger.debug(f"处理块 {chunk_id} 的小块 {sub_idx}/{len(sub_chunks)}")

                start_time = datetime.now()

                # 生成提示
                prompt = chunk_prompt(sub_chunk.content, chunk_id)

                # 调用LLM（使用唯一标识符以便追踪）
                llm_response = await self.llm_service.call(prompt, chunk_id)
                response = llm_response.content

                # 累计token使用情况
                if llm_response.token_usage:
                    prompt_tokens = llm_response.token_usage.get("prompt_tokens", 0) or 0
                    completion_tokens = llm_response.token_usage.get("completion_tokens", 0) or 0
                    total_tokens = llm_response.token_usage.get("total_tokens", 0) or 0

                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += total_tokens

                    # 记录子块级别的token使用情况
                    logger.debug(
                        f"子块 {sub_chunk_id} Token使用: 输入={prompt_tokens}, "
                        f"输出={completion_tokens}, 总计={total_tokens}"
                    )

                # 解析响应
                sub_outline = self._parse_llm_response(response, chunk_id)

                # 记录处理时间和子块元数据
                processing_time = (datetime.now() - start_time).total_seconds()
                sub_outline["raw_response"] = response
                sub_outline["processing_time"] = processing_time
                sub_outline["sub_chunk_index"] = sub_idx  # 小块索引
                sub_outline["sub_chunk_id"] = sub_chunk_id  # 唯一标识符
                sub_outline["original_chunk_id"] = chunk_id  # 原始块ID

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

        # 检查是否有成功的小块
        if not successful_sub_outlines:
            logger.warning(f"块 {chunk_id} 所有小块都处理失败")
            return []

        # 更新部分完成状态
        progress_data.partial_indices.add(chunk_id)
        # 注意：部分完成的块不应该添加到completed_indices，避免重复计数和恢复时的混淆
        progress_data.partial_outlines.extend(successful_sub_outlines)
        processing_state.update_partial(1)

        # 注意：部分完成不计入processed_chunks，只计入partial_chunks
        # 这样可以区分完全完成和部分完成的块

        logger.info(
            f"块 {chunk_id} 部分完成: 成功 {len(successful_sub_outlines)}/{len(sub_chunks)} 个小块，失败 {failed_sub_chunks} 个小块"
        )
        self._emit_progress(chunk_id=chunk_id, partial_info=f"{chunk_id}块部分完成")

        return successful_sub_outlines

    def _split_chunk_into_five(self, chunk: TextChunk) -> list[TextChunk]:
        """将一个块拆分为多个小块"""
        text = chunk.content
        total_length = len(text)
        chunk_size = total_length // SUB_CHUNK_COUNT

        sub_chunks: list[TextChunk] = []
        start_position = chunk.start_position

        for idx in range(SUB_CHUNK_COUNT):
            start = idx * chunk_size
            if idx == SUB_CHUNK_COUNT - 1:  # 最后一块包含剩余所有内容
                end = total_length
            else:
                end = start + chunk_size

            sub_content = text[start:end]
            sub_chunk = TextChunk(
                id=chunk.id,  # 保持原始chunk_id
                content=sub_content,
                token_count=count_tokens(sub_content),
                start_position=start_position + start,
                end_position=start_position + end,
            )
            sub_chunks.append(sub_chunk)

        return sub_chunks

    def _merge_partial_outlines(
        self, partial_outlines: list[dict[str, Any]], original_chunk_id: int
    ) -> dict[str, Any]:
        """将部分完成的小块大纲合并为一个完整大纲"""
        all_plot: list[str] = []
        all_characters: set[str] = set()
        all_relationships: set[tuple[str, str, str]] = set()

        for outline in partial_outlines:
            # 合并剧情（保持顺序）
            plot = outline.get("plot", [])
            if isinstance(plot, list):
                all_plot.extend([p for p in plot if isinstance(p, str)])

            # 合并人物
            characters = outline.get("characters", [])
            if isinstance(characters, list):
                all_characters.update(c for c in characters if isinstance(c, str))

            # 合并关系（保留完整的3元组：人物A、人物B、关系描述）
            relationships = outline.get("relationships", [])
            if isinstance(relationships, list):
                for rel in relationships:
                    if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                        all_relationships.add((str(rel[0]), str(rel[1]), str(rel[2])))

        # 创建合并后的大纲
        merged_outline: dict[str, Any] = {
            "chunk_id": original_chunk_id,
            "is_partial": True,
            "sub_chunk_count": len(partial_outlines),
            "plot": all_plot,
            "characters": sorted(all_characters),
            "relationships": [list(rel) for rel in sorted(all_relationships)],
            "partial_outlines": partial_outlines,  # 保留原始小块大纲
        }

        # 如果有原始响应，合并它们
        if all("raw_response" in outline for outline in partial_outlines):
            merged_outline["raw_response"] = "\n\n".join(
                [outline["raw_response"] for outline in partial_outlines]
            )

        # 处理时间取平均值
        processing_times = [
            outline.get("processing_time", 0)
            for outline in partial_outlines
            if "processing_time" in outline
        ]
        if processing_times:
            merged_outline["processing_time"] = sum(processing_times) / len(processing_times)

        return merged_outline

    def _parse_llm_response(self, response: str, chunk_id: int) -> dict[str, Any]:
        """解析LLM响应"""
        import json
        import re

        try:
            # 尝试直接解析JSON
            data = json.loads(response)
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            raise ValueError("LLM响应不是JSON对象")

        except (json.JSONDecodeError, ValueError, TypeError):
            # 尝试提取JSON部分
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if isinstance(data, dict):
                        return cast(dict[str, Any], data)
                except (json.JSONDecodeError, ValueError, TypeError):
                    # JSON解析失败，继续使用默认结构
                    pass

            # 如果无法解析，创建基础结构
            logger.warning(f"块 {chunk_id} 响应无法解析为JSON，使用原始文本")
            return {
                "chunk_id": chunk_id,
                "plot": [response],
                "characters": [],
                "relationships": [],
            }

    async def merge_outlines_recursive(
        self,
        outlines: list[dict[str, Any]] | list[str],
        level: int = 1,
        is_text_mode: bool = False,
    ) -> str:
        """递归合并大纲"""
        if not outlines:
            return ""

        # 检查是否被取消
        if self.cancel_event.is_set():
            logger.info("任务已被取消")
            raise asyncio.CancelledError()

        # 检查处理状态
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")

        # 更新合并层级
        self.processing_state.merge_level += 1
        self.processing_state.merge_outlines_count = len(outlines)
        self._emit_progress()

        # 判断模式
        if not is_text_mode and len(outlines) > 0:
            if isinstance(outlines[0], str):
                is_text_mode = True
            elif isinstance(outlines[0], dict) and "merged_content" in outlines[0]:
                outlines_dicts = cast(list[dict[str, Any]], outlines)
                outlines = [item["merged_content"] for item in outlines_dicts]
                is_text_mode = True

        # 生成合并提示
        if is_text_mode:
            merge_prompt_text = merge_text_prompt(cast(list[str], outlines))
        else:
            outlines_json = json.dumps(outlines, ensure_ascii=False)
            merge_prompt_text = merge_prompt(outlines_json)

        # 检查token数量
        input_tokens = count_tokens(merge_prompt_text)
        max_input_tokens = int(self.processing_config.model_max_tokens * 0.8)

        if input_tokens <= max_input_tokens:
            # 直接合并
            logger.debug(f"层级 {level}: 合并 {len(outlines)} 个大纲块")
            llm_response = await self.llm_service.call(merge_prompt_text)

            # 检查是否被取消（在LLM调用后）
            if self.cancel_event.is_set():
                logger.info("任务已被取消")
                raise asyncio.CancelledError()

            # 累计token使用情况
            if llm_response.token_usage:
                prompt_tokens = llm_response.token_usage.get("prompt_tokens", 0) or 0
                completion_tokens = llm_response.token_usage.get("completion_tokens", 0) or 0
                total_tokens = llm_response.token_usage.get("total_tokens", 0) or 0

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens

                logger.debug(
                    f"合并 Token使用: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}"
                )

            # 减少合并层级
            self.processing_state.merge_level -= 1
            self._emit_progress()
            return llm_response.content

        # 需要拆分
        logger.warning(f"层级 {level}: 输入过大，拆分为多个批次")
        batch_size = max(1, int(max_input_tokens / (input_tokens / len(outlines)) * 0.8))

        # 分批处理
        batches = []
        for i in range(0, len(outlines), batch_size):
            batches.append(outlines[i : i + batch_size])

        # 更新批次信息
        self.processing_state.merge_batch_total = len(batches)
        self.processing_state.merge_batch_current = 0
        self._emit_progress()

        # 递归处理每批
        merged_batches = []
        for idx, batch in enumerate(batches, 1):
            # 检查是否被取消
            if self.cancel_event.is_set():
                logger.info("任务已被取消")
                raise asyncio.CancelledError()

            # 更新当前批次
            self.processing_state.merge_batch_current = idx
            self._emit_progress()
            logger.debug(f"处理批次 {idx}/{len(batches)}")
            merged = await self.merge_outlines_recursive(batch, level + 1, is_text_mode)
            merged_batches.append(merged)

        # 如果只有一个批次，直接返回
        if len(merged_batches) == 1:
            # 减少合并层级
            self.processing_state.merge_level -= 1
            self._emit_progress()
            return merged_batches[0]

        # 合并批次结果
        logger.debug(f"合并 {len(merged_batches)} 个批次的结果")
        result = await self.merge_outlines_recursive(merged_batches, level + 1, is_text_mode=True)
        # 减少合并层级
        self.processing_state.merge_level -= 1
        self._emit_progress()
        return result

    async def _save_results(
        self, outlines: list[dict[str, Any]], final_outline: str, original_file: str
    ) -> None:
        """保存结果文件"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")

        processing_state = self.processing_state
        processing_state.current_phase = "saving"
        self._emit_progress()

        # 确保输出目录存在
        output_dir = self.file_service.ensure_output_directory()

        # 保存中间结果
        chunk_outlines_path = output_dir / "chunk_outlines.json"
        self.file_service.write_json_file(chunk_outlines_path, outlines)

        # 保存最终大纲
        original_path = Path(original_file)
        final_outline_filename = f"{original_path.stem}-提纲{original_path.suffix}"
        final_outline_path = output_dir / final_outline_filename
        self.file_service.write_text_file(final_outline_path, final_outline)

        # 保存处理元数据
        metadata = {
            "original_file": original_file,
            "processing_time": processing_state.elapsed_time,
            "total_chunks": len(outlines),
            "success_rate": processing_state.success_rate,
            "completed_at": datetime.now().isoformat(),
            "summary": processing_state.get_summary(),
        }
        metadata_path = output_dir / "processing_metadata.json"
        self.file_service.write_json_file(metadata_path, metadata)

        logger.info(f"结果已保存到: {output_dir}")

    def _cleanup_intermediate_outputs(self, output_dir: Path) -> list[str]:
        """删除中间产物，保留最终大纲"""
        targets = [
            output_dir / "chunk_outlines.json",
            output_dir / "processing_metadata.json",
        ]
        removed: list[str] = []
        for path in targets:
            try:
                if path.exists():
                    path.unlink()
                    removed.append(path.name)
            except Exception as err:  # noqa: BLE001
                logger.debug(f"删除中间文件失败 {path}: {err}")
        return removed

    def get_processing_summary(self) -> dict[str, Any]:
        """获取处理摘要"""
        if not self.processing_state:
            return {"status": "not_started"}

        return self.processing_state.get_summary()

    def _emit_progress(
        self,
        chunk_id: int | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        partial_info: str | None = None,
    ) -> None:
        """向外部回调当前进度，便于 Web UI 实时显示。
        设计为尽量轻量、容错，不影响主流程。
        """
        if not self.progress_callback or not self.processing_state:
            return

        total = self.processing_state.total_chunks or 0
        completed = self.processing_state.processed_chunks
        failed = self.processing_state.failed_chunks
        partial = self.processing_state.partial_chunks

        # 计算部分完成块的权重
        # 每个块拆分为5个小块，每个小块权重为 1/5 = 0.2
        partial_weight = 0.0
        if self.current_progress_data and partial > 0:
            # 按chunk_id分组统计成功的小块数量
            from collections import defaultdict

            chunk_sub_counts: dict[int, int] = defaultdict(int)

            for outline in self.current_progress_data.partial_outlines:
                original_chunk_id = outline.get("original_chunk_id") or outline.get("chunk_id")
                if original_chunk_id in self.current_progress_data.partial_indices:
                    chunk_sub_counts[original_chunk_id] += 1

            # 计算总权重：每个小块按 1/SUB_CHUNK_COUNT 计算
            for _chunk_id, sub_count in chunk_sub_counts.items():
                partial_weight += sub_count / SUB_CHUNK_COUNT

        effective_completed = completed + partial_weight
        progress = (effective_completed / total) if total else 0.0

        # 使用 ETA 估算器计算剩余时间
        eta_result = self.eta_estimator.estimate(
            total_chunks=total,
            completed_chunks=completed,
            failed_chunks=failed,
        )

        payload: dict[str, Any] = {
            "progress": progress,
            "completed_chunks": completed,
            "failed_chunks": failed,
            "partial_chunks": partial,
            "total_chunks": total,
            "phase": self.processing_state.current_phase,
            "merge_level": self.processing_state.merge_level,
            "merge_batch_current": self.processing_state.merge_batch_current,
            "merge_batch_total": self.processing_state.merge_batch_total,
            "merge_outlines_count": self.processing_state.merge_outlines_count,
        }
        if eta_result["eta_seconds"] is not None:
            payload["eta_seconds"] = eta_result["eta_seconds"]
            payload["eta_confidence"] = eta_result["confidence"]
            payload["eta_method"] = eta_result["method"]
        if chunk_id is not None:
            payload["last_chunk_id"] = chunk_id
        if error is not None:
            payload["last_error"] = error
        if token_usage is not None:
            payload["token_usage"] = token_usage
        if partial_info is not None:
            payload["partial_info"] = partial_info

        try:
            self.progress_callback(payload)
        except Exception:
            # 避免外部回调异常中断主流程
            logger.debug("Progress callback failed", exc_info=True)
