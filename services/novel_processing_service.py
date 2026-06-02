"""
小说处理服务模块
核心业务逻辑，处理小说文本并生成大纲
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_processing_config
from exceptions import ProcessingError
from models.outline import TextChunk
from models.processing_state import ProcessingState, ProgressData
from services.chunk_processor import ChunkProcessor
from services.eta_estimator import ETAEstimator
from services.file_service import FileService
from services.llm_service import LLMService, create_llm_service
from services.outline_merger import OutlineMerger
from services.progress_service import ProgressService
from services.progress_tracker import ProgressTracker
from splitter import split_text, split_text_stream
from tokenizer import count_tokens

logger = logging.getLogger(__name__)


class NovelProcessingService:
    """小说处理服务类"""

    _MAX_MERGE_LEVELS: int = 10

    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: asyncio.Event | None = None,
        llm_service: LLMService | None = None,
    ):
        self.processing_config = get_processing_config()
        self.llm_service = llm_service or create_llm_service()
        self.progress_service = ProgressService()
        self.file_service = FileService()
        self.processing_state: ProcessingState | None = None
        self.progress_callback = progress_callback
        self.current_progress_data: ProgressData | None = None
        self.cancel_event = cancel_event or asyncio.Event()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.force_complete: bool = False
        self.eta_estimator = ETAEstimator(
            window_size=20,
            outlier_threshold=2.5,
            min_samples=3,
        )
        self.eta_estimator.set_parallel_limit(self.processing_config.parallel_limit)
        self._init_sub_components()

    def _init_sub_components(self) -> None:
        """初始化子组件（将组装逻辑从构造器中分离）"""
        self._outline_merger = OutlineMerger(
            llm_service=self.llm_service,
            processing_config=self.processing_config,
            cancel_event=self.cancel_event,
        )
        self._progress_tracker = ProgressTracker(
            progress_callback=self.progress_callback,
            eta_estimator=self.eta_estimator,
            progress_service=self.progress_service,
            processing_config=self.processing_config,
        )
        self._chunk_processor = ChunkProcessor(
            llm_service=self.llm_service,
            processing_config=self.processing_config,
            cancel_event=self.cancel_event,
            progress_service=self.progress_service,
            eta_estimator=self.eta_estimator,
            emit_progress_fn=self.emit_progress,
            accumulate_tokens_fn=self._accumulate_token_usage,
        )

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            logger.info("任务已被取消")
            raise asyncio.CancelledError()

    def _accumulate_token_usage(
        self, token_usage: dict[str, int] | None, context: str = ""
    ) -> None:
        """累加token使用情况

        Args:
            token_usage: LLM响应中的token使用情况
            context: 日志上下文信息（如块ID）
        """
        if not token_usage:
            return

        prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
        completion_tokens = token_usage.get("completion_tokens", 0) or 0
        total_tokens = token_usage.get("total_tokens", 0) or 0

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        context_str = f" {context}" if context else ""
        logger.debug(
            f"Token使用{context_str}: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}"
        )

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
        self._initialize_processing(file_path)

        try:
            chunks, encoding = await self._prepare_chunks(file_path)
            outlines = await self._collect_outlines(file_path, chunks, resume, encoding)
            final_outline = await self._merge_outlines(outlines)
            await self._save_and_cleanup(file_path, outlines, final_outline, output_dir)

            if self.processing_state is None:
                raise ProcessingError("处理状态未初始化")
            self.processing_state.complete()
            self.emit_progress()

            return self._build_success_result(final_outline, len(chunks))

        except asyncio.CancelledError:
            logger.info("Novel processing cancelled")
            raise
        except Exception as e:
            if self.processing_state:
                self.processing_state.fail(str(e))
                self.emit_progress()
            logger.error(f"处理小说失败: {e}")
            raise ProcessingError(f"处理小说失败: {str(e)}") from e

    def _initialize_processing(self, file_path: str) -> None:
        """初始化处理上下文。"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        self.processing_state = ProcessingState(file_path=file_path, total_chunks=0)
        self.processing_state.current_phase = "loading"
        self.emit_progress()

    async def _prepare_chunks(self, file_path: str) -> tuple[list[TextChunk], str]:
        """加载文件并切分为文本块。"""
        if self._should_use_streaming(file_path):
            return self._prepare_chunks_streaming(file_path)

        text, encoding = await self.load_and_validate_file(file_path)
        chunks = self.split_text_into_chunks(text)

        if self.processing_state:
            self.processing_state.total_chunks = len(chunks)
        if not chunks:
            raise ProcessingError("未检测到可处理的内容")

        self.emit_progress()
        return chunks, encoding

    def _should_use_streaming(self, file_path: str) -> bool:
        """是否启用流式切块。"""
        file_size = self.file_service.get_file_size(file_path)
        threshold = self.processing_config.stream_split_threshold_mb * 1024 * 1024
        return file_size >= threshold

    def _prepare_chunks_streaming(self, file_path: str) -> tuple[list[TextChunk], str]:
        """大文件流式切块，降低内存峰值。"""
        logger.info("检测到大文件，启用流式切块")
        encoding = self.file_service.detect_file_encoding(file_path)

        raw_chunks = split_text_stream(self.file_service.iter_text_file(file_path))
        chunks: list[TextChunk] = []
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

        if self.processing_state:
            self.processing_state.total_chunks = len(chunks)
        if not chunks:
            raise ProcessingError("未检测到可处理的内容")

        self.emit_progress()
        return chunks, encoding

    async def _collect_outlines(
        self, file_path: str, chunks: list[TextChunk], resume: bool, encoding: str
    ) -> list[dict[str, Any]]:
        """处理或恢复文本块并返回排序后的大纲列表。"""
        progress_data = await self.handle_progress_resume(file_path, chunks, resume, encoding)
        if progress_data is None:
            outlines = await self.process_chunks(chunks)
        else:
            outlines = await self._process_remaining_chunks(chunks, progress_data)

        outlines.sort(key=lambda item: (item.get("chunk_id", 0) if isinstance(item, dict) else 0))
        return outlines

    async def _process_remaining_chunks(
        self, chunks: list[TextChunk], progress_data: ProgressData
    ) -> list[dict[str, Any]]:
        """恢复处理时，仅处理未完成文本块。"""
        outlines = list(progress_data.outlines)
        failed_ids = {
            error["chunk_id"]
            for error in progress_data.errors
            if isinstance(error, dict) and "chunk_id" in error
        }
        completed_ids = progress_data.completed_indices | progress_data.partial_indices | failed_ids
        remaining_chunks = [chunk for chunk in chunks if chunk.id not in completed_ids]
        if remaining_chunks:
            logger.info(f"恢复进度: 剩余 {len(remaining_chunks)} 个块待处理")
            new_outlines = await self.process_chunks(
                remaining_chunks,
                progress_data=progress_data,
                total_chunks=len(chunks),
            )
            outlines.extend(new_outlines)
        else:
            logger.info("恢复进度: 所有块已处理，直接进入合并")
            self.progress_service.finalize_progress(progress_data)
        return outlines

    async def _merge_outlines(self, outlines: list[dict[str, Any]]) -> str:
        """合并分块大纲并上报 token 统计"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")
        self.processing_state.current_phase = "merging"
        self.emit_progress()
        final_outline = await self.merge_outlines_recursive(outlines)
        self.emit_progress(
            token_usage={
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            }
        )
        return final_outline

    async def _save_and_cleanup(
        self,
        file_path: str,
        outlines: list[dict[str, Any]],
        final_outline: str,
        output_dir: str | None,
    ) -> None:
        """保存结果并执行成功后的清理。"""
        if output_dir:
            self.processing_config.output_dir = output_dir

        await self.save_results(outlines, final_outline, file_path)

        try:
            removed = self.file_service.remove_backups(self.processing_config.output_dir, "*.bak")
            logger.debug(f"已清理备份文件: {removed} 个")
        except Exception as cleanup_err:
            logger.warning(f"清理备份文件失败: {cleanup_err}")

        try:
            cleaned = self.cleanup_intermediate_outputs(Path(self.processing_config.output_dir))
            if cleaned:
                logger.info(f"已清理中间结果文件: {', '.join(cleaned)}")
        except Exception as cleanup_err:
            logger.warning(f"清理中间结果文件失败: {cleanup_err}")

    def _build_success_result(self, final_outline: str, chunk_count: int) -> dict[str, Any]:
        """构建成功返回结果。"""
        processing_time = self.processing_state.elapsed_time if self.processing_state else 0.0
        return {
            "success": True,
            "final_outline": final_outline,
            "chunk_count": chunk_count,
            "processing_time": processing_time,
            "output_dir": self.processing_config.output_dir,
            "token_usage": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }

    async def load_and_validate_file(self, file_path: str) -> tuple[str, str]:
        logger.info(f"正在读取文件: {file_path}")
        try:
            text, encoding = self.file_service.read_text_file(file_path)
            if not text.strip():
                raise ProcessingError("文件内容为空")
            return text, encoding
        except Exception as e:
            raise ProcessingError(f"读取文件失败: {str(e)}") from e

    def split_text_into_chunks(self, text: str) -> list[TextChunk]:
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

    async def handle_progress_resume(
        self,
        file_path: str,
        chunks: list[TextChunk],
        resume: bool,
        encoding: str,
    ) -> ProgressData | None:
        """处理进度恢复（委托给 ProgressTracker）"""
        progress_data = await self._progress_tracker.handle_progress_resume(
            file_path=file_path,
            chunks=chunks,
            resume=resume,
            encoding=encoding,
            processing_state=self.processing_state,
        )
        if progress_data is not None:
            self.current_progress_data = progress_data
            self.emit_progress()
        return progress_data

    async def process_chunks(
        self,
        chunks: list[TextChunk],
        progress_data: ProgressData | None = None,
        total_chunks: int | None = None,
    ) -> list[dict[str, Any]]:
        """处理所有文本块（委托给 ChunkProcessor）"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")
        outlines, updated_progress = await self._chunk_processor.process_chunks(
            chunks,
            self.processing_state,
            progress_data=progress_data,
            total_chunks=total_chunks,
            force_complete=self.force_complete,
        )
        self.current_progress_data = updated_progress
        return outlines

    async def _process_single_chunk(
        self, chunk: TextChunk, sem: asyncio.Semaphore, progress_data: Any
    ) -> dict[str, Any]:
        """处理单个文本块（委托给 ChunkProcessor）"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")
        return await self._chunk_processor.process_single_chunk(
            chunk, sem, self.processing_state, progress_data
        )

    async def _process_failing_chunk_as_partial(
        self,
        chunk: TextChunk,
        sem: asyncio.Semaphore,
        progress_data: Any,
        processing_state: ProcessingState,
    ) -> list[dict[str, Any]]:
        """将失败的分块拆分为多个小块（委托给 ChunkProcessor）"""
        return await self._chunk_processor.process_failing_chunk_as_partial(
            chunk, sem, processing_state, progress_data
        )

    def _split_chunk_into_sub_chunks(self, chunk: TextChunk) -> list[TextChunk]:
        """将一个块拆分为多个小块（委托给 ChunkProcessor）"""
        return self._chunk_processor.split_chunk_into_sub_chunks(chunk)

    def _merge_partial_outlines(
        self, partial_outlines: list[dict[str, Any]], original_chunk_id: int
    ) -> dict[str, Any]:
        """将部分完成的小块大纲合并为一个完整大纲（委托给 OutlineMerger）"""
        return OutlineMerger.merge_partial_outlines(partial_outlines, original_chunk_id)

    def _parse_llm_response(self, response: str, chunk_id: int | str) -> dict[str, Any]:
        """解析LLM响应（委托给 ChunkProcessor）"""
        return ChunkProcessor.parse_llm_response(response, chunk_id)

    async def merge_outlines_recursive(
        self,
        outlines: list[dict[str, Any]] | list[str],
        level: int = 1,
        is_text_mode: bool = False,
    ) -> str:
        """递归合并大纲（委托给 OutlineMerger）"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")
        return await self._outline_merger.merge_outlines_recursive(
            outlines=outlines,
            processing_state=self.processing_state,
            emit_progress_fn=self.emit_progress,
            accumulate_tokens_fn=self._accumulate_token_usage,
            level=level,
            is_text_mode=is_text_mode,
        )

    async def save_results(
        self, outlines: list[dict[str, Any]], final_outline: str, original_file: str
    ) -> None:
        """保存结果文件"""
        if self.processing_state is None:
            raise ProcessingError("处理状态未初始化")

        processing_state = self.processing_state
        processing_state.current_phase = "saving"
        self.emit_progress()

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

    def cleanup_intermediate_outputs(self, output_dir: Path) -> list[str]:
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

    def emit_progress(
        self,
        chunk_id: int | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        partial_info: str | None = None,
    ) -> None:
        """向外部回调当前进度（委托给 ProgressTracker）"""
        self._progress_tracker.emit_progress(
            processing_state=self.processing_state,
            current_progress_data=self.current_progress_data,
            chunk_id=chunk_id,
            error=error,
            token_usage=token_usage,
            partial_info=partial_info,
        )
