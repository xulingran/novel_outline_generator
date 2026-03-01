"""
处理管道模块
实现小说处理执行管道，协调各个处理阶段
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exceptions import ProcessingError

if TYPE_CHECKING:
    from services.novel_processing_service import NovelProcessingService

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """小说处理执行管道

    协调小说处理的各个阶段：
    1. loading - 加载和验证文件
    2. splitting - 文本分块
    3. resuming - 处理进度恢复
    4. processing - 处理文本块
    5. merging - 合并大纲
    6. saving - 保存结果

    每个阶段都统一处理进度报告和错误转换。
    """

    def __init__(self, service: "NovelProcessingService") -> None:
        """初始化处理管道

        Args:
            service: 小说处理服务实例
        """
        self.service = service
        self.state = service.processing_state
        self.config = service.processing_config

    async def execute(
        self,
        file_path: str,
        output_dir: str | None,
        resume: bool,
    ) -> dict[str, Any]:
        """执行完整处理流程

        Args:
            file_path: 小说文件路径
            output_dir: 输出目录（可选）
            resume: 是否恢复进度

        Returns:
            处理结果字典
        """
        # 1. 加载文件
        text, encoding = await self._execute_phase(
            "loading",
            self._load_file,
            file_path,
        )

        # 2. 分块
        chunks = await self._execute_phase(
            "splitting",
            self._split_text,
            text,
        )
        self.state.total_chunks = len(chunks)
        if not chunks:
            raise ProcessingError("未检测到可处理的内容")

        # 3. 处理或恢复进度
        outlines = await self._handle_processing(chunks, resume, encoding)

        # 4. 合并大纲
        final_outline = await self._execute_phase(
            "merging",
            self._merge,
            outlines,
        )

        # 5. 保存结果
        await self._execute_phase(
            "saving",
            self._save,
            outlines,
            final_outline,
            file_path,
            output_dir,
        )

        # 6. 完成处理
        return self._build_result(outlines, final_outline, chunks)

    async def _execute_phase(
        self,
        phase_name: str,
        phase_func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """执行单个阶段，统一处理进度和错误

        Args:
            phase_name: 阶段名称
            phase_func: 阶段执行函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            阶段执行结果

        Raises:
            ProcessingError: 阶段执行失败
        """
        self.state.current_phase = phase_name
        self.service._emit_progress()

        try:
            result = await phase_func(*args, **kwargs)
            logger.debug(f"阶段 '{phase_name}' 完成")
            return result
        except Exception as e:
            logger.error(f"阶段 '{phase_name}' 失败: {e}")
            raise ProcessingError(f"阶段 '{phase_name}' 失败: {e}") from e

    async def _load_file(self, file_path: str) -> tuple[str, str]:
        """加载并验证文件"""
        return await self.service._load_and_validate_file(file_path)

    async def _split_text(self, text: str) -> list:
        """将文本分割成块"""
        return self.service._split_text_into_chunks(text)

    async def _handle_processing(
        self,
        chunks: list,
        resume: bool,
        encoding: str,
    ) -> list[dict[str, Any]]:
        """处理文本块或恢复进度"""
        file_path = self.state.file_path

        # 处理或恢复进度
        progress_data = await self.service._handle_progress_resume(
            file_path, chunks, resume, encoding
        )

        if progress_data is None:
            # 全新处理
            return await self.service._process_chunks(chunks)

        # 恢复进度
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
            new_outlines = await self.service._process_chunks(
                remaining_chunks,
                progress_data=progress_data,
                total_chunks=len(chunks),
            )
            outlines.extend(new_outlines)
        else:
            logger.info("恢复进度: 所有块已处理，直接进入合并")
            self.service.progress_service.finalize_progress(progress_data)

        if outlines:
            outlines.sort(
                key=lambda item: (item.get("chunk_id", 0) if isinstance(item, dict) else 0)
            )

        return outlines

    async def _merge(self, outlines: list[dict[str, Any]]) -> str:
        """递归合并大纲"""
        result = await self.service.merge_outlines_recursive(outlines)
        # 报告 token 使用情况
        self.service._emit_progress(
            token_usage={
                "prompt_tokens": self.service.total_prompt_tokens,
                "completion_tokens": self.service.total_completion_tokens,
                "total_tokens": self.service.total_tokens,
            }
        )
        return result

    async def _save(
        self,
        outlines: list[dict[str, Any]],
        final_outline: str,
        file_path: str,
        output_dir: str | None,
    ) -> None:
        """保存结果"""
        # 更新输出目录
        if output_dir:
            self.config.output_dir = output_dir

        # 保存结果
        await self.service._save_results(outlines, final_outline, file_path)

        # 清理备份文件
        try:
            removed = self.service.file_service.remove_backups(self.config.output_dir, "*.bak")
            logger.debug(f"已清理备份文件: {removed} 个")
        except Exception as e:
            logger.warning(f"清理备份文件失败: {e}")

        # 清理中间结果文件
        try:
            cleaned = self.service._cleanup_intermediate_outputs(Path(self.config.output_dir))
            if cleaned:
                logger.info(f"已清理中间结果文件: {', '.join(cleaned)}")
        except Exception as e:
            logger.warning(f"清理中间结果文件失败: {e}")

    def _build_result(
        self,
        outlines: list[dict[str, Any]],
        final_outline: str,
        chunks: list,
    ) -> dict[str, Any]:
        """构建处理结果字典"""
        return {
            "success": True,
            "final_outline": final_outline,
            "chunk_count": len(chunks),
            "processing_time": self.state.elapsed_time,
            "output_dir": self.config.output_dir,
            "token_usage": {
                "prompt_tokens": self.service.total_prompt_tokens,
                "completion_tokens": self.service.total_completion_tokens,
                "total_tokens": self.service.total_tokens,
            },
        }
