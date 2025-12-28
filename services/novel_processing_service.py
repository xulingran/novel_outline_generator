"""
小说处理服务模块
核心业务逻辑，处理小说文本并生成大纲
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
import json
from datetime import datetime
from pathlib import Path

from models.outline import TextChunk
from models.processing_state import ProcessingState, ProgressData
from services.llm_service import create_llm_service
from services.progress_service import ProgressService
from services.file_service import FileService
from splitter import split_text
from tokenizer import count_tokens
from prompts import chunk_prompt, merge_prompt, merge_text_prompt
from config import get_processing_config
from exceptions import ProcessingError, APIError

logger = logging.getLogger(__name__)


class NovelProcessingService:
    """小说处理服务类"""

    def __init__(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.processing_config = get_processing_config()
        self.llm_service = create_llm_service()
        self.progress_service = ProgressService()
        self.file_service = FileService()
        self.processing_state: Optional[ProcessingState] = None
        self.progress_callback = progress_callback
        self.current_progress_data: Optional[ProgressData] = None
        # Token统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    async def process_novel(self,
                          file_path: str,
                          output_dir: Optional[str] = None,
                          resume: bool = True) -> Dict[str, Any]:
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
            outlines = await self._handle_progress_resume(
                file_path, chunks, resume
            )

            # 3. 处理文本块
            if outlines is None:
                outlines = await self._process_chunks(chunks)

            # 4. 合并大纲
            self.processing_state.current_phase = "merging"
            self._emit_progress()
            final_outline = await self.merge_outlines_recursive(outlines)

            # 5. 保存结果
            if output_dir:
                self.processing_config.output_dir = output_dir

            await self._save_results(outlines, final_outline, file_path)
            # 5.1 清理备份文件（成功完成后删除 outputs 下的 .bak）
            try:
                removed = self.file_service.remove_backups(self.processing_config.output_dir, "*.bak")
                logger.debug(f"已清理备份文件: {removed} 个")
            except Exception as cleanup_err:
                logger.warning(f"清理备份文件失败: {cleanup_err}")

            # 5.2 清理中间结果文件
            try:
                cleaned = self._cleanup_intermediate_outputs(Path(self.processing_config.output_dir))
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
                    "total_tokens": self.total_tokens
                }
            }

        except Exception as e:
            if self.processing_state:
                self.processing_state.fail(str(e))
                self._emit_progress()
            logger.error(f"处理小说失败: {e}")
            raise ProcessingError(f"处理小说失败: {str(e)}") from e

    async def _load_and_validate_file(self, file_path: str) -> Tuple[str, str]:
        """加载并验证文件"""
        logger.info(f"正在读取文件: {file_path}")
        try:
            text, encoding = self.file_service.read_text_file(file_path)
            if not text.strip():
                raise ProcessingError("文件内容为空")
            return text, encoding
        except Exception as e:
            raise ProcessingError(f"读取文件失败: {str(e)}") from e

    def _split_text_into_chunks(self, text: str) -> List[TextChunk]:
        """分割文本为块"""
        logger.info("正在分割文本...")
        try:
            raw_chunks = split_text(text)

            # 转换为TextChunk对象
            chunks = []
            position = 0
            for idx, chunk_content in enumerate(raw_chunks, 1):
                token_count = count_tokens(chunk_content)
                chunks.append(TextChunk(
                    id=idx,
                    content=chunk_content,
                    token_count=token_count,
                    start_position=position,
                    end_position=position + len(chunk_content)
                ))
                position += len(chunk_content)

            logger.info(f"文本已分割为 {len(chunks)} 个块")
            return chunks

        except Exception as e:
            raise ProcessingError(f"分割文本失败: {str(e)}") from e

    async def _handle_progress_resume(self,
                                     file_path: str,
                                     chunks: List[TextChunk],
                                     resume: bool) -> Optional[List[Dict[str, Any]]]:
        """处理进度恢复逻辑"""
        if not resume:
            return None

        # 加载进度
        progress_data = self.progress_service.load_progress()
        if not progress_data:
            return None

        # 计算当前哈希
        chunks_hash = ProgressData.calculate_chunks_hash([c.content for c in chunks])

        # 验证进度是否有效
        if not self.progress_service.is_progress_valid(
            progress_data, file_path, [c.content for c in chunks], chunks_hash
        ):
            logger.info("进度无效，将重新开始")
            self.progress_service.clear_progress()
            return None

        logger.info(f"恢复进度: {progress_data.completed_count}/{progress_data.total_chunks}")
        return progress_data.outlines

    async def _process_chunks(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """处理所有文本块"""
        self.processing_state.current_phase = "processing"
        self.processing_state.total_chunks = len(chunks)
        self._emit_progress()

        # 创建进度数据
        progress_data = self.progress_service.create_progress(
            self.processing_state.file_path,
            len(chunks),
            ProgressData.calculate_chunks_hash([c.content for c in chunks])
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
            successful_outlines = []
            for idx, result in enumerate(outlines, 1):
                if isinstance(result, Exception):
                    logger.error(f"块 {idx} 处理失败: {result}")
                    self.processing_state.add_error(f"块 {idx}: {str(result)}")
                    self.processing_state.update_progress(processed=0, failed=1)
                    self.progress_service.add_progress_error(progress_data, idx, str(result))
                    self._emit_progress(chunk_id=idx, error=str(result))
                else:
                    successful_outlines.append(result)

        except Exception as e:
            raise ProcessingError(f"处理文本块失败: {str(e)}") from e
        finally:
            # 保存最终进度
            self.progress_service.finalize_progress(progress_data)

        # 按chunk_id排序
        successful_outlines.sort(key=lambda x: x.get('chunk_id', 0))

        logger.info(f"成功处理 {len(successful_outlines)}/{len(chunks)} 个块")
        return successful_outlines

    async def _process_single_chunk(self,
                                   chunk: TextChunk,
                                   sem: asyncio.Semaphore,
                                   progress_data: Any) -> Dict[str, Any]:
        """处理单个文本块"""
        async with sem:
            chunk_id = chunk.id
            logger.debug(f"开始处理块 {chunk_id}")

            start_time = datetime.now()

            try:
                # 生成提示
                prompt = chunk_prompt(chunk.content, chunk_id)

                # 调用LLM
                llm_response = await self.llm_service.call(prompt, chunk_id)
                response = llm_response.content

                # 累计token使用情况
                if llm_response.token_usage:
                    prompt_tokens = llm_response.token_usage.get('prompt_tokens', 0) or 0
                    completion_tokens = llm_response.token_usage.get('completion_tokens', 0) or 0
                    total_tokens = llm_response.token_usage.get('total_tokens', 0) or 0

                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += total_tokens

                    logger.debug(f"块 {chunk_id} Token使用: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")

                # 尝试解析JSON响应
                outline_data = self._parse_llm_response(response, chunk_id)

                # 记录处理时间
                processing_time = (datetime.now() - start_time).total_seconds()

                # 保存原始响应
                outline_data['raw_response'] = response
                outline_data['processing_time'] = processing_time

                # 更新进度
                self.progress_service.update_chunk_completed(
                    progress_data, chunk_id, outline_data, processing_time
                )
                self.processing_state.update_progress(processed=1)
                self._emit_progress(chunk_id=chunk_id)

                logger.debug(f"块 {chunk_id} 处理完成，耗时: {processing_time:.2f}秒")
                return outline_data

            except APIError as e:
                logger.error(f"块 {chunk_id} API调用失败: {e}")
                self.processing_state.update_progress(processed=0, failed=1)
                self._emit_progress(chunk_id=chunk_id, error=str(e))
                raise ProcessingError(f"块 {chunk_id} API调用失败: {str(e)}") from e
            except Exception as e:
                logger.error(f"块 {chunk_id} 处理失败: {e}")
                self.processing_state.update_progress(processed=0, failed=1)
                self._emit_progress(chunk_id=chunk_id, error=str(e))
                raise ProcessingError(f"块 {chunk_id} 处理失败: {str(e)}") from e

    def _parse_llm_response(self, response: str, chunk_id: int) -> Dict[str, Any]:
        """解析LLM响应"""
        import json
        import re

        try:
            # 尝试直接解析JSON
            return json.loads(response)

        except json.JSONDecodeError:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except (json.JSONDecodeError, ValueError, TypeError):
                    # JSON解析失败，继续使用默认结构
                    pass

            # 如果无法解析，创建基础结构
            logger.warning(f"块 {chunk_id} 响应无法解析为JSON，使用原始文本")
            return {
                "chunk_id": chunk_id,
                "events": [response],
                "characters": [],
                "relationships": [],
                "conflicts": []
            }

    async def merge_outlines_recursive(self,
                                     outlines: List[Dict[str, Any]],
                                     level: int = 1,
                                     is_text_mode: bool = False) -> str:
        """递归合并大纲"""
        if not outlines:
            return ""

        # 判断模式
        if not is_text_mode and len(outlines) > 0:
            if isinstance(outlines[0], str):
                is_text_mode = True
            elif isinstance(outlines[0], dict) and "merged_content" in outlines[0]:
                outlines = [item["merged_content"] for item in outlines]
                is_text_mode = True

        # 生成合并提示
        if is_text_mode:
            merge_prompt_text = merge_text_prompt(outlines)
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

            # 累计token使用情况
            if llm_response.token_usage:
                prompt_tokens = llm_response.token_usage.get('prompt_tokens', 0) or 0
                completion_tokens = llm_response.token_usage.get('completion_tokens', 0) or 0
                total_tokens = llm_response.token_usage.get('total_tokens', 0) or 0

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens

                logger.debug(f"合并 Token使用: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")

            return llm_response.content

        # 需要拆分
        logger.warning(f"层级 {level}: 输入过大，拆分为多个批次")
        batch_size = max(1, int(max_input_tokens / (input_tokens / len(outlines)) * 0.8))

        # 分批处理
        batches = []
        for i in range(0, len(outlines), batch_size):
            batches.append(outlines[i:i + batch_size])

        # 递归处理每批
        merged_batches = []
        for idx, batch in enumerate(batches, 1):
            logger.debug(f"处理批次 {idx}/{len(batches)}")
            merged = await self.merge_outlines_recursive(batch, level + 1, is_text_mode)
            merged_batches.append(merged)

        # 如果只有一个批次，直接返回
        if len(merged_batches) == 1:
            return merged_batches[0]

        # 合并批次结果
        logger.debug(f"合并 {len(merged_batches)} 个批次的结果")
        return await self.merge_outlines_recursive(merged_batches, level + 1, is_text_mode=True)

    async def _save_results(self,
                           outlines: List[Dict[str, Any]],
                           final_outline: str,
                           original_file: str) -> None:
        """保存结果文件"""
        self.processing_state.current_phase = "saving"
        self._emit_progress()

        # 确保输出目录存在
        output_dir = self.file_service.ensure_output_directory()

        # 保存中间结果
        chunk_outlines_path = output_dir / "chunk_outlines.json"
        self.file_service.write_json_file(chunk_outlines_path, outlines)

        # 保存最终大纲
        final_outline_path = output_dir / "final_outline.txt"
        self.file_service.write_text_file(final_outline_path, final_outline)

        # 保存处理元数据
        metadata = {
            "original_file": original_file,
            "processing_time": self.processing_state.elapsed_time,
            "total_chunks": len(outlines),
            "success_rate": self.processing_state.success_rate,
            "completed_at": datetime.now().isoformat(),
            "summary": self.processing_state.get_summary()
        }
        metadata_path = output_dir / "processing_metadata.json"
        self.file_service.write_json_file(metadata_path, metadata)

        logger.info(f"结果已保存到: {output_dir}")

    def _cleanup_intermediate_outputs(self, output_dir: Path) -> List[str]:
        """删除中间产物，保留最终大纲"""
        targets = [
            output_dir / "chunk_outlines.json",
            output_dir / "processing_metadata.json",
        ]
        removed: List[str] = []
        for path in targets:
            try:
                if path.exists():
                    path.unlink()
                    removed.append(path.name)
            except Exception as err:  # noqa: BLE001
                logger.debug(f"删除中间文件失败 {path}: {err}")
        return removed

    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        if not self.processing_state:
            return {"status": "not_started"}

        return self.processing_state.get_summary()

    def _emit_progress(self, chunk_id: Optional[int] = None, error: Optional[str] = None) -> None:
        """向外部回调当前进度，便于 Web UI 实时显示。
        设计为尽量轻量、容错，不影响主流程。
        """
        if not self.progress_callback or not self.processing_state:
            return

        total = self.processing_state.total_chunks or 0
        completed = self.processing_state.processed_chunks
        failed = self.processing_state.failed_chunks
        progress = (completed / total) if total else 0.0

        eta_seconds: Optional[float] = None
        if self.current_progress_data and self.current_progress_data.processing_times:
            avg_time = sum(self.current_progress_data.processing_times) / len(self.current_progress_data.processing_times)
            remaining = max(total - completed - failed, 0)
            eta_seconds = remaining * avg_time

        payload: Dict[str, Any] = {
            "progress": progress,
            "completed_chunks": completed,
            "failed_chunks": failed,
            "total_chunks": total,
            "phase": self.processing_state.current_phase,
        }
        if eta_seconds is not None:
            payload["eta_seconds"] = eta_seconds
        if chunk_id is not None:
            payload["last_chunk_id"] = chunk_id
        if error is not None:
            payload["last_error"] = error

        try:
            self.progress_callback(payload)
        except Exception:
            # 避免外部回调异常中断主流程
            logger.debug("Progress callback failed", exc_info=True)
