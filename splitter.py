"""
文本分割器模块
将长文本分割成适合处理的块
"""

import logging
import re

from config import ProcessingConfig, get_processing_config
from exceptions import ProcessingError
from tokenizer import count_tokens, get_encoder

logger = logging.getLogger(__name__)


class TextSplitter:
    """智能文本分割器"""

    def __init__(self):
        self.processing_config: ProcessingConfig = get_processing_config()
        self.encoder = get_encoder()
        self._sentence_end_tokens: set[int] | None = None

    @property
    def sentence_end_tokens(self) -> set[int]:
        """动态获取句子结束标记的 tokens"""
        if self._sentence_end_tokens is None:
            # 常见的句子结束标点
            sentence_endings = "。！？.!?；;…"
            tokens = set()
            for char in sentence_endings:
                try:
                    encoded = self.encoder.encode(char)
                    tokens.update(encoded)
                except Exception:
                    pass
            self._sentence_end_tokens = tokens
        return self._sentence_end_tokens

    def split_text(self, text: str) -> list[str]:
        """
        分割文本为适合处理的块

        Args:
            text: 要分割的文本

        Returns:
            List[str]: 文本块列表
        """
        if not text or not text.strip():
            raise ProcessingError("文本内容为空")

        logger.debug(f"开始分割文本，总长度: {len(text)} 字符")

        try:
            # 1. 优先按章节分割
            chapter_chunks = self._split_by_chapters(text)
            logger.debug(f"按章节分割得到 {len(chapter_chunks)} 个块")

            # 2. 检查章节大小，必要时进行二级分割
            final_chunks = []
            for idx, chunk in enumerate(chapter_chunks, 1):
                token_count = count_tokens(chunk)
                if token_count > self.processing_config.target_tokens_per_chunk:
                    logger.debug(f"章节 {idx} 过大 ({token_count} tokens)，进行二级分割")
                    sub_chunks = self._split_by_tokens(chunk)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)

            logger.info(f"文本分割完成，共 {len(final_chunks)} 个块")
            return final_chunks

        except Exception as e:
            logger.error(f"分割文本失败: {e}")
            raise ProcessingError(f"分割文本失败: {str(e)}") from e

    def _split_by_chapters(self, text: str) -> list[str]:
        """
        按章节分割文本

        Args:
            text: 文本内容

        Returns:
            List[str]: 章节块列表
        """
        # 多种章节匹配模式
        patterns = [
            r"(第[\d一二三四五六七八九十百千万零]+章[^\n]*)",  # 第X章
            r"(第[\d一二三四五六七八九十百千万零]+节[^\n]*)",  # 第X节
            r"(Chapter\s+\d+[^\n]*)",  # Chapter X
            r"(第\d+卷[^\n]*)",  # 第X卷
        ]

        # 尝试每种模式
        for pattern in patterns:
            chunks = self._try_split_pattern(text, pattern)
            if len(chunks) > 1:
                logger.debug(f"使用模式 '{pattern}' 成功分割")
                return chunks

        # 如果没有找到章节，按段落分割
        logger.debug("未找到章节标记，使用段落分割")
        return self._split_by_paragraphs(text)

    def _try_split_pattern(self, text: str, pattern: str) -> list[str]:
        """尝试使用特定模式分割"""
        parts = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        if len(parts) <= 1:
            return [text]

        chunks = []

        # 重新组合分割结果
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""

            if title or content:
                chunk = f"{title}\n{content}" if title else content
                chunks.append(chunk.strip())

        return chunks if chunks else [text]

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """按段落分割文本"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块为空，直接添加
            if not current_chunk:
                current_chunk = para
            else:
                # 检查添加后是否会超限
                test_chunk = current_chunk + "\n\n" + para
                test_tokens = count_tokens(test_chunk)

                if test_tokens > self.processing_config.target_tokens_per_chunk:
                    # 保存当前块，开始新块
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    # 添加到当前块
                    current_chunk = test_chunk

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _split_by_tokens(self, text: str) -> list[str]:
        """
        基于token数量精确分割文本

        Args:
            text: 文本内容

        Returns:
            List[str]: 分割后的文本块
        """
        target_tokens = self.processing_config.target_tokens_per_chunk
        tokens = self.encoder.encode(text)

        if len(tokens) <= target_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + target_tokens, len(tokens))

            # 尝试在句子边界分割
            if end < len(tokens):
                end = self._find_sentence_boundary(tokens, start, end)

            # 解码tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text.strip())

            start = end

        logger.debug(f"基于tokens分割得到 {len(chunks)} 个块")
        return chunks

    def _find_sentence_boundary(self, tokens: list[int], start: int, end: int) -> int:
        """在tokens中寻找最近的句子边界"""
        # 从后向前搜索
        search_start = max(start, end - min(100, end - start))  # 最多向前搜索100个token

        for i in range(end - 1, search_start - 1, -1):
            if tokens[i] in self.sentence_end_tokens:
                return i + 1

        # 如果没找到句子边界，尝试在空格处分割
        try:
            space_tokens = set(self.encoder.encode(" "))
            for i in range(end - 1, search_start - 1, -1):
                if tokens[i] in space_tokens:
                    return i + 1
        except Exception:
            pass

        # 如果都没找到，返回原始end
        return end

    def estimate_chunk_count(self, text: str) -> int:
        """
        估算文本会被分割成多少块

        Args:
            text: 文本内容

        Returns:
            int: 估算的块数
        """
        total_tokens = count_tokens(text)
        target_size = self.processing_config.target_tokens_per_chunk
        return max(1, (total_tokens + target_size - 1) // target_size)


# 全局分割器实例
_splitter: TextSplitter | None = None


def get_splitter() -> TextSplitter:
    """获取分割器实例（单例模式）"""
    global _splitter
    if _splitter is None:
        _splitter = TextSplitter()
    return _splitter


def split_text(text: str) -> list[str]:
    """
    分割文本（向后兼容的函数）

    Args:
        text: 要分割的文本

    Returns:
        List[str]: 文本块列表
    """
    splitter = get_splitter()
    return splitter.split_text(text)


# 保留旧函数以向后兼容
def try_split_by_chapter(text: str) -> list[str]:
    """按章节分割（向后兼容）"""
    splitter = get_splitter()
    return splitter._split_by_chapters(text)


def split_by_tokens(text: str) -> list[str]:
    """按token分割（向后兼容）"""
    splitter = get_splitter()
    return splitter._split_by_tokens(text)
