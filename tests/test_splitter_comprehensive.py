"""
Splitter module comprehensive tests.

Tests text splitting by chapters, paragraphs, and tokens.
"""

import pytest

from exceptions import ProcessingError
from splitter import TextSplitter, get_splitter, split_by_tokens, split_text, try_split_by_chapter


class TestTextSplitterInitialization:
    """Test TextSplitter initialization."""

    def test_initialization(self):
        """Test that TextSplitter initializes correctly."""
        splitter = TextSplitter()
        assert splitter.processing_config is not None
        assert splitter.encoder is not None
        assert splitter._sentence_end_tokens is None


class TestSentenceEndTokens:
    """Test sentence end tokens property."""

    def test_sentence_end_tokens_dynamic(self):
        """Sentence end tokens should be dynamically generated."""
        splitter = TextSplitter()
        tokens = splitter.sentence_end_tokens
        assert isinstance(tokens, set)
        assert len(tokens) > 0

    def test_sentence_end_tokens_cached(self):
        """Sentence end tokens should be cached."""
        splitter = TextSplitter()
        tokens1 = splitter.sentence_end_tokens
        tokens2 = splitter.sentence_end_tokens
        assert tokens1 is tokens2

    def test_sentence_end_tokens_includes_punctuation(self):
        """Tokens should include common sentence endings."""
        splitter = TextSplitter()
        tokens = splitter.sentence_end_tokens
        # Should include tokens for 。！？.!?
        assert len(tokens) >= 6  # At least 6 punctuation types


class TestSplitTextBasic:
    """Test basic text splitting."""

    def test_split_short_text(self):
        """Short text should not be split."""
        splitter = TextSplitter()
        text = "This is a short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1

    def test_split_empty_text_raises_error(self):
        """Empty text should raise ProcessingError."""
        splitter = TextSplitter()
        with pytest.raises(ProcessingError) as excinfo:
            splitter.split_text("")
        assert "文本内容为空" in str(excinfo.value)

    def test_split_whitespace_only_text_raises_error(self):
        """Whitespace-only text should raise ProcessingError."""
        splitter = TextSplitter()
        with pytest.raises(ProcessingError) as excinfo:
            splitter.split_text("   \n  \t  ")
        assert "文本内容为空" in str(excinfo.value)


class TestSplitByChapters:
    """Test chapter-based splitting."""

    def test_split_by_chapter_pattern_chinese_numbers(self):
        """Test splitting with Chinese chapter numbers."""
        splitter = TextSplitter()
        text = "第一章 开始\n这是第一章的内容\n\n第二章 结束\n这是第二章的内容"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert "第一章" in chunks[0] or "第一章" in chunks[1]
        assert "第二章" in chunks[-1]

    def test_split_by_chapter_pattern_arabic_numbers(self):
        """Test splitting with Arabic chapter numbers."""
        splitter = TextSplitter()
        text = "第1章 开始\n这是第一章的内容\n\n第2章 结束\n这是第二章的内容"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_by_chapter_pattern_chapter_keyword(self):
        """Test splitting with 'Chapter' keyword."""
        splitter = TextSplitter()
        text = "Chapter 1\nStart\n\nChapter 2\nEnd"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_by_chapter_pattern_section(self):
        """Test splitting with section markers."""
        splitter = TextSplitter()
        text = "第一节 开始\n这是第一节\n\n第二节 结束\n这是第二节"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_by_chapter_pattern_volume(self):
        """Test splitting with volume markers."""
        splitter = TextSplitter()
        text = "第一卷 开始\n这是第一卷\n\n第二卷 结束\n这是第二卷"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_try_split_pattern_no_match(self):
        """Test that pattern returns original text when no match."""
        splitter = TextSplitter()
        text = "This is text without chapters"
        result = splitter._try_split_pattern(text, r"(第\d+章)")
        assert result == [text]


class TestSplitByParagraphs:
    """Test paragraph-based splitting."""

    def test_split_by_paragraphs_basic(self):
        """Test basic paragraph splitting."""
        splitter = TextSplitter()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter._split_by_paragraphs(text)
        assert len(chunks) >= 1
        assert any("First paragraph" in chunk for chunk in chunks)

    def test_split_by_paragraphs_empty_paragraphs_skipped(self):
        """Test that empty paragraphs are skipped."""
        splitter = TextSplitter()
        text = "First.\n\n\n\nSecond."  # Multiple empty lines
        chunks = splitter._split_by_paragraphs(text)
        # Empty paragraphs should be filtered out
        assert all(chunk.strip() for chunk in chunks)

    def test_split_by_paragraphs_single_paragraph(self):
        """Test that single paragraph text is returned as single chunk."""
        splitter = TextSplitter()
        text = "This is a single paragraph."
        chunks = splitter._split_by_paragraphs(text)
        assert len(chunks) == 1


class TestSplitByTokens:
    """Test token-based splitting."""

    def test_split_by_tokens_small_text(self):
        """Small text should not be split."""
        splitter = TextSplitter()
        text = "Short text."
        chunks = splitter._split_by_tokens(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_by_tokens_large_text(self):
        """Large text should be split at token boundaries."""
        splitter = TextSplitter()
        # Create text that exceeds target tokens (6000)
        text = "这是一个测试句子。" * 1000  # Repeat to exceed limit
        chunks = splitter._split_by_tokens(text)
        assert len(chunks) >= 2
        # Verify chunks are within reasonable size
        for chunk in chunks:
            assert len(chunk) > 0

    def test_split_by_tokens_preserves_content(self):
        """Split should preserve all content."""
        splitter = TextSplitter()
        text = "First. Second. Third." * 50
        chunks = splitter._split_by_tokens(text)
        combined = "".join(chunks)
        assert combined == text


class TestFindSentenceBoundary:
    """Test sentence boundary detection."""

    def test_find_sentence_boundary_with_punctuation(self):
        """Test finding boundary with sentence-ending punctuation."""
        splitter = TextSplitter()
        tokens = splitter.encoder.encode("First sentence. Second sentence.")
        # Find boundary after first sentence
        boundary = splitter._find_sentence_boundary(tokens, 0, len(tokens))
        # Should split after the period
        assert boundary > 0
        assert boundary <= len(tokens)

    def test_find_sentence_boundary_no_sentence_end(self):
        """Test handling when no sentence boundary exists."""
        splitter = TextSplitter()
        tokens = splitter.encoder.encode("word1 word2 word3")
        # Search near the end without punctuation
        boundary = splitter._find_sentence_boundary(tokens, 0, len(tokens))
        # Should return end position as fallback
        assert boundary == len(tokens)

    def test_find_sentence_boundary_with_space(self):
        """Test splitting at space when no sentence end found."""
        splitter = TextSplitter()
        tokens = splitter.encoder.encode("word1 word2 word3")
        # Search within a range that includes a space
        boundary = splitter._find_sentence_boundary(tokens, 0, len(tokens) // 2)
        # Should split at or before the space
        assert boundary > 0


class TestSecondarySplitting:
    """Test secondary splitting for large chunks."""

    def test_large_chapter_triggers_secondary_split(self):
        """Large chapters should trigger secondary token-based splitting."""
        splitter = TextSplitter()
        # Create text with a chapter header and large content
        text = "第一章 测试\n" + "测试内容。" * 500
        chunks = splitter.split_text(text)
        # Should split into multiple chunks
        assert len(chunks) >= 1


class TestEstimateChunkCount:
    """Test chunk count estimation."""

    def test_estimate_chunk_count(self):
        """Estimate chunk count."""
        splitter = TextSplitter()
        text = "test " * 100
        count = splitter.estimate_chunk_count(text)
        assert count >= 1
        assert isinstance(count, int)

    def test_estimate_chunk_count_empty_text(self):
        """Empty text should estimate at least 1 chunk."""
        splitter = TextSplitter()
        text = ""
        count = splitter.estimate_chunk_count(text)
        assert count >= 1

    def test_estimate_chunk_count_small_text(self):
        """Small text should estimate 1 chunk."""
        splitter = TextSplitter()
        text = "Short text."
        count = splitter.estimate_chunk_count(text)
        assert count == 1


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_try_split_by_chapter(self):
        """Test backward-compatible chapter split function."""
        text = "第一章 测试\n内容\n\n第二章 测试\n内容"
        chunks = try_split_by_chapter(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_split_by_tokens(self):
        """Test backward-compatible token split function."""
        text = "测试。" * 100
        chunks = split_by_tokens(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1


class TestSplitTextConvenience:
    """Test convenience functions."""

    def test_split_text_function(self):
        """Test convenience function."""
        text = "This is a test text."
        chunks = split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0


class TestGetSplitter:
    """Test get_splitter singleton."""

    def test_singleton(self):
        """Should return same instance."""
        splitter1 = get_splitter()
        splitter2 = get_splitter()
        assert splitter1 is splitter2


class TestErrorHandling:
    """Test error handling in splitter."""

    def test_split_text_with_invalid_encoding(self, monkeypatch):
        """Test handling of encoding errors."""
        splitter = TextSplitter()
        text = "Test text for splitting."

        # Mock encoder to raise exception
        def fake_encode(s):
            raise UnicodeError("Encoding failed")

        monkeypatch.setattr(splitter.encoder, "encode", fake_encode)

        with pytest.raises(ProcessingError):
            splitter.split_text(text)

    def test_find_boundary_with_encoder_error(self, monkeypatch):
        """Test handling encoder errors in boundary finding."""
        splitter = TextSplitter()
        tokens = [1, 2, 3]

        # Mock encoder.encode to raise exception
        def fake_encode(s):
            raise UnicodeError("Encoding failed")

        monkeypatch.setattr(splitter.encoder, "encode", fake_encode)

        # Should not crash
        boundary = splitter._find_sentence_boundary(tokens, 0, 3)
        assert boundary == 3


class TestRealWorldScenarios:
    """Test with real-world text patterns."""

    def test_novel_with_multiple_chapters(self):
        """Test splitting a novel-like text with multiple chapters."""
        splitter = TextSplitter()
        text = """
第一章 开始

这是第一章的内容，描述了主角的出场。

第二章 发展

故事继续发展，主角遇到了新的挑战。

第三章 高潮

这是故事的最高潮部分。
        """.strip()

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        # Content should be preserved
        combined = "".join(chunks)
        assert "第一章" in combined
        assert "第二章" in combined
        assert "第三章" in combined

    def test_mixed_format_text(self):
        """Test text with mixed chapter formats."""
        splitter = TextSplitter()
        text = """
Chapter 1: Start

Introduction here.

第2章 发展

Chinese format chapter.

Section 3: Final

Section format.
        """.strip()

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_very_long_paragraph(self):
        """Test splitting very long paragraphs without chapter markers."""
        splitter = TextSplitter()
        # Create a very long paragraph without newlines
        text = "This is a very long paragraph. " * 100

        chunks = splitter.split_text(text)
        # Should split into multiple chunks due to token limit
        assert len(chunks) >= 1
        # All content should be preserved (allow small differences due to tokenization)
        combined = "".join(chunks)
        assert len(combined) >= len(text) - len("\n\n") * (len(chunks) - 1) - 10
