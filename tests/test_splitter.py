# -*- coding: utf-8 -*-
"""
Splitter module unit tests
"""
import pytest
from splitter import TextSplitter, get_splitter, split_text


class TestTextSplitter:
    """Test TextSplitter class"""

    def test_sentence_end_tokens_dynamic(self):
        """Sentence end tokens should be dynamically generated"""
        splitter = TextSplitter()
        tokens = splitter.sentence_end_tokens
        assert isinstance(tokens, set)
        assert len(tokens) > 0

    def test_sentence_end_tokens_cached(self):
        """Sentence end tokens should be cached"""
        splitter = TextSplitter()
        tokens1 = splitter.sentence_end_tokens
        tokens2 = splitter.sentence_end_tokens
        assert tokens1 is tokens2

    def test_split_short_text(self):
        """Short text should not be split"""
        splitter = TextSplitter()
        text = "This is a short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1

    def test_estimate_chunk_count(self):
        """Estimate chunk count"""
        splitter = TextSplitter()
        text = "test " * 100
        count = splitter.estimate_chunk_count(text)
        assert count >= 1
        assert isinstance(count, int)


class TestGetSplitter:
    """Test get_splitter function"""

    def test_singleton(self):
        """Should return same instance"""
        splitter1 = get_splitter()
        splitter2 = get_splitter()
        assert splitter1 is splitter2


class TestSplitText:
    """Test split_text convenience function"""

    def test_split_text_function(self):
        """Test convenience function"""
        text = "This is a test text."
        chunks = split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
