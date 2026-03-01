"""
Splitter module unit tests
"""

import pytest

import splitter as splitter_module
from splitter import TextSplitter, get_splitter, split_text, split_text_stream


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

    def test_split_text_stream_function(self):
        """Test streaming convenience function"""
        text_stream = ["First paragraph.\n\nSecond paragraph."]
        chunks = list(split_text_stream(text_stream))
        assert len(chunks) > 0
        assert "First paragraph" in "".join(chunks)

    def test_split_text_stream_supports_crlf_separator(self, monkeypatch: pytest.MonkeyPatch):
        """CRLF 段落分隔应被正确识别，不应退化到 token 强制切分。"""
        text_splitter = TextSplitter()
        text_splitter.processing_config.target_tokens_per_chunk = 100

        monkeypatch.setattr(
            splitter_module,
            "count_tokens",
            lambda text: 120 if ("第一段" in text and "第二段" in text) else 50,
        )

        def _should_not_call_split_by_tokens(_text: str) -> list[str]:
            raise AssertionError("CRLF 段落分隔失效，意外触发 _split_by_tokens")

        monkeypatch.setattr(text_splitter, "_split_by_tokens", _should_not_call_split_by_tokens)

        chunks = list(text_splitter.split_text_stream(["第一段\r\n\r\n第二段"]))
        assert chunks == ["第一段", "第二段"]
