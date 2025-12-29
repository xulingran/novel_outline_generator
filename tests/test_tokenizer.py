"""
Tokenizer module unit tests
"""

import pytest

from tokenizer import count_tokens, count_tokens_batch, get_encoder, truncate_by_tokens


class TestCountTokens:
    """Test count_tokens function"""

    def test_empty_string(self):
        """Empty string should return 0"""
        assert count_tokens("") == 0

    def test_simple_english(self):
        """Simple English text"""
        result = count_tokens("Hello, world!")
        assert result > 0
        assert isinstance(result, int)

    def test_chinese_text(self):
        """Chinese text"""
        result = count_tokens("Hello World Test")
        assert result > 0
        assert isinstance(result, int)

    def test_invalid_input(self):
        """Non-string input should raise exception"""
        with pytest.raises(ValueError):
            count_tokens(123)

        with pytest.raises(ValueError):
            count_tokens(None)


class TestGetEncoder:
    """Test get_encoder function"""

    def test_singleton(self):
        """Should return same encoder instance"""
        encoder1 = get_encoder()
        encoder2 = get_encoder()
        assert encoder1 is encoder2

    def test_encoder_works(self):
        """Encoder should work properly"""
        encoder = get_encoder()
        tokens = encoder.encode("test")
        assert len(tokens) > 0


class TestTruncateByTokens:
    """Test truncate_by_tokens function"""

    def test_no_truncation_needed(self):
        """No truncation needed case"""
        text = "Hello"
        result = truncate_by_tokens(text, 100)
        assert result == text

    def test_truncation(self):
        """Truncation needed case"""
        text = "Hello world, this is a test message that might need truncation."
        result = truncate_by_tokens(text, 5)
        assert count_tokens(result) <= 5


class TestCountTokensBatch:
    """Test count_tokens_batch function"""

    def test_empty_list(self):
        """Empty list"""
        result = count_tokens_batch([])
        assert result == {}

    def test_batch_counting(self):
        """Batch counting"""
        texts = ["Hello", "World", "Test"]
        result = count_tokens_batch(texts)
        assert len(result) == 3

    def test_invalid_input(self):
        """Non-list input should raise exception"""
        with pytest.raises(ValueError):
            count_tokens_batch("not a list")
