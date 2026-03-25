"""
Tokenizer module unit tests
"""

import pytest

from tokenizer import (
    count_tokens,
    count_tokens_batch,
    estimate_tokens_from_chars,
    get_encoder,
    truncate_by_tokens,
)


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

    def test_non_string_element_returns_zero(self):
        """批量计算时非字符串元素应返回 0"""
        result = count_tokens_batch(["hello", 123, "world"])
        assert result[0] > 0
        assert result[1] == 0
        assert result[2] > 0


class TestEstimateTokensFromChars:
    """Test estimate_tokens_from_chars function"""

    def test_basic_estimation(self):
        """字符数转 token 估算"""
        result = estimate_tokens_from_chars(300)
        assert result == 100  # 300 // 3

    def test_empty_returns_one(self):
        """0 字符应返回最小值 1"""
        result = estimate_tokens_from_chars(0)
        assert result == 1

    def test_small_count_returns_at_least_one(self):
        """小字符数保证不返回 0"""
        result = estimate_tokens_from_chars(1)
        assert result >= 1

    def test_large_count(self):
        """大字符数按比例估算"""
        result = estimate_tokens_from_chars(9000)
        assert result == 3000


class TestTruncateByTokensEdgeCases:
    """Edge cases for truncate_by_tokens"""

    def test_exact_limit_no_truncation(self):
        """文本恰好等于限制时不截断"""
        text = "Hello"
        result = truncate_by_tokens(text, count_tokens(text))
        assert result == text

    def test_truncation_result_within_limit(self):
        """截断后 token 数不超过 max_tokens"""
        long_text = "word " * 100
        result = truncate_by_tokens(long_text, 10)
        assert count_tokens(result) <= 10


class TestCountTokensEdgeCases:
    """Edge cases for count_tokens"""

    def test_mixed_language(self):
        """中英混合文本"""
        result = count_tokens("Hello 你好 World 世界")
        assert result > 0

    def test_special_characters(self):
        """特殊字符"""
        result = count_tokens("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result > 0


class TestFallbackEncoder:
    """Test _FallbackEncoder behavior"""

    def test_fallback_encode_decode_roundtrip(self):
        """_FallbackEncoder 的 encode/decode 应当可逆"""
        import tokenizer as tok_module
        fb = tok_module._FallbackEncoder()
        text = "Hello 世界"
        encoded = fb.encode(text)
        decoded = fb.decode(encoded)
        assert decoded == text

    def test_fallback_used_when_tiktoken_none(self, monkeypatch):
        """当 tiktoken 为 None 时，get_encoder 使用 FallbackEncoder"""
        import tokenizer as tok_module
        original_encoder = tok_module._encoder
        original_tiktoken = tok_module.tiktoken
        try:
            tok_module._encoder = None
            tok_module.tiktoken = None
            encoder = tok_module.get_encoder()
            assert isinstance(encoder, tok_module._FallbackEncoder)
        finally:
            tok_module._encoder = original_encoder
            tok_module.tiktoken = original_tiktoken

    def test_fallback_used_when_tiktoken_get_encoding_fails(self, monkeypatch):
        """tiktoken.get_encoding 失败时应回退到 FallbackEncoder"""
        import tokenizer as tok_module

        class FakeTiktoken:
            def get_encoding(self, name):
                raise RuntimeError("encoding not found")

        original_encoder = tok_module._encoder
        original_tiktoken = tok_module.tiktoken
        try:
            tok_module._encoder = None
            tok_module.tiktoken = FakeTiktoken()
            encoder = tok_module.get_encoder()
            assert isinstance(encoder, tok_module._FallbackEncoder)
        finally:
            tok_module._encoder = original_encoder
            tok_module.tiktoken = original_tiktoken


class TestEncoderErrorPaths:
    """Test exception handling in token functions"""

    def test_count_tokens_raises_encoding_error_on_encoder_failure(self, monkeypatch):
        """编码器失败时 count_tokens 应抛出 EncodingError"""
        import tokenizer as tok_module
        from exceptions import EncodingError

        class BadEncoder:
            def encode(self, text):
                raise RuntimeError("encoder broken")

        original_encoder = tok_module._encoder
        try:
            tok_module._encoder = BadEncoder()
            with pytest.raises(EncodingError):
                count_tokens("test text")
        finally:
            tok_module._encoder = original_encoder

    def test_count_tokens_batch_raises_encoding_error_on_failure(self, monkeypatch):
        """编码器失败时 count_tokens_batch 应抛出 EncodingError"""
        import tokenizer as tok_module
        from exceptions import EncodingError

        class BadEncoder:
            def encode(self, text):
                raise RuntimeError("encoder broken")

        original_encoder = tok_module._encoder
        try:
            tok_module._encoder = BadEncoder()
            with pytest.raises(EncodingError):
                count_tokens_batch(["test"])
        finally:
            tok_module._encoder = original_encoder

    def test_truncate_by_tokens_falls_back_to_char_truncation_on_error(self, monkeypatch):
        """编码器失败时 truncate_by_tokens 应使用字符截断作为回退"""
        import tokenizer as tok_module

        class BadEncoder:
            def encode(self, text):
                raise RuntimeError("encoder broken")

        original_encoder = tok_module._encoder
        try:
            tok_module._encoder = BadEncoder()
            result = truncate_by_tokens("Hello world this is a long text", max_tokens=2)
            # 应返回截断的字符串（不抛出异常）
            assert isinstance(result, str)
            assert len(result) <= 2 * 4  # max_tokens * avg_chars_per_token
        finally:
            tok_module._encoder = original_encoder
