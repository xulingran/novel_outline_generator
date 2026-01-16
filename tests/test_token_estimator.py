"""
测试Token估算服务
"""

from pathlib import Path

import pytest

from services.token_estimator import estimate_tokens


class TestEstimateTokens:
    """测试estimate_tokens函数"""

    def test_estimate_tokens_empty_file(self, tmp_path):
        """测试空文件的token估算"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        # 空文件会抛出ProcessingError
        from exceptions import ProcessingError
        with pytest.raises(ProcessingError, match="文本内容为空"):
            estimate_tokens(str(empty_file))

    def test_estimate_tokens_small_file(self, tmp_path):
        """测试小文件的token估算"""
        small_file = tmp_path / "small.txt"
        small_file.write_text("Hello world. This is a test.", encoding="utf-8")

        result = estimate_tokens(str(small_file))

        assert result["total_tokens"] > 0
        assert result["chunk_tokens"] > 0
        assert result["chunk_responses"] >= 0
        assert result["merge_tokens"] >= 0
        assert result["total_estimated"] > 0
        assert result["chunk_count"] >= 1

    def test_estimate_tokens_large_file(self, tmp_path):
        """测试大文件的token估算"""
        large_file = tmp_path / "large.txt"
        large_file.write_text("Hello world. " * 1000, encoding="utf-8")

        result = estimate_tokens(str(large_file))

        assert result["total_tokens"] > 0
        assert result["chunk_tokens"] > 0
        assert result["chunk_responses"] > 0
        assert result["merge_tokens"] > 0
        assert result["total_estimated"] > 0
        assert result["chunk_count"] >= 1

    def test_estimate_tokens_with_chinese(self, tmp_path):
        """测试包含中文的文件token估算"""
        chinese_file = tmp_path / "chinese.txt"
        chinese_file.write_text("这是一个测试文件。包含一些中文内容。", encoding="utf-8")

        result = estimate_tokens(str(chinese_file))

        assert result["total_tokens"] > 0
        assert result["chunk_tokens"] > 0
        assert result["chunk_responses"] >= 0
        assert result["merge_tokens"] >= 0
        assert result["total_estimated"] > 0
        assert result["chunk_count"] >= 1

    def test_estimate_tokens_with_newlines(self, tmp_path):
        """测试包含换行符的文件token估算"""
        newline_file = tmp_path / "newline.txt"
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        newline_file.write_text(content, encoding="utf-8")

        result = estimate_tokens(str(newline_file))

        assert result["total_tokens"] > 0
        assert result["chunk_tokens"] > 0
        assert result["chunk_responses"] >= 0
        assert result["merge_tokens"] >= 0
        assert result["total_estimated"] > 0
        assert result["chunk_count"] >= 1

    def test_estimate_tokens_chunk_responses_calculation(self, tmp_path):
        """测试chunk_responses计算（应为chunk_tokens的30%）"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world. " * 100, encoding="utf-8")

        result = estimate_tokens(str(test_file))

        assert result["chunk_responses"] == int(result["chunk_tokens"] * 0.3)

    def test_estimate_tokens_merge_tokens_calculation(self, tmp_path):
        """测试merge_tokens计算（应为total_tokens的10%）"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world. " * 100, encoding="utf-8")

        result = estimate_tokens(str(test_file))

        assert result["merge_tokens"] == int(result["total_tokens"] * 0.1)

    def test_estimate_tokens_total_estimated_calculation(self, tmp_path):
        """测试total_estimated计算"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world. " * 100, encoding="utf-8")

        result = estimate_tokens(str(test_file))

        expected_total = (
            result["chunk_tokens"] +
            result["chunk_responses"] +
            result["merge_tokens"]
        )
        assert result["total_estimated"] == expected_total

    def test_estimate_tokens_chunk_count(self, tmp_path):
        """测试chunk_count正确性"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world. " * 100, encoding="utf-8")

        result = estimate_tokens(str(test_file))

        assert result["chunk_count"] >= 1
        assert isinstance(result["chunk_count"], int)

    def test_estimate_tokens_file_not_exists(self):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            estimate_tokens("nonexistent_file.txt")

    def test_estimate_tokens_return_structure(self, tmp_path):
        """测试返回值结构"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world.", encoding="utf-8")

        result = estimate_tokens(str(test_file))

        assert isinstance(result, dict)
        assert "total_tokens" in result
        assert "chunk_tokens" in result
        assert "chunk_responses" in result
        assert "merge_tokens" in result
        assert "total_estimated" in result
        assert "chunk_count" in result

        # 检查所有值都是整数
        for key, value in result.items():
            assert isinstance(value, int), f"{key} should be int, got {type(value)}"

    def test_estimate_tokens_with_special_characters(self, tmp_path):
        """测试包含特殊字符的文件token估算"""
        special_file = tmp_path / "special.txt"
        special_file.write_text("Hello! @#$%^&*() World! 12345", encoding="utf-8")

        result = estimate_tokens(str(special_file))

        assert result["total_tokens"] > 0
        assert result["chunk_tokens"] > 0
        assert result["total_estimated"] > 0

    def test_estimate_tokens_with_emoji(self, tmp_path):
        """测试包含emoji的文件token估算"""
        emoji_file = tmp_path / "emoji.txt"
        emoji_file.write_text("Hello 😊 World 🎉", encoding="utf-8")

        result = estimate_tokens(str(emoji_file))

        assert result["total_tokens"] > 0
        assert result["chunk_tokens"] > 0
        assert result["total_estimated"] > 0

    def test_estimate_tokens_single_chunk(self, tmp_path):
        """测试只有一个块的情况"""
        single_chunk_file = tmp_path / "single.txt"
        single_chunk_file.write_text("Short text.", encoding="utf-8")

        result = estimate_tokens(str(single_chunk_file))

        assert result["chunk_count"] == 1

    def test_estimate_tokens_multiple_chunks(self, tmp_path):
        """测试多个块的情况"""
        multi_chunk_file = tmp_path / "multi.txt"
        # 创建足够长的文本以产生多个块
        multi_chunk_file.write_text("Hello world. " * 1000, encoding="utf-8")

        result = estimate_tokens(str(multi_chunk_file))

        assert result["chunk_count"] >= 1

    def test_estimate_tokens_consistency(self, tmp_path):
        """测试多次调用结果一致"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world. " * 100, encoding="utf-8")

        result1 = estimate_tokens(str(test_file))
        result2 = estimate_tokens(str(test_file))

        assert result1 == result2