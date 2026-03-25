"""
Validators module unit tests
"""

import tempfile
from pathlib import Path

import pytest

from exceptions import FileValidationError
from validators import (
    sanitize_filename,
    validate_api_provider,
    validate_encoding_list,
    validate_file_path,
    validate_non_negative_int,
    validate_output_dir,
    validate_positive_int,
)


class TestValidateFilePath:
    """Test validate_file_path function"""

    def test_nonexistent_file(self):
        """Non-existent file should raise exception"""
        with pytest.raises(FileValidationError):
            validate_file_path("/nonexistent/file.txt")

    def test_empty_path(self):
        """Empty path should raise exception"""
        with pytest.raises(FileValidationError):
            validate_file_path("")

    def test_path_traversal(self):
        """Path traversal attack should be detected"""
        with pytest.raises(FileValidationError):
            validate_file_path("../../../etc/passwd")

    def test_valid_file(self):
        """Valid file should return Path object"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = validate_file_path(temp_path, allowed_extensions=[".txt"])
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            Path(temp_path).unlink()


class TestValidateOutputDir:
    """Test validate_output_dir function"""

    def test_creates_directory(self):
        """Should automatically create directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir"
            result = validate_output_dir(str(new_dir))
            assert result.exists()
            assert result.is_dir()


class TestValidateApiProvider:
    """Test validate_api_provider function"""

    def test_valid_providers(self):
        """Valid providers"""
        assert validate_api_provider("openai") == "openai"
        assert validate_api_provider("OPENAI") == "openai"
        assert validate_api_provider("gemini") == "gemini"
        assert validate_api_provider("zhipu") == "zhipu"
        assert validate_api_provider("aihubmix") == "aihubmix"
        assert validate_api_provider("AIHUBMIX") == "aihubmix"

    def test_invalid_provider(self):
        """Invalid provider should raise exception"""
        with pytest.raises(FileValidationError):
            validate_api_provider("invalid_provider")

    def test_non_string_provider(self):
        """Non-string provider should raise exception"""
        with pytest.raises(FileValidationError):
            validate_api_provider(123)  # type: ignore[arg-type]


class TestValidateEncodingList:
    """Test validate_encoding_list function"""

    def test_mixed_type_encodings(self):
        """Mixed type list should keep valid encodings and skip invalid types"""
        assert validate_encoding_list(["utf-8", 123]) == ["utf-8"]  # type: ignore[list-item]

    def test_non_list_encodings(self):
        """Non-list input should raise exception"""
        with pytest.raises(FileValidationError):
            validate_encoding_list("utf-8")  # type: ignore[arg-type]


class TestValidatePositiveInt:
    """Test validate_positive_int function"""

    def test_valid_positive(self):
        """Valid positive integers"""
        assert validate_positive_int(1, "test") == 1
        assert validate_positive_int(100, "test") == 100

    def test_zero(self):
        """Zero should raise exception"""
        with pytest.raises(FileValidationError):
            validate_positive_int(0, "test")

    def test_negative(self):
        """Negative should raise exception"""
        with pytest.raises(FileValidationError):
            validate_positive_int(-1, "test")

    def test_non_int(self):
        """Non-int should raise exception"""
        with pytest.raises(FileValidationError):
            validate_positive_int("1", "test")  # type: ignore[arg-type]


class TestValidateNonNegativeInt:
    """Test validate_non_negative_int function"""

    def test_valid_values(self):
        """Valid non-negative integers"""
        assert validate_non_negative_int(0, "test") == 0
        assert validate_non_negative_int(1, "test") == 1

    def test_negative(self):
        """Negative should raise exception"""
        with pytest.raises(FileValidationError):
            validate_non_negative_int(-1, "test")

    def test_non_int(self):
        """Non-int should raise exception"""
        with pytest.raises(FileValidationError):
            validate_non_negative_int("0", "test")  # type: ignore[arg-type]


class TestSanitizeFilename:
    """Test sanitize_filename function"""

    def test_safe_filename(self):
        """Safe filename unchanged"""
        assert sanitize_filename("test.txt") == "test.txt"

    def test_removes_unsafe_chars(self):
        """Remove unsafe characters"""
        result = sanitize_filename("test<>|.txt")
        assert "<" not in result
        assert ">" not in result

    def test_empty_filename(self):
        """Empty filename returns default"""
        assert sanitize_filename("") == "output"

    def test_long_filename(self):
        """Long filename should be truncated"""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestValidateFilePathEdgeCases:
    """Edge cases for validate_file_path"""

    def test_path_is_directory_raises(self):
        """路径是目录而非文件时应抛出异常"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileValidationError, match="路径不是文件"):
                validate_file_path(tmpdir)

    def test_file_exceeds_max_size_raises(self, tmp_path):
        """文件超过大小限制时应抛出异常"""
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"x" * 1024)  # 1KB
        with pytest.raises(FileValidationError, match="文件过大"):
            validate_file_path(str(large_file), max_size_mb=0.0001)

    def test_unsupported_extension_raises(self, tmp_path):
        """不支持的扩展名应抛出异常"""
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(FileValidationError, match="不支持的文件扩展名"):
            validate_file_path(str(f), allowed_extensions=[".txt"])

    def test_double_slash_path_is_normalized(self, tmp_path):
        """双斜杠路径应被 normpath 规范化后正常访问"""
        f = tmp_path / "test.txt"
        f.write_text("content")
        normalized = str(f).replace("/", "//", 1)
        result = validate_file_path(normalized)
        assert result.exists()


class TestValidateOutputDirEdgeCases:
    """Edge cases for validate_output_dir"""

    def test_empty_path_raises(self):
        """空路径应抛出异常"""
        with pytest.raises(FileValidationError, match="路径不能为空"):
            validate_output_dir("")


class TestValidateEncodingListEdgeCases:
    """Edge cases for validate_encoding_list"""

    def test_unsupported_encoding_is_filtered(self):
        """不支持的编码应被过滤掉，保留有效编码"""
        result = validate_encoding_list(["utf-8", "xyz-unknown", "gbk"])
        assert "utf-8" in result
        assert "gbk" in result
        assert "xyz-unknown" not in result

    def test_all_unsupported_raises(self):
        """全部为不支持编码时应抛出异常"""
        with pytest.raises(FileValidationError, match="没有找到有效的编码"):
            validate_encoding_list(["xyz-123", "abc-456"])

    def test_empty_list_raises(self):
        """空列表应抛出异常"""
        with pytest.raises(FileValidationError):
            validate_encoding_list([])


class TestValidateChunkSizeAndParallelLimit:
    """Test validate_chunk_size and validate_parallel_limit"""

    def test_chunk_size_exceeds_max_tokens_raises(self):
        """块大小超过模型最大 token 数时应抛出异常"""
        from validators import validate_chunk_size

        with pytest.raises(FileValidationError, match="块大小"):
            validate_chunk_size(chunk_size=8192, max_tokens=4096)

    def test_chunk_size_warning_near_limit(self):
        """块大小接近模型限制时应触发 UserWarning"""
        import warnings

        from validators import validate_chunk_size

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_chunk_size(chunk_size=7000, max_tokens=8192)
            assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_parallel_limit_exceeds_recommendation_warns(self):
        """并发限制超过建议值时应触发 UserWarning"""
        import warnings

        from validators import validate_parallel_limit

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_parallel_limit(21)  # RECOMMENDED_MAX_PARALLEL_LIMIT = 20，严格大于才触发
            assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_parallel_limit_at_boundary_no_warning(self):
        """并发限制恰好等于建议值时不触发警告"""
        import warnings

        from validators import validate_parallel_limit

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_parallel_limit(20)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0
        assert result == 20
