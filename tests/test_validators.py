# -*- coding: utf-8 -*-
"""
Validators module unit tests
"""
import pytest
import tempfile
from pathlib import Path

from validators import (
    validate_file_path,
    validate_output_dir,
    validate_api_provider,
    validate_positive_int,
    validate_non_negative_int,
    sanitize_filename,
)
from exceptions import FileValidationError


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

    def test_invalid_provider(self):
        """Invalid provider should raise exception"""
        with pytest.raises(FileValidationError):
            validate_api_provider("invalid_provider")


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
