"""
Utils module unit tests
"""

import json
import tempfile
from pathlib import Path

import pytest

from utils import (
    atomic_write_json,
    atomic_write_text,
    format_file_size,
    get_file_info,
    safe_read_json,
    safe_read_text,
    truncate_text,
)


class TestAtomicWriteJson:
    """Test atomic_write_json function"""

    def test_write_json(self):
        """Normal JSON write"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 123}

            atomic_write_json(file_path, data)

            assert file_path.exists()
            with open(file_path, encoding="utf-8") as f:
                result = json.load(f)
            assert result == data

    def test_creates_directory(self):
        """Auto-create directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.json"
            data = {"test": True}

            atomic_write_json(file_path, data)

            assert file_path.exists()


class TestAtomicWriteText:
    """Test atomic_write_text function"""

    def test_write_text(self):
        """Normal text write"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            content = "Hello, World!"

            atomic_write_text(file_path, content)

            assert file_path.exists()
            with open(file_path, encoding="utf-8") as f:
                result = f.read()
            assert result == content


class TestSafeReadJson:
    """Test safe_read_json function"""

    def test_read_valid_json(self):
        """Read valid JSON"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"key": "value"}, f)
            temp_path = f.name

        try:
            result = safe_read_json(temp_path)
            assert result == {"key": "value"}
        finally:
            Path(temp_path).unlink()

    def test_read_nonexistent(self):
        """Read non-existent file returns default"""
        result = safe_read_json("/nonexistent/file.json", default={"default": True})
        assert result == {"default": True}


class TestSafeReadText:
    """Test safe_read_text function"""

    def test_read_utf8(self):
        """Read UTF-8 file"""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("Hello World")
            temp_path = f.name

        try:
            result = safe_read_text(temp_path)
            assert result == "Hello World"
        finally:
            Path(temp_path).unlink()

    def test_nonexistent_file(self):
        """Read non-existent file raises exception"""
        with pytest.raises(FileNotFoundError):
            safe_read_text("/nonexistent/file.txt")


class TestFormatFileSize:
    """Test format_file_size function"""

    def test_bytes(self):
        """Bytes"""
        assert format_file_size(500) == "500.0B"

    def test_kilobytes(self):
        """Kilobytes"""
        assert format_file_size(1024) == "1.0KB"

    def test_megabytes(self):
        """Megabytes"""
        assert format_file_size(1024 * 1024) == "1.0MB"


class TestTruncateText:
    """Test truncate_text function"""

    def test_no_truncation(self):
        """No truncation needed"""
        text = "Hello"
        assert truncate_text(text, 100) == text

    def test_truncation(self):
        """Truncation needed"""
        text = "Hello, this is a long text"
        result = truncate_text(text, 10)
        assert len(result) == 10
        assert result.endswith("...")


class TestGetFileInfo:
    """Test get_file_info function"""

    def test_nonexistent_file(self):
        """Non-existent file"""
        result = get_file_info("/nonexistent/file.txt")
        assert result["exists"] is False

    def test_existing_file(self):
        """Existing file"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = get_file_info(temp_path)
            assert result["exists"] is True
            assert result["is_file"] is True
            assert result["size"] == 12
        finally:
            Path(temp_path).unlink()
