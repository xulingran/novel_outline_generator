"""
File service module tests.

Tests file I/O, encoding detection, and backup management.
"""

import json

import pytest

from exceptions import FileValidationError
from services.file_service import FileService


@pytest.fixture
def file_service():
    """Create a FileService instance."""
    return FileService()


class TestFileServiceInitialization:
    """Test FileService initialization."""

    def test_default_encodings_set(self, file_service):
        """Test that default encodings are set."""
        assert file_service.processing_config.encodings
        assert len(file_service.processing_config.encodings) > 0


class TestReadTextFile:
    """Test text file reading with encoding detection."""

    def test_read_utf8_file_success(self, file_service, tmp_path):
        """Test reading a UTF-8 encoded file."""
        test_file = tmp_path / "test.txt"
        content = "Hello World 你好世界"
        test_file.write_text(content, encoding="utf-8")

        result, encoding = file_service.read_text_file(test_file)
        assert result == content
        assert encoding == "utf-8"

    def test_read_gbk_file_success(self, file_service, tmp_path):
        """Test reading a GBK encoded file."""
        test_file = tmp_path / "test.txt"
        content = "你好世界"
        test_file.write_text(content, encoding="gbk")

        result, encoding = file_service.read_text_file(test_file)
        assert result == content
        assert encoding == "gbk"

    def test_read_file_fallback_encoding(self, file_service, tmp_path):
        """Test reading file with fallback encoding."""
        test_file = tmp_path / "test.txt"
        content = "GB2312内容"
        test_file.write_text(content, encoding="gb2312")

        result, encoding = file_service.read_text_file(test_file)
        assert result == content

    def test_read_nonexistent_file_raises_error(self, file_service):
        """Test reading a nonexistent file raises FileValidationError."""
        with pytest.raises(FileValidationError):
            file_service.read_text_file("nonexistent.txt")

    def test_read_invalid_extension_raises_error(self, file_service, tmp_path):
        """Test reading file with invalid extension raises error."""
        test_file = tmp_path / "test.exe"
        test_file.write_text("content")

        with pytest.raises(FileValidationError):
            file_service.read_text_file(test_file)

    def test_read_file_too_large_raises_error(self, file_service, tmp_path, monkeypatch):
        """Test reading a file that's too large raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("small content")

        # Mock validate_file_path to raise size error
        from validators import validate_file_path

        def fake_validate(file_path, **kwargs):
            if kwargs.get("max_size_mb") == 100:
                raise FileValidationError(f"File too large: {file_path}")
            return validate_file_path.__wrapped__(file_path, **kwargs)

        monkeypatch.setattr("services.file_service.validate_file_path", fake_validate)

        with pytest.raises(FileValidationError):
            file_service.read_text_file(test_file)

    def test_read_file_all_encodings_fail(self, file_service, tmp_path):
        """Test when all encodings fail."""
        test_file = tmp_path / "test.txt"
        # Write binary data that's not valid text
        test_file.write_bytes(b"\x80\x81\x82\x83")

        with pytest.raises(UnicodeError) as excinfo:
            file_service.read_text_file(test_file)
        # UnicodeError is raised when utf-16 decoding fails without BOM
        assert "UTF-16" in str(excinfo.value) or "无法使用任何编码读取文件" in str(excinfo.value)


class TestWriteTextFile:
    """Test text file writing."""

    def test_write_text_file_success(self, file_service, tmp_path):
        """Test writing text to file."""
        test_file = tmp_path / "output.txt"
        content = "Test content 你好"

        file_service.write_text_file(test_file, content, encoding="utf-8")

        assert test_file.read_text(encoding="utf-8") == content

    def test_write_text_file_creates_directory(self, file_service, tmp_path):
        """Test that write creates parent directories."""
        test_file = tmp_path / "subdir" / "nested" / "output.txt"
        content = "Test content"

        file_service.write_text_file(test_file, content)

        assert test_file.exists()
        assert test_file.read_text() == content

    def test_write_text_file_gbk_encoding(self, file_service, tmp_path):
        """Test writing file with GBK encoding."""
        test_file = tmp_path / "output.txt"
        content = "测试内容"

        file_service.write_text_file(test_file, content, encoding="gbk")

        assert test_file.read_text(encoding="gbk") == content


class TestReadJsonFile:
    """Test JSON file reading."""

    def test_read_valid_json_file(self, file_service, tmp_path):
        """Test reading a valid JSON file."""
        test_file = tmp_path / "data.json"
        data = {"key": "value", "number": 123}
        test_file.write_text(json.dumps(data), encoding="utf-8")

        result = file_service.read_json_file(test_file)
        assert result == data

    def test_read_json_with_default(self, file_service, tmp_path):
        """Test reading JSON file with default value."""
        test_file = tmp_path / "nonexistent.json"

        result = file_service.read_json_file(test_file, default={"default": True})
        assert result == {"default": True}

    def test_read_json_file_invalid_json(self, file_service, tmp_path):
        """Test reading invalid JSON file returns default."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json")

        # safe_read_json returns default instead of raising
        result = file_service.read_json_file(test_file, default={"error": True})
        assert result == {"error": True}


class TestWriteJsonFile:
    """Test JSON file writing."""

    def test_write_json_file_success(self, file_service, tmp_path):
        """Test writing data to JSON file."""
        test_file = tmp_path / "output.json"
        data = {"result": "success", "items": [1, 2, 3]}

        file_service.write_json_file(test_file, data)

        result = json.loads(test_file.read_text(encoding="utf-8"))
        assert result == data

    def test_write_json_file_creates_directory(self, file_service, tmp_path):
        """Test that write creates parent directories."""
        test_file = tmp_path / "nested" / "output.json"
        data = {"test": True}

        file_service.write_json_file(test_file, data)

        assert test_file.exists()
        result = json.loads(test_file.read_text())
        assert result == data

    def test_write_json_with_backup(self, file_service, tmp_path):
        """Test writing JSON with backup creates .bak file."""
        test_file = tmp_path / "data.json"

        # Write initial file
        file_service.write_json_file(test_file, {"version": 1})
        # Update
        file_service.write_json_file(test_file, {"version": 2})

        # Check backup exists (format: data.YYYYMMDD_HHMMSS.bak)
        backup_files = list(tmp_path.glob("data.*.bak"))
        assert len(backup_files) >= 1

    def test_write_json_without_backup(self, file_service, tmp_path):
        """Test writing JSON without backup."""
        test_file = tmp_path / "data.json"

        file_service.write_json_file(test_file, {"version": 1}, backup=False)

        # Check no backup exists
        backup_files = list(tmp_path.glob("*.json.bak"))
        assert len(backup_files) == 0


class TestEnsureOutputDirectory:
    """Test output directory creation."""

    def test_ensure_output_directory_exists(self, file_service, tmp_path, monkeypatch):
        """Test ensuring output directory exists."""
        monkeypatch.setattr(file_service.processing_config, "output_dir", str(tmp_path / "output"))

        result = file_service.ensure_output_directory()

        assert result.exists()
        assert result.is_dir()

    def test_ensure_output_directory_with_subdirectory(self, file_service, tmp_path, monkeypatch):
        """Test ensuring output directory with subdirectory."""
        monkeypatch.setattr(file_service.processing_config, "output_dir", str(tmp_path / "output"))

        result = file_service.ensure_output_directory("subdir")

        assert result.exists()
        assert result.is_dir()
        assert "subdir" in str(result)


class TestListTxtFiles:
    """Test listing text files in directory."""

    def test_list_txt_files_empty_directory(self, file_service, tmp_path):
        """Test listing files in empty directory."""
        result = file_service.list_txt_files(tmp_path)
        assert result == []

    def test_list_txt_files_with_txt(self, file_service, tmp_path):
        """Test listing .txt files."""
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        result = file_service.list_txt_files(tmp_path)
        assert len(result) == 2
        assert all(f.suffix == ".txt" for f in result)

    def test_list_txt_files_with_md(self, file_service, tmp_path):
        """Test listing .md files."""
        (tmp_path / "doc1.md").write_text("# Title")
        (tmp_path / "doc2.md").write_text("# Title 2")

        result = file_service.list_txt_files(tmp_path)
        assert len(result) == 2
        assert all(f.suffix == ".md" for f in result)

    def test_list_txt_files_with_text_extension(self, file_service, tmp_path):
        """Test listing .text files."""
        (tmp_path / "readme.text").write_text("readme")

        result = file_service.list_txt_files(tmp_path)
        assert len(result) == 1
        assert result[0].suffix == ".text"

    def test_list_txt_files_mixed_extensions(self, file_service, tmp_path):
        """Test listing files with mixed extensions."""
        (tmp_path / "novel.txt").write_text("novel")
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "readme.text").write_text("readme")
        (tmp_path / "data.json").write_text("{}")  # Should be ignored

        result = file_service.list_txt_files(tmp_path)
        assert len(result) == 3
        assert all(f.suffix in [".txt", ".md", ".text"] for f in result)

    def test_list_txt_files_nonexistent_directory(self, file_service):
        """Test listing files in nonexistent directory."""
        result = file_service.list_txt_files("/nonexistent/path")
        assert result == []

    def test_list_txt_files_sorted(self, file_service, tmp_path):
        """Test that files are sorted."""
        (tmp_path / "c.txt").write_text("")
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")

        result = file_service.list_txt_files(tmp_path)
        assert result[0].name == "a.txt"
        assert result[1].name == "b.txt"
        assert result[2].name == "c.txt"


class TestGetFileSize:
    """Test getting file size."""

    def test_get_file_size_existing_file(self, file_service, tmp_path):
        """Test getting size of existing file."""
        test_file = tmp_path / "test.txt"
        content = "Hello World"
        test_file.write_text(content, encoding="utf-8")

        size = file_service.get_file_size(test_file)
        assert size == len(content.encode("utf-8"))

    def test_get_file_size_nonexistent_file(self, file_service):
        """Test getting size of nonexistent file."""
        size = file_service.get_file_size("/nonexistent/file.txt")
        assert size == 0

    def test_get_file_size_empty_file(self, file_service, tmp_path):
        """Test getting size of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        size = file_service.get_file_size(test_file)
        assert size == 0


class TestGetFileInfo:
    """Test getting file information."""

    def test_get_file_info_existing_file(self, file_service, tmp_path):
        """Test getting info for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        info = file_service.get_file_info(test_file)
        assert info["exists"] is True
        assert info["size"] > 0
        assert "modified" in info

    def test_get_file_info_nonexistent_file(self, file_service):
        """Test getting info for nonexistent file."""
        info = file_service.get_file_info("/nonexistent/file.txt")
        assert info["exists"] is False
        assert info == {"exists": False}


class TestRemoveBackups:
    """Test removing backup files."""

    def test_remove_backups_from_directory(self, file_service, tmp_path):
        """Test removing backup files."""
        # Create some backup files
        (tmp_path / "file1.txt.bak").write_text("backup1")
        (tmp_path / "file2.txt.bak").write_text("backup2")
        (tmp_path / "data.json").write_text("{}")  # Not a backup

        removed = file_service.remove_backups(tmp_path, "*.bak")

        assert removed == 2
        assert not (tmp_path / "file1.txt.bak").exists()
        assert not (tmp_path / "file2.txt.bak").exists()
        assert (tmp_path / "data.json").exists()

    def test_remove_backups_recursive(self, file_service, tmp_path):
        """Test removing backups recursively."""
        subdir = tmp_path / "nested"
        subdir.mkdir()

        (tmp_path / "backup1.bak").write_text("")
        (subdir / "backup2.bak").write_text("")

        removed = file_service.remove_backups(tmp_path, "*.bak")

        assert removed == 2

    def test_remove_backups_nonexistent_directory(self, file_service):
        """Test removing backups from nonexistent directory."""
        removed = file_service.remove_backups("/nonexistent/path", "*.bak")
        assert removed == 0

    def test_remove_backups_no_backups(self, file_service, tmp_path):
        """Test removing when no backup files exist."""
        (tmp_path / "data.json").write_text("{}")

        removed = file_service.remove_backups(tmp_path, "*.bak")

        assert removed == 0

    def test_remove_backups_custom_pattern(self, file_service, tmp_path):
        """Test removing backups with custom pattern."""
        (tmp_path / "backup1.tmp").write_text("")
        (tmp_path / "backup2.tmp").write_text("")
        (tmp_path / "data.txt").write_text("")

        removed = file_service.remove_backups(tmp_path, "*.tmp")

        assert removed == 2
        assert not (tmp_path / "backup1.tmp").exists()
        assert (tmp_path / "data.txt").exists()
