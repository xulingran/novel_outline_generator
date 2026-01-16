"""
Progress service module tests.

Tests progress tracking, persistence, and recovery functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.processing_state import ProgressData
from services.progress_service import ProgressService


@pytest.fixture
def progress_service():
    """Create a ProgressService instance."""
    return ProgressService()


@pytest.fixture
def mock_progress_data():
    """Create mock progress data for testing."""
    return ProgressData(
        txt_file="test.txt",
        total_chunks=10,
        completed_count=5,
        completed_indices={0, 1, 2, 3, 4},
        outlines=[
            {"chunk_id": 0, "outline": "Chapter 1"},
            {"chunk_id": 1, "outline": "Chapter 2"},
            {"chunk_id": 2, "outline": "Chapter 3"},
            {"chunk_id": 3, "outline": "Chapter 4"},
            {"chunk_id": 4, "outline": "Chapter 5"},
        ],
        last_update=datetime.now(),
        chunks_hash="test_hash_123",
    )


class TestProgressServiceInitialization:
    """Test ProgressService initialization."""

    def test_initialization_creates_progress_dir(self, progress_service):
        """Test that initialization creates progress directory."""
        progress_file = Path(progress_service.processing_config.progress_file)
        assert progress_file.parent.exists()


class TestCreateProgress:
    """Test progress creation."""

    def test_create_progress_basic(self, progress_service):
        """Test creating basic progress data."""
        progress = progress_service.create_progress("novel.txt", 20, "hash_abc")

        assert progress.txt_file == "novel.txt"
        assert progress.total_chunks == 20
        assert progress.completed_count == 0
        assert len(progress.completed_indices) == 0
        assert len(progress.outlines) == 0
        assert progress.chunks_hash == "hash_abc"

    def test_create_progress_with_datetime(self, progress_service):
        """Test that create_progress sets last_update."""
        progress = progress_service.create_progress("novel.txt", 10, "hash_xyz")

        assert isinstance(progress.last_update, datetime)


class TestSaveProgress:
    """Test progress persistence."""

    def test_save_progress_writes_file(
        self, progress_service, mock_progress_data, tmp_path, monkeypatch
    ):
        """Test that save_progress writes to file."""
        # Redirect progress file to tmp_path
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        progress_service.save_progress(mock_progress_data)

        assert progress_file.exists()
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        assert data["txt_file"] == "test.txt"
        assert data["total_chunks"] == 10
        assert data["completed_count"] == 5

    def test_save_progress_updates_last_update(
        self, progress_service, mock_progress_data, tmp_path, monkeypatch
    ):
        """Test that save_progress updates last_update timestamp."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        original_time = mock_progress_data.last_update
        import time

        time.sleep(0.01)  # Small delay

        progress_service.save_progress(mock_progress_data)

        assert mock_progress_data.last_update > original_time

    def test_save_progress_creates_backup(
        self, progress_service, mock_progress_data, tmp_path, monkeypatch
    ):
        """Test that save_progress creates backup file."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        # Save twice to create backup
        progress_service.save_progress(mock_progress_data)
        mock_progress_data.completed_count = 6
        progress_service.save_progress(mock_progress_data)

        # Check for backup (format: progress.YYYYMMDD_HHMMSS.bak)
        backup_files = list(tmp_path.glob("progress.*.bak"))
        assert len(backup_files) >= 1


class TestLoadProgress:
    """Test progress loading."""

    def test_load_progress_nonexistent_file(self, progress_service, tmp_path, monkeypatch):
        """Test loading when progress file doesn't exist."""
        progress_file = tmp_path / "nonexistent_progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        result = progress_service.load_progress()
        assert result is None

    def test_load_progress_existing_file(
        self, progress_service, mock_progress_data, tmp_path, monkeypatch
    ):
        """Test loading existing progress file."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        # First save
        progress_service.save_progress(mock_progress_data)

        # Then load
        result = progress_service.load_progress()

        assert result is not None
        assert result.txt_file == "test.txt"
        assert result.total_chunks == 10
        assert result.completed_count == 5
        assert len(result.outlines) == 5

    def test_load_progress_creates_corrupt_backup_on_failure(
        self, progress_service, tmp_path, monkeypatch
    ):
        """Test that corrupt progress file gets backed up with .corrupt suffix."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        # Create corrupt progress file
        progress_file.write_text("{ invalid json", encoding="utf-8")

        # Load should return None
        result = progress_service.load_progress()
        assert result is None

        # Check that a .corrupt backup was created
        corrupt_backups = list(tmp_path.glob("*.corrupt_*"))
        assert len(corrupt_backups) == 1

    def test_load_progress_no_backup_on_failure(self, progress_service, tmp_path, monkeypatch):
        """Test loading when both file and backup are corrupt."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        # Create corrupt progress file
        progress_file.write_text("{ invalid json", encoding="utf-8")

        # Load should return None
        result = progress_service.load_progress()
        assert result is None


class TestUpdateChunkCompleted:
    """Test updating completed chunk."""

    def test_update_chunk_completed_basic(self, progress_service, mock_progress_data):
        """Test basic chunk completion update."""
        chunk_id = 5
        outline_data = {"chunk_id": 5, "outline": "Chapter 6"}
        processing_time = 2.5

        progress_service.update_chunk_completed(
            mock_progress_data, chunk_id, outline_data, processing_time
        )

        assert chunk_id in mock_progress_data.completed_indices
        assert mock_progress_data.completed_count == 6
        assert mock_progress_data.outlines[-1] == outline_data
        assert len(mock_progress_data.processing_times) == 1
        assert mock_progress_data.processing_times[-1] == processing_time

    def test_update_chunk_without_processing_time(self, progress_service, mock_progress_data):
        """Test updating chunk without processing time."""
        chunk_id = 5
        outline_data = {"chunk_id": 5, "outline": "Chapter 6"}

        progress_service.update_chunk_completed(mock_progress_data, chunk_id, outline_data)

        assert mock_progress_data.completed_count == 6
        assert len(mock_progress_data.processing_times) == 0


class TestIsProgressValid:
    """Test progress validation."""

    def test_is_progress_valid_with_none(self, progress_service):
        """Test validation with None progress data."""
        result = progress_service.is_progress_valid(None, "test.txt", ["chunk1"], "hash")
        assert result is False

    def test_is_progress_valid_path_mismatch(self, progress_service, mock_progress_data, tmp_path):
        """Test validation with mismatched file path."""
        # Create a different file in tmp_path
        different_file = tmp_path / "different.txt"
        different_file.write_text("content")

        result = progress_service.is_progress_valid(
            mock_progress_data,
            str(different_file),
            ["chunk1"] * 10,
            "test_hash_123",
        )
        assert result is False

    def test_is_progress_valid_chunk_count_mismatch(self, progress_service, mock_progress_data):
        """Test validation with mismatched chunk count."""
        result = progress_service.is_progress_valid(
            mock_progress_data,
            "test.txt",
            ["chunk"] * 20,  # Different count
            "test_hash_123",
        )
        assert result is False

    def test_is_progress_valid_hash_mismatch(self, progress_service, mock_progress_data):
        """Test validation with mismatched hash."""
        result = progress_service.is_progress_valid(
            mock_progress_data,
            "test.txt",
            ["chunk"] * 10,
            "different_hash",  # Different hash
        )
        assert result is False

    def test_is_progress_valid_all_match(self, progress_service, mock_progress_data):
        """Test validation when all parameters match."""
        chunks = ["chunk"] * 10
        result = progress_service.is_progress_valid(
            mock_progress_data,
            "test.txt",
            chunks,
            "test_hash_123",
        )
        assert result is True


class TestClearProgress:
    """Test clearing progress file."""

    def test_clear_progress_existing_file(
        self, progress_service, mock_progress_data, tmp_path, monkeypatch
    ):
        """Test clearing existing progress file."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        # Create file
        progress_service.save_progress(mock_progress_data)
        assert progress_file.exists()

        # Clear it
        progress_service.clear_progress()
        assert not progress_file.exists()

    def test_clear_progress_nonexistent_file(self, progress_service, tmp_path, monkeypatch):
        """Test clearing nonexistent progress file."""
        progress_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        # Should not raise error
        progress_service.clear_progress()
        assert not progress_file.exists()


class TestGetProgressSummary:
    """Test progress summary generation."""

    def test_get_progress_summary_no_progress(self, progress_service):
        """Test summary when no progress data."""
        summary = progress_service.get_progress_summary(None)

        assert summary["has_progress"] is False
        assert "message" in summary

    def test_get_progress_summary_with_progress(self, progress_service, mock_progress_data):
        """Test summary with progress data."""
        summary = progress_service.get_progress_summary(mock_progress_data)

        assert summary["has_progress"] is True
        assert summary["file"] == "test.txt"
        assert summary["total_chunks"] == 10
        assert summary["completed_chunks"] == 5
        assert "completion_rate" in summary
        assert "last_update" in summary
        assert "average_processing_time" in summary

    def test_get_progress_summary_with_eta(self, progress_service, mock_progress_data):
        """Test summary includes ETA when some chunks completed."""
        # Add some processing time to enable ETA calculation
        mock_progress_data.processing_times = [1.0, 2.0, 1.5, 2.5, 1.8]

        summary = progress_service.get_progress_summary(mock_progress_data)

        assert "estimated_remaining_time" in summary


class TestFinalizeProgress:
    """Test finalizing progress."""

    def test_finalize_progress_saves_and_clears(
        self, progress_service, mock_progress_data, tmp_path, monkeypatch
    ):
        """Test that finalize saves progress and clears file."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        progress_service.finalize_progress(mock_progress_data)

        # File should be cleared
        assert not progress_file.exists()


class TestAddProgressError:
    """Test adding progress errors."""

    def test_add_progress_error(self, progress_service, mock_progress_data, tmp_path, monkeypatch):
        """Test adding error to progress data."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(progress_service.processing_config, "progress_file", str(progress_file))

        chunk_id = 7
        error_message = "API timeout"

        progress_service.add_progress_error(mock_progress_data, chunk_id, error_message)

        assert len(mock_progress_data.errors) == 1
        error = mock_progress_data.errors[0]
        assert error["chunk_id"] == chunk_id
        assert error_message in error["error"]
