"""Progress tracking service.

Provides simple persistence and recovery for processing progress.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_processing_config
from models.processing_state import ProgressData
from services.file_service import FileService
from utils import ProgressTracker

logger = logging.getLogger(__name__)


class ProgressService:
    """Manage progress persistence for long running jobs."""

    def __init__(self) -> None:
        self.processing_config = get_processing_config()
        self.file_service = FileService()
        self.progress_tracker = ProgressTracker(batch_size=5)
        self._ensure_progress_dir()

    def _ensure_progress_dir(self) -> None:
        """Make sure the progress directory exists."""
        progress_file = Path(self.processing_config.progress_file)
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    def save_progress(self, progress_data: ProgressData) -> None:
        """Persist the latest progress to disk."""
        try:
            progress_data.last_update = datetime.now()
            self.file_service.write_json_file(
                self.processing_config.progress_file,
                progress_data.to_dict(),
                backup=True,
            )
            logger.debug(
                "Progress saved: %s/%s",
                progress_data.completed_count,
                progress_data.total_chunks,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to save progress: %s", e)
            raise

    def load_progress(self) -> ProgressData | None:
        """Load progress from disk if available."""
        progress_path = Path(self.processing_config.progress_file)
        try:
            if not progress_path.exists():
                return None

            data = self.file_service.read_json_file(str(progress_path))
            if data:
                progress = ProgressData.from_dict(data)
                logger.info(
                    "Loaded progress: %s/%s",
                    progress.completed_count,
                    progress.total_chunks,
                )
                return progress
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load progress: %s", e)
            self._try_recover_progress(progress_path)

        return None

    def _try_recover_progress(self, progress_path: Path) -> None:
        """Attempt to recover progress from the newest backup."""
        backup_files = sorted(
            progress_path.parent.glob(f"{progress_path.stem}*.bak"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not backup_files:
            return

        latest_backup = backup_files[0]
        try:
            import shutil

            shutil.copy2(latest_backup, progress_path)
            logger.info("Recovered progress from backup: %s", latest_backup)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to recover progress: %s", e)
            try:
                progress_path.unlink()
            except Exception:
                pass

    def create_progress(self, txt_file: str, total_chunks: int, chunks_hash: str) -> ProgressData:
        """Create a new progress record."""
        return ProgressData(
            txt_file=txt_file,
            total_chunks=total_chunks,
            completed_count=0,
            completed_indices=set(),
            outlines=[],
            last_update=datetime.now(),
            chunks_hash=chunks_hash,
        )

    def update_chunk_completed(
        self,
        progress_data: ProgressData,
        chunk_id: int,
        outline_data: dict[str, Any],
        processing_time: float | None = None,
    ) -> None:
        """Mark a chunk as completed and optionally record time."""
        progress_data.completed_indices.add(chunk_id)
        progress_data.completed_count += 1
        progress_data.outlines.append(outline_data)

        if processing_time is not None:
            progress_data.processing_times.append(processing_time)

        self.progress_tracker.add_update(
            {"chunk_id": chunk_id, "timestamp": datetime.now().isoformat()}
        )

        if progress_data.completed_count % 5 == 0:
            self.save_progress(progress_data)

    def is_progress_valid(
        self,
        progress_data: ProgressData | None,
        current_file: str,
        current_chunks: list[str],
        current_hash: str,
    ) -> bool:
        """Validate that existing progress matches current input."""
        if not progress_data:
            return False

        if progress_data.txt_file != current_file:
            logger.debug("Progress file path mismatch")
            return False

        if progress_data.total_chunks != len(current_chunks):
            logger.debug("Progress chunk count mismatch")
            return False

        if progress_data.chunks_hash != current_hash:
            logger.debug("Progress chunk hash mismatch")
            return False

        return True

    def clear_progress(self) -> None:
        """Delete the progress file if it exists."""
        progress_file = Path(self.processing_config.progress_file)
        try:
            if progress_file.exists():
                progress_file.unlink()
                logger.info("Progress file cleared")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to clear progress file: %s", e)

    def get_progress_summary(self, progress_data: ProgressData | None) -> dict[str, Any]:
        """Return a lightweight summary of current progress."""
        if not progress_data:
            return {"has_progress": False, "message": "暂无进度数据"}

        summary = {
            "has_progress": True,
            "file": progress_data.txt_file,
            "total_chunks": progress_data.total_chunks,
            "completed_chunks": progress_data.completed_count,
            "completion_rate": f"{progress_data.completion_rate * 100:.1f}%",
            "last_update": progress_data.last_update.strftime("%Y-%m-%d %H:%M:%S"),
            "average_processing_time": f"{progress_data.average_processing_time:.2f}s",
            "errors_count": len(progress_data.errors),
        }

        if progress_data.completed_count > 0:
            remaining_chunks = progress_data.total_chunks - progress_data.completed_count
            avg_time = progress_data.average_processing_time
            estimated_remaining = remaining_chunks * avg_time
            summary["estimated_remaining_time"] = f"{estimated_remaining / 60:.1f}m"

        return summary

    def finalize_progress(self, progress_data: ProgressData) -> None:
        """Flush in-memory updates, persist, then clear the progress file."""
        self.progress_tracker.force_flush()
        self.save_progress(progress_data)
        self.clear_progress()
        logger.info(
            "Progress completed: %s/%s",
            progress_data.completed_count,
            progress_data.total_chunks,
        )

    def add_progress_error(
        self,
        progress_data: ProgressData,
        chunk_id: int,
        error_message: str,
    ) -> None:
        """Record an error for a specific chunk and persist."""
        progress_data.add_error(chunk_id, error_message)
        logger.warning("Chunk %s failed: %s", chunk_id, error_message)
        self.save_progress(progress_data)
