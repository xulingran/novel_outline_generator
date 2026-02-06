"""测试进度条组件。"""

from unittest.mock import MagicMock

from gui.widgets.progress_bar import ProgressBar


def test_update_progress() -> None:
    bar = ProgressBar(MagicMock())
    bar.update_progress(completed=3, total=10, failed=1, partial=1, phase="processing")

    assert bar.completed_chunks == 3
    assert bar.total_chunks == 10
    assert bar.failed_chunks == 1
    assert bar.partial_chunks == 1
    assert bar.current_phase == "processing"


def test_eta_format() -> None:
    bar = ProgressBar(MagicMock())
    assert "1分钟" in bar._format_eta(60, 0.7)


def test_reset() -> None:
    bar = ProgressBar(MagicMock())
    bar.update_progress(completed=5, total=10)
    bar.reset()
    assert bar.completed_chunks == 0
    assert bar.total_chunks == 0
