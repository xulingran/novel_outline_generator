"""测试日志查看组件。"""

from pathlib import Path
from unittest.mock import MagicMock

from gui.widgets.log_viewer import LogViewer


def test_refresh_log(temp_log_file: Path) -> None:
    viewer = LogViewer(MagicMock(), log_file=temp_log_file, auto_refresh=False)
    viewer.refresh_log()
    assert "Test message" in viewer.get_text()


def test_filter_by_level(temp_log_file: Path) -> None:
    viewer = LogViewer(MagicMock(), log_file=temp_log_file, auto_refresh=False)
    lines = [
        "2025-01-31 10:00:00 - test - DEBUG - d\n",
        "2025-01-31 10:00:00 - test - ERROR - e\n",
    ]
    filtered = viewer._filter_by_level(lines, "ERROR")
    assert len(filtered) == 1
    assert "ERROR" in filtered[0]


def test_search_keyword(temp_log_file: Path) -> None:
    viewer = LogViewer(MagicMock(), log_file=temp_log_file, auto_refresh=False)
    viewer.refresh_log()
    assert viewer.search("ERROR") >= 1
