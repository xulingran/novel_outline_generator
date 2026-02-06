"""测试文件选择组件。"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gui.widgets.file_selector import FileSelector


def test_set_and_get_file(temp_test_file: Path) -> None:
    selector = FileSelector(MagicMock())
    selector.set_file(temp_test_file)
    assert selector.get_file() == temp_test_file


def test_set_nonexistent_file() -> None:
    selector = FileSelector(MagicMock())
    selector.set_file(Path("/no/such/file.txt"))
    assert selector.get_file() is None


def test_callback_invoked(temp_test_file: Path) -> None:
    callback = MagicMock()
    selector = FileSelector(MagicMock(), on_file_selected=callback)
    selector.set_file(temp_test_file)
    callback.assert_called_once()


def test_clear(temp_test_file: Path) -> None:
    selector = FileSelector(MagicMock())
    selector.set_file(temp_test_file)
    selector.clear()
    assert selector.get_file() is None


@pytest.mark.parametrize(
    ("token_count", "target", "expected"),
    [
        (2000, 1000, 2),
        (2001, 1000, 3),
        (0, 1000, 1),
    ],
)
def test_chunk_estimate_uses_ceil(
    monkeypatch, tmp_path: Path, token_count: int, target: int, expected: int
):
    test_file = tmp_path / "novel.txt"
    test_file.write_text("content", encoding="utf-8")

    selector = FileSelector(MagicMock())
    selector.current_file = test_file

    monkeypatch.setattr("gui.widgets.file_selector.count_tokens", lambda _text: token_count)
    monkeypatch.setattr(
        "gui.widgets.file_selector.get_processing_config",
        lambda: MagicMock(target_tokens_per_chunk=target),
    )

    selector._update_file_info()

    assert selector.chunks_label.cget("text") == f"预估块数: {expected}"
