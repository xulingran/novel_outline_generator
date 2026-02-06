"""测试配置对话框。"""

from pathlib import Path
from unittest.mock import MagicMock

from gui.config_dialog import ConfigDialog


def test_dialog_init() -> None:
    dialog = ConfigDialog(MagicMock())
    assert hasattr(dialog, "provider_var")
    assert hasattr(dialog, "chunk_size_var")
    assert hasattr(dialog, "proxy_enabled_var")


def test_collect_env_lines() -> None:
    dialog = ConfigDialog(MagicMock())
    lines = dialog._collect_env_lines()
    assert any(line.startswith("API_PROVIDER=") for line in lines)
    assert any(line.startswith("PARALLEL_LIMIT=") for line in lines)


def test_update_env_file_preserves_comments_and_unmodified_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# global comment\n"
        "API_PROVIDER=openai # provider comment\n"
        "PARALLEL_LIMIT=5\n"
        "UNRELATED=value\n",
        encoding="utf-8",
    )

    dialog = ConfigDialog(MagicMock())
    updates = {
        "API_PROVIDER": "gemini",
        "PARALLEL_LIMIT": "5",
        "NEW_KEY": "new_value",
    }

    dialog._update_env_file(env_file, updates)

    content = env_file.read_text(encoding="utf-8")
    assert "# global comment" in content
    assert "API_PROVIDER=gemini # provider comment" in content
    assert "PARALLEL_LIMIT=5" in content
    assert "UNRELATED=value" in content
    assert "NEW_KEY=new_value" in content
