"""文件上传处理"""

import logging
import shutil
from pathlib import Path

from config import load_env_file as load_env_file_from_config

logger = logging.getLogger(__name__)

ENV_PATH = Path(".env")
UPLOAD_DIR = Path("outputs/uploads")
_UPLOAD_ROOT = UPLOAD_DIR.resolve()


def load_env_file() -> dict[str, str]:
    """读取 .env 原始配置值（不掩码）。"""
    import web_api

    return load_env_file_from_config(str(web_api.ENV_PATH))


def _resolve_upload_path(path: str) -> Path | None:
    """返回在 uploads 目录下的路径，其他情况返回 None。"""
    import web_api

    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None

    upload_root = web_api._UPLOAD_ROOT
    if resolved == upload_root:
        return None

    try:
        resolved.relative_to(upload_root)
        return resolved
    except ValueError:
        return None


def cleanup_uploads(protected_paths: set[Path] | None = None) -> int:
    """删除上传目录中的内容，保留目录本身。

    Args:
        protected_paths: 需要保留的上传文件路径集合（已解析的绝对路径）
    """
    import web_api

    upload_dir = web_api.UPLOAD_DIR
    if not upload_dir.exists():
        return 0

    keep = {p.resolve() for p in protected_paths} if protected_paths else set()

    cleaned = 0
    for item in upload_dir.iterdir():
        item_path = item.resolve()
        if any(item_path == kept_path or item_path in kept_path.parents for kept_path in keep):
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
            cleaned += 1
        except Exception as e:
            # 忽略清理失败，避免影响主流程
            logger.warning(f"清理上传文件失败: {e}")
    return cleaned
