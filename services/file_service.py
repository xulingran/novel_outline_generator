"""
文件服务模块
提供文件读写、编码检测等功能的封装
"""

import logging
import os
from pathlib import Path
from typing import Any

from config import get_processing_config
from exceptions import EncodingError, FileValidationError
from utils import (
    atomic_write_json,
    atomic_write_text,
    safe_read_json,
    safe_read_text,
)
from utils import (
    get_file_info as utils_get_file_info,
)
from validators import validate_encoding_list, validate_file_path, validate_output_dir

logger = logging.getLogger(__name__)


class FileService:
    """文件服务类"""

    def __init__(self):
        self.processing_config = get_processing_config()
        self._validate_encodings()

    def _validate_encodings(self) -> None:
        """验证编码列表"""
        try:
            validate_encoding_list(self.processing_config.encodings)
        except FileValidationError as e:
            logger.error(f"编码配置无效: {e}")
            # 使用默认编码
            self.processing_config.encodings = ["utf-8", "gbk", "gb2312"]

    def read_text_file(self, file_path: str | Path) -> tuple[str, str]:
        """
        读取文本文件，自动检测编码

        Args:
            file_path: 文件路径

        Returns:
            Tuple[str, str]: (文件内容, 实际使用的编码)

        Raises:
            FileNotFoundError: 文件不存在
            EncodingError: 所有编码都失败
        """
        file_path = validate_file_path(
            file_path,
            allowed_extensions=[".txt", ".md", ".text"],
            max_size_mb=100,  # 限制100MB
        )

        logger.debug(f"尝试读取文件: {file_path}")

        # 使用备选编码列表
        fallback_encodings = self.processing_config.encodings[1:]  # 跳过第一个编码
        try:
            content, actual_encoding = safe_read_text(
                file_path,
                encoding=self.processing_config.encodings[0],
                fallback_encodings=fallback_encodings,
            )
            logger.info(f"成功读取文件: {file_path}，使用编码: {actual_encoding}")
            return content, actual_encoding
        except UnicodeDecodeError as e:
            raise EncodingError(
                f"无法读取文件 {file_path}，已尝试编码: {', '.join(self.processing_config.encodings)}"
            ) from e
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            raise

    def write_text_file(self, file_path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """
        写入文本文件（原子性操作）

        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 文件编码
        """
        try:
            atomic_write_text(file_path, content, encoding=encoding)
            logger.debug(f"成功写入文件: {file_path}")
        except Exception as e:
            logger.error(f"写入文件失败: {file_path}, 错误: {e}")
            raise

    def read_json_file(self, file_path: str | Path, default: dict | None = None) -> dict:
        """
        读取JSON文件（带容错处理）

        Args:
            file_path: 文件路径
            default: 默认值（如果读取失败）

        Returns:
            dict: JSON数据
        """
        try:
            data = safe_read_json(file_path, default=default or {})
            logger.debug(f"成功读取JSON文件: {file_path}")
            return data
        except Exception as e:
            logger.error(f"读取JSON文件失败: {file_path}, 错误: {e}")
            raise

    def write_json_file(
        self, file_path: str | Path, data: dict[str, Any] | list[Any], backup: bool = True
    ) -> None:
        """
        写入JSON文件（原子性操作）

        Args:
            file_path: 文件路径
            data: JSON数据
            backup: 是否创建备份
        """
        try:
            atomic_write_json(file_path, data, backup=backup)
            logger.debug(f"成功写入JSON文件: {file_path}")
        except Exception as e:
            logger.error(f"写入JSON文件失败: {file_path}, 错误: {e}")
            raise

    def ensure_output_directory(self, subdirectory: str | None = None) -> Path:
        """
        确保输出目录存在

        Args:
            subdirectory: 子目录名（可选）

        Returns:
            Path: 输出目录路径
        """
        output_dir = self.processing_config.output_dir
        if subdirectory:
            output_dir = os.path.join(output_dir, subdirectory)

        return validate_output_dir(output_dir)

    def list_txt_files(self, directory: str | Path) -> list[Path]:
        """
        列出目录中的所有文本文件

        Args:
            directory: 目录路径

        Returns:
            List[Path]: 文本文件列表
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"目录不存在: {directory}")
            return []

        txt_files: list[Path] = []
        for pattern in ["*.txt", "*.md", "*.text"]:
            txt_files.extend(directory.glob(pattern))

        logger.debug(f"在目录 {directory} 中找到 {len(txt_files)} 个文本文件")
        return sorted(txt_files)

    def get_file_size(self, file_path: str | Path) -> int:
        """
        获取文件大小（字节）

        Args:
            file_path: 文件路径

        Returns:
            int: 文件大小
        """
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.debug(f"获取文件大小失败: {e}")
            return 0

    def get_file_info(self, file_path: str | Path) -> dict:
        """获取文件信息（存在性、大小、时间等），包装 utils.get_file_info。"""
        return utils_get_file_info(file_path)

    def remove_backups(self, directory: str | Path, pattern: str = "*.bak") -> int:
        """删除目录下的备份文件，返回删除数量。"""
        directory = Path(directory)
        if not directory.exists():
            return 0

        removed = 0
        for bak in directory.rglob(pattern):
            try:
                bak.unlink()
                removed += 1
                logger.debug(f"删除备份文件: {bak}")
            except Exception as e:
                logger.warning(f"删除备份文件失败: {bak}, 错误: {e}")
        return removed
