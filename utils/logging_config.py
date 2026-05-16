"""日志配置模块"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_logging_configured = False


_DEFAULT = object()


def setup_logging(level=None, log_dir=_DEFAULT, log_backup_days=30):
    """统一配置日志系统，支持按天自动轮转

    Args:
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
        log_dir: 日志目录，默认从环境变量 LOG_DIR 读取
        log_backup_days: 日志保留天数，默认从环境变量 LOG_BACKUP_DAYS 读取
    """
    global _logging_configured
    if _logging_configured:
        return

    # 从环境变量读取配置（仅当参数为默认值时）
    if log_dir is _DEFAULT:
        log_dir = os.getenv("LOG_DIR", "logs")
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
    try:
        log_backup_days = int(os.getenv("LOG_BACKUP_DAYS", str(log_backup_days)))
    except ValueError:
        log_backup_days = 30

    # 确保日志目录存在
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # 如果创建目录失败，回退到项目根目录
        print(f"警告：无法创建日志目录 {log_dir}，将使用项目根目录: {e}")
        log_path = Path.cwd()

    # 日志文件名：当前日志使用基础名称，轮转后自动添加日期后缀
    # 例如：novel_outline.log（当前）-> novel_outline.log.2026-01-16（历史）
    log_filename = "novel_outline.log"
    log_filepath = log_path / log_filename

    # 创建按天轮转的文件处理器
    try:
        file_handler = TimedRotatingFileHandler(
            log_filepath,
            when="midnight",
            interval=1,
            backupCount=log_backup_days,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(level)
    except (OSError, ValueError) as e:
        print(f"警告：无法创建日志文件处理器: {e}")
        file_handler = None

    # 控制台处理器（只显示 INFO 及以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if file_handler:
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _logging_configured = True


def init_logging(level=None, log_dir=_DEFAULT, log_backup_days=30):
    """显式初始化日志系统（应在应用入口处调用）

    Args:
        level: 日志级别，默认从环境变量 LOG_LEVEL 读取
        log_dir: 日志目录，默认从环境变量 LOG_DIR 读取
        log_backup_days: 日志保留天数，默认从环境变量 LOG_BACKUP_DAYS 读取

    Returns:
        logging.Logger: 根日志器
    """
    setup_logging(level, log_dir, log_backup_days)
    return logging.getLogger()
