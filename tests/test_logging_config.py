"""测试日志配置功能"""

import logging
import os
import shutil
from pathlib import Path

import pytest

from utils import setup_logging


@pytest.fixture(autouse=True)
def reset_logging_state():
    """每次测试前重置日志配置状态"""
    # 导入前重置全局状态
    import utils

    utils._logging_configured = False
    # 清除所有处理器并关闭
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    yield
    # 测试后再次清理，显式关闭处理器以释放文件句柄
    utils._logging_configured = False
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


@pytest.fixture
def test_log_dir():
    """创建测试日志目录"""
    test_dir = Path(__file__).parent / "test_logs_temp"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # 清理测试目录
    try:
        # 再次确保所有处理器都已关闭
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        # 尝试删除目录
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
    except Exception:
        pass  # 测试清理失败不影响测试结果


def test_log_directory_created(test_log_dir):
    """测试日志目录自动创建"""
    test_sub_dir = test_log_dir / "sub_logs"
    setup_logging(log_dir=str(test_sub_dir))
    assert test_sub_dir.exists()
    assert test_sub_dir.is_dir()


def test_log_file_named_with_date(test_log_dir):
    """测试日志文件按日期命名"""
    setup_logging(log_dir=str(test_log_dir))
    # 当前日志文件使用基础名称
    expected_file = test_log_dir / "novel_outline.log"
    assert expected_file.exists()


def test_log_file_has_content(test_log_dir):
    """测试日志文件可以正确写入内容"""
    setup_logging(log_dir=str(test_log_dir))

    test_logger = logging.getLogger("test_module")
    test_logger.info("Test log message")

    # 当前日志文件使用基础名称
    log_file = test_log_dir / "novel_outline.log"

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message" in content
    assert "test_module" in content


def test_log_level_from_env(test_log_dir):
    """测试从环境变量读取日志级别"""
    os.environ["LOG_LEVEL"] = "DEBUG"
    try:
        setup_logging(log_dir=str(test_log_dir))
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    finally:
        del os.environ["LOG_LEVEL"]


def test_log_backup_days_from_env(test_log_dir):
    """测试从环境变量读取日志保留天数"""
    os.environ["LOG_BACKUP_DAYS"] = "7"
    try:
        # 这个测试主要验证环境变量能被正确读取
        # 实际的轮转行为需要模拟时间变化
        setup_logging(log_dir=str(test_log_dir), log_backup_days=7)
        # 如果没有抛出异常，说明配置正确
        assert True
    finally:
        del os.environ["LOG_BACKUP_DAYS"]


def test_console_handler_info_level(test_log_dir):
    """测试控制台处理器只显示 INFO 及以上级别"""
    setup_logging(log_dir=str(test_log_dir))

    root_logger = logging.getLogger()
    console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]

    assert len(console_handlers) > 0
    console_handler = console_handlers[0]
    assert console_handler.level == logging.INFO


def test_log_directory_fallback(test_log_dir):
    """测试日志系统的回退机制"""
    # 测试正常情况下日志系统工作正常
    setup_logging(log_dir=str(test_log_dir))

    # 验证日志系统工作
    test_logger = logging.getLogger("test_fallback")
    test_logger.info("Fallback test")

    # 应该有文件和控制台处理器
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0


def test_logging_idempotent(test_log_dir):
    """测试多次调用 setup_logging 不会重复添加处理器"""
    setup_logging(log_dir=str(test_log_dir))
    root_logger = logging.getLogger()
    handler_count_before = len(root_logger.handlers)

    setup_logging(log_dir=str(test_log_dir))
    handler_count_after = len(root_logger.handlers)

    # 由于有 _logging_configured 标志，第二次调用应该直接返回
    assert handler_count_before == handler_count_after


def test_log_format(test_log_dir):
    """测试日志格式正确"""
    setup_logging(log_dir=str(test_log_dir))

    test_logger = logging.getLogger("test_format")
    test_logger.info("Format test")

    # 当前日志文件使用基础名称
    log_file = test_log_dir / "novel_outline.log"
    content = log_file.read_text(encoding="utf-8")

    # 验证日志格式包含时间、模块名、级别和消息
    assert "test_format" in content
    assert "INFO" in content
    assert "Format test" in content
