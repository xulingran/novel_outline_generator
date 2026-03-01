"""
GUI 页面模块

提供各个功能页面：处理、配置、日志、关于。
"""

from gui.pages.about_page import AboutPage
from gui.pages.config_page import ConfigPage
from gui.pages.log_page import LogPage
from gui.pages.process_page import ProcessPage

__all__ = [
    "ProcessPage",
    "ConfigPage",
    "LogPage",
    "AboutPage",
]
