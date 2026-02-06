#!/usr/bin/env python
"""GUI 桌面应用启动入口。"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main() -> None:
    """启动 GUI。"""
    from config import init_config
    from gui.main_window import MainWindow
    from utils import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        init_config()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"配置初始化失败: {exc}")
        print(f"错误: 配置初始化失败: {exc}")
        print("请检查 .env 文件配置。")
        sys.exit(1)

    try:
        app = MainWindow()
        app.mainloop()
    except Exception as exc:  # noqa: BLE001
        logger.exception("GUI 启动失败")
        print(f"错误: GUI 启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
