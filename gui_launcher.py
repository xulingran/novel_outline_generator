#!/usr/bin/env python
"""
桌面应用启动脚本

小说大纲生成器 GUI 应用入口。
"""

import logging
import sys

# 添加项目根目录到路径
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """启动 GUI 应用"""
    from config import init_config
    from gui.main_window import MainWindow
    from utils import init_logging

    # 初始化日志
    init_logging()

    # 记录启动
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("启动小说大纲生成器 GUI 应用")
    logger.info("=" * 60)

    # 初始化配置
    try:
        init_config()
        logger.info("配置初始化成功")
    except Exception as e:
        logger.error(f"配置初始化失败: {e}")
        print(f"错误: 配置初始化失败: {e}")
        print("\n请检查 .env 文件是否存在并包含必要的配置。")
        print("可以从 .env.sample 复制一个模板:")
        print("  cp .env.sample .env")
        sys.exit(1)

    # 创建并启动 GUI
    try:
        logger.info("创建主窗口...")
        app = MainWindow()
        logger.info("启动 GUI 主循环...")
        app.mainloop()
        logger.info("GUI 应用已退出")
    except Exception as e:
        logger.exception("GUI 启动失败")
        print(f"错误: GUI 启动失败: {e}")
        sys.exit(1)

    # 检查重启标志
    restart_flag = project_root / ".restart_flag"
    if restart_flag.exists():
        restart_flag.unlink()
        logger.info("重启应用...")
        import os

        # 重启应用
        os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()
