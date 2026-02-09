#!/bin/bash
# 小说大纲生成器启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="venv_system/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "未检测到 venv_system 虚拟环境。"
    echo "请先执行："
    echo "  python3 -m venv venv_system"
    echo "  venv_system/bin/python -m pip install -e \".[dev]\""
    exit 1
fi

echo "========================================"
echo "小说大纲生成器启动器"
echo "========================================"
echo "1) WebUI 模式"
echo "2) 命令行模式"
echo "3) GUI 桌面模式"
echo "========================================"
read -r -p "请输入选项 (1/2/3，默认 1): " choice

case "${choice:-1}" in
1)
    echo "启动 WebUI 模式..."
    "$PYTHON_BIN" -m uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
    ;;
2)
    echo "启动命令行模式..."
    "$PYTHON_BIN" main.py --mode process
    ;;
3)
    echo "启动 GUI 桌面模式..."
    "$PYTHON_BIN" gui_launcher.py
    ;;
*)
    echo "无效选项: ${choice}"
    exit 1
    ;;
esac
