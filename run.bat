@echo off
REM 小说大纲生成器启动脚本

cd /d "%~dp0"
set "PYTHON_BIN=venv_system\Scripts\python.exe"

if not exist "%PYTHON_BIN%" (
    echo 未检测到 venv_system 虚拟环境。
    echo 请先执行：
    echo   py -3.12 -m venv venv_system
    echo   venv_system\Scripts\python -m pip install -e ".[dev]"
    exit /b 1
)

echo ========================================
echo 小说大纲生成器启动器
echo ========================================
echo 1^) WebUI 模式
echo 2^) 命令行模式
echo 3^) GUI 桌面模式
echo ========================================
set /p choice=请输入选项 (1/2/3，默认 1): 
if "%choice%"=="" set choice=1

if "%choice%"=="1" goto WEBUI
if "%choice%"=="2" goto CLI
if "%choice%"=="3" goto GUI

echo 无效选项: %choice%
exit /b 1

:WEBUI
echo 启动 WebUI 模式...
"%PYTHON_BIN%" -Xutf8 -m uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
goto :EOF

:CLI
echo 启动命令行模式...
"%PYTHON_BIN%" -Xutf8 main.py --mode process
goto :EOF

:GUI
echo 启动 GUI 桌面模式...
"%PYTHON_BIN%" gui_launcher.py
goto :EOF
