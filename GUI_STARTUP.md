# GUI 桌面应用启动指南

## 问题说明

macOS 上 Python 3.14 没有内置 tkinter 支持，这是 Python 3.14 在 macOS 上的已知问题。

## 解决方案（按推荐顺序）

### 方案 1：使用 PyInstaller 打包（推荐）

打包后的应用会自动包含所有依赖，无需 tkinter：

```bash
# 1. 安装 PyInstaller
venv_system/bin/python -m pip install pyinstaller

# 2. 打包应用
venv_system/bin/python -m PyInstaller build/app.spec --onefile --windowed

# 3. 运行打包后的应用
./dist/小说大纲生成器
```

### 方案 2：使用 pyenv 安装带 tkinter 的 Python

```bash
# 1. 安装 pyenv
brew install pyenv

# 2. 安装 Python 3.12（带 tkinter）
pyenv install 3.12.8

# 3. 在项目中使用
pyenv local 3.12.8

# 4. 启动 GUI
python gui_launcher.py
```

### 方案 3：使用命令行模式（立即可用）

```bash
# 使用命令行模式，无需 GUI
venv_system/bin/python main.py
# 选择模式 2
```

### 方案 4：使用 Web UI 模式（立即可用）

```bash
# 启动 Web 服务
venv_system/bin/python -m uvicorn web_api:app --reload --port 8000

# 在浏览器中打开
open http://localhost:8000
```

## 当前状态

✅ 项目代码：支持 Python 3.12+  
✅ 类型注解：使用 `str | None` 等现代语法  
✅ 依赖安装：完整（包括 customtkinter）  
⚠️  GUI 启动：需要 tkinter（打包后会有）

## 开发建议

在开发期间，推荐使用：
- **功能开发**：使用命令行模式或 Web UI 模式
- **GUI 测试**：使用 PyInstaller 打包后测试
- **调试 GUI**：建议安装 pyenv 使用带 tkinter 的 Python

## 打包后的优势

使用 PyInstaller 打包后的应用：
1. ✅ 包含所有依赖，无需用户安装 Python
2. ✅ 自动处理 tkinter 兼容性问题
3. ✅ 独立可执行文件，双击即可运行
4. ✅ 跨平台支持（Windows/macOS/Linux）
