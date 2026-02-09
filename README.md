## 小说大纲生成工具

支持三种运行模式：`WebUI`、`命令行`、`桌面 GUI`。  
本项目已统一为 **Python 官方 `venv + pip` 工作流**，不再使用 `uv`。

## 快速开始（推荐）

### 1. 创建并激活虚拟环境

macOS / Linux：

```bash
python3 -m venv venv_system
source venv_system/bin/activate
```

Windows（PowerShell）：

```powershell
py -3.12 -m venv venv_system
venv_system\Scripts\Activate.ps1
```

Windows（CMD）：

```bat
py -3.12 -m venv venv_system
venv_system\Scripts\activate.bat
```

### 2. 安装依赖

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 3. 配置环境变量

```bash
cp .env.sample .env
```

Windows:

```bat
copy .env.sample .env
```

填写至少一个可用提供商的 API Key（`openai/gemini/zhipu/aihubmix`）。

---

## 启动方式

### 方式一：启动脚本（三选一菜单）

Windows:

```bat
run.bat
```

macOS / Linux:

```bash
./run.sh
```

菜单中可选：
1. WebUI
2. 命令行
3. GUI

### 方式二：手动命令

WebUI：

```bash
venv_system/bin/python -m uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
```

命令行：

```bash
venv_system/bin/python main.py --mode process
```

GUI：

```bash
venv_system/bin/python gui_launcher.py
```

Windows（对应写法）：

```bat
venv_system\Scripts\python -m uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
venv_system\Scripts\python -Xutf8 main.py --mode process
venv_system\Scripts\python gui_launcher.py
```

---

## 开发命令

```bash
venv_system/bin/python -m ruff check . --fix
venv_system/bin/python -m black .
venv_system/bin/python -m black . --check
venv_system/bin/python -m mypy .
venv_system/bin/python -m pytest tests/ -v
```

Windows：

```bat
venv_system\Scripts\python -m ruff check . --fix
venv_system\Scripts\python -m black .
venv_system\Scripts\python -m black . --check
venv_system\Scripts\python -m mypy .
venv_system\Scripts\python -m pytest tests/ -v
```

---

## 常见问题

### 1) GUI 启动失败（`_tkinter` / `init.tcl`）
- 请确认使用的是 `venv_system` 对应的 Python。
- macOS 建议使用 python.org 官方安装包或 pyenv 安装带 Tcl/Tk 的 Python。

### 2) 点开始处理没反应
- 先在处理页选择文件，按钮会启用。
- 看日志区是否有错误（配置/API Key/文件读取等）。

### 3) 修改 `.env` 后不生效
- 需要重启应用。

---

## 目录结构

```text
novel_outline_generator/
├── main.py
├── web_api.py
├── gui_launcher.py
├── run.bat
├── run.sh
├── gui/
├── services/
├── models/
├── tests/
└── ui/index.html
```

## 许可证

MIT
