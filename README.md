## 小说大纲生成工具

支持命令行与 Web 界面，使用LLM大模型自动分块、生成和合并大纲。

## 功能概览
- 多模型：OpenAI / 智谱 / Gemini，支持代理、中转URL。
- 自动分块，逐块生成大纲并递归合并总纲。
- Web UI：上传文件、查看进度与日志。
- 在线编辑 .env（含提示词模板）。
- 支持断点恢复、失败重试、并发处理。
- 处理完成后自动清理中间文件与上传缓存。

## 环境准备

### 1. 安装 Python 和 uv
- Python 3.10+（推荐 3.11+）
- 安装 [uv](https://github.com/astral-sh/uv)：
```bash
pip install uv
```

### 2. 创建虚拟环境并安装依赖
```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 激活虚拟环境（Linux/Mac）
source .venv/bin/activate

# 安装项目依赖
uv pip install openai tiktoken httpx python-dotenv google-generativeai zhipuai fastapi uvicorn python-multipart

# 安装开发依赖（可选）
uv pip install pytest pytest-asyncio
```

### 3. 配置环境变量
```bash
# Windows
python -Xutf8 create_env.py

# Linux/Mac
python create_env.py
```

## 使用方式

### 命令行
```bash
# Windows
python -Xutf8 main.py

# Linux/Mac
python main.py
```
按提示选择2，处理新文件，输入 txt/md 路径

### Web UI
在`main.py`中选择1使用webui
浏览器自动打开` ui/index.html`
**Web 界面支持上传、进度日志、Token 预估。**
**支持在webui编辑 .env 与提示词模板。**

## uv 常用命令

| 命令 | 说明 |
|------|------|
| `uv venv` | 创建虚拟环境 |
| `uv pip install <package>` | 安装包 |
| `uv pip freeze` | 查看已安装的包 |
| `uv pip freeze > requirements.lock` | 锁定依赖版本 |
| `uv pip sync requirements.lock` | 从锁定文件同步依赖 |

## 目录结构
```
- main.py：命令行入口，选择 CLI/Web。
- web_api.py：FastAPI 后端接口。
- ui/index.html：Web 前端界面。
- services/：分块、LLM 调用、进度管理、文件处理等。
- models/：数据模型与状态。
- prompts.py：提示词模板与自定义支持。
- outputs/：输出目录（已在 .gitignore）。
- pyproject.toml：项目配置和依赖管理。
- requirements.lock：依赖版本锁定文件。
```

## 许可证
本项目采用 MIT License，详见 LICENSE 文件。
