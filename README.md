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
1. Python 3.10+（推荐 3.11+）。
2. 安装依赖：
```python
pip install -r requirements.txt
```
3. 配置环境变量：运行
```python
python -Xutf8 create_env.py
```

## 使用方式
### 命令行
```python
python -Xutf8 main.py
```
按提示选择2，处理新文件，输入 txt/md 路径

### Web UI
在`main.py`中选择1使用webui
浏览器自动打开` ui/index.html`
**Web 界面支持上传、进度日志、Token 预估。**
**支持在webui编辑 .env 与提示词模板。**

## 目录结构
```
- main.py：命令行入口，选择 CLI/Web。
- web_api.py：FastAPI 后端接口。
- ui/index.html：Web 前端界面。
- services/：分块、LLM 调用、进度管理、文件处理等。
- models/：数据模型与状态。
- prompts.py：提示词模板与自定义支持。
- outputs/：输出目录（已在 .gitignore）。
```

## 许可证
本项目采用 MIT License，详见 LICENSE 文件。
