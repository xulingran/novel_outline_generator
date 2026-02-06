# CLAUDE.md

本文件用于指导 Claude Code 在本仓库中协作开发。

## 代码质量检查（提交前必须执行）

```bash
# 1. Ruff
.venv/bin/python -m ruff check . --fix

# 2. Black
.venv/bin/python -m black .

# 3. Mypy
.venv/bin/python -m mypy .

# 4. Pytest
.venv/bin/python -m pytest tests/ -v
```

Windows 路径可用 `.venv\Scripts\python`。

## 常用命令

```bash
# 运行主程序（菜单选择模式）
.venv/bin/python main.py

# 直接启动 GUI
.venv/bin/python gui_launcher.py

# 直接启动 Web API
.venv/bin/python -m uvicorn web_api:app --reload --port 8000

# 运行 GUI 测试
.venv/bin/python -m pytest tests/test_gui -v
```

## 运行模式

`main.py` 支持三种模式：

1. 模式 1：Web UI（FastAPI + 本地 `ui/index.html`）
2. 模式 2：CLI 文件处理
3. 模式 3：桌面 GUI（CustomTkinter）

## GUI 当前实现

### 关键文件

- `gui/main_window.py`：主窗口（处理、配置、日志、关于）
- `gui/async_worker.py`：后台线程运行协程
- `gui/config_dialog.py`：配置编辑与 `.env` 更新
- `gui/widgets/file_selector.py`：文件选择与 token/块数估算
- `gui/widgets/progress_bar.py`：进度、阶段、ETA 展示
- `gui/widgets/log_viewer.py`：日志查看和级别过滤
- `gui_launcher.py`：GUI 启动入口

### 处理流程（GUI）

1. 在主窗口选择文件并点击开始
2. 若检测到同文件历史进度，弹窗询问是否恢复
3. `AsyncWorker` 在线程中运行协程
4. 调用 `NovelProcessingService.process_novel(...)`
5. 通过 `progress_callback` 回传进度，主线程 `after()` 更新 UI
6. 可通过 `cancel_event` + `worker.stop()` 取消

### 配置保存行为

`ConfigDialog` 保存 `.env` 时采用“增量更新”策略：

- 只更新变更项
- 保留原注释与未改动行
- 不存在的键会追加到文件末尾
- 保存后会调用 `_refresh_config_cache()`

## 测试

### GUI 测试文件

- `tests/test_gui/test_main_window.py`
- `tests/test_gui/test_async_worker.py`
- `tests/test_gui/test_config_dialog.py`
- `tests/test_gui/test_file_selector.py`
- `tests/test_gui/test_progress_bar.py`
- `tests/test_gui/test_log_viewer.py`

### 说明

- GUI 测试使用 `tests/test_gui/conftest.py` 中的 mock `customtkinter`，可在无图形环境运行。
- 当前测试覆盖重点是主流程行为、回调调度、取消语义和配置写入策略。

## 架构与业务要点

- 核心处理服务：`services/novel_processing_service.py`
- LLM 抽象与熔断：`services/llm_service.py`
- 进度持久化与恢复：`services/progress_service.py`
- ETA 估算：`services/eta_estimator.py`
- 文本切分：`splitter.py`
- Token 计数：`tokenizer.py`

最终输出仍为纯文本大纲（非 Markdown）。

## 注意事项

- 修改 `.env` 后需重启对应进程（CLI/Web/GUI）。
- 不要在后台线程直接操作 GUI，必须通过 `after()` 回到主线程。
- 打包配置文件为 `build/app.spec`；如需图标资源，请先补齐对应文件路径。
