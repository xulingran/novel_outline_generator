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

### 运行应用

```bash
# 菜单选择模式（推荐）
./run.sh          # Unix
run.bat           # Windows

# 直接运行
.venv/bin/python main.py --mode web_ui    # Web UI
.venv/bin/python main.py --mode process   # CLI
.venv/bin/python gui_launcher.py          # GUI
```

### 开发命令

```bash
# 代码质量检查
.venv/bin/python -m ruff check . --fix
.venv/bin/python -m black .
.venv/bin/python -m mypy .

# 运行测试
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/test_gui -v

# 运行单个测试
.venv/bin/python -m pytest tests/test_config.py::TestConfig::test_get_api_config -v
```

## 运行模式

`main.py` 支持三种模式：

1. 模式 1：Web UI（FastAPI + 本地 `ui/index.html`）
2. 模式 2：CLI 文件处理
3. 模式 3：桌面 GUI（CustomTkinter）

## GUI 当前实现

### 架构概述（组件化设计）

GUI 采用**三层组件架构**：

1. **Components**（`gui/components/`）：底层可复用组件
2. **Widgets**（`gui/widgets/`）：业务组件
3. **Pages**（`gui/pages/`）：页面组件

### 底层组件（Components）

- `animation.py`：动画管理器（缓动函数、帧动画）
- `icon.py`：Phosphor 图标组件（30+ 图标，支持尺寸/粗细）
- `card.py`：卡片容器（统一圆角、阴影、间距）
- `button.py`：按钮组件（primary/secondary/ghost 变体）
- `sidebar.py`：侧边导航栏（导航项、主题切换）

### 业务组件（Widgets）

- `file_selector.py`：文件选择器（拖放区域、token/块数估算）
- `progress_bar.py`：线性进度条（阶段、ETA、百分比、统计信息）
- `log_viewer.py`：日志查看器（级别、过滤、自动滚动）

### 页面组件（Pages）

- `process_page.py`：处理页（文件选择 → 进度可视化 → 实时日志）
- `config_page.py`：配置页（API/处理参数）
- `log_page.py`：日志页（系统日志查看）
- `about_page.py`：关于页（应用信息）

### 核心系统

- `main_window.py`：主窗口（侧边导航 + 主内容区）
- `theme_manager.py`：主题管理（Nord 配色、设计系统、主题切换）
- `async_worker.py`：后台线程运行协程
- `gui_launcher.py`：GUI 启动入口

### 设计系统

- **配色**：Nord 主题（dark/light 双模式）
- **间距**：xs(4) / sm(8) / md(16) / lg(24) / xl(32) / 2xl(48)
- **字体**：SF Pro Display (macOS) / Segoe UI (Windows) / Inter (Linux)
- **圆角**：4px (sm) / 8px (md) / 12px (lg)

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

### 测试结构

- `tests/test_gui/`：GUI 组件测试（使用 mock customtkinter）
- `tests/test_services/`：服务层单元测试
- `tests/test_models/`：数据模型测试
- `tests/conftest.py`：共享测试 fixtures

### GUI 测试文件

- `test_main_window.py`：主窗口测试
- `test_progress_bar.py`：进度条组件测试
- 其他组件测试：`test_async_worker.py`、`test_config_dialog.py`、`test_file_selector.py`、`test_log_viewer.py`

### 测试说明

- GUI 测试使用 `tests/test_gui/conftest.py` 中的 mock `customtkinter`，可在无图形环境运行
- 测试覆盖重点：主流程行为、回调调度、取消语义、配置写入策略
- 覆盖率报告：运行 `pytest` 时自动生成（`--cov-report=term-missing`）

## 架构与业务要点

### 服务层（`services/`）

- `novel_processing_service.py`：核心处理服务（切分 → 生成 → 合并）
- `llm_service.py`：LLM 抽象与熔断器（多提供商支持）
- `progress_service.py`：进度持久化与恢复
- `eta_estimator.py`：ETA 估算（移动平均 + 异常值剔除）
- `file_service.py`：文件 I/O（编码检测）
- `token_estimator.py`：Token 预估（成本估算）
- `task_queue.py`：异步任务队列
- `job_manager.py`：作业生命周期管理

### 数据模型（`models/`）

- `processing_state.py`：处理状态和进度数据
- `outline.py`：文本块和大纲模型
- `character.py`：角色相关模型

### 核心工具

- `splitter.py`：智能文本切分（章/段/token）
- `tokenizer.py`：Token 计数
- `config.py`：配置管理（延迟验证）
- `prompts.py`：LLM 提示词模板
- `validators.py`：输入验证
- `exceptions.py`：自定义异常层级

**输出格式**：纯文本大纲（非 Markdown）。

## LLM 提供商架构

项目支持多个 LLM 提供商，通过统一的 `LLMService` 抽象层：

- **OpenAI**：GPT 系列（默认）
- **Gemini**：Google Gemini API
- **Zhipu**（智谱）：智谱 AI GLM 系列
- **AiHubMix**：混合 API 服务

**关键特性**：
- 熔断器模式：自动故障隔离和恢复
- 延迟验证：API 密钥在首次使用时验证
- 注册机制：通过 `@register_llm_provider` 装饰器扩展新提供商

## 关键约定

**项目结构：**
- 扁平包结构（无 `src/` 目录），核心模块在根目录
- `services/`、`models/`、`gui/` 是主要子目录
- 使用标准 Python `venv + pip` 工作流（不再使用 `uv`）

**包管理：**
- 虚拟环境：`venv_system/`（推荐）或 `.venv/`
- 安装：`python -m pip install -e ".[dev]"`
- 构建系统：hatchling（`pyproject.toml`）

**代码风格：**
- 行长度限制：100 字符（Black/Ruff）
- 导入顺序：标准库 → 第三方 → 本地（严格）
- 类型注解：所有公共 API 需要（现代 `str | None` 语法）
- 中文文档字符串

**异步编程：**
- I/O 操作必须使用 `async/await`
- HTTP 客户端：`httpx.AsyncClient`（共享，关闭时清理）
- 同步第三方调用用 `loop.run_in_executor` 包装

**配置：**
- 通过 `config.get_api_config()` 和 `config.get_processing_config()` 访问
- **延迟验证**：API 密钥在首次访问时验证，非导入时
- 环境变量在 `.env` 中（永不硬编码密钥）
- 配置变更后调用 `_refresh_config_cache()` 或重启

**错误处理：**
- 部分块完成：失败块拆分为 5 个子块，若部分成功则合并
- 进度恢复：支持从部分完成状态恢复
- 熔断器模式：`llm_service.py` 中的 `CircuitBreaker` 用于容错
- 异常层级：`NovelOutlineError` → `APIError` / `ConfigurationError` / `APIKeyError`

## 启动脚本

项目提供了两个启动脚本：

- **run.sh**（Unix）：检查 `venv_system` 存在后提供三选一菜单
- **run.bat**（Windows）：对应 Windows 版本

脚本启动流程：
1. 检测 `venv_system` 虚拟环境
2. 提供模式选择（WebUI/CLI/GUI）
3. 启动对应模式

## 注意事项

- 修改 `.env` 后需重启对应进程（CLI/Web/GUI）。
- 不要在后台线程直接操作 GUI，必须通过 `after()` 回到主线程。
- **GUI 打包**：macOS Python 3.14 无内置 tkinter，建议使用 PyInstaller 打包
  ```bash
  venv_system/bin/python -m pip install pyinstaller
  venv_system/bin/python -m PyInstaller build/app.spec --onefile --windowed
  ./dist/小说大纲生成器
  ```
- 打包配置文件为 `build/app.spec`；如需图标资源，请先补齐对应文件路径。
