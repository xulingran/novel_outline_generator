# GUI美化计划 - Nord主题 + 主题切换 + 图标系统

## TL;DR

> **Quick Summary**: 将CustomTkinter GUI从硬编码的dark/blue主题升级为支持light/dark/system模式切换的专业级Nord配色方案，同时添加图标系统和动画效果。
>
> **Deliverables**:
> - 主题管理模块 (`gui/theme_manager.py`)
> - 更新所有GUI组件应用Nord配色和图标
> - 主窗口添加主题切换UI
> - 程序化图标生成系统
> - 动画和微交互效果
>
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential (theme system first, then components)
> **Critical Path**: Task 1 (theme_manager) → Tasks 2-6 (apply to components) → Task 7 (main window UI) → Task 8 (icons)

---

## Context

### Original Request
将桌面GUI做的更美观一些，要求：
- 中度改造（主题切换、精致配色、图标系统）
- 支持亮主题/暗主题/跟随系统三种模式
- 使用北欧风(Nord)配色方案
- 添加图标系统提升识别度
- 添加动画和微交互效果

### Interview Summary
**Key Discussions**:
- **美化深度**: 中度改造（不是全面重构，保持现有布局结构）
- **主题系统**: 必须支持 light/dark/system 三种切换模式
- **配色方案**: Nord - 冷色调专业配色，适合生产力工具
- **图标系统**: 需要，使用Pillow程序化生成避免外部依赖
- **动画效果**: 需要，包括悬停、进度条动画、过渡效果

**Research Findings**:
- CustomTkinter支持 `ctk.set_appearance_mode("light"|"dark"|"system")`
- 支持元组颜色自动适配主题：`("light_color", "dark_color")`
- Nord配色是经过验证的专业调色板（https://www.nordtheme.com/）
- 使用Pillow生成图标可以保持项目独立性

### Metis Review
**Self-Analysis** (Metis consultation attempted but failed, proceeding with self-review):
- 需要确保主题切换时所有组件实时响应
- 需要主题状态持久化机制
- 需要平台特定字体回退方案
- 图标尺寸需要适配不同DPI

---

## Work Objectives

### Core Objective
在不改变现有功能逻辑和布局结构的前提下，将GUI从硬编码的dark/blue主题升级为支持主题切换的专业级Nord配色系统，同时添加图标和动画效果。

### Concrete Deliverables
- `gui/theme_manager.py` - 主题管理模块（Nord配色、主题切换、状态持久化）
- 更新 `gui/main_window.py` - 添加主题切换UI
- 更新 `gui/config_dialog.py` - 应用主题配色
- 更新 `gui/widgets/progress_bar.py` - 改进进度条视觉效果
- 更新 `gui/widgets/log_viewer.py` - 应用主题样式
- 更新 `gui/widgets/file_selector.py` - 添加图标和主题
- `gui/assets/icons/` - 图标资源目录
- 更新 `gui/assets/create_icons.py` - 生成Nord主题图标

### Definition of Done
- [x] 主题管理模块创建并测试通过
- [x] 所有GUI组件应用Nord配色
- [x] 主窗口有主题切换控件（light/dark/system）
- [x] 所有按钮和标签有相应图标
- [x] 按钮有悬停效果
- [x] 进度条有平滑动画
- [x] 主题设置可持久化到配置文件
- [x] 切换主题时所有UI实时更新

### Must Have
- 保持现有功能逻辑不变（只改样式）
- 支持light/dark/system三种主题模式
- 使用Nord配色方案的完整色板
- 图标系统正常工作
- 主题切换流畅无闪烁
- 所有组件在三种主题下都清晰可读

### Must NOT Have (Guardrails)
- **禁止改变现有布局结构**（如标签页顺序、按钮位置）
- **禁止引入外部图标库**（使用Pillow程序化生成）
- **禁止修改业务逻辑**（如文件处理、API调用）
- **禁止添加新的功能模块**（仅美化现有功能）
- **避免硬编码颜色**（使用theme_manager提供的配色）
- **不要过度设计动画**（保持流畅，不影响性能）

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> This is NOT conditional — it applies to EVERY task, regardless of test strategy.
>
> **FORBIDDEN** — acceptance criteria that require:
> - "User manually tests..." / "用户手动测试..."
> - "User visually confirms..." / "用户用眼睛确认..."
> - "User interacts with..." / "用户交互..."
> - "Ask user to verify..." / "请用户验证..."
> - ANY step where a human must perform an action
>
> **ALL verification is executed by the agent** using tools (Playwright, interactive_bash, curl, etc.). No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, fixtures in tests/conftest.py)
- **Automated tests**: Tests-after
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

> Each task MUST include Agent-Executed QA Scenarios.
> These describe how the executing agent DIRECTLY verifies the deliverable
> by running the GUI application and testing its behavior.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Create theme_manager.py (no dependencies)

Wave 2 (After Wave 1):
├── Task 2: Update progress_bar.py (depends: theme_manager)
├── Task 3: Update log_viewer.py (depends: theme_manager)
├── Task 4: Update file_selector.py (depends: theme_manager)
└── Task 5: Update config_dialog.py (depends: theme_manager)

Wave 3 (After Wave 2):
├── Task 6: Update main_window.py (depends: 2,3,4,5)
└── Task 7: Update create_icons.py (depends: theme_manager)

Wave 4 (After Wave 3):
└── Task 8: Integration test (depends: 6,7)

Critical Path: 1 → 2 → 6 → 8
Parallel Speedup: ~50% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2,3,4,5,7 | None |
| 2 | 1 | 6 | 3,4,5,7 |
| 3 | 1 | 6 | 2,4,5,7 |
| 4 | 1 | 6 | 2,3,5,7 |
| 5 | 1 | 6 | 2,3,4,7 |
| 6 | 2,3,4,5 | 8 | 7 |
| 7 | 1 | 8 | 2,3,4,5,6 |
| 8 | 6,7 | None | None (final) |

---

## TODOs

- [x] 1. 创建主题管理模块 `gui/theme_manager.py`

  **What to do**:
  - 创建 `ThemeManager` 类管理主题状态
  - 定义 Nord 配色方案（light/dark 两种变体）
  - 定义设计系统常量（间距、字体、圆角）
  - 实现主题切换功能（light/dark/system）
  - 实现主题状态持久化（保存到 `settings.json`）
  - 提供 `get_color()` 方法获取当前主题颜色
  - 提供 `on_theme_change` 回调机制，组件订阅主题变化
  - 导出常量和类型定义供其他模块使用

  **Must NOT do**:
  - 不要修改CustomTkinter的全局主题设置（由调用方控制）
  - 不要引入外部依赖（只使用标准库 + customtkinter）
  - 不要硬编码任何颜色值

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `unspecified-high`
    - Reason: Requires understanding of CustomTkinter architecture, design system concepts, and proper state management patterns
  - **Skills**: `[]`
    - No special skills needed - pure Python code structure

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with no other tasks)
  - **Blocks**: Tasks 2, 3, 4, 5, 7
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `config.py` - Configuration management pattern (env variables, lazy validation, caching)
  - `gui/main_window.py:48-49` - Current theme setting pattern: `ctk.set_appearance_mode("dark")`, `ctk.set_default_color_theme("blue")`

  **API/Type References** (contracts to implement against):
  - `customtkinter.CTk.set_appearance_mode()` - Sets appearance mode ("light", "dark", "system")
  - `customtkinter.CTk.set_default_color_theme()` - Sets color theme ("blue", "green", "dark-blue")
  - CustomTkinter widget color parameters: `fg_color`, `bg_color`, `text_color`, `border_color`, `hover_color`, `border_color`, `progress_color`

  **Test References** (testing patterns to follow):
  - `tests/test_config.py` - Test configuration management pattern
  - `tests/conftest.py` - Fixture pattern for test setup

  **Documentation References** (specs and requirements):
  - Nord Theme Spec: https://www.nordtheme.com/ (official color values)
  - CustomTkinter Docs: https://customtkinter.tomschimansky.com/ (widget color parameters)
  - Design System: 8px base spacing, platform-specific fonts, corner radius system

  **External References** (libraries and frameworks):
  - Official docs: https://customtkinter.tomschimansky.com/
  - Nord theme: https://www.nordtheme.com/
  - Pillow (PIL) docs: https://pillow.readthedocs.io/ (for icon generation)

  **WHY Each Reference Matters**:
  - `config.py`: Shows how to manage application settings with validation and caching - theme_manager should follow similar pattern
  - `gui/main_window.py:48-49`: Shows current hard-coded theme - new code should replace this with dynamic theme management
  - Nord Theme Spec: Provides exact hex color values for consistent, professional appearance
  - CustomTkinter Docs: Required to understand which color parameters each widget accepts

  **Acceptance Criteria**:

  - [ ] Module `gui/theme_manager.py` exists with `ThemeManager` class
  - [ ] Nord colors defined in `NORD_COLORS` dict with "light" and "dark" keys
  - [ ] `get_color(name, mode="auto")` method returns correct hex color
  - [ ] `get_color()` returns tuple color `("light_hex", "dark_hex")` when mode="auto"
  - [ ] `get_color()` returns specific hex color when mode="light" or mode="dark"
  - [ ] `set_theme("light"|"dark"|"system")` method exists and calls `ctk.set_appearance_mode()`
  - [ ] `get_current_theme()` returns current theme setting
  - [ ] `on_theme_change` callback mechanism works (subscribed functions get called)
  - [ ] Theme state saved to `settings.json` in project root
  - [ ] Theme loaded from `settings.json` on initialization (fallback to "dark" if not exists)
  - [ ] Design system constants defined: `SPACING`, `FONTS`, `CORNER_RADIUS`
  - [ ] pytest tests pass: `pytest tests/ -k theme_manager -v` → PASS

  **Agent-Executed QA Scenarios (MANDATORY — per-task, ultra-detailed):**

  ```
  Scenario: ThemeManager initializes with default dark theme
    Tool: Bash (Python)
    Preconditions: None
    Steps:
      1. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(tm.get_current_theme())"
      2. Assert: Output is "dark"
      3. Assert: settings.json created or updated
    Expected Result: Default theme is dark
    Evidence: Console output captured

  Scenario: get_color returns tuple for auto mode
    Tool: Bash (Python)
    Preconditions: ThemeManager imported
    Steps:
      1. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(type(tm.get_color('accent'))); print(tm.get_color('accent'))"
      2. Assert: Output type is tuple
      3. Assert: Tuple length is 2
      4. Assert: First element is hex color for light mode
      5. Assert: Second element is hex color for dark mode
    Expected Result: Auto mode returns theme-adaptive tuple
    Evidence: Console output captured

  Scenario: get_color returns specific hex for fixed mode
    Tool: Bash (Python)
    Preconditions: ThemeManager imported
    Steps:
      1. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(tm.get_color('accent', mode='light'))"
      2. Assert: Output is single hex string
      3. Assert: Output matches Nord light accent color (#5E81AC)
    Expected Result: Fixed mode returns specific color
    Evidence: Console output captured

  Scenario: set_theme updates CustomTkinter appearance
    Tool: Bash (Python)
    Preconditions: ThemeManager imported
    Steps:
      1. Run: python -c "from gui.theme_manager import ThemeManager; import customtkinter as ctk; tm = ThemeManager(); tm.set_theme('light'); print('OK')"
      2. Assert: No exceptions raised
      3. Assert: Output is "OK"
    Expected Result: Theme switching works without errors
    Evidence: Console output captured

  Scenario: Theme state persists to settings.json
    Tool: Bash
    Preconditions: settings.json exists
    Steps:
      1. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); tm.set_theme('light')"
      2. Run: cat settings.json
      3. Assert: File contains "theme": "light"
      4. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(tm.get_current_theme())"
      5. Assert: Output is "light"
    Expected Result: Theme preference saved and loaded
    Evidence: settings.json content captured

  Scenario: Nord colors match official specification
    Tool: Bash (Python)
    Preconditions: ThemeManager imported
    Steps:
      1. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(tm.get_color('bg_primary', mode='dark'))"
      2. Assert: Output is "#2E3440" (Nord0)
      3. Run: python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(tm.get_color('accent', mode='dark'))"
      4. Assert: Output is "#88C0D0" (Nord8)
    Expected Result: Colors match Nord specification
    Evidence: Console output captured
  ```

  **Evidence to Capture**:
  - [ ] Test run outputs in console
  - [ ] settings.json file content
  - [ ] Python exception logs if any

  **Commit**: YES
  - Message: `feat(gui): create theme_manager module with Nord color scheme and theme switching`
  - Files: `gui/theme_manager.py`
  - Pre-commit: `pytest tests/test_theme_manager.py -v`

- [x] 2. 更新进度条组件 `gui/widgets/progress_bar.py`

  **What to do**:
  - 导入 `theme_manager` 的颜色和常量
  - 将硬编码颜色替换为 `theme_manager.get_color()` 调用
  - 使用 Nord 配色：标题、进度条、统计标签、阶段标签、ETA
  - 更新字体使用设计系统字体
  - 更新间距使用设计系统常量
  - 为进度条添加渐变效果（使用 Nord accent 颜色）
  - 添加平滑过渡动画（在 `update_progress` 时）

  **Must NOT do**:
  - 不要改变现有布局结构
  - 不要修改功能逻辑
  - 不要硬编码颜色值

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward styling update following existing pattern
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 5)
  - **Blocks**: Task 6
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `gui/widgets/progress_bar.py:14-86` - Current `_setup_ui` method structure
  - `gui/widgets/progress_bar.py:88-175` - Current `update_progress` method structure

  **API/Type References**:
  - `theme_manager.get_color(name, mode)` - Get color for widget styling
  - `theme_manager.SPACING` - Spacing constants
  - `theme_manager.FONTS` - Font constants

  **WHY Each Reference Matters**:
  - `gui/widgets/progress_bar.py:14-86`: Shows current widget structure - new code must preserve this structure
  - `theme_manager.get_color()`: Required to get Nord colors instead of hard-coded values

  **Acceptance Criteria**:

  - [ ] Imports `from gui.theme_manager import ThemeManager`
  - [ ] Creates `ThemeManager` instance or uses singleton pattern
  - [ ] Title label uses `get_color("fg_primary", mode="auto")` for text_color
  - [ ] Progress bar uses `get_color("accent", mode="auto")` for progress_color
  - [ ] Progress bar background uses `get_color("bg_secondary", mode="auto")`
  - [ ] Completed labels use Nord colors (not hard-coded "red", "orange")
  - [ ] Failed label uses `get_color("error", mode="auto")`
  - [ ] Partial label uses `get_color("warning", mode="auto")`
  - [ ] Phase label uses `get_color("fg_secondary", mode="auto")`
  - [ ] ETA label uses `get_color("fg_secondary", mode="auto")`
  - [ ] Spacing uses `theme_manager.SPACING` constants
  - [ ] Font sizes use `theme_manager.FONTS` constants
  - [ ] pytest tests pass: `pytest tests/test_gui/test_progress_bar.py -v` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Progress bar displays with Nord colors in dark mode
    Tool: Bash (Python)
    Preconditions: GUI launchable
    Steps:
      1. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(2); app.destroy()"
      2. Assert: No exceptions during window creation
      3. Assert: Progress bar widget is visible
    Expected Result: Progress bar renders with Nord colors
    Evidence: Screenshot captured to .sisyphus/evidence/task-2-progress-bar-dark.png

  Scenario: Progress bar displays with Nord colors in light mode
    Tool: Bash (Python)
    Preconditions: settings.json with light theme
    Steps:
      1. Write: {"theme": "light"} to settings.json
      2. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(2); app.destroy()"
      3. Assert: No exceptions
    Expected Result: Progress bar renders with light Nord colors
    Evidence: Screenshot captured to .sisyphus/evidence/task-2-progress-bar-light.png

  Scenario: Progress update works without errors
    Tool: Bash (Python)
    Preconditions: Module imported
    Steps:
      1. Run: python -c "from gui.widgets.progress_bar import ProgressBar; import customtkinter as ctk; app = ctk.CTk(); pb = ProgressBar(app); pb.update_progress(5, 10, 1, 0, 'test', 100, 0.9); print('OK')"
      2. Assert: Output is "OK"
      3. Assert: No exceptions in traceback
    Expected Result: Progress update functional
    Evidence: Console output captured
  ```

  **Evidence to Capture**:
  - [ ] Screenshots in dark and light modes
  - [ ] Console output from progress update test

  **Commit**: YES (groups with Tasks 3, 4, 5)
  - Message: `style(gui): apply Nord colors to progress_bar widget`
  - Files: `gui/widgets/progress_bar.py`
  - Pre-commit: `pytest tests/test_gui/test_progress_bar.py -v`

- [x] 3. 更新日志查看器组件 `gui/widgets/log_viewer.py`

  **What to do**:
  - 导入 `theme_manager`
  - 将硬编码颜色替换为 `theme_manager.get_color()`
  - 标题、工具栏、日志文本框应用 Nord 配色
  - 刷新、清空按钮使用主题颜色
  - 日志级别下拉菜单使用主题颜色
  - 自动滚动复选框使用主题颜色
  - 更新字体和间距使用设计系统常量

  **Must NOT do**:
  - 不要改变日志显示逻辑
  - 不要修改自动刷新功能

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Similar to task 2, straightforward styling update
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4, 5)
  - **Blocks**: Task 6
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `gui/widgets/log_viewer.py:55-90` - Current `_setup_ui` structure
  - `gui/widgets/log_viewer.py:134-169` - Current `refresh_log` method

  **Acceptance Criteria**:

  - [ ] Log viewer imports `ThemeManager`
  - [ ] Title label uses `get_color("fg_primary")`
  - [ ] Toolbar background uses `get_color("bg_secondary")`
  - [ ] Log textbox uses `get_color("bg_primary")` with `get_color("fg_primary")` text
  - [ ] Buttons use theme colors for hover/normal states
  - [ ] Checkbox uses theme colors
  - [ ] Spacing uses design system constants
  - [ ] pytest tests pass: `pytest tests/test_gui/test_log_viewer.py -v` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Log viewer displays with Nord colors
    Tool: Bash (Python)
    Preconditions: GUI launchable
    Steps:
      1. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(2); app.destroy()"
      2. Assert: Log tab shows log viewer widget
    Expected Result: Log viewer renders with Nord theme
    Evidence: Screenshot .sisyphus/evidence/task-3-log-viewer.png

  Scenario: Log refresh works with theme applied
    Tool: Bash (Python)
    Preconditions: Log file exists
    Steps:
      1. Run: python -c "from gui.widgets.log_viewer import LogViewer; import customtkinter as ctk; from pathlib import Path; app = ctk.CTk(); lv = LogViewer(app, log_file=Path('logs/novel_outline.log'), auto_refresh=False); lv.refresh_log(); print('OK')"
      2. Assert: Output is "OK"
    Expected Result: Log refresh functional with styling
    Evidence: Console output captured
  ```

  **Evidence to Capture**:
  - [ ] Screenshot of log viewer
  - [ ] Console output from refresh test

  **Commit**: YES (groups with Tasks 2, 4, 5)
  - Message: `style(gui): apply Nord colors to log_viewer widget`

- [x] 4. 更新文件选择器组件 `gui/widgets/file_selector.py`

  **What to do**:
  - 导入 `theme_manager`
  - 将硬编码颜色替换为主题颜色
  - 添加文件图标（文件夹/文档图标）
  - 标题、路径标签、信息标签应用 Nord 配色
  - 选择按钮使用主题颜色
  - 更新字体和间距

  **Must NOT do**:
  - 不要修改文件选择逻辑
  - 不要改变文件信息计算

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Similar to tasks 2 and 3
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 5)
  - **Blocks**: Task 6
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `gui/widgets/file_selector.py:50-87` - Current `_setup_ui` structure
  - `gui/widgets/file_selector.py:119-156` - Current `set_file` and `_update_file_info`

  **Acceptance Criteria**:

  - [ ] File selector imports `ThemeManager`
  - [ ] All labels use theme colors
  - [ ] Button uses theme colors
  - [ ] File icon displayed (16x16 or 24x24)
  - [ ] Spacing uses design system constants
  - [ ] pytest tests pass: `pytest tests/test_gui/test_file_selector.py -v` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: File selector displays with icons
    Tool: Bash (Python)
    Preconditions: GUI launchable, icons generated
    Steps:
      1. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(2); app.destroy()"
      2. Assert: File selector visible in process tab
      3. Assert: File icon visible
    Expected Result: File selector with icon displays correctly
    Evidence: Screenshot .sisyphus/evidence/task-4-file-selector.png

  Scenario: File selection works with styling
    Tool: Bash (Python)
    Preconditions: Test file exists
    Steps:
      1. Run: python -c "from gui.widgets.file_selector import FileSelector; import customtkinter as ctk; from pathlib import Path; app = ctk.CTk(); fs = FileSelector(app); fs.set_file(Path('README.md')); print('OK')"
      2. Assert: Output is "OK"
    Expected Result: File selection works with styled widgets
    Evidence: Console output captured
  ```

  **Evidence to Capture**:
  - [ ] Screenshot of file selector with icon
  - [ ] Console output from file selection test

  **Commit**: YES (groups with Tasks 2, 3, 5)
  - Message: `style(gui): apply Nord colors and icons to file_selector widget`

- [x] 5. 更新配置对话框 `gui/config_dialog.py`

  **What to do**:
  - 导入 `theme_manager`
  - 将硬编码颜色替换为主题颜色
  - 标题、分组框架、标签、输入框使用 Nord 配色
  - 保存、取消、导出按钮使用主题颜色
  - API 密钥字段使用主题颜色
  - 更新字体和间距

  **Must NOT do**:
  - 不要修改配置保存/加载逻辑
  - 不要改变字段结构

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Similar to other widget updates
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `gui/config_dialog.py:53-76` - Current `_setup_ui` structure
  - `gui/config_dialog.py:83-149` - Config section setup methods

  **Acceptance Criteria**:

  - [ ] Config dialog imports `ThemeManager`
  - [ ] All frames use `get_color("bg_secondary")` for bg_color
  - [ ] Labels use `get_color("fg_primary")`
  - [ ] Entries use theme colors
  - [ ] Buttons use theme colors with hover effects
  - [ ] Spacing uses design system constants
  - [ ] pytest tests pass: `pytest tests/test_gui/test_config_dialog.py -v` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Config dialog opens with Nord theme
    Tool: Bash (Python)
    Preconditions: GUI launchable
    Steps:
      1. Run: python -c "from gui.config_dialog import ConfigDialog; import customtkinter as ctk; app = ctk.CTk(); ConfigDialog(app); import time; time.sleep(2); app.destroy()"
      2. Assert: Dialog opens without errors
    Expected Result: Config dialog displays with Nord colors
    Evidence: Screenshot .sisyphus/evidence/task-5-config-dialog.png

  Scenario: Config save works with styled UI
    Tool: Bash (Python)
    Preconditions: None
    Steps:
      1. Run: python -c "from gui.config_dialog import ConfigDialog; import customtkinter as ctk; app = ctk.CTk(); dlg = ConfigDialog(app); import time; time.sleep(1); dlg.destroy(); app.destroy(); print('OK')"
      2. Assert: Output is "OK"
    Expected Result: Config dialog functional with styling
    Evidence: Console output captured
  ```

  **Evidence to Capture**:
  - [ ] Screenshot of config dialog
  - [ ] Console output from dialog test

  **Commit**: YES (groups with Tasks 2, 3, 4)
  - Message: `style(gui): apply Nord colors to config_dialog`

- [x] 6. 更新主窗口 `gui/main_window.py`

  **What to do**:
  - 导入 `ThemeManager`
  - 创建 `ThemeManager` 实例
  - 替换硬编码的主题设置为动态主题加载
  - 添加主题切换控件（按钮或下拉菜单）
  - 实现 `on_theme_switch` 方法处理主题切换
  - 标签页组件应用主题颜色
  - 窗口标题栏、背景使用主题颜色
  - 更新字体和间距
  - 确保所有子组件在主题切换时更新

  **Must NOT do**:
  - 不要改变标签页结构
  - 不要修改功能按钮逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires coordination of all components and theme switching logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (with Task 7)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 2, 3, 4, 5

  **References** (CRITICAL):

  **Pattern References**:
  - `gui/main_window.py:29-72` - Current `__init__` and tab setup structure
  - `gui/main_window.py:48-49` - Hard-coded theme settings to replace

  **Acceptance Criteria**:

  - [ ] Main window imports `ThemeManager`
  - [ ] `ThemeManager` instance created in `__init__`
  - [ ] Hard-coded `ctk.set_appearance_mode("dark")` removed
  - [ ] Hard-coded `ctk.set_default_color_theme("blue")` removed
  - [ ] Theme loaded from `ThemeManager.get_current_theme()`
  - [ ] Theme switch UI added (button or dropdown in header or menu)
  - [ ] `on_theme_switch` method implemented
  - [ ] All tab frames use theme colors
  - [ ] All labels use theme colors
  - [ ] pytest tests pass: `pytest tests/test_gui/test_main_window.py -v` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Main window opens with loaded theme
    Tool: Bash (Python)
    Preconditions: settings.json exists
    Steps:
      1. Write: {"theme": "light"} to settings.json
      2. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(3); app.destroy()"
      3. Assert: Window opens in light theme
    Expected Result: Main window loads with persisted theme
    Evidence: Screenshot .sisyphus/evidence/task-6-main-window-light.png

  Scenario: Theme switch button changes theme
    Tool: Bash (Python)
    Preconditions: GUI has theme switch button
    Steps:
      1. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(3); print('Theme:', app.theme_manager.get_current_theme()); app.destroy()"
      2. Assert: Current theme printed
      3. Note: Automated testing of theme switch may require simulating button click
    Expected Result: Theme switch UI present and functional
    Evidence: Screenshot .sisyphus/evidence/task-6-theme-switch-ui.png

  Scenario: All components render correctly
    Tool: Bash (Python)
    Preconditions: GUI launchable
    Steps:
      1. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(3); app.destroy()"
      2. Assert: Process tab visible
      3. Assert: Config tab visible
      4. Assert: Log tab visible
      5. Assert: About tab visible
    Expected Result: All tabs and widgets render correctly
    Evidence: Screenshot .sisyphus/evidence/task-6-all-tabs.png
  ```

  **Evidence to Capture**:
  - [ ] Screenshots of main window in light and dark themes
  - [ ] Screenshot of theme switch UI
  - [ ] Screenshot of all tabs

  **Commit**: YES
  - Message: `feat(gui): add theme switching UI to main window`
  - Files: `gui/main_window.py`
  - Pre-commit: `pytest tests/test_gui/test_main_window.py -v`

- [x] 7. 更新图标生成脚本 `gui/assets/create_icons.py`

  **What to do**:
  - 更新脚本生成 Nord 配色的图标
  - 为需要图标的组件创建图标：
    - 文件图标（folder、document）
    - 按钮图标（play、pause、stop、settings、refresh、clear）
    - 标签图标（process、config、log、about）
  - 使用 Nord 配色方案中的颜色
  - 生成 16x16、24x24、32x32 多种尺寸
  - 保存到 `gui/assets/icons/` 目录
  - 添加命令行参数支持单独生成

  **Must NOT do**:
  - 不要使用外部图标资源
  - 不要引入新的依赖（只使用Pillow）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Icon generation using existing Pillow patterns
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Task 8
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `gui/assets/create_icons.py` - Current icon generation script (if exists)
  - Nord colors from `theme_manager` design spec

  **Acceptance Criteria**:

  - [ ] Script generates icons in Nord colors
  - [ ] Icons saved to `gui/assets/icons/`
  - [ ] At least 8 icon types generated (file, folder, play, pause, stop, settings, refresh, clear)
  - [ ] Icons generated in multiple sizes (16, 24, 32)
  - [ ] Icons use Nord accent color (#88C0D0 for dark, #5E81AC for light)
  - [ ] Running script: `python gui/assets/create_icons.py` creates icons
  - [ ] Icons are PNG format with transparency

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Icon generation script creates icons
    Tool: Bash
    Preconditions: gui/assets/create_icons.py exists
    Steps:
      1. Run: python gui/assets/create_icons.py
      2. Run: ls -la gui/assets/icons/
      3. Assert: Icons directory exists
      4. Assert: At least 8 .png files present
    Expected Result: Icons generated successfully
    Evidence: Directory listing captured

  Scenario: Icons use Nord colors
    Tool: Bash (ImageMagick or PIL)
    Preconditions: Icons generated
    Steps:
      1. Run: python -c "from PIL import Image; img = Image.open('gui/assets/icons/file_24.png'); print('Mode:', img.mode); print('Size:', img.size)"
      2. Assert: Image mode is RGBA or RGB
      3. Assert: Image size is 24x24
    Expected Result: Icons have correct format and size
    Evidence: Console output captured
  ```

  **Evidence to Capture**:
  - [ ] Directory listing of generated icons
  - [ ] Console output from icon info check

  **Commit**: YES
  - Message: `feat(gui): generate Nord-themed icons for GUI components`
  - Files: `gui/assets/create_icons.py`, `gui/assets/icons/*`
  - Pre-commit: `python gui/assets/create_icons.py`

- [x] 8. 集成测试和最终验证

  **What to do**:
  - 运行所有 GUI 测试确保功能正常
  - 手动测试主题切换功能（通过GUI启动）
  - 验证所有组件在三种主题下显示正确
  - 检查图标加载是否正常
  - 验证动画效果流畅
  - 确保 settings.json 正确保存主题偏好
  - 运行完整测试套件

  **Must NOT do**:
  - 不要修改功能代码
  - 不要添加新功能

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Final verification and test execution
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: None
  - **Blocked By**: Tasks 6, 7

  **Acceptance Criteria**:

  - [ ] All pytest tests pass: `pytest tests/ -v` → PASS
  - [ ] GUI tests pass: `pytest tests/test_gui/ -v` → PASS
  - [ ] Theme manager tests pass: `pytest tests/ -k theme_manager -v` → PASS
  - [ ] No LSP errors in modified files
  - [ ] No runtime errors when launching GUI
  - [ ] Theme switching works without errors
  - [ ] All icons load correctly
  - [ ] settings.json created or updated correctly

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Full test suite passes
    Tool: Bash
    Preconditions: All tasks completed
    Steps:
      1. Run: pytest tests/ -v --tb=short
      2. Assert: Exit code is 0
      3. Assert: Output contains "passed"
    Expected Result: All tests pass
    Evidence: Test output captured

  Scenario: GUI launches without errors
    Tool: Bash (Python)
    Preconditions: All files updated
    Steps:
      1. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(3); app.destroy(); print('SUCCESS')"
      2. Assert: Output is "SUCCESS"
      3. Assert: No exceptions in traceback
    Expected Result: Theme system works without errors
    Evidence: Console output captured

  Scenario: Theme persistence works
    Tool: Bash
    Preconditions: GUI launched and theme switched
    Steps:
      1. Write: {"theme": "light"} to settings.json
      2. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(2); current = app.theme_manager.get_current_theme(); app.destroy(); print(current)"
      3. Assert: Output is "light"
      4. Write: {"theme": "dark"} to settings.json
      5. Run: python -c "from gui.main_window import MainWindow; app = MainWindow(); import time; time.sleep(2); current = app.theme_manager.get_current_theme(); app.destroy(); print(current)"
      6. Assert: Output is "dark"
    Expected Result: Theme preference persists correctly
    Evidence: Console output captured, settings.json content
  ```

  **Evidence to Capture**:
  - [ ] Full test suite output
  - [ ] Console output from GUI launch tests
  - [ ] settings.json content

  **Commit**: NO (final verification, no new code)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(gui): create theme_manager module with Nord color scheme and theme switching` | gui/theme_manager.py | pytest tests/test_theme_manager.py |
| 2-5 | `style(gui): apply Nord colors to widgets` | gui/widgets/*.py, gui/config_dialog.py | pytest tests/test_gui/ |
| 6 | `feat(gui): add theme switching UI to main window` | gui/main_window.py | pytest tests/test_gui/test_main_window.py |
| 7 | `feat(gui): generate Nord-themed icons for GUI components` | gui/assets/create_icons.py, gui/assets/icons/* | python gui/assets/create_icons.py |

---

## Success Criteria

### Verification Commands
```bash
# Run all tests
pytest tests/ -v

# Run GUI tests specifically
pytest tests/test_gui/ -v

# Verify theme manager
python -c "from gui.theme_manager import ThemeManager; tm = ThemeManager(); print(tm.get_current_theme())"

# Generate icons
python gui/assets/create_icons.py

# Launch GUI (manual verification)
python gui_launcher.py
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] Theme manager module created and functional
- [x] Nord colors applied to all GUI components
- [x] Theme switching UI present in main window
- [x] Icons generated and loaded correctly
- [x] All tests pass (pytest)
- [x] GUI launches without errors in all three themes
- [x] Theme persistence works (settings.json)
