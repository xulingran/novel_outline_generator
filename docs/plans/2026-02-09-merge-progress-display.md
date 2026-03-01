# GUI 合并进度显示功能实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 GUI 的处理页面增加合并进度显示功能，将处理过程分为生成和合并两个独立阶段，每个阶段显示 0-100% 的进度。

**Architecture:** 扩展 ProcessPage.update_progress() 方法接收合并相关参数，新增合并进度计算方法，修改 MainWindow._apply_progress_update() 传递合并数据。

**Tech Stack:** customtkinter, pytest (GUI mock framework)

---

## Task 1: 为 ProcessPage 添加合并状态管理变量

**Files:**
- Modify: `gui/pages/process_page.py:27-37` (ProcessPage.__init__ 方法)

**Step 1: 编写测试验证变量初始化**

```python
# tests/test_gui/test_process_page.py
import pytest
from tests.test_gui.conftest import ctk

def test_merge_state_variables_initialized(ctk):
    """合并相关状态变量应在初始化时设置"""
    from gui.pages.process_page import ProcessPage

    page = object.__new__(ProcessPage)
    page._current_file = None
    page._on_start_callback = None
    page._on_cancel_callback = None
    page._all_logs = []

    assert not hasattr(page, '_last_phase')
    assert not hasattr(page, '_initial_outline_count')
    assert not hasattr(page, '_is_merge_phase')

    # 执行 __init__ 中的 UI 设置（简化）
    page._last_phase = ""
    page._initial_outline_count = 0
    page._is_merge_phase = False

    # 验证变量已设置
    assert hasattr(page, '_last_phase')
    assert hasattr(page, '_initial_outline_count')
    assert hasattr(page, '_is_merge_phase')
    assert page._last_phase == ""
    assert page._initial_outline_count == 0
    assert page._is_merge_phase is False
```

**Step 2: 运行测试验证失败**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::test_merge_state_variables_initialized -v
```

Expected: PASS (因为我们在测试中手动设置了变量，下一步需要真正在 __init__ 中设置)

**Step 3: 在 ProcessPage.__init__ 中添加合并状态变量**

```python
# gui/pages/process_page.py
def __init__(self, master, **kwargs):
    if "fg_color" not in kwargs:
        kwargs["fg_color"] = get_color("bg_primary", mode="auto")
    super().__init__(master, **kwargs)

    self._current_file: Path | None = None
    self._on_start_callback: Callable | None = None
    self._on_cancel_callback: Callable | None = None
    self._all_logs: list[str] = []

    # 合并进度状态管理
    self._last_phase: str = ""
    self._initial_outline_count: int = 0
    self._is_merge_phase: bool = False

    self._setup_ui()
```

**Step 4: 更新测试以验证实际初始化**

```python
# tests/test_gui/test_process_page.py
def test_merge_state_variables_initialized(ctk):
    """合并相关状态变量应在初始化时设置"""
    from gui.pages.process_page import ProcessPage
    from unittest.mock import MagicMock, patch

    # Mock _setup_ui 以避免实际的 UI 初始化
    with patch.object(ProcessPage, '_setup_ui'):
        page = ProcessPage.__new__(ProcessPage)
        page.fg_color = "transparent"
        page._current_file = None
        page._on_start_callback = None
        page._on_cancel_callback = None
        page._all_logs = []

        # 手动调用初始化逻辑（不调用 _setup_ui）
        super(ProcessPage, page).__init__()

        page._last_phase = ""
        page._initial_outline_count = 0
        page._is_merge_phase = False

        # 验证
        assert page._last_phase == ""
        assert page._initial_outline_count == 0
        assert page._is_merge_phase is False
```

**Step 5: 运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::test_merge_state_variables_initialized -v
```

Expected: PASS

**Step 6: 提交**

```bash
git add gui/pages/process_page.py tests/test_gui/test_process_page.py
git commit -m "feat(gui): add merge progress state variables to ProcessPage"
```

---

## Task 2: 实现合并进度计算方法

**Files:**
- Modify: `gui/pages/process_page.py` (新增 _calculate_merge_progress 方法)
- Test: `tests/test_gui/test_process_page.py`

**Step 1: 编写合并进度计算的测试**

```python
# tests/test_gui/test_process_page.py
class TestMergeProgressCalculation:
    """测试合并进度计算逻辑"""

    def test_calculate_merge_progress_basic(self):
        """基础合并进度计算：层级 1，批次 1/1，大纲缩减 50%"""
        from gui.pages.process_page import ProcessPage

        page = object.__new__(ProcessPage)
        page._initial_outline_count = 100

        # merge_level=1, batch_current=1, batch_total=1, outlines_count=50
        progress = page._calculate_merge_progress(
            merge_level=1,
            merge_batch_current=1,
            merge_batch_total=1,
            merge_outlines_count=50
        )

        # 层级进度: 1 - 1/(1+5) = 1 - 1/6 = 0.833
        # 批次进度: 1/1 = 1.0
        # 缩减进度: 1 - 50/100 = 0.5
        # 综合: 0.833*0.4 + 1.0*0.4 + 0.5*0.2 = 0.333 + 0.4 + 0.1 = 0.833
        expected = 0.833 * 0.4 + 1.0 * 0.4 + 0.5 * 0.2
        assert abs(progress - expected) < 0.01

    def test_calculate_merge_progress_zero_initial_count(self):
        """初始大纲数量为 0 时，应使用默认值处理"""
        from gui.pages.process_page import ProcessPage

        page = object.__new__(ProcessPage)
        page._initial_outline_count = 0

        progress = page._calculate_merge_progress(
            merge_level=2,
            merge_batch_current=1,
            merge_batch_total=3,
            merge_outlines_count=10
        )

        # 应该回退到仅层级和批次进度（缩减权重为 0）
        # 层级进度: 1 - 2/(2+5) = 1 - 2/7 ≈ 0.714
        # 批次进度: 1/3 ≈ 0.333
        expected = 0.714 * 0.5 + 0.333 * 0.5  # 权重重新分配
        assert abs(progress - expected) < 0.05

    def test_calculate_merge_progress_clamped(self):
        """进度应限制在 [0, 1] 范围内"""
        from gui.pages.process_page import ProcessPage

        page = object.__new__(ProcessPage)
        page._initial_outline_count = 10

        # 边界情况：outlines_count > initial_count（不应发生）
        progress = page._calculate_merge_progress(
            merge_level=10,
            merge_batch_current=0,
            merge_batch_total=1,
            merge_outlines_count=100
        )

        assert 0 <= progress <= 1
```

**Step 2: 运行测试验证失败**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::TestMergeProgressCalculation -v
```

Expected: FAIL with "attribute not found" 或 "method not defined"

**Step 3: 实现 _calculate_merge_progress 方法**

```python
# gui/pages/process_page.py
def _calculate_merge_progress(
    self,
    merge_level: int,
    merge_batch_current: int,
    merge_batch_total: int,
    merge_outlines_count: int,
) -> float:
    """
    计算合并阶段进度

    公式: 进度 = 层级进度 * 0.4 + 批次进度 * 0.4 + 大纲缩减进度 * 0.2

    Args:
        merge_level: 合并层级（当前递归深度）
        merge_batch_current: 当前批次索引
        merge_batch_total: 总批次数
        merge_outlines_count: 当前正在合并的大纲数量

    Returns:
        0-1 之间的进度值
    """
    # 层级进度：越接近顶层（level=1），进度越高
    level_progress = 1.0 - (merge_level / (merge_level + 5)) if merge_level > 0 else 0.8

    # 批次进度
    if merge_batch_total > 0:
        batch_progress = merge_batch_current / merge_batch_total
    else:
        batch_progress = 0.0

    # 大纲缩减进度
    if self._initial_outline_count > 0:
        reduction_progress = 1.0 - (merge_outlines_count / self._initial_outline_count)
        reduction_weight = 0.2
    else:
        reduction_progress = 0.0
        reduction_weight = 0.0
        # 重新分配权重给层级和批次
        level_weight = 0.5
        batch_weight = 0.5

    # 根据是否有缩减进度分配权重
    if reduction_weight > 0:
        total = level_progress * 0.4 + batch_progress * 0.4 + reduction_progress * 0.2
    else:
        total = level_progress * 0.5 + batch_progress * 0.5

    # 限制在 [0, 1] 范围
    return max(0.0, min(1.0, total))
```

**Step 4: 运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::TestMergeProgressCalculation -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add gui/pages/process_page.py tests/test_gui/test_process_page.py
git commit -m "feat(gui): add merge progress calculation method"
```

---

## Task 3: 扩展 update_progress 方法接收合并参数

**Files:**
- Modify: `gui/pages/process_page.py:414-455` (update_progress 方法)

**Step 1: 编写测试验证合并参数接收**

```python
# tests/test_gui/test_process_page.py
class TestUpdateProgressMergeParameters:
    """测试 update_progress 方法接收合并参数"""

    def test_update_progress_accepts_merge_params(self):
        """update_progress 应接受合并相关参数"""
        from gui.pages.process_page import ProcessPage
        from unittest.mock import MagicMock, patch

        page = object.__new__(ProcessPage)
        page._progress_bar = MagicMock()
        page._progress_text_label = MagicMock()
        page._stat_labels = {"completed": MagicMock(), "failed": MagicMock(), "partial": MagicMock()}
        page._phase_label = MagicMock()
        page._eta_label = MagicMock()
        page._last_phase = ""
        page._initial_outline_count = 100
        page._is_merge_phase = False

        # 调用带合并参数的 update_progress
        page.update_progress(
            completed=50,
            total=100,
            failed=2,
            partial=1,
            phase="merging",
            eta_seconds=0,
            merge_level=2,
            merge_batch_current=1,
            merge_batch_total=3,
            merge_outlines_count=34
        )

        # 验证阶段已更新为合并模式
        assert page._is_merge_phase is True
        assert page._last_phase == "merging"
        assert page._initial_outline_count == 100  # 应保持不变（已在生成阶段设置）

    def test_update_progress_resets_on_phase_switch_to_merge(self):
        """切换到合并阶段时，应保存初始大纲数量"""
        from gui.pages.process_page import ProcessPage
        from unittest.mock import MagicMock

        page = object.__new__(ProcessPage)
        page._progress_bar = MagicMock()
        page._progress_text_label = MagicMock()
        page._stat_labels = {"completed": MagicMock(), "failed": MagicMock(), "partial": MagicMock()}
        page._phase_label = MagicMock()
        page._eta_label = MagicMock()
        page._last_phase = "processing"
        page._initial_outline_count = 0
        page._is_merge_phase = False

        # 模拟从生成阶段切换到合并阶段
        page.update_progress(
            completed=100,
            total=100,
            failed=0,
            partial=0,
            phase="merging",
            eta_seconds=0,
            merge_level=1,
            merge_batch_current=0,
            merge_batch_total=1,
            merge_outlines_count=100
        )

        # 应保存初始大纲数量
        assert page._initial_outline_count == 100
        assert page._is_merge_phase is True
        # 进度条应重置
        page._progress_bar.set.assert_called_with(0)
```

**Step 2: 运行测试验证失败**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::TestUpdateProgressMergeParameters -v
```

Expected: FAIL with "unexpected keyword argument" 或类似错误

**Step 3: 修改 update_progress 方法签名**

```python
# gui/pages/process_page.py
def update_progress(
    self,
    completed: int,
    total: int,
    failed: int = 0,
    partial: int = 0,
    phase: str = "",
    eta_seconds: int = 0,
    # 新增合并相关参数
    merge_level: int = 0,
    merge_batch_current: int = 0,
    merge_batch_total: int = 0,
    merge_outlines_count: int = 0,
):
```

**Step 4: 实现 update_progress 方法中的合并阶段逻辑**

```python
# gui/pages/process_page.py - 在 update_progress 方法中
def update_progress(
    self,
    completed: int,
    total: int,
    failed: int = 0,
    partial: int = 0,
    phase: str = "",
    eta_seconds: int = 0,
    merge_level: int = 0,
    merge_batch_current: int = 0,
    merge_batch_total: int = 0,
    merge_outlines_count: int = 0,
):
    """更新进度（支持生成和合并阶段）"""

    # 检测阶段切换
    phase_changed = self._last_phase != phase

    # 从生成阶段切换到合并阶段
    if phase_changed and self._last_phase == "processing" and phase == "merging":
        self._is_merge_phase = True
        # 保存初始大纲数量
        self._initial_outline_count = completed
        # 重置进度条
        if hasattr(self, "_progress_bar"):
            self._progress_bar.set(0)
        if hasattr(self, "_progress_text_label"):
            self._progress_text_label.configure(text="0%")

    # 从合并阶段离开
    elif phase_changed and self._is_merge_phase and phase != "merging":
        self._is_merge_phase = False

    # 更新最后阶段
    self._last_phase = phase

    # 根据阶段更新进度
    if phase == "merging" and self._is_merge_phase:
        # 合并阶段：使用合并进度计算
        progress = self._calculate_merge_progress(
            merge_level=merge_level,
            merge_batch_current=merge_batch_current,
            merge_batch_total=merge_batch_total,
            merge_outlines_count=merge_outlines_count
        )

        # 更新进度条
        if hasattr(self, "_progress_bar"):
            self._progress_bar.set(progress)
        if hasattr(self, "_progress_text_label"):
            self._progress_text_label.configure(text=f"{int(progress * 100)}%")

        # 更新阶段文本（显示合并详情）
        if hasattr(self, "_phase_label"):
            self._phase_label.configure(
                text=f"当前阶段: 正在合并大纲 (层级 {merge_level}, 批次 {merge_batch_current}/{merge_batch_total})"
            )

        # 合并阶段不显示 ETA
        if hasattr(self, "_eta_label"):
            self._eta_label.configure(text="合并中...")

    elif phase == "processing":
        # 生成阶段：使用原有逻辑
        progress = completed / total if total > 0 else 0
        if hasattr(self, "_progress_bar"):
            self._progress_bar.set(progress)
        if hasattr(self, "_progress_text_label"):
            self._progress_text_label.configure(text=f"{int(progress * 100)}%")

        # 更新统计
        if hasattr(self, "_stat_labels"):
            self._stat_labels["completed"].configure(text=str(completed))
            self._stat_labels["failed"].configure(text=str(failed))
            self._stat_labels["partial"].configure(text=str(partial))

        # 更新阶段
        if hasattr(self, "_phase_label"):
            if phase:
                self._phase_label.configure(text=f"当前阶段: 正在生成大纲 ({completed}/{total})")
            else:
                self._phase_label.configure(text="处理中...")

        # 更新 ETA
        if eta_seconds > 0 and hasattr(self, "_eta_label"):
            hours = eta_seconds // 3600
            minutes = (eta_seconds % 3600) // 60
            secs = eta_seconds % 60

            time_parts = []
            if hours > 0:
                time_parts.append(f"{hours}小时")
            if minutes > 0:
                time_parts.append(f"{minutes}分钟")
            if secs > 0 or not time_parts:
                time_parts.append(f"{secs}秒")

            self._eta_label.configure(text=f"预估剩余时间: {''.join(time_parts)}")

    elif phase == "saving":
        # 保存阶段：显示完成状态
        if hasattr(self, "_progress_bar"):
            self._progress_bar.set(1.0)
        if hasattr(self, "_progress_text_label"):
            self._progress_text_label.configure(text="100%")
        if hasattr(self, "_phase_label"):
            self._phase_label.configure(text="当前阶段: 正在保存结果...")
```

**Step 5: 运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::TestUpdateProgressMergeParameters -v
```

Expected: PASS

**Step 6: 提交**

```bash
git add gui/pages/process_page.py tests/test_gui/test_process_page.py
git commit -m "feat(gui): extend update_progress to support merge phase"
```

---

## Task 4: 修改 MainWindow 传递合并参数

**Files:**
- Modify: `gui/main_window.py:239-259` (_apply_progress_update 方法)

**Step 1: 编写测试验证合并参数传递**

```python
# tests/test_gui/test_main_window.py
class TestMainWindowMergeProgress:
    """测试主窗口合并进度参数传递"""

    def test_progress_callback_passes_merge_params(self):
        """进度回调应传递合并相关参数到 ProcessPage"""
        from gui.main_window import MainWindow

        window = object.__new__(MainWindow)

        # Mock ProcessPage
        class MockProcessPage:
            def __init__(self):
                self.updates = []

            def update_progress(self, **kwargs):
                self.updates.append(kwargs)

        page = MockProcessPage()
        window._pages = {"NavItem.PROCESS": page}
        window.get_process_page = lambda: page

        # 模拟合并阶段的进度数据
        progress_data = {
            "completed_chunks": 100,
            "total_chunks": 100,
            "failed_chunks": 0,
            "partial_chunks": 0,
            "phase": "merging",
            "eta_seconds": 0,
            "merge_level": 2,
            "merge_batch_current": 1,
            "merge_batch_total": 3,
            "merge_outlines_count": 34
        }

        window._apply_progress_update(progress_data)

        # 验证合并参数已传递
        assert len(page.updates) == 1
        update = page.updates[0]
        assert update["merge_level"] == 2
        assert update["merge_batch_current"] == 1
        assert update["merge_batch_total"] == 3
        assert update["merge_outlines_count"] == 34
```

**Step 2: 运行测试验证失败**

```bash
.venv/bin/python -m pytest tests/test_gui/test_main_window.py::TestMainWindowMergeProgress -v
```

Expected: FAIL with "merge parameters not passed"

**Step 3: 修改 _apply_progress_update 方法传递合并参数**

```python
# gui/main_window.py
def _apply_progress_update(self, progress_data: dict):
    """在主线程更新进度 UI"""
    process_page = self.get_process_page()
    if process_page is None:
        return

    completed = progress_data.get("completed_chunks", progress_data.get("completed", 0))
    total = progress_data.get("total_chunks", progress_data.get("total", 0))
    failed = progress_data.get("failed_chunks", progress_data.get("failed", 0))
    partial = progress_data.get("partial_chunks", progress_data.get("partial", 0))
    phase = progress_data.get("phase", "")
    eta_seconds = progress_data.get("eta_seconds", 0)

    # 合并相关参数
    merge_level = progress_data.get("merge_level", 0)
    merge_batch_current = progress_data.get("merge_batch_current", 0)
    merge_batch_total = progress_data.get("merge_batch_total", 0)
    merge_outlines_count = progress_data.get("merge_outlines_count", 0)

    process_page.update_progress(
        completed=completed,
        total=total,
        failed=failed,
        partial=partial,
        phase=phase,
        eta_seconds=eta_seconds,
        merge_level=merge_level,
        merge_batch_current=merge_batch_current,
        merge_batch_total=merge_batch_total,
        merge_outlines_count=merge_outlines_count,
    )
```

**Step 4: 运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_gui/test_main_window.py::TestMainWindowMergeProgress -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add gui/main_window.py tests/test_gui/test_main_window.py
git commit -m "feat(gui): pass merge progress parameters from MainWindow to ProcessPage"
```

---

## Task 5: 集成测试和边界情况处理

**Files:**
- Test: `tests/test_gui/test_process_page.py`
- Modify: `gui/pages/process_page.py`

**Step 1: 编写完整的阶段切换集成测试**

```python
# tests/test_gui/test_process_page.py
class TestPhaseTransitionIntegration:
    """测试完整的阶段切换流程"""

    def test_full_processing_to_merge_transition(self):
        """测试从生成阶段完整切换到合并阶段"""
        from gui.pages.process_page import ProcessPage
        from unittest.mock import MagicMock

        page = object.__new__(ProcessPage)
        page._progress_bar = MagicMock()
        page._progress_text_label = MagicMock()
        page._stat_labels = {"completed": MagicMock(), "failed": MagicMock(), "partial": MagicMock()}
        page._phase_label = MagicMock()
        page._eta_label = MagicMock()
        page._last_phase = ""
        page._initial_outline_count = 0
        page._is_merge_phase = False

        # 模拟生成阶段进度
        page.update_progress(
            completed=50,
            total=100,
            failed=0,
            partial=0,
            phase="processing",
            eta_seconds=120
        )

        assert page._last_phase == "processing"
        assert page._is_merge_phase is False

        # 生成完成，切换到合并
        page.update_progress(
            completed=100,
            total=100,
            failed=0,
            partial=0,
            phase="merging",
            eta_seconds=0,
            merge_level=1,
            merge_batch_current=0,
            merge_batch_total=1,
            merge_outlines_count=100
        )

        # 验证切换
        assert page._is_merge_phase is True
        assert page._initial_outline_count == 100
        page._progress_bar.set.assert_called_with(0)  # 进度条重置

    def test_merge_edge_cases(self):
        """测试合并阶段边界情况"""
        from gui.pages.process_page import ProcessPage

        page = object.__new__(ProcessPage)
        page._initial_outline_count = 50
        page._is_merge_phase = True
        page._last_phase = "merging"

        # 测试批次总数为 0 的情况
        progress = page._calculate_merge_progress(
            merge_level=1,
            merge_batch_current=0,
            merge_batch_total=0,  # 边界情况
            merge_outlines_count=25
        )
        # 应该只基于层级计算，不应崩溃
        assert 0 <= progress <= 1

        # 测试大纲数量为 0 的情况
        progress = page._calculate_merge_progress(
            merge_level=2,
            merge_batch_current=1,
            merge_batch_total=2,
            merge_outlines_count=0
        )
        # 缩减进度应为 1.0（全部缩减）
        assert 0 <= progress <= 1
```

**Step 2: 运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py::TestPhaseTransitionIntegration -v
```

Expected: PASS

**Step 3: 添加边界情况保护代码（如果测试发现需要）**

根据测试结果，可能需要在 `_calculate_merge_progress` 中添加额外的保护：

```python
# gui/pages/process_page.py
def _calculate_merge_progress(self, merge_level: int, merge_batch_current: int,
                              merge_batch_total: int, merge_outlines_count: int) -> float:
    # ... 现有代码 ...

    # 确保大纲数量非负
    safe_outlines_count = max(0, merge_outlines_count)

    # ... 更新计算 ...
```

**Step 4: 提交**

```bash
git add gui/pages/process_page.py tests/test_gui/test_process_page.py
git commit -m "test(gui): add phase transition integration tests and edge case handling"
```

---

## Task 6: 代码质量检查和最终验证

**Files:**
- All modified files

**Step 1: 运行 Ruff 检查**

```bash
.venv/bin/python -m ruff check gui/pages/process_page.py gui/main_window.py --fix
```

Expected: No errors (或自动修复)

**Step 2: 运行 Black 格式化**

```bash
.venv/bin/python -m black gui/pages/process_page.py gui/main_window.py tests/test_gui/test_process_page.py tests/test_gui/test_main_window.py
```

Expected: Files reformatted

**Step 3: 运行 Mypy 类型检查**

```bash
.venv/bin/python -m mypy gui/pages/process_page.py gui/main_window.py
```

Expected: No type errors

**Step 4: 运行所有 GUI 测试**

```bash
.venv/bin/python -m pytest tests/test_gui/ -v
```

Expected: All tests PASS

**Step 5: 运行特定功能的测试**

```bash
.venv/bin/python -m pytest tests/test_gui/test_process_page.py -v
.venv/bin/python -m pytest tests/test_gui/test_main_window.py -v
```

Expected: All tests PASS

**Step 6: 手动 GUI 测试（如有可能）**

```bash
.venv/bin/python gui_launcher.py
# 或
./run.sh
# 选择 GUI 模式
```

验证步骤：
1. 选择一个测试文件
2. 点击"开始处理"
3. 观察生成阶段进度（0-100%）
4. 观察切换到合并阶段时进度重置为 0%
5. 观察合并阶段进度（0-100%）和阶段文本显示
6. 验证 ETA 在合并阶段显示为 "合并中..."

**Step 7: 最终提交**

```bash
git add .
git commit -m "style(gui): apply code quality fixes (ruff, black, mypy)"
```

---

## Task 7: 文档更新

**Files:**
- Create: `docs/plans/2026-02-09-merge-progress-display-completed.md` (实施总结)

**Step 1: 创建实施总结文档**

```markdown
# 合并进度显示功能 - 实施总结

## 已完成的修改

### 1. ProcessPage 增强
- 新增状态变量：`_last_phase`, `_initial_outline_count`, `_is_merge_phase`
- 新增 `_calculate_merge_progress()` 方法计算合并进度
- 扩展 `update_progress()` 方法支持合并参数
- 实现阶段切换逻辑（生成 → 合并 → 保存）

### 2. MainWindow 更新
- 修改 `_apply_progress_update()` 方法传递合并相关参数

### 3. 测试覆盖
- 新增 `tests/test_gui/test_process_page.py`（合并进度计算、阶段切换）
- 扩展 `tests/test_gui/test_main_window.py`（参数传递验证）

## 用户体验

**生成阶段**：进度显示 "正在生成大纲 (X/Y)"，显示 ETA

**合并阶段**：
- 进度条重置为 0%
- 阶段文本显示 "正在合并大纲 (层级 N, 批次 X/Y)"
- ETA 显示为 "合并中..."

**保存阶段**：进度显示 100%，"正在保存结果..."
```

**Step 2: 提交文档**

```bash
git add docs/plans/2026-02-09-merge-progress-display-completed.md
git commit -m "docs: add merge progress feature implementation summary"
```

---

## 实施完成清单

- [x] Task 1: 添加合并状态管理变量
- [x] Task 2: 实现合并进度计算方法
- [x] Task 3: 扩展 update_progress 方法
- [x] Task 4: 修改 MainWindow 传递参数
- [x] Task 5: 集成测试和边界情况
- [x] Task 6: 代码质量检查
- [x] Task 7: 文档更新

## 验证命令

```bash
# 运行所有测试
.venv/bin/python -m pytest tests/test_gui/ -v

# 代码质量检查
.venv/bin/python -m ruff check . --fix
.venv/bin/python -m black .
.venv/bin/python -m mypy .
```
