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

## 验证命令

```bash
# 运行所有测试
.venv/bin/python -m pytest tests/test_gui/ -v

# 代码质量检查
.venv/bin/python -m ruff check . --fix
.venv/bin/python -m black .
.venv/bin/python -m mypy .
```
