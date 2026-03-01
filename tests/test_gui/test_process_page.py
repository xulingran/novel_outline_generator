"""
ProcessPage 组件测试
"""

from unittest.mock import MagicMock

from gui.pages.process_page import ProcessPage


def test_merge_state_variables_initialized():
    """合并相关状态变量应在初始化时设置"""
    from gui.pages.process_page import MergeProgressState

    page = object.__new__(ProcessPage)
    page._current_file = None
    page._on_start_callback = None
    page._on_cancel_callback = None
    page._all_logs = []
    page._current_layout_mode = "normal"
    page._merge_state = MergeProgressState()

    assert hasattr(page, "_merge_state")

    assert page._merge_state.last_phase == ""
    assert page._merge_state.initial_outline_count == 0
    assert page._merge_state.is_merge_phase is False


class TestMergeProgressCalculation:
    """测试合并进度计算逻辑"""

    def test_calculate_merge_progress_basic(self):
        """基础合并进度计算：层级 1，批次 1/1，大纲缩减 50%"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(initial_outline_count=100)

        # merge_level=1, batch_current=1, batch_total=1, outlines_count=50
        progress = page._calculate_merge_progress(
            merge_level=1, merge_batch_current=1, merge_batch_total=1, merge_outlines_count=50
        )

        # 层级进度: 1 - 1/(1+5) = 1 - 1/6 = 0.833
        # 批次进度: 1/1 = 1.0
        # 缩减进度: 1 - 50/100 = 0.5
        # 综合: 0.833*0.4 + 1.0*0.4 + 0.5*0.2 = 0.333 + 0.4 + 0.1 = 0.833
        expected = 0.833 * 0.4 + 1.0 * 0.4 + 0.5 * 0.2
        assert abs(progress - expected) < 0.01

    def test_calculate_merge_progress_zero_initial_count(self):
        """初始大纲数量为 0 时，应使用默认值处理"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(initial_outline_count=0)

        progress = page._calculate_merge_progress(
            merge_level=2, merge_batch_current=1, merge_batch_total=3, merge_outlines_count=10
        )

        # 应该回退到仅层级和批次进度（缩减权重为 0）
        # 层级进度: 1 - 2/(2+5) = 1 - 2/7 ≈ 0.714
        # 批次进度: 1/3 ≈ 0.333
        expected = 0.714 * 0.5 + 0.333 * 0.5  # 权重重新分配
        assert abs(progress - expected) < 0.05

    def test_calculate_merge_progress_clamped(self):
        """进度应限制在 [0, 1] 范围内"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(initial_outline_count=10)

        # 边界情况：outlines_count > initial_count（不应发生）
        progress = page._calculate_merge_progress(
            merge_level=10, merge_batch_current=0, merge_batch_total=1, merge_outlines_count=100
        )

        assert 0 <= progress <= 1

    def test_calculate_merge_progress_zero_level(self):
        """层级为 0 时应返回 0.8"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(initial_outline_count=100)

        progress = page._calculate_merge_progress(
            merge_level=0, merge_batch_current=1, merge_batch_total=1, merge_outlines_count=50
        )

        # merge_level=0 时 level_progress = 0.8
        # 批次进度: 1/1 = 1.0
        # 缩减进度: 1 - 50/100 = 0.5
        # 综合: 0.8*0.4 + 1.0*0.4 + 0.5*0.2 = 0.32 + 0.4 + 0.1 = 0.82
        expected = 0.8 * 0.4 + 1.0 * 0.4 + 0.5 * 0.2
        assert abs(progress - expected) < 0.01

    def test_calculate_merge_progress_zero_total_batches(self):
        """总批次数为 0 时，批次进度应为 0"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(initial_outline_count=100)

        progress = page._calculate_merge_progress(
            merge_level=1, merge_batch_current=0, merge_batch_total=0, merge_outlines_count=50
        )

        # 应该只使用层级进度（批次进度为 0）
        # 层级进度: 1 - 1/(1+5) ≈ 0.833
        # 缩减进度: 1 - 50/100 = 0.5
        expected = 0.833 * 0.4 + 0.0 * 0.4 + 0.5 * 0.2
        assert abs(progress - expected) < 0.01


class TestUpdateProgressMergeParameters:
    """测试 update_progress 方法接收合并参数"""

    def test_update_progress_accepts_merge_params(self):
        """update_progress 应接受合并相关参数"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState()
        page._progress_bar = MagicMock()
        page._progress_text_label = MagicMock()
        page._stat_labels = {
            "completed": MagicMock(),
            "failed": MagicMock(),
            "partial": MagicMock(),
        }
        page._phase_label = MagicMock()
        page._eta_label = MagicMock()

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
            merge_outlines_count=34,
        )

        # 验证阶段已更新为合并模式
        assert page._merge_state.is_merge_phase is True
        assert page._merge_state.last_phase == "merging"
        assert page._merge_state.initial_outline_count == 50  # 首次进入合并时使用 completed

    def test_update_progress_resets_on_phase_switch_to_merge(self):
        """切换到合并阶段时，应保存初始大纲数量"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(last_phase="processing", initial_outline_count=0)
        page._progress_bar = MagicMock()
        page._progress_text_label = MagicMock()
        page._stat_labels = {
            "completed": MagicMock(),
            "failed": MagicMock(),
            "partial": MagicMock(),
        }
        page._phase_label = MagicMock()
        page._eta_label = MagicMock()

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
            merge_outlines_count=100,
        )

        # 应保存初始大纲数量
        assert page._merge_state.initial_outline_count == 100
        assert page._merge_state.is_merge_phase is True
        # 进度条应被重置为 0（至少调用过一次）
        page._progress_bar.set.assert_any_call(0)
        # 最后的调用应该是合并进度计算的结果
        # merge_level=1, batch_current=0, batch_total=1, outlines_count=100, initial=100
        # level_progress = 1 - 1/(1+5) = 0.833, batch_progress = 0/1 = 0, reduction_progress = 0
        # expected = 0.833 * 0.4 + 0 * 0.4 + 0 * 0.2 = 0.333
        assert abs(page._progress_bar.set.call_args_list[-1][0][0] - 0.333) < 0.01


class TestPhaseTransitionIntegration:
    """测试完整的阶段切换流程"""

    def test_full_processing_to_merge_transition(self):
        """测试从生成阶段完整切换到合并阶段"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState()
        page._progress_bar = MagicMock()
        page._progress_text_label = MagicMock()
        page._stat_labels = {
            "completed": MagicMock(),
            "failed": MagicMock(),
            "partial": MagicMock(),
        }
        page._phase_label = MagicMock()
        page._eta_label = MagicMock()

        # 模拟生成阶段进度
        page.update_progress(
            completed=50, total=100, failed=0, partial=0, phase="processing", eta_seconds=120
        )

        assert page._merge_state.last_phase == "processing"
        assert page._merge_state.is_merge_phase is False

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
            merge_outlines_count=100,
        )

        # 验证切换
        assert page._merge_state.is_merge_phase is True
        assert page._merge_state.initial_outline_count == 100
        # 进度条应被重置为 0（至少调用过一次）
        page._progress_bar.set.assert_any_call(0)  # 进度条重置

    def test_merge_edge_cases(self):
        """测试合并阶段边界情况"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        from gui.pages.process_page import MergeProgressState

        page._merge_state = MergeProgressState(
            initial_outline_count=50, is_merge_phase=True, last_phase="merging"
        )

        # 测试批次总数为 0 的情况
        progress = page._calculate_merge_progress(
            merge_level=1, merge_batch_current=0, merge_batch_total=0, merge_outlines_count=25
        )
        # 应该只基于层级计算，不应崩溃
        assert 0 <= progress <= 1

        # 测试大纲数量为 0 的情况
        progress = page._calculate_merge_progress(
            merge_level=2, merge_batch_current=1, merge_batch_total=2, merge_outlines_count=0
        )
        # 缩减进度应为 1.0（全部缩减）
        assert 0 <= progress <= 1

        # 测试大纲数量为负数的情况（边界保护）
        progress = page._calculate_merge_progress(
            merge_level=2, merge_batch_current=1, merge_batch_total=2, merge_outlines_count=-10
        )
        # 应该被限制为 0，不应崩溃
        assert 0 <= progress <= 1


class TestLogFiltering:
    """测试日志级别过滤逻辑"""

    def test_filter_logs_does_not_match_body_keywords(self):
        """正文中的 error 关键词不应被误判为 ERROR 级别"""
        page = object.__new__(ProcessPage)
        page._log_text = MagicMock()
        page._log_level_var = MagicMock()
        page._log_level_var.get.return_value = "ERROR"
        page._all_logs = [
            "2026-02-13 10:00:00 - app - INFO - request finished with error details",
            "2026-02-13 10:00:01 - app - ERROR - processing failed",
        ]

        page._filter_logs()

        inserted = [call.args[1] for call in page._log_text.insert.call_args_list]
        assert "2026-02-13 10:00:01 - app - ERROR - processing failed\n" in inserted
        assert (
            "2026-02-13 10:00:00 - app - INFO - request finished with error details\n"
            not in inserted
        )

    def test_filter_logs_excludes_unrecognized_format_in_specific_level(self):
        """筛选特定级别时应排除无法识别级别的日志"""
        page = object.__new__(ProcessPage)
        page._log_text = MagicMock()
        page._log_level_var = MagicMock()
        page._log_level_var.get.return_value = "ERROR"
        page._all_logs = [
            "plain log without level prefix",
            "2026-02-13 10:00:01 - app - ERROR - processing failed",
        ]

        page._filter_logs()

        inserted = [call.args[1] for call in page._log_text.insert.call_args_list]
        assert "2026-02-13 10:00:01 - app - ERROR - processing failed\n" in inserted
        assert "plain log without level prefix\n" not in inserted


class TestResponsiveLayout:
    """测试响应式布局关键网格位置。"""

    def test_compact_mode_places_action_row_below_log(self):
        """compact 模式下操作栏应位于日志区下方，避免重叠。"""
        page = object.__new__(ProcessPage)
        page._top_left = MagicMock()
        page._top_right = MagicMock()
        page._bottom_log = MagicMock()
        page._action_row = MagicMock()
        page._main_container = MagicMock()

        page._apply_layout_mode("compact")

        assert page._bottom_log.grid.call_args.kwargs["row"] == 2
        assert page._action_row.grid.call_args.kwargs["row"] == 3

    def test_extract_log_level_supports_common_head_formats(self):
        """支持常见日志头格式的级别提取"""
        page = object.__new__(ProcessPage)

        assert page._extract_log_level("[WARNING] disk almost full") == "WARNING"
        assert page._extract_log_level("ERROR: failed to open file") == "ERROR"
        assert page._extract_log_level("(INFO) startup complete") == "INFO"
