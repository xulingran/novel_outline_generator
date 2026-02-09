"""
ProcessPage 组件测试
"""

from unittest.mock import MagicMock, patch

from gui.pages.process_page import ProcessPage


def test_merge_state_variables_initialized():
    """合并相关状态变量应在初始化时设置"""
    # Mock _setup_ui 以避免实际的 UI 初始化
    with patch.object(ProcessPage, "_setup_ui"):
        # 创建一个 mock master
        mock_master = MagicMock()

        # 初始化 ProcessPage
        page = ProcessPage(mock_master)

        # 验证合并状态变量已初始化
        assert hasattr(page, "_last_phase")
        assert hasattr(page, "_initial_outline_count")
        assert hasattr(page, "_is_merge_phase")

        # 验证初始值
        assert page._last_phase == ""
        assert page._initial_outline_count == 0
        assert page._is_merge_phase is False


class TestMergeProgressCalculation:
    """测试合并进度计算逻辑"""

    def test_calculate_merge_progress_basic(self):
        """基础合并进度计算：层级 1，批次 1/1，大纲缩减 50%"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        page._initial_outline_count = 100

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
        page._initial_outline_count = 0

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
        page._initial_outline_count = 10

        # 边界情况：outlines_count > initial_count（不应发生）
        progress = page._calculate_merge_progress(
            merge_level=10, merge_batch_current=0, merge_batch_total=1, merge_outlines_count=100
        )

        assert 0 <= progress <= 1

    def test_calculate_merge_progress_zero_level(self):
        """层级为 0 时应返回 0.8"""
        # 使用 object.__new__ 跳过 __init__ 避免实际 UI 初始化
        page = object.__new__(ProcessPage)
        page._initial_outline_count = 100

        progress = page._calculate_merge_progress(
            merge_level=0, merge_batch_current=1, merge_batch_total=1, merge_outlines_count=50
        )

        # merge_level=0 时 level_progress = 0.8
        # 批次进度: 1/1 = 1.0
        # 缩减进度: 1 - 50/100 = 0.5
        # 综合: 0.8*0.4 + 1.0*0.4 + 0.5*0.2 = 0.32 + 0.4 + 0.1 = 0.82
        expected = 0.8 * 0.4 + 1.0 * 0.4 + 0.5 * 0.2
        assert abs(progress - expected) < 0.01
