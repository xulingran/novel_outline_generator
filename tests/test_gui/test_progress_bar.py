"""
测试 progress_bar 组件

测试进度条组件，包括进度更新、统计显示、ETA 格式化等功能。
"""

from unittest.mock import MagicMock

import pytest

from gui.widgets.progress_bar import ProgressBar
from tests.test_gui.conftest import create_mock_progress_data, ctk


@pytest.fixture
def progress_bar():
    """创建进度条实例"""
    master = MagicMock()
    bar = ProgressBar(master)
    return bar


class TestProgressBar:
    """测试 ProgressBar 组件"""

    def test_init(self, skip_if_no_gui):
        """测试 ProgressBar 初始化"""
        master = MagicMock()
        bar = ProgressBar(master)

        assert bar.total_chunks == 0
        assert bar.completed_chunks == 0
        assert bar.failed_chunks == 0
        assert bar.partial_chunks == 0
        assert bar.current_phase == ""
        assert bar.eta_seconds == 0
        assert bar.eta_confidence == 0.0

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_update_progress_basic(self, progress_bar):
        """测试基本进度更新"""
        progress_bar.update_progress(
            completed=5,
            total=10,
            failed=0,
            partial=0,
            phase="处理中",
        )

        assert progress_bar.completed_chunks == 5
        assert progress_bar.total_chunks == 10
        assert progress_bar.failed_chunks == 0
        assert progress_bar.partial_chunks == 0
        assert progress_bar.current_phase == "处理中"

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_update_progress_with_eta(self, progress_bar):
        """测试带 ETA 的进度更新"""
        progress_bar.update_progress(
            completed=7,
            total=10,
            failed=1,
            partial=0,
            phase="合并中",
            eta_seconds=60,
            eta_confidence=0.9,
        )

        assert progress_bar.eta_seconds == 60
        assert progress_bar.eta_confidence == 0.9
        assert progress_bar.current_phase == "合并中"

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_reset(self, progress_bar):
        """测试重置进度"""
        # 先设置一些进度
        progress_bar.update_progress(
            completed=5,
            total=10,
            failed=1,
            partial=0,
            phase="处理中",
            eta_seconds=120,
        )

        # 重置
        progress_bar.reset()

        assert progress_bar.total_chunks == 0
        assert progress_bar.completed_chunks == 0
        assert progress_bar.failed_chunks == 0
        assert progress_bar.partial_chunks == 0
        assert progress_bar.current_phase == "等待开始..."


class TestProgressBarETAFormatting:
    """测试 ETA 格式化功能"""

    def test_format_eta_seconds_only(self, progress_bar):
        """测试仅秒数的 ETA"""
        eta_text = progress_bar._format_eta(30, 0.3)

        assert "30秒" in eta_text
        assert "置信度: 低" in eta_text

    def test_format_eta_minutes(self, progress_bar):
        """测试分钟的 ETA"""
        eta_text = progress_bar._format_eta(150, 0.6)

        assert "2分钟" in eta_text
        assert "置信度: 中" in eta_text

    def test_format_eta_hours(self, progress_bar):
        """测试小时的 ETA"""
        eta_text = progress_bar._format_eta(3665, 0.9)

        assert "1小时" in eta_text
        assert "置信度: 高" in eta_text

    def test_format_eta_combined(self, progress_bar):
        """测试组合时间的 ETA"""
        eta_text = progress_bar._format_eta(3720, 0.8)

        # 应该包含小时和分钟
        assert "1小时" in eta_text
        assert "2分钟" in eta_text

    def test_format_eta_confidence_levels(self, progress_bar):
        """测试不同置信度级别"""
        # 高置信度
        assert "高" in progress_bar._format_eta(60, 0.85)
        assert "高" in progress_bar._format_eta(60, 0.8)

        # 中置信度
        assert "中" in progress_bar._format_eta(60, 0.5)
        assert "中" in progress_bar._format_eta(60, 0.7)

        # 低置信度
        assert "低" in progress_bar._format_eta(60, 0.3)
        assert "低" in progress_bar._format_eta(60, 0.0)


class TestProgressBarStatistics:
    """测试统计功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_completed_count(self, progress_bar):
        """测试完成计数"""
        progress_bar.update_progress(completed=10, total=20)
        assert progress_bar.completed_chunks == 10

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_failed_count(self, progress_bar):
        """测试失败计数"""
        progress_bar.update_progress(completed=8, total=10, failed=2)
        assert progress_bar.failed_chunks == 2

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_partial_count(self, progress_bar):
        """测试部分完成计数"""
        progress_bar.update_progress(completed=7, total=10, partial=1)
        assert progress_bar.partial_chunks == 1

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_all_statistics(self, progress_bar):
        """测试所有统计信息"""
        progress_bar.update_progress(
            completed=15,
            total=20,
            failed=3,
            partial=2,
            phase="完成",
        )

        assert progress_bar.completed_chunks == 15
        assert progress_bar.total_chunks == 20
        assert progress_bar.failed_chunks == 3
        assert progress_bar.partial_chunks == 2
        assert progress_bar.current_phase == "完成"


class TestProgressBarEdgeCases:
    """测试边界情况"""

    def test_zero_progress(self, progress_bar):
        """测试零进度"""
        progress_bar.update_progress(completed=0, total=100)

        assert progress_bar.completed_chunks == 0
        assert progress_bar.total_chunks == 100

    def test_complete_progress(self, progress_bar):
        """测试完成进度"""
        progress_bar.update_progress(completed=100, total=100)

        assert progress_bar.completed_chunks == 100
        assert progress_bar.total_chunks == 100

    def test_zero_eta(self, progress_bar):
        """测试零 ETA"""
        eta_text = progress_bar._format_eta(0, 0.0)

        assert "0秒" in eta_text

    def test_very_large_eta(self, progress_bar):
        """测试非常大的 ETA"""
        # 10 小时
        eta_text = progress_bar._format_eta(36000, 0.7)

        assert "10小时" in eta_text

    def test_negative_eta(self, progress_bar):
        """测试负 ETA（不应该发生，但要处理）"""
        # 在实际使用中不应该有负的 ETA
        eta_text = progress_bar._format_eta(-10, 0.5)

        # 应该仍然返回某种格式化的字符串
        assert isinstance(eta_text, str)

    def test_none_eta(self, progress_bar):
        """测试 None ETA"""
        # 在更新时传递 None
        progress_bar.update_progress(completed=5, total=10, eta_seconds=None)

        assert progress_bar.eta_seconds == 0


class TestProgressBarProgressCalculation:
    """测试进度计算"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_progress_percentage(self, progress_bar):
        """测试进度百分比"""
        # 50%
        progress_bar.update_progress(completed=5, total=10)
        # 假设 get_progress 返回 0-1 之间的值
        if hasattr(progress_bar, "get_progress"):
            assert isinstance(progress_bar.get_progress(), float)

    @pytest.mark.skipif(not hasattr(ctk, "CTkProgressBar"), reason="CustomTkinter not available")
    def test_progress_zero_denominator(self, progress_bar):
        """测试零分母情况"""
        # 避免除以零
        progress_bar.update_progress(completed=0, total=0)

        # 不应该崩溃
        assert progress_bar.total_chunks == 0
        assert progress_bar.completed_chunks == 0


class TestProgressBarIntegration:
    """测试集成功能"""

    def test_with_mock_progress_data(self, progress_bar):
        """测试使用模拟进度数据"""
        mock_data = create_mock_progress_data()

        progress_bar.update_progress(
            completed=mock_data["completed_chunks"],
            total=mock_data["total_chunks"],
            failed=mock_data["failed_chunks"],
            partial=mock_data["partial_chunks"],
            phase=mock_data["phase"],
            eta_seconds=mock_data["eta_seconds"],
            eta_confidence=mock_data["eta_confidence"],
        )

        assert progress_bar.completed_chunks == 5
        assert progress_bar.total_chunks == 10
        assert progress_bar.failed_chunks == 1

    def test_series_of_updates(self, progress_bar):
        """测试一系列更新"""
        updates = [
            {"completed": 1, "total": 10},
            {"completed": 2, "total": 10},
            {"completed": 5, "total": 10},
            {"completed": 8, "total": 10},
            {"completed": 10, "total": 10},
        ]

        for update in updates:
            progress_bar.update_progress(**update)

        # 最终状态应该是最后一次更新
        assert progress_bar.completed_chunks == 10
        assert progress_bar.total_chunks == 10
