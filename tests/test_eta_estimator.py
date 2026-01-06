"""测试 ETA 估算器模块"""

import time
from datetime import datetime

from services.eta_estimator import ETAEstimator


class TestETAEstimator:
    """ETAEstimator 测试类"""

    def test_initialization(self):
        """测试初始化"""
        estimator = ETAEstimator(window_size=10, outlier_threshold=2.0, min_samples=5)
        assert estimator.window_size == 10
        assert estimator.outlier_threshold == 2.0
        assert estimator.min_samples == 5
        assert len(estimator.processing_times) == 0
        assert estimator.processing_start_time is None
        assert estimator.parallel_limit == 1

    def test_set_parallel_limit(self):
        """测试设置并发限制"""
        estimator = ETAEstimator()
        estimator.set_parallel_limit(5)
        assert estimator.parallel_limit == 5
        estimator.set_parallel_limit(0)
        assert estimator.parallel_limit == 1
        estimator.set_parallel_limit(-1)
        assert estimator.parallel_limit == 1

    def test_start_processing(self):
        """测试开始处理"""
        estimator = ETAEstimator()
        assert estimator.processing_start_time is None
        estimator.start_processing()
        assert estimator.processing_start_time is not None

    def test_add_completion(self):
        """测试添加完成记录"""
        estimator = ETAEstimator()
        estimator.add_completion(1.5, 1)
        assert len(estimator.processing_times) == 1
        assert estimator.processing_times[0] == 1.5
        assert len(estimator.timestamps) == 1
        estimator.add_completion(2.0, 2)
        assert len(estimator.processing_times) == 2

    def test_add_retry(self):
        """测试添加重试时间"""
        estimator = ETAEstimator()
        estimator.add_retry(0.5)
        assert len(estimator.retry_times) == 1
        estimator.add_retry(1.0)
        assert len(estimator.retry_times) == 2

    def test_filter_outliers(self):
        """测试异常值过滤"""
        estimator = ETAEstimator()
        # 测试样本不足4个
        result = estimator._filter_outliers([1.0, 2.0, 3.0])
        assert len(result) == 3

        # 测试正常数据
        times = [1.0, 2.0, 100.0, 2.0, 1.5]
        result = estimator._filter_outliers(times)
        assert 100.0 not in result

        # 测试IQR为0的情况
        times = [5.0, 5.0, 5.0, 5.0]
        result = estimator._filter_outliers(times)
        assert len(result) == 4

        # 测试所有值都是异常值的情况
        times = [1.0, 1000.0, 2000.0, 3000.0]
        result = estimator._filter_outliers(times)
        # 当所有值被过滤后，返回原始列表
        assert len(result) == 4

    def test_estimate_with_empty_filtered_times(self):
        """测试过滤后时间为空的情况"""
        estimator = ETAEstimator(min_samples=1)
        # 直接测试 processing_times 为空时的 _filter_outliers 返回值
        filtered = estimator._filter_outliers([])
        assert filtered == []
        assert not filtered

        # 现在测试 estimate 中 processing_times 为空的情况
        # 当 processing_times 为空时，会因为样本不足返回 None
        # 但这不会执行到 filtered_times 检查
        # 所以第195行的代码是防御性编程，实际上很难触发
        # 但我们仍然保留这个检查以防万一
        estimator2 = ETAEstimator(min_samples=0)  # 设置 min_samples=0 使样本检查通过
        result = estimator2.estimate(total_chunks=10, completed_chunks=0)
        # 此时 processing_times 为空，会在 filtered_times 检查时返回
        assert result["eta_seconds"] is None

    def test_calculate_weighted_average(self):
        """测试加权平均计算"""
        estimator = ETAEstimator()
        # 测试空列表
        result = estimator._calculate_weighted_average([])
        assert result == 0.0

        # 测试单个样本
        result = estimator._calculate_weighted_average([5.0])
        assert result == 5.0

        # 测试多个样本
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = estimator._calculate_weighted_average(times)
        # 权重: [1,2,3,4,5], 加权平均应该偏向更大的值
        assert result > 3.0
        assert result < 5.0

    def test_calculate_throughput(self):
        """测试吞吐量计算"""
        estimator = ETAEstimator()
        # 测试时间戳不足
        result = estimator._calculate_throughput()
        assert result is None

        # 添加时间戳
        estimator.timestamps.append((datetime.now(), 0))
        result = estimator._calculate_throughput()
        assert result is None

        # 添加时间戳但数量相同
        estimator.timestamps.append((datetime.now(), 0))
        result = estimator._calculate_throughput()
        assert result is None

        # 添加时间戳但更早的数量更大
        estimator.timestamps.clear()
        estimator.timestamps.append((datetime.now(), 10))
        time.sleep(0.01)
        estimator.timestamps.append((datetime.now(), 5))
        result = estimator._calculate_throughput()
        assert result is None

        # 添加足够的时间戳
        estimator.timestamps.clear()
        estimator.timestamps.append((datetime.now(), 0))
        time.sleep(0.01)
        estimator.timestamps.append((datetime.now(), 5))
        result = estimator._calculate_throughput()
        assert result is not None
        assert result > 0

    def test_estimate_total_chunks_zero(self):
        """测试总块数为0的情况"""
        estimator = ETAEstimator()
        result = estimator.estimate(total_chunks=0, completed_chunks=0)
        assert result["eta_seconds"] is None
        assert result["confidence"] == "low"

    def test_estimate_completed(self):
        """测试已完成的情况"""
        estimator = ETAEstimator()
        result = estimator.estimate(total_chunks=10, completed_chunks=10)
        assert result["eta_seconds"] == 0.0
        assert result["confidence"] == "high"
        assert result["method"] == "completed"

    def test_estimate_insufficient_samples(self):
        """测试样本不足的情况"""
        estimator = ETAEstimator(min_samples=5)
        estimator.add_completion(1.0, 1)
        estimator.add_completion(1.5, 2)
        result = estimator.estimate(total_chunks=10, completed_chunks=2)
        assert result["eta_seconds"] is None
        assert result["confidence"] == "low"

    def test_estimate_throughput_method(self):
        """测试使用吞吐量方法估算"""
        estimator = ETAEstimator(min_samples=2)
        estimator.processing_start_time = datetime.now()
        estimator.add_completion(1.0, 1)
        time.sleep(0.01)
        estimator.add_completion(1.2, 2)
        result = estimator.estimate(total_chunks=10, completed_chunks=2)
        assert result["eta_seconds"] is not None
        assert result["method"] == "throughput"

    def test_estimate_weighted_average_method(self):
        """测试使用加权平均方法估算"""
        estimator = ETAEstimator(min_samples=3, window_size=10)
        # 不设置时间戳，强制使用加权平均
        for i in range(5):
            estimator.add_completion(2.0, i + 1)
        result = estimator.estimate(total_chunks=10, completed_chunks=5)
        assert result["eta_seconds"] is not None
        assert result["method"] == "weighted_average"

    def test_estimate_with_parallel(self):
        """测试并发情况下的估算"""
        estimator = ETAEstimator(min_samples=3)
        estimator.set_parallel_limit(4)
        for i in range(5):
            estimator.add_completion(4.0, i + 1)
        result = estimator.estimate(total_chunks=10, completed_chunks=5)
        assert result["eta_seconds"] is not None
        # 并发应该减少预估时间

    def test_estimate_with_retries(self):
        """测试包含重试时间的估算"""
        estimator = ETAEstimator(min_samples=3)
        for i in range(5):
            estimator.add_completion(2.0, i + 1)
        estimator.add_retry(5.0)
        estimator.add_retry(3.0)
        result = estimator.estimate(total_chunks=10, completed_chunks=5, failed_chunks=2)
        assert result["eta_seconds"] is not None
        # 应该考虑了失败块的重试时间

    def test_estimate_high_confidence_with_window_full(self):
        """测试窗口满时提高置信度"""
        estimator = ETAEstimator(min_samples=2, window_size=5)
        for i in range(5):
            estimator.add_completion(2.0, i + 1)
        result = estimator.estimate(total_chunks=10, completed_chunks=5)
        assert result["confidence"] == "high"

    def test_reset(self):
        """测试重置估算器"""
        estimator = ETAEstimator()
        estimator.add_completion(1.0, 1)
        estimator.add_completion(2.0, 2)
        estimator.add_retry(0.5)
        estimator.start_processing()
        estimator.processing_start_time = datetime.now()

        estimator.reset()

        assert len(estimator.processing_times) == 0
        assert len(estimator.timestamps) == 0
        assert len(estimator.retry_times) == 0
        assert estimator.processing_start_time is None
