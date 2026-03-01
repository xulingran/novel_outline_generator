"""
ETA 估算器 - 提供稳健的剩余时间估算
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ETAEstimator:
    """ETA 估算器，使用多种策略提供准确的剩余时间估算"""

    def __init__(
        self,
        window_size: int = 20,
        outlier_threshold: float = 2.5,
        min_samples: int = 3,
    ) -> None:
        """
        初始化 ETA 估算器

        Args:
            window_size: 移动平均窗口大小
            outlier_threshold: 异常值阈值（标准差倍数）
            min_samples: 最小样本数
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.min_samples = min_samples

        # 记录处理时间（最近N个）
        self.processing_times: deque[float] = deque(maxlen=window_size)

        # 记录处理时间戳（用于计算实际吞吐量）
        self.timestamps: deque[tuple[datetime, int]] = deque(maxlen=window_size)

        # 记录失败块的重试时间
        self.retry_times: list[float] = []

        # 实际处理开始时间
        self.processing_start_time: datetime | None = None

        # 并发数
        self.parallel_limit: int = 1

    def set_parallel_limit(self, limit: int) -> None:
        """设置并发限制"""
        self.parallel_limit = max(1, limit)

    def start_processing(self) -> None:
        """标记处理开始"""
        if self.processing_start_time is None:
            self.processing_start_time = datetime.now()
            logger.debug("ETA 估算器：处理开始")

    def add_completion(
        self, processing_time: float, completed_count: int, record_timestamp: bool = True
    ) -> None:
        """
        添加一个完成的块

        Args:
            processing_time: 处理时间（秒）
            completed_count: 已完成块数
            record_timestamp: 是否记录时间戳用于吞吐量计算
        """
        self.processing_times.append(processing_time)
        if record_timestamp:
            self.timestamps.append((datetime.now(), completed_count))
        logger.debug(f"ETA 估算器：添加完成，时间={processing_time:.2f}s，已完成={completed_count}")

    def add_retry(self, retry_time: float) -> None:
        """添加重试时间"""
        self.retry_times.append(retry_time)
        logger.debug(f"ETA 估算器：添加重试，时间={retry_time:.2f}s")

    def _filter_outliers(self, times: list[float]) -> list[float]:
        """
        过滤异常值

        使用 IQR（四分位距）方法过滤异常值
        """
        if len(times) < 4:
            return times

        sorted_times = sorted(times)
        n = len(sorted_times)
        q1 = sorted_times[n // 4]
        q3 = sorted_times[3 * n // 4]
        iqr = q3 - q1

        if iqr == 0:
            return times

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = [t for t in times if lower_bound <= t <= upper_bound]
        return filtered if filtered else times

    def _calculate_weighted_average(self, times: list[float]) -> float:
        """
        计算加权平均，最近的样本权重更高

        Args:
            times: 处理时间列表

        Returns:
            加权平均时间
        """
        if not times:
            return 0.0

        n = len(times)
        weights = [i + 1 for i in range(n)]  # 线性权重，最近的权重最高
        total_weight = sum(weights)

        weighted_sum = sum(t * w for t, w in zip(times, weights, strict=True))
        return weighted_sum / total_weight

    def _calculate_throughput(self) -> float | None:
        """
        基于实际吞吐量计算 ETA

        Returns:
            每秒处理的块数
        """
        if len(self.timestamps) < 2:
            return None

        # 使用最近的两个时间戳计算吞吐量
        latest_time, latest_count = self.timestamps[-1]
        earliest_time, earliest_count = self.timestamps[0]

        time_diff = (latest_time - earliest_time).total_seconds()
        if time_diff <= 0:
            return None

        count_diff = latest_count - earliest_count
        if count_diff <= 0:
            return None

        throughput = count_diff / time_diff
        logger.debug(f"ETA 估算器：吞吐量={throughput:.2f} 块/秒")
        return throughput

    def estimate(
        self,
        total_chunks: int,
        completed_chunks: int,
        failed_chunks: int = 0,
    ) -> dict[str, Any]:
        """
        估算剩余时间

        Args:
            total_chunks: 总块数
            completed_chunks: 已完成块数
            failed_chunks: 失败块数

        Returns:
            包含 eta_seconds, confidence, method 等信息的字典
        """
        result: dict[str, Any] = {
            "eta_seconds": None,
            "confidence": "low",
            "method": "none",
        }

        if total_chunks == 0:
            return result

        remaining = max(total_chunks - completed_chunks, 0)
        if remaining == 0:
            result["eta_seconds"] = 0.0
            result["confidence"] = "high"
            result["method"] = "completed"
            return result

        # 样本不足
        if len(self.processing_times) < self.min_samples:
            return result

        # 策略1：基于吞吐量的估算（最准确）
        throughput = self._calculate_throughput()
        if throughput is not None and throughput > 0:
            eta = remaining / throughput
            result["eta_seconds"] = eta
            result["confidence"] = "high"
            result["method"] = "throughput"
            return result

        # 策略2：基于加权平均时间的估算
        filtered_times = self._filter_outliers(list(self.processing_times))
        if not filtered_times:
            return result

        weighted_avg = self._calculate_weighted_average(filtered_times)

        # 考虑并发处理
        effective_avg = weighted_avg / self.parallel_limit

        # 考虑失败块的重试时间
        if self.retry_times:
            avg_retry = sum(self.retry_times) / len(self.retry_times)
            # 假设失败块需要重试一次
            estimated_retry_time = failed_chunks * avg_retry
        else:
            estimated_retry_time = 0.0

        eta = remaining * effective_avg + estimated_retry_time

        result["eta_seconds"] = eta
        result["confidence"] = "medium"
        result["method"] = "weighted_average"

        # 如果有足够样本，提高置信度
        if len(filtered_times) >= self.window_size:
            result["confidence"] = "high"

        logger.debug(
            f"ETA 估算：剩余={remaining}，平均时间={weighted_avg:.2f}s，"
            f"并发={self.parallel_limit}，ETA={eta:.2f}s，置信度={result['confidence']}"
        )

        return result

    def reset(self) -> None:
        """重置估算器"""
        self.processing_times.clear()
        self.timestamps.clear()
        self.retry_times.clear()
        self.processing_start_time = None
        logger.debug("ETA 估算器：已重置")
