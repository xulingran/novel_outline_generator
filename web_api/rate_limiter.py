"""请求限流器"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException


class RateLimiter:
    """简单内存限流器，按 IP 在时间窗口内计数。"""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check_rate_limit(self, client_ip: str, max_requests: int, window_seconds: int) -> None:
        now = time.time()
        window_start = now - window_seconds
        bucket = self._requests[client_ip]
        # 清理窗口外的请求时间戳
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= max_requests:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        bucket.append(now)


rate_limiter = RateLimiter()
