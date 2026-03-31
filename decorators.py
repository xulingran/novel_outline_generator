"""
通用装饰器模块
提供重试、日志等常用装饰器
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """重试装饰器

    为异步函数提供自动重试功能，支持指数退避。

    Args:
        max_attempts: 最大尝试次数（默认 3）
        backoff_base: 退避基数（秒）（默认 2.0）
            第 n 次重试等待时间 = backoff_base * (n-1)
        exceptions: 需要重试的异常类型元组（默认捕获所有 Exception）
        on_retry: 重试回调函数，接收 (exception, attempt) 参数

    Returns:
        装饰后的函数

    Example:
        @with_retry(
            max_attempts=3,
            backoff_base=2.0,
            exceptions=(APIError, ProcessingError),
            on_retry=lambda e, a: logger.warning(f"Retry {a}: {e}")
        )
        async def process_chunk(chunk: TextChunk) -> dict:
            # 业务逻辑
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        wait_time = backoff_base * (attempt - 1)
                        if on_retry:
                            on_retry(e, attempt)
                        logger.debug(
                            f"Function {func.__name__} attempt {attempt} failed: {e}, "
                            f"retrying in {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.debug(
                            f"Function {func.__name__} failed after {max_attempts} attempts"
                        )

            # 所有重试都失败，抛出最后一个错误
            raise last_error  # type: ignore

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        wait_time = backoff_base * (attempt - 1)
                        if on_retry:
                            on_retry(e, attempt)
                        logger.debug(
                            f"Function {func.__name__} attempt {attempt} failed: {e}, "
                            f"retrying in {wait_time:.1f}s"
                        )
                        import time

                        time.sleep(wait_time)
                    else:
                        logger.debug(
                            f"Function {func.__name__} failed after {max_attempts} attempts"
                        )

            raise last_error  # type: ignore

        # 根据被装饰函数是同步还是异步返回对应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
