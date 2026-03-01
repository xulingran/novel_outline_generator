"""
HTTP 连接池管理器
管理 HTTP 客户端连接池，支持代理和连接复用
"""

import asyncio
import atexit
import inspect
import logging
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _httpx_proxy_kwargs(
    client_class: type[httpx.Client] | type[httpx.AsyncClient], proxy_url: str
) -> dict[str, Any]:
    """Support both proxy/proxies kwargs across httpx versions."""
    params = inspect.signature(client_class.__init__).parameters
    if "proxy" in params:
        return {"proxy": proxy_url}
    if "proxies" in params:
        return {"proxies": proxy_url}
    return {}


class HTTPConnectionPool:
    """HTTP 客户端连接池管理器

    管理主客户端和代理客户端的生命周期，支持：
    - 延迟创建客户端
    - 代理配置管理
    - 统一资源清理
    """

    _instance: "HTTPConnectionPool | None" = None

    def __new__(cls) -> "HTTPConnectionPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._main_client: httpx.AsyncClient | None = None
        self._proxy_clients: dict[str, httpx.AsyncClient] = {}
        self._cleanup_registered: bool = False
        self._initialized = True

    def get_client(
        self,
        proxy_url: str | None = None,
        limits: httpx.Limits | None = None,
        timeout: float = 60.0,
    ) -> httpx.AsyncClient:
        """获取 HTTP 客户端

        Args:
            proxy_url: 代理 URL，如果为 None 则返回主客户端
            limits: 连接限制，默认 max_connections=100
            timeout: 超时时间（秒）

        Returns:
            httpx.AsyncClient: HTTP 客户端实例
        """
        if limits is None:
            limits = httpx.Limits(max_connections=100)

        if proxy_url:
            client = self._proxy_clients.get(proxy_url)
            if client is None:
                proxy_kwargs = _httpx_proxy_kwargs(httpx.AsyncClient, proxy_url)
                client = httpx.AsyncClient(
                    **proxy_kwargs,
                    limits=limits,
                    timeout=timeout,
                )
                self._proxy_clients[proxy_url] = client
                logger.debug(f"创建代理客户端: {proxy_url}")
            return client

        if self._main_client is None:
            self._main_client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
            )
            logger.debug("创建主 HTTP 客户端")
        return self._main_client

    async def close_all(self) -> None:
        """关闭所有 HTTP 客户端连接"""
        # 关闭代理客户端
        for proxy_url, client in list(self._proxy_clients.items()):
            try:
                await client.aclose()
                logger.debug(f"已关闭代理客户端: {proxy_url}")
            except Exception as e:
                logger.warning(f"关闭代理客户端失败 ({proxy_url}): {e}")
        self._proxy_clients.clear()

        # 关闭主客户端
        if self._main_client is not None:
            try:
                await self._main_client.aclose()
                logger.debug("已关闭主 HTTP 客户端")
            except Exception as e:
                logger.warning(f"关闭主 HTTP 客户端失败: {e}")
            self._main_client = None

    def register_cleanup(self, cleanup_func: Callable[[], None]) -> None:
        """注册程序退出清理函数（仅注册一次）

        Args:
            cleanup_func: 清理函数
        """
        if not self._cleanup_registered:
            atexit.register(cleanup_func)
            self._cleanup_registered = True
            logger.debug("已注册 HTTP 连接池清理函数")

    def _cleanup_on_exit(self) -> None:
        """程序退出时清理资源（同步方法）"""
        logger.debug("程序退出，清理 HTTP 连接池资源")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.close_all())
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"退出时清理 HTTP 连接池失败: {e}")


# 全局默认连接池实例
_default_pool: HTTPConnectionPool | None = None


def get_default_connection_pool() -> HTTPConnectionPool:
    """获取全局默认连接池

    Returns:
        HTTPConnectionPool: 全局默认连接池实例（单例）
    """
    global _default_pool
    if _default_pool is None:
        _default_pool = HTTPConnectionPool()
    return _default_pool
