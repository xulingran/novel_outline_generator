"""
LLM服务模块
提供统一的LLM调用接口
"""
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from config import get_api_config, get_processing_config
from exceptions import APIKeyError, RateLimitError, APIError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应模型"""
    content: str
    token_usage: Optional[Dict[str, int]] = None
    response_time: Optional[float] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class CircuitBreaker:
    """熔断器模式实现"""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call_allowed(self) -> bool:
        """检查是否允许调用"""
        if self.state == 'CLOSED':
            return True

        if self.state == 'OPEN':
            if (self.last_failure_time and
                datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_seconds)):
                self.state = 'HALF_OPEN'
                return True
            return False

        # HALF_OPEN state
        return True

    def record_success(self) -> None:
        """记录成功"""
        self.failure_count = 0
        self.state = 'CLOSED'

    def record_failure(self) -> None:
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"熔断器打开，失败次数: {self.failure_count}")


class LLMService(ABC):
    """LLM服务基类"""

    def __init__(self):
        self.api_config = get_api_config()
        self.processing_config = get_processing_config()
        self.circuit_breaker = CircuitBreaker()
        self._init_client()

    @abstractmethod
    def _init_client(self) -> None:
        """初始化客户端"""
        pass

    @abstractmethod
    async def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        """调用API的具体实现"""
        pass

    async def call(self, prompt: str, chunk_id: Optional[int] = None) -> LLMResponse:
        """统一的调用接口，返回完整的响应对象（包含token使用情况）"""
        chunk_info = f" [块 {chunk_id}]" if chunk_id else ""

        # 检查熔断器
        if not self.circuit_breaker.call_allowed():
            raise APIError("服务暂时不可用，请稍后再试", is_retryable=True)

        # 重试逻辑
        max_retries = self.processing_config.max_retry
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    logger.debug(f"调用LLM API{chunk_info}...")

                start_time = datetime.now()
                llm_response = await self._call_api(prompt)
                response_time = (datetime.now() - start_time).total_seconds()

                # 记录成功
                self.circuit_breaker.record_success()

                logger.debug(f"LLM API调用成功{chunk_info}，耗时: {response_time:.2f}秒")
                return llm_response

            except RateLimitError as e:
                # 速率限制错误，等待更长时间
                wait_time = e.retry_after or (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API速率限制{chunk_info}，等待 {wait_time:.1f} 秒后重试")
                await asyncio.sleep(wait_time)

            except APIError as e:
                if not e.is_retryable:
                    # 不可重试的错误，直接抛出
                    self.circuit_breaker.record_failure()
                    raise

                # 可重试的错误
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API调用失败{chunk_info}，{wait_time:.1f}秒后重试: {e.message}")
                await asyncio.sleep(wait_time)

            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.error(f"未知错误{chunk_info}，{wait_time:.1f}秒后重试: {str(e)}")
                self.circuit_breaker.record_failure()
                await asyncio.sleep(wait_time)

        # 所有重试都失败
        self.circuit_breaker.record_failure()
        raise APIError(f"LLM API多次失败{chunk_info}", is_retryable=False)


class OpenAIService(LLMService):
    """OpenAI服务实现"""

    _http_client: Optional["httpx.AsyncClient"] = None
    _proxy_clients: Dict[str, "httpx.AsyncClient"] = {}

    @classmethod
    def get_http_client(cls, proxy_url: Optional[str] = None) -> "httpx.AsyncClient":
        import httpx

        if proxy_url:
            client = cls._proxy_clients.get(proxy_url)
            if client is None:
                client = httpx.AsyncClient(
                    proxies=proxy_url,
                    limits=httpx.Limits(max_connections=100),
                    timeout=60.0
                )
                cls._proxy_clients[proxy_url] = client
            return client

        if cls._http_client is None:
            cls._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=100),
                timeout=60.0
            )
        return cls._http_client

    @classmethod
    async def close_http_clients(cls) -> None:
        """关闭所有HTTP客户端连接池"""
        import httpx

        # 关闭代理客户端
        for proxy_url, client in list(cls._proxy_clients.items()):
            try:
                await client.aclose()
                logger.debug(f"已关闭代理客户端: {proxy_url}")
            except Exception as e:
                logger.warning(f"关闭代理客户端失败 ({proxy_url}): {e}")
        cls._proxy_clients.clear()

        # 关闭主客户端
        if cls._http_client is not None:
            try:
                await cls._http_client.aclose()
                logger.debug("已关闭主HTTP客户端")
            except Exception as e:
                logger.warning(f"关闭主HTTP客户端失败: {e}")
            cls._http_client = None

    def _init_client(self) -> None:
        """初始化OpenAI客户端"""
        try:
            from openai import AsyncOpenAI

            proxy_url = self.processing_config.proxy_url if self.processing_config.use_proxy else None
            http_client = self.get_http_client(proxy_url)
            if proxy_url:
                logger.debug(f"OpenAI客户端配置代理: {proxy_url}")

            self.client = AsyncOpenAI(
                api_key=self.api_config.api_key,
                base_url=self.api_config.base_url,
                http_client=http_client
            )
            logger.info(f"OpenAI API客户端初始化成功 (模型: {self.api_config.model_name})")

        except ImportError:
            raise APIKeyError("未安装openai库，请运行: pip install openai")
        except Exception as e:
            raise APIKeyError(f"OpenAI API配置失败: {str(e)}")

    async def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        """调用OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.api_config.model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
                **kwargs
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage=response.usage.model_dump() if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason
            )

        except Exception as e:
            error_str = str(e).lower()

            # 检查特定错误类型
            if "authentication" in error_str or "unauthorized" in error_str:
                raise APIKeyError("API密钥无效或已过期")

            if "rate" in error_str and "limit" in error_str:
                # 尝试从错误中提取重试时间
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')
                    if retry_after:
                        retry_after = int(retry_after)
                raise RateLimitError("API速率限制", retry_after=retry_after)

            if "quota" in error_str or "exceeded" in error_str:
                raise APIError("API配额已用尽", is_retryable=False)

            # 其他错误
            raise APIError(f"OpenAI API错误: {str(e)}", is_retryable=True)


class GeminiService(LLMService):
    """Gemini服务实现"""

    def _init_client(self) -> None:
        """初始化Gemini客户端"""
        try:
            import google.generativeai as genai

            # 配置API密钥
            genai.configure(api_key=self.api_config.api_key)

            # 配置安全设置
            safety_mapping = {
                "BLOCK_NONE": genai.types.HarmBlockThreshold.BLOCK_NONE,
                "BLOCK_ONLY_HIGH": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                "BLOCK_MEDIUM_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                "BLOCK_LOW_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }

            safety_threshold = safety_mapping.get(
                self.api_config.gemini_safety,
                genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            )

            self.safety_settings = [
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": safety_threshold,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": safety_threshold,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": safety_threshold,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": safety_threshold,
                },
            ]

            logger.info(f"Gemini API初始化成功 (模型: {self.api_config.model_name})")
            logger.info(f"安全设置: {self.api_config.gemini_safety}")

        except ImportError:
            raise APIKeyError("未安装google-generativeai库，请运行: pip install google-generativeai")
        except Exception as e:
            raise APIKeyError(f"Gemini API配置失败: {str(e)}")

    async def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        """调用Gemini API"""
        import google.generativeai as genai

        try:
            # 创建模型实例
            model = genai.GenerativeModel(self.api_config.model_name)

            # 在线程池中执行同步调用
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    safety_settings=self.safety_settings
                )
            )

            # 检查内容是否被阻止
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason') and feedback.block_reason:
                    raise APIError(f"内容被安全过滤器阻止: {feedback.block_reason}", is_retryable=False)

            # 提取文本内容
            if hasattr(response, 'text') and response.text:
                return LLMResponse(
                    content=response.text,
                    model=self.api_config.model_name
                )
            else:
                raise APIError("Gemini API返回空内容", is_retryable=True)

        except Exception as e:
            error_str = str(e).lower()

            # 检查特定错误类型
            if "permission" in error_str or "forbidden" in error_str:
                raise APIKeyError("API密钥无效或无权限")

            if "quota" in error_str or "exceeded" in error_str:
                raise APIError("API配额已用尽", is_retryable=False)

            if "safety" in error_str and "filter" in error_str:
                raise APIError("内容被安全过滤器阻止", is_retryable=False)

            raise APIError(f"Gemini API错误: {str(e)}", is_retryable=True)


class ZhipuService(LLMService):
    """智谱清言服务实现"""

    def _init_client(self) -> None:
        """初始化智谱客户端"""
        try:
            from zhipuai import ZhipuAI

            # 配置代理
            http_client = None
            if self.processing_config.use_proxy and self.processing_config.proxy_url:
                import httpx
                http_client = httpx.Client(proxies=self.processing_config.proxy_url)
                logger.debug(f"智谱客户端配置代理: {self.processing_config.proxy_url}")

            self.client = ZhipuAI(
                api_key=self.api_config.api_key,
                base_url=self.api_config.base_url,
                http_client=http_client
            )
            logger.info(f"智谱API客户端初始化成功 (模型: {self.api_config.model_name})")

        except ImportError:
            raise APIKeyError("未安装zhipuai库，请运行: pip install zhipuai")
        except Exception as e:
            raise APIKeyError(f"智谱API配置失败: {str(e)}")

    async def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        """调用智谱API"""
        try:
            # 智谱AI SDK是同步的，需要在executor中运行以避免阻塞事件循环
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.api_config.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=None,  # 让API自行决定
                    **kwargs
                )
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None
                } if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason
            )

        except Exception as e:
            error_str = str(e).lower()

            # 检查特定错误类型
            if "invalid_api_key" in error_str or "unauthorized" in error_str:
                raise APIKeyError("API密钥无效或已过期")

            if "rate" in error_str and "limit" in error_str:
                # 尝试从错误中提取重试时间
                retry_after = None
                # 智谱API可能会在响应中包含重试时间
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')
                    if retry_after:
                        retry_after = int(retry_after)
                raise RateLimitError("API速率限制", retry_after=retry_after)

            if "insufficient_quota" in error_str or "exceeded" in error_str:
                raise APIError("API配额已用尽", is_retryable=False)

            if "content_filter" in error_str or "sensitive" in error_str:
                raise APIError("内容被安全过滤器阻止", is_retryable=False)

            # 其他错误
            raise APIError(f"智谱API错误: {str(e)}", is_retryable=True)


def create_llm_service() -> LLMService:
    """工厂函数：创建LLM服务实例"""
    api_config = get_api_config()

    if api_config.provider == 'openai':
        return OpenAIService()
    elif api_config.provider == 'gemini':
        return GeminiService()
    elif api_config.provider == 'zhipu':
        return ZhipuService()
    else:
        raise ValueError(f"不支持的API提供商: {api_config.provider}")
