import sys
import types

import httpx
import pytest

import config
import services.llm_service as llm_service
from exceptions import APIError, RateLimitError
from services.llm_service import LLMResponse, LLMService


class DummyService(LLMService):
    def __init__(self, responses: list[LLMResponse | Exception]):
        self._responses: list[LLMResponse | Exception] = list(responses)
        self.call_count = 0
        super().__init__()

    def _init_client(self) -> None:
        self.client = object()

    async def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_call_success():
    service = DummyService([LLMResponse(content="ok")])
    service.processing_config.max_retry = 1
    response = await service.call("ping")
    assert response.content == "ok"
    assert service.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_retryable_error(monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)

    service = DummyService(
        [
            APIError("retry", is_retryable=True),
            LLMResponse(content="ok"),
        ]
    )
    service.processing_config.max_retry = 2

    response = await service.call("ping")
    assert response.content == "ok"
    assert service.call_count == 2
    assert sleep_calls


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after(monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_service.random, "uniform", lambda _a, _b: 0)

    service = DummyService(
        [
            RateLimitError("limit", retry_after=2),
            LLMResponse(content="ok"),
        ]
    )
    service.processing_config.max_retry = 2

    response = await service.call("ping")
    assert response.content == "ok"
    assert sleep_calls == [2]


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failure():
    service = DummyService([APIError("fatal", is_retryable=False)])
    service.processing_config.max_retry = 1
    service.circuit_breaker.failure_threshold = 1

    with pytest.raises(APIError):
        await service.call("ping")

    with pytest.raises(APIError) as excinfo:
        await service.call("ping again")

    assert excinfo.value.is_retryable is True


def test_openai_service_initialization_uses_http_client(monkeypatch):
    class DummyOpenAIClient:
        def __init__(self, api_key=None, base_url=None, http_client=None):
            self.api_key = api_key
            self.base_url = base_url
            self.http_client = http_client

    fake_openai = types.SimpleNamespace(AsyncOpenAI=DummyOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()
    assert isinstance(service.client, DummyOpenAIClient)
    assert service.client.api_key == "sk-test"
    assert service.client.http_client is not None


def test_default_providers_registered():
    providers = llm_service.get_registered_llm_providers()
    assert "openai" in providers
    assert "gemini" in providers
    assert "zhipu" in providers
    assert "aihubmix" in providers


def test_register_custom_provider(monkeypatch):
    class CustomService(LLMService):
        def _init_client(self) -> None:
            self.client = object()

        async def _call_api(self, prompt: str, **kwargs) -> LLMResponse:
            return LLMResponse(content="custom")

    llm_service.register_llm_provider("custom-provider")(CustomService)
    monkeypatch.setenv("API_PROVIDER", "custom-provider")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    service = llm_service.create_llm_service()
    assert isinstance(service, CustomService)


# ──────────────────────────── CircuitBreaker 状态机 ────────────────────────────


def test_circuit_breaker_closed_allows_calls():
    """CLOSED 状态允许调用"""
    cb = llm_service.CircuitBreaker(failure_threshold=3)
    assert cb.state == "CLOSED"
    assert cb.call_allowed() is True


def test_circuit_breaker_open_blocks_calls():
    """OPEN 状态且未超时时阻断调用"""

    cb = llm_service.CircuitBreaker(failure_threshold=1, timeout_seconds=3600)
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.call_allowed() is False


def test_circuit_breaker_open_to_half_open_after_timeout():
    """OPEN 状态超时后转为 HALF_OPEN"""
    from datetime import datetime, timedelta

    cb = llm_service.CircuitBreaker(failure_threshold=1, timeout_seconds=60)
    cb.record_failure()
    assert cb.state == "OPEN"

    # 模拟超时
    cb.last_failure_time = datetime.now() - timedelta(seconds=61)
    assert cb.call_allowed() is True
    assert cb.state == "HALF_OPEN"


def test_circuit_breaker_half_open_to_closed_on_success():
    """HALF_OPEN 状态成功后转为 CLOSED"""
    from datetime import datetime, timedelta

    cb = llm_service.CircuitBreaker(failure_threshold=1, timeout_seconds=60)
    cb.record_failure()
    cb.last_failure_time = datetime.now() - timedelta(seconds=61)
    cb.call_allowed()  # 触发 HALF_OPEN 转变
    assert cb.state == "HALF_OPEN"

    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0


def test_circuit_breaker_half_open_allows_calls():
    """HALF_OPEN 状态允许调用"""
    cb = llm_service.CircuitBreaker()
    cb.state = "HALF_OPEN"
    assert cb.call_allowed() is True


# ──────────────────────── call() 错误路径 ────────────────────────


@pytest.mark.asyncio
async def test_content_filter_error_not_recorded_in_circuit_breaker():
    """ContentFilterError 不触发熔断器计数"""
    service = DummyService([llm_service.ContentFilterError("blocked")])
    service.processing_config.max_retry = 1

    with pytest.raises(llm_service.ContentFilterError):
        await service.call("prompt")

    assert service.circuit_breaker.state == "CLOSED"
    assert service.circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_all_retries_exhausted_raises_api_error(monkeypatch):
    """所有重试都失败后抛出 APIError"""
    from exceptions import APIError

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)

    service = DummyService(
        [
            APIError("err", is_retryable=True),
            APIError("err", is_retryable=True),
        ]
    )
    service.processing_config.max_retry = 2

    with pytest.raises(APIError):
        await service.call("prompt")


@pytest.mark.asyncio
async def test_unknown_exception_triggers_circuit_breaker(monkeypatch):
    """未知异常触发熔断器记录失败"""

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)

    service = DummyService([RuntimeError("boom"), RuntimeError("boom")])
    service.processing_config.max_retry = 2
    service.circuit_breaker.failure_threshold = 100  # 防止自动打开

    with pytest.raises(Exception):  # noqa: B017
        await service.call("prompt")

    assert service.circuit_breaker.failure_count > 0


# ──────────────────────── create_llm_service ────────────────────────


def test_create_llm_service_unknown_provider_raises(monkeypatch):
    """未知 provider 应抛出 ValueError"""
    monkeypatch.setenv("API_PROVIDER", "unknown-xyz")
    config._api_config = None

    with pytest.raises(ValueError, match="不支持的API提供商"):
        llm_service.create_llm_service()


# ──────────────────────── OpenAIService._call_api ────────────────────────


@pytest.mark.asyncio
async def test_openai_call_api_success(monkeypatch):
    """OpenAIService._call_api 成功路径"""
    import types

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    # 构造伪造的 response 对象
    fake_usage = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 10})
    fake_message = types.SimpleNamespace(content="hello")
    fake_choice = types.SimpleNamespace(message=fake_message, finish_reason="stop")
    fake_response = types.SimpleNamespace(choices=[fake_choice], usage=fake_usage, model="gpt-4")

    class FakeCompletion:
        async def create(self, **kwargs):
            return fake_response

    class FakeChatClient:
        completions = FakeCompletion()

    class FakeOpenAIClient:
        chat = FakeChatClient()

        def __init__(self, api_key=None, base_url=None, http_client=None):
            pass

    fake_openai_mod = types.SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()
    response = await service._call_api("test prompt")

    assert response.content == "hello"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_call_api_auth_error(monkeypatch):
    """OpenAIService._call_api 认证错误转为 APIKeyError"""
    import types

    from exceptions import APIKeyError

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")
    config._api_config = None

    class FakeCompletion:
        async def create(self, **kwargs):
            raise RuntimeError("authentication failed")

    class FakeChatClient:
        completions = FakeCompletion()

    class FakeOpenAIClient:
        chat = FakeChatClient()

        def __init__(self, **kwargs):
            pass

    fake_openai_mod = types.SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()

    with pytest.raises(APIKeyError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_openai_call_api_rate_limit_error(monkeypatch):
    """OpenAIService._call_api 速率限制转为 RateLimitError"""
    import types

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    class FakeCompletion:
        async def create(self, **kwargs):
            raise RuntimeError("rate limit exceeded")

    class FakeChatClient:
        completions = FakeCompletion()

    class FakeOpenAIClient:
        chat = FakeChatClient()

        def __init__(self, **kwargs):
            pass

    fake_openai_mod = types.SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()

    with pytest.raises(RateLimitError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_openai_call_api_quota_error(monkeypatch):
    """OpenAIService._call_api 配额用尽转为非重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    class FakeCompletion:
        async def create(self, **kwargs):
            raise RuntimeError("quota exceeded")

    class FakeChatClient:
        completions = FakeCompletion()

    class FakeOpenAIClient:
        chat = FakeChatClient()

        def __init__(self, **kwargs):
            pass

    fake_openai_mod = types.SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()

    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is False


@pytest.mark.asyncio
async def test_openai_call_api_content_filter_421(monkeypatch):
    """OpenAIService._call_api 421 错误转为 ContentFilterError"""
    import types

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    class FakeCompletion:
        async def create(self, **kwargs):
            raise RuntimeError("421 content blocked")

    class FakeChatClient:
        completions = FakeCompletion()

    class FakeOpenAIClient:
        chat = FakeChatClient()

        def __init__(self, **kwargs):
            pass

    fake_openai_mod = types.SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()

    with pytest.raises(llm_service.ContentFilterError):
        await service._call_api("prompt")


# ──────────────────────── GeminiService._call_api ────────────────────────


def _make_fake_genai_types():
    """构造伪 google.generativeai.types"""
    import types

    return types.SimpleNamespace(
        HarmBlockThreshold=types.SimpleNamespace(
            BLOCK_NONE="BLOCK_NONE",
            BLOCK_ONLY_HIGH="BLOCK_ONLY_HIGH",
            BLOCK_MEDIUM_AND_ABOVE="BLOCK_MEDIUM_AND_ABOVE",
            BLOCK_LOW_AND_ABOVE="BLOCK_LOW_AND_ABOVE",
        ),
        HarmCategory=types.SimpleNamespace(
            HARM_CATEGORY_HARASSMENT="harassment",
            HARM_CATEGORY_HATE_SPEECH="hate",
            HARM_CATEGORY_SEXUALLY_EXPLICIT="sexual",
            HARM_CATEGORY_DANGEROUS_CONTENT="dangerous",
        ),
    )


def _make_fake_genai(response_obj):
    """构造伪 google.generativeai 模块（使用 response 对象的简便版本）"""
    import types

    class FakeModel:
        def generate_content(self, prompt, safety_settings=None):
            return response_obj

    return types.SimpleNamespace(
        configure=lambda api_key: None,
        GenerativeModel=lambda model_name: FakeModel(),
        types=_make_fake_genai_types(),
    )


def _make_fake_genai_with_model(model_instance):
    """构造伪 google.generativeai 模块（使用预创建的 model 实例）"""
    import types

    return types.SimpleNamespace(
        configure=lambda api_key: None,
        GenerativeModel=lambda model_name: model_instance,
        types=_make_fake_genai_types(),
    )


@pytest.mark.asyncio
async def test_gemini_call_api_success(monkeypatch):
    """GeminiService._call_api 成功路径"""
    import types

    monkeypatch.setenv("API_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "gemini-key")
    config._api_config = None

    fake_response = types.SimpleNamespace(text="gemini result", prompt_feedback=None)
    fake_genai = _make_fake_genai(fake_response)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    # 需要同时注入 google 包
    if "google" not in sys.modules:
        monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(generativeai=fake_genai))

    service = llm_service.GeminiService()
    response = await service._call_api("test prompt")

    assert response.content == "gemini result"


@pytest.mark.asyncio
async def test_gemini_call_api_empty_text_raises(monkeypatch):
    """Gemini 返回空内容时应抛出 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "gemini-key")
    config._api_config = None

    fake_response = types.SimpleNamespace(text="", prompt_feedback=None)
    fake_genai = _make_fake_genai(fake_response)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    service = llm_service.GeminiService()
    with pytest.raises(APIError, match="空内容"):
        await service._call_api("prompt")


# ──────────────────────── ZhipuService._call_api ────────────────────────


@pytest.mark.asyncio
async def test_zhipu_call_api_success(monkeypatch):
    """ZhipuService._call_api 成功路径"""
    import types

    monkeypatch.setenv("API_PROVIDER", "zhipu")
    monkeypatch.setenv("OPENAI_API_KEY", "zhipu-key")
    config._api_config = None

    fake_usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    fake_message = types.SimpleNamespace(content="zhipu result")
    fake_choice = types.SimpleNamespace(message=fake_message, finish_reason="stop")
    fake_response = types.SimpleNamespace(choices=[fake_choice], usage=fake_usage, model="glm-4")

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    class FakeChat:
        completions = FakeCompletions()

    class FakeZhipuAI:
        chat = FakeChat()

        def __init__(self, **kwargs):
            pass

    fake_zhipuai = types.SimpleNamespace(ZhipuAI=FakeZhipuAI)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)

    service = llm_service.ZhipuService()
    response = await service._call_api("test prompt")

    assert response.content == "zhipu result"
    assert response.token_usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_zhipu_call_api_auth_error(monkeypatch):
    """ZhipuService._call_api 认证错误转为 APIKeyError"""
    import types

    from exceptions import APIKeyError

    monkeypatch.setenv("API_PROVIDER", "zhipu")
    monkeypatch.setenv("OPENAI_API_KEY", "zhipu-bad")
    config._api_config = None

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("invalid_api_key error")

    class FakeChat:
        completions = FakeCompletions()

    class FakeZhipuAI:
        chat = FakeChat()

        def __init__(self, **kwargs):
            pass

    fake_zhipuai = types.SimpleNamespace(ZhipuAI=FakeZhipuAI)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)

    service = llm_service.ZhipuService()
    with pytest.raises(APIKeyError):
        await service._call_api("prompt")


# ──────────────────────── AiHubMixService._call_api ────────────────────────


@pytest.mark.asyncio
async def test_aihubmix_call_api_success(monkeypatch):
    """AiHubMixService._call_api 成功路径"""
    import types

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    fake_data = {
        "choices": [{"message": {"content": "aihubmix result"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 20},
        "model": "gpt-4",
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_data

    class FakeRequests:
        exceptions = types.SimpleNamespace(
            HTTPError=Exception,
            Timeout=Exception,
            ConnectionError=Exception,
            RequestException=Exception,
        )

        def post(self, *args, **kwargs):
            return FakeResponse()

    fake_requests = FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    service = llm_service.AiHubMixService()
    response = await service._call_api("test prompt")

    assert response.content == "aihubmix result"
    assert response.token_usage == {"total_tokens": 20}


@pytest.mark.asyncio
async def test_aihubmix_call_api_timeout(monkeypatch):
    """AiHubMixService._call_api 超时转为可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeTimeout(Exception):  # noqa: N818
        pass

    class FakeRequests:
        exceptions = types.SimpleNamespace(
            HTTPError=type("HTTPError", (Exception,), {}),
            Timeout=FakeTimeout,
            ConnectionError=type("ConnError", (Exception,), {}),
            RequestException=type("ReqError", (Exception,), {}),
        )

        def post(self, *args, **kwargs):
            raise FakeTimeout("timeout")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


@pytest.mark.asyncio
async def test_gemini_call_api_permission_error(monkeypatch):
    """Gemini permission error 转为 APIKeyError"""

    from exceptions import APIKeyError

    monkeypatch.setenv("API_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "gemini-key")
    config._api_config = None

    class FakeModel:
        def generate_content(self, prompt, safety_settings=None):
            raise RuntimeError("permission denied")

    fake_genai = _make_fake_genai_with_model(FakeModel())
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    service = llm_service.GeminiService()
    with pytest.raises(APIKeyError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_gemini_call_api_quota_error(monkeypatch):
    """Gemini quota error 转为不可重试 APIError"""

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "gemini-key")
    config._api_config = None

    class FakeModel:
        def generate_content(self, prompt, safety_settings=None):
            raise RuntimeError("quota exceeded")

    fake_genai = _make_fake_genai_with_model(FakeModel())
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    service = llm_service.GeminiService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is False


@pytest.mark.asyncio
async def test_zhipu_call_api_rate_limit_error(monkeypatch):
    """ZhipuService 速率限制转为 RateLimitError"""
    import types

    monkeypatch.setenv("API_PROVIDER", "zhipu")
    monkeypatch.setenv("OPENAI_API_KEY", "zhipu-key")
    config._api_config = None

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("rate limit reached")

    class FakeChat:
        completions = FakeCompletions()

    class FakeZhipuAI:
        chat = FakeChat()

        def __init__(self, **kwargs):
            pass

    fake_zhipuai = types.SimpleNamespace(ZhipuAI=FakeZhipuAI)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)

    service = llm_service.ZhipuService()
    with pytest.raises(RateLimitError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_zhipu_call_api_quota_error(monkeypatch):
    """ZhipuService 配额用尽转为不可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "zhipu")
    monkeypatch.setenv("OPENAI_API_KEY", "zhipu-key")
    config._api_config = None

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("insufficient_quota exceeded")

    class FakeChat:
        completions = FakeCompletions()

    class FakeZhipuAI:
        chat = FakeChat()

        def __init__(self, **kwargs):
            pass

    fake_zhipuai = types.SimpleNamespace(ZhipuAI=FakeZhipuAI)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)

    service = llm_service.ZhipuService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is False


@pytest.mark.asyncio
async def test_zhipu_call_api_content_filter_error(monkeypatch):
    """ZhipuService 内容过滤转为不可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "zhipu")
    monkeypatch.setenv("OPENAI_API_KEY", "zhipu-key")
    config._api_config = None

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("content_filter triggered sensitive")

    class FakeChat:
        completions = FakeCompletions()

    class FakeZhipuAI:
        chat = FakeChat()

        def __init__(self, **kwargs):
            pass

    fake_zhipuai = types.SimpleNamespace(ZhipuAI=FakeZhipuAI)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)

    service = llm_service.ZhipuService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is False


@pytest.mark.asyncio
async def test_aihubmix_call_api_auth_error(monkeypatch):
    """AiHubMixService 401 认证错误转为 APIKeyError"""
    import types

    from exceptions import APIKeyError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeHTTPError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            self.response = types.SimpleNamespace(status_code=401)

    class FakeRequests:
        class exceptions:  # noqa: N801
            HTTPError = FakeHTTPError
            Timeout = type("Timeout", (Exception,), {})
            ConnectionError = type("ConnError", (Exception,), {})
            RequestException = type("ReqError", (Exception,), {})

        def post(self, *args, **kwargs):
            raise FakeHTTPError("401 unauthorized")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIKeyError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_aihubmix_call_api_rate_limit(monkeypatch):
    """AiHubMixService 429 速率限制转为 RateLimitError"""
    import types

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeHTTPError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            self.response = types.SimpleNamespace(status_code=429, headers={})

    class FakeRequests:
        class exceptions:  # noqa: N801
            HTTPError = FakeHTTPError
            Timeout = type("Timeout", (Exception,), {})
            ConnectionError = type("ConnError", (Exception,), {})
            RequestException = type("ReqError", (Exception,), {})

        def post(self, *args, **kwargs):
            raise FakeHTTPError("429 rate limit")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(RateLimitError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_aihubmix_call_api_connection_error(monkeypatch):
    """AiHubMixService 连接错误转为可重试 APIError"""

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeConnectionError(Exception):
        pass

    class FakeRequests:
        class exceptions:  # noqa: N801
            HTTPError = type("HTTPError", (Exception,), {})
            Timeout = type("Timeout", (Exception,), {})
            ConnectionError = FakeConnectionError
            RequestException = type("ReqError", (Exception,), {})

        def post(self, *args, **kwargs):
            raise FakeConnectionError("connection refused")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


@pytest.mark.asyncio
async def test_openai_call_api_general_error(monkeypatch):
    """OpenAI 其他错误转为可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config._api_config = None

    class FakeCompletion:
        async def create(self, **kwargs):
            raise RuntimeError("some unexpected error")

    class FakeChatClient:
        completions = FakeCompletion()

    class FakeOpenAIClient:
        chat = FakeChatClient()

        def __init__(self, **kwargs):
            pass

    fake_openai_mod = types.SimpleNamespace(AsyncOpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    llm_service.OpenAIService._http_client = None
    llm_service.OpenAIService._proxy_clients = {}

    service = llm_service.OpenAIService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


@pytest.mark.asyncio
async def test_gemini_call_api_block_reason(monkeypatch):
    """Gemini block_reason 触发时抛出 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "gemini-key")
    config._api_config = None

    fake_feedback = types.SimpleNamespace(block_reason="SAFETY")
    fake_response = types.SimpleNamespace(text="", prompt_feedback=fake_feedback)
    fake_genai = _make_fake_genai(fake_response)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    service = llm_service.GeminiService()
    # block_reason 触发的 APIError 被外层 except 重包为可重试 APIError
    with pytest.raises(APIError):
        await service._call_api("prompt")


@pytest.mark.asyncio
async def test_gemini_call_api_safety_filter_error(monkeypatch):
    """Gemini safety filter 错误转为不可重试 APIError"""

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "gemini-key")
    config._api_config = None

    class FakeModel:
        def generate_content(self, prompt, safety_settings=None):
            raise RuntimeError("safety filter blocked")

    fake_genai = _make_fake_genai_with_model(FakeModel())
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    service = llm_service.GeminiService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is False


@pytest.mark.asyncio
async def test_zhipu_call_api_general_error(monkeypatch):
    """ZhipuService 未知错误转为可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "zhipu")
    monkeypatch.setenv("OPENAI_API_KEY", "zhipu-key")
    config._api_config = None

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("something unexpected happened")

    class FakeChat:
        completions = FakeCompletions()

    class FakeZhipuAI:
        chat = FakeChat()

        def __init__(self, **kwargs):
            pass

    fake_zhipuai = types.SimpleNamespace(ZhipuAI=FakeZhipuAI)
    monkeypatch.setitem(sys.modules, "zhipuai", fake_zhipuai)

    service = llm_service.ZhipuService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


@pytest.mark.asyncio
async def test_aihubmix_call_api_quota_error(monkeypatch):
    """AiHubMixService 402 配额错误转为不可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeHTTPError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            self.response = types.SimpleNamespace(status_code=402)

    class FakeRequests:
        class exceptions:  # noqa: N801
            HTTPError = FakeHTTPError
            Timeout = type("Timeout", (Exception,), {})
            ConnectionError = type("ConnError", (Exception,), {})
            RequestException = type("ReqError", (Exception,), {})

        def post(self, *args, **kwargs):
            raise FakeHTTPError("402 quota exceeded")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is False


@pytest.mark.asyncio
async def test_aihubmix_call_api_other_http_error(monkeypatch):
    """AiHubMixService 500 等其他 HTTP 错误转为可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeHTTPError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            self.response = types.SimpleNamespace(status_code=500)

    class FakeRequests:
        class exceptions:  # noqa: N801
            HTTPError = FakeHTTPError
            Timeout = type("Timeout", (Exception,), {})
            ConnectionError = type("ConnError", (Exception,), {})
            RequestException = type("ReqError", (Exception,), {})

        def post(self, *args, **kwargs):
            raise FakeHTTPError("500 internal server error")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


@pytest.mark.asyncio
async def test_aihubmix_call_api_request_exception(monkeypatch):
    """AiHubMixService RequestException 转为可重试 APIError"""

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeRequestException(Exception):  # noqa: N818
        pass

    class FakeRequests:
        class exceptions:  # noqa: N801
            HTTPError = type("HTTPError", (Exception,), {})
            Timeout = type("Timeout", (Exception,), {})
            ConnectionError = type("ConnError", (Exception,), {})
            RequestException = FakeRequestException

        def post(self, *args, **kwargs):
            raise FakeRequestException("request failed")

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


@pytest.mark.asyncio
async def test_aihubmix_call_api_key_error(monkeypatch):
    """AiHubMixService 响应缺少字段时转为可重试 APIError"""
    import types

    from exceptions import APIError

    monkeypatch.setenv("API_PROVIDER", "aihubmix")
    monkeypatch.setenv("OPENAI_API_KEY", "aihub-key")
    monkeypatch.setenv("AIHUBMIX_API_KEY", "real-aihubmix-key-12345")
    config._api_config = None

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {}  # 缺少 "choices" 字段 → KeyError

    class FakeRequests:
        exceptions = types.SimpleNamespace(
            HTTPError=type("HTTPError", (Exception,), {}),
            Timeout=type("Timeout", (Exception,), {}),
            ConnectionError=type("ConnError", (Exception,), {}),
            RequestException=type("ReqError", (Exception,), {}),
        )

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests())

    service = llm_service.AiHubMixService()
    with pytest.raises(APIError) as exc:
        await service._call_api("prompt")
    assert exc.value.is_retryable is True


# ──────────────────────── httpx proxy helper ────────────────────────


def test_httpx_proxy_kwargs_with_proxy_param():
    """_httpx_proxy_kwargs 应返回 proxy 或 proxies 关键字"""
    result = llm_service._httpx_proxy_kwargs(httpx.AsyncClient, "http://proxy:8080")
    assert "proxy" in result or "proxies" in result


def test_httpx_proxy_kwargs_unknown_class_returns_empty():
    """不支持代理参数的类应返回空字典"""

    class NoProxyClass:
        def __init__(self):
            pass

    result = llm_service._httpx_proxy_kwargs(NoProxyClass, "http://proxy:8080")
    assert result == {}


# ──────────────────────── ContentFilterError ────────────────────────


def test_content_filter_error_is_not_retryable():
    """ContentFilterError 应为不可重试的 APIError"""
    from exceptions import APIError

    err = llm_service.ContentFilterError("blocked")
    assert isinstance(err, APIError)
    assert err.is_retryable is False
