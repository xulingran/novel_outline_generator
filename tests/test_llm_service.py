import sys
import types

import pytest

import config
import services.llm_service as llm_service
from exceptions import APIError, RateLimitError
from services.llm_service import LLMResponse, LLMService


class DummyService(LLMService):
    def __init__(self, responses):
        self._responses = list(responses)
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
    assert await service.call("ping") == "ok"
    assert service.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_retryable_error(monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)

    service = DummyService([
        APIError("retry", is_retryable=True),
        LLMResponse(content="ok"),
    ])
    service.processing_config.max_retry = 2

    assert await service.call("ping") == "ok"
    assert service.call_count == 2
    assert sleep_calls


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after(monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_service.random, "uniform", lambda _a, _b: 0)

    service = DummyService([
        RateLimitError("limit", retry_after=2),
        LLMResponse(content="ok"),
    ])
    service.processing_config.max_retry = 2

    assert await service.call("ping") == "ok"
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
