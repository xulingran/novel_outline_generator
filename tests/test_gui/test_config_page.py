"""
测试 ConfigPage 关键逻辑。
"""

from unittest.mock import MagicMock

from gui.pages.config_page import ConfigPage
from tests.test_gui.conftest import ctk


class TestConfigPageProvider:
    """测试 provider 处理逻辑。"""

    def test_normalize_provider_handles_case(self):
        """应将不同大小写统一为支持的小写值。"""
        page = object.__new__(ConfigPage)

        assert page._normalize_provider("OpenAI") == "openai"
        assert page._normalize_provider("GEMINI") == "gemini"
        assert page._normalize_provider("zhipu") == "zhipu"
        assert page._normalize_provider("AiHubMix") == "aihubmix"

    def test_normalize_provider_fallback(self):
        """非法 provider 应回退到 openai。"""
        page = object.__new__(ConfigPage)

        assert page._normalize_provider("unknown") == "openai"
        assert page._normalize_provider(None) == "openai"

    def test_on_provider_change_normalizes_before_update(self):
        """切换 provider 时应先归一化再刷新字段。"""
        page = object.__new__(ConfigPage)
        page._provider_var = ctk.StringVar(value="OpenAI")
        page._update_api_fields = MagicMock()

        page._on_provider_change()

        assert page._provider_var.get() == "openai"
        page._update_api_fields.assert_called_once()
