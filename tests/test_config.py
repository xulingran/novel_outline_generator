"""
测试配置管理模块
"""

import os
from unittest.mock import patch

import pytest

from config import (
    API_KEY,
    API_PROVIDER,
    SUPPORTED_API_PROVIDERS,
    APIConfig,
    ProcessingConfig,
    _APIKeyWrapper,
    create_env_file,
    get_api_config,
    get_processing_config,
    init_config,
)
from exceptions import ConfigurationError


class TestAPIConfig:
    """测试APIConfig类"""

    def test_api_config_initialization_with_defaults(self):
        """测试APIConfig使用默认值初始化"""
        with patch.dict(os.environ, {}, clear=True):
            config = APIConfig()
            assert config.provider == "openai"
            assert config.openai_key is None
            assert config.openai_base is None
            assert config.openai_model == "gpt-4o-mini"
            assert config.gemini_key is None
            assert config.gemini_model == "gemini-2.5-flash"
            assert config.gemini_safety == "BLOCK_NONE"
            assert config.zhipu_key is None
            assert config.zhipu_base == "https://open.bigmodel.cn/api/paas/v4"
            assert config.zhipu_model == "glm-4-flash"
            assert config.aihubmix_api_key is None
            assert config.aihubmix_model == "gpt-3.5-turbo"
            assert config.aihubmix_api_base == "https://aihubmix.com/v1"

    def test_api_config_initialization_from_env(self):
        """测试APIConfig从环境变量初始化"""
        with patch.dict(
            os.environ,
            {
                "API_PROVIDER": "gemini",
                "OPENAI_API_KEY": "sk-test1",
                "OPENAI_API_BASE": "https://custom.openai.com",
                "OPENAI_MODEL": "gpt-4",
                "GEMINI_API_KEY": "gemini-test",
                "GEMINI_MODEL": "gemini-pro",
                "GEMINI_SAFETY_SETTINGS": "BLOCK_MEDIUM",
                "ZHIPU_API_KEY": "zhipu-test",
                "ZHIPU_API_BASE": "https://custom.zhipu.com",
                "ZHIPU_MODEL": "glm-4",
                "AIHUBMIX_API_KEY": "aihubmix-test",
                "AIHUBMIX_MODEL": "gpt-4",
                "AIHUBMIX_API_BASE": "https://custom.aihubmix.com",
            },
        ):
            config = APIConfig()
            assert config.provider == "gemini"
            assert config.openai_key == "sk-test1"
            assert config.openai_base == "https://custom.openai.com"
            assert config.openai_model == "gpt-4"
            assert config.gemini_key == "gemini-test"
            assert config.gemini_model == "gemini-pro"
            assert config.gemini_safety == "BLOCK_MEDIUM"
            assert config.zhipu_key == "zhipu-test"
            assert config.zhipu_base == "https://custom.zhipu.com"
            assert config.zhipu_model == "glm-4"
            assert config.aihubmix_api_key == "aihubmix-test"
            assert config.aihubmix_model == "gpt-4"
            assert config.aihubmix_api_base == "https://custom.aihubmix.com"

    def test_validate_unsupported_provider(self):
        """测试验证不支持的API提供商"""
        with patch.dict(os.environ, {"API_PROVIDER": "unsupported"}):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="不支持的API提供商"):
                config.validate()

    def test_validate_openai_missing_key(self):
        """测试验证OpenAI缺少密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": ""}):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="使用OpenAI API时必须设置OPENAI_API_KEY"):
                config.validate()

    def test_validate_openai_placeholder_key(self):
        """测试验证OpenAI占位符密钥"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "your_api_key_here"}
        ):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="当前值看起来像是占位符"):
                config.validate()

    def test_validate_openai_valid_key(self):
        """测试验证OpenAI有效密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-abc123"}):
            config = APIConfig()
            config.validate()
            assert config._validated is True

    def test_validate_gemini_missing_key(self):
        """测试验证Gemini缺少密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "gemini", "GEMINI_API_KEY": ""}):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="使用Gemini API时必须设置GEMINI_API_KEY"):
                config.validate()

    def test_validate_gemini_placeholder_key(self):
        """测试验证Gemini占位符密钥"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "gemini", "GEMINI_API_KEY": "your_gemini_key_here"}
        ):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="当前值看起来像是占位符"):
                config.validate()

    def test_validate_gemini_valid_key(self):
        """测试验证Gemini有效密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "gemini", "GEMINI_API_KEY": "AIzaSyTestKey"}):
            config = APIConfig()
            config.validate()
            assert config._validated is True

    def test_validate_zhipu_missing_key(self):
        """测试验证智谱缺少密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_API_KEY": ""}):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="使用智谱API时必须设置ZHIPU_API_KEY"):
                config.validate()

    def test_validate_zhipu_placeholder_key(self):
        """测试验证智谱占位符密钥"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_API_KEY": "your_zhipu_key_here"}
        ):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="当前值看起来像是占位符"):
                config.validate()

    def test_validate_zhipu_valid_key(self):
        """测试验证智谱有效密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_API_KEY": "zhipu.test.key"}):
            config = APIConfig()
            config.validate()
            assert config._validated is True

    def test_validate_aihubmix_missing_key(self):
        """测试验证AiHubMix缺少密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "aihubmix", "AIHUBMIX_API_KEY": ""}):
            config = APIConfig()
            with pytest.raises(
                ConfigurationError, match="使用AiHubMix API时必须设置AIHUBMIX_API_KEY"
            ):
                config.validate()

    def test_validate_aihubmix_placeholder_key(self):
        """测试验证AiHubMix占位符密钥"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "aihubmix", "AIHUBMIX_API_KEY": "your_aihubmix_key_here"}
        ):
            config = APIConfig()
            with pytest.raises(ConfigurationError, match="当前值看起来像是占位符"):
                config.validate()

    def test_validate_aihubmix_valid_key(self):
        """测试验证AiHubMix有效密钥"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "aihubmix", "AIHUBMIX_API_KEY": "aihubmix.test.key"}
        ):
            config = APIConfig()
            config.validate()
            assert config._validated is True

    def test_validate_already_validated(self):
        """测试验证已验证的配置"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            config = APIConfig()
            config.validate()
            # 第二次调用应该不会再次验证
            config.validate()
            assert config._validated is True

    def test_api_key_property_openai(self):
        """测试获取OpenAI API密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            config = APIConfig()
            assert config.api_key == "sk-test"

    def test_api_key_property_openai_missing(self):
        """测试获取OpenAI API密钥（缺失）"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": ""}):
            config = APIConfig()
            # validate会先检查并抛出ConfigurationError
            with pytest.raises(ConfigurationError, match="使用OpenAI API时必须设置OPENAI_API_KEY"):
                _ = config.api_key

    def test_api_key_property_gemini(self):
        """测试获取Gemini API密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "gemini", "GEMINI_API_KEY": "AIzaSyTest"}):
            config = APIConfig()
            assert config.api_key == "AIzaSyTest"

    def test_api_key_property_gemini_missing(self):
        """测试获取Gemini API密钥（缺失）"""
        with patch.dict(os.environ, {"API_PROVIDER": "gemini", "GEMINI_API_KEY": ""}):
            config = APIConfig()
            # validate会先检查并抛出ConfigurationError
            with pytest.raises(ConfigurationError, match="使用Gemini API时必须设置GEMINI_API_KEY"):
                _ = config.api_key

    def test_api_key_property_zhipu(self):
        """测试获取智谱API密钥"""
        with patch.dict(os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_API_KEY": "zhipu.test"}):
            config = APIConfig()
            assert config.api_key == "zhipu.test"

    def test_api_key_property_zhipu_missing(self):
        """测试获取智谱API密钥（缺失）"""
        with patch.dict(os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_API_KEY": ""}):
            config = APIConfig()
            # validate会先检查并抛出ConfigurationError
            with pytest.raises(ConfigurationError, match="使用智谱API时必须设置ZHIPU_API_KEY"):
                _ = config.api_key

    def test_api_key_property_aihubmix(self):
        """测试获取AiHubMix API密钥"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "aihubmix", "AIHUBMIX_API_KEY": "aihubmix.test"}
        ):
            config = APIConfig()
            assert config.api_key == "aihubmix.test"

    def test_api_key_property_aihubmix_missing(self):
        """测试获取AiHubMix API密钥（缺失）"""
        with patch.dict(os.environ, {"API_PROVIDER": "aihubmix", "AIHUBMIX_API_KEY": ""}):
            config = APIConfig()
            # validate会先检查并抛出ConfigurationError
            with pytest.raises(
                ConfigurationError, match="使用AiHubMix API时必须设置AIHUBMIX_API_KEY"
            ):
                _ = config.api_key

    def test_base_url_property_openai(self):
        """测试获取OpenAI基础URL"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "openai", "OPENAI_API_BASE": "https://custom.com"}
        ):
            config = APIConfig()
            assert config.base_url == "https://custom.com"

    def test_base_url_property_openai_none(self):
        """测试获取OpenAI基础URL（None）"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai"}, clear=True):
            config = APIConfig()
            assert config.base_url is None

    def test_base_url_property_gemini(self):
        """测试获取Gemini基础URL"""
        with patch.dict(os.environ, {"API_PROVIDER": "gemini"}):
            config = APIConfig()
            assert config.base_url is None

    def test_base_url_property_zhipu(self):
        """测试获取智谱基础URL"""
        with patch.dict(
            os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_API_BASE": "https://custom.zhipu.com"}
        ):
            config = APIConfig()
            assert config.base_url == "https://custom.zhipu.com"

    def test_base_url_property_zhipu_default(self):
        """测试获取智谱基础URL（默认）"""
        with patch.dict(os.environ, {"API_PROVIDER": "zhipu"}, clear=True):
            config = APIConfig()
            assert config.base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_base_url_property_aihubmix(self):
        """测试获取AiHubMix基础URL"""
        with patch.dict(
            os.environ,
            {"API_PROVIDER": "aihubmix", "AIHUBMIX_API_BASE": "https://custom.aihubmix.com"},
        ):
            config = APIConfig()
            assert config.base_url == "https://custom.aihubmix.com"

    def test_base_url_property_aihubmix_default(self):
        """测试获取AiHubMix基础URL（默认）"""
        with patch.dict(os.environ, {"API_PROVIDER": "aihubmix"}, clear=True):
            config = APIConfig()
            assert config.base_url == "https://aihubmix.com/v1"

    def test_model_name_property_openai(self):
        """测试获取OpenAI模型名称"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_MODEL": "gpt-4"}):
            config = APIConfig()
            assert config.model_name == "gpt-4"

    def test_model_name_property_gemini(self):
        """测试获取Gemini模型名称"""
        with patch.dict(os.environ, {"API_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-pro"}):
            config = APIConfig()
            assert config.model_name == "gemini-pro"

    def test_model_name_property_zhipu(self):
        """测试获取智谱模型名称"""
        with patch.dict(os.environ, {"API_PROVIDER": "zhipu", "ZHIPU_MODEL": "glm-4"}):
            config = APIConfig()
            assert config.model_name == "glm-4"

    def test_model_name_property_aihubmix(self):
        """测试获取AiHubMix模型名称"""
        with patch.dict(os.environ, {"API_PROVIDER": "aihubmix", "AIHUBMIX_MODEL": "gpt-4"}):
            config = APIConfig()
            assert config.model_name == "gpt-4"


class TestProcessingConfig:
    """测试ProcessingConfig类"""

    def test_processing_config_initialization_with_defaults(self):
        """测试ProcessingConfig使用默认值初始化"""
        with patch.dict(os.environ, {}, clear=True):
            config = ProcessingConfig()
            assert config.default_txt_file == "novel.txt"
            assert config.output_dir == "outputs"
            assert config.progress_file == os.path.join("outputs", "progress.json")
            assert config.encodings == [
                "utf-8",
                "gbk",
                "gb2312",
                "gb18030",
                "big5",
                "utf-16",
                "utf-16-le",
                "utf-16-be",
                "latin1",
                "cp1252",
            ]
            assert config.model_max_tokens == 200000
            assert config.target_tokens_per_chunk == 64000
            assert config.parallel_limit == 5
            assert config.max_retry == 5
            assert config.log_every == 1
            assert config.use_proxy is False
            assert config.proxy_url == "http://127.0.0.1:7897"

    def test_processing_config_initialization_from_env(self):
        """测试ProcessingConfig从环境变量初始化"""
        with patch.dict(
            os.environ,
            {
                "MODEL_MAX_TOKENS": "300000",
                "TARGET_TOKENS_PER_CHUNK": "8000",
                "PARALLEL_LIMIT": "10",
                "MAX_RETRY": "3",
                "LOG_EVERY": "5",
                "USE_PROXY": "true",
                "PROXY_URL": "http://proxy.example.com:8080",
            },
        ):
            config = ProcessingConfig()
            assert config.model_max_tokens == 300000
            assert config.target_tokens_per_chunk == 8000
            assert config.parallel_limit == 10
            assert config.max_retry == 3
            assert config.log_every == 5
            assert config.use_proxy is True
            assert config.proxy_url == "http://proxy.example.com:8080"

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = ProcessingConfig()
        config.validate()  # 不应该抛出异常

    def test_validate_model_max_tokens_zero(self):
        """测试验证model_max_tokens为0"""
        with patch.dict(os.environ, {"MODEL_MAX_TOKENS": "0"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="MODEL_MAX_TOKENS必须大于0"):
                config.validate()

    def test_validate_model_max_tokens_negative(self):
        """测试验证model_max_tokens为负数"""
        with patch.dict(os.environ, {"MODEL_MAX_TOKENS": "-100"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="MODEL_MAX_TOKENS必须大于0"):
                config.validate()

    def test_validate_target_tokens_per_chunk_zero(self):
        """测试验证target_tokens_per_chunk为0"""
        with patch.dict(os.environ, {"TARGET_TOKENS_PER_CHUNK": "0"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="TARGET_TOKENS_PER_CHUNK必须大于0"):
                config.validate()

    def test_validate_target_tokens_per_chunk_negative(self):
        """测试验证target_tokens_per_chunk为负数"""
        with patch.dict(os.environ, {"TARGET_TOKENS_PER_CHUNK": "-100"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="TARGET_TOKENS_PER_CHUNK必须大于0"):
                config.validate()

    def test_validate_target_tokens_per_chunk_greater_than_model_max_tokens(self):
        """测试验证target_tokens_per_chunk大于model_max_tokens"""
        with patch.dict(
            os.environ, {"MODEL_MAX_TOKENS": "10000", "TARGET_TOKENS_PER_CHUNK": "15000"}
        ):
            config = ProcessingConfig()
            with pytest.raises(
                ConfigurationError, match="TARGET_TOKENS_PER_CHUNK必须小于MODEL_MAX_TOKENS"
            ):
                config.validate()

    def test_validate_target_tokens_per_chunk_equal_to_model_max_tokens(self):
        """测试验证target_tokens_per_chunk等于model_max_tokens"""
        with patch.dict(
            os.environ, {"MODEL_MAX_TOKENS": "10000", "TARGET_TOKENS_PER_CHUNK": "10000"}
        ):
            config = ProcessingConfig()
            with pytest.raises(
                ConfigurationError, match="TARGET_TOKENS_PER_CHUNK必须小于MODEL_MAX_TOKENS"
            ):
                config.validate()

    def test_validate_parallel_limit_zero(self):
        """测试验证parallel_limit为0"""
        with patch.dict(os.environ, {"PARALLEL_LIMIT": "0"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="PARALLEL_LIMIT必须大于0"):
                config.validate()

    def test_validate_parallel_limit_negative(self):
        """测试验证parallel_limit为负数"""
        with patch.dict(os.environ, {"PARALLEL_LIMIT": "-5"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="PARALLEL_LIMIT必须大于0"):
                config.validate()

    def test_validate_max_retry_negative(self):
        """测试验证max_retry为负数"""
        with patch.dict(os.environ, {"MAX_RETRY": "-1"}):
            config = ProcessingConfig()
            with pytest.raises(ConfigurationError, match="MAX_RETRY不能小于0"):
                config.validate()

    def test_validate_max_retry_zero(self):
        """测试验证max_retry为0（有效）"""
        with patch.dict(os.environ, {"MAX_RETRY": "0"}):
            config = ProcessingConfig()
            config.validate()  # 不应该抛出异常


class TestGetAPIConfig:
    """测试get_api_config函数"""

    def test_get_api_config_singleton(self):
        """测试get_api_config单例模式"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            config1 = get_api_config()
            config2 = get_api_config()
            assert config1 is config2

    def test_get_api_config_different_instances(self):
        """测试get_api_config不同实例"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            config1 = get_api_config()
            # 修改全局变量后，应该返回新实例
            import config

            config._api_config = None
            config2 = get_api_config()
            assert config1 is not config2


class TestGetProcessingConfig:
    """测试get_processing_config函数"""

    def test_get_processing_config_singleton(self):
        """测试get_processing_config单例模式"""
        config1 = get_processing_config()
        config2 = get_processing_config()
        assert config1 is config2


class TestSupportedAPIProviders:
    """测试SUPPORTED_API_PROVIDERS常量"""

    def test_supported_api_providers(self):
        """测试支持的API提供商列表"""
        assert isinstance(SUPPORTED_API_PROVIDERS, list)
        assert "openai" in SUPPORTED_API_PROVIDERS
        assert "gemini" in SUPPORTED_API_PROVIDERS
        assert "zhipu" in SUPPORTED_API_PROVIDERS
        assert "aihubmix" in SUPPORTED_API_PROVIDERS


class TestAPIKeyWrapper:
    """测试_APIKeyWrapper类"""

    def test_api_key_wrapper_str(self):
        """测试_APIKeyWrapper的__str__方法"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            wrapper = _APIKeyWrapper()
            assert str(wrapper) == "sk-test"

    def test_api_key_wrapper_repr(self):
        """测试_APIKeyWrapper的__repr__方法"""
        with patch.dict(os.environ, {"API_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            wrapper = _APIKeyWrapper()
            assert repr(wrapper) == "'sk-test'"


class TestCreateEnvFile:
    """测试create_env_file函数"""

    def test_create_env_file_when_not_exists(self, tmp_path, monkeypatch):
        """测试创建.env文件（文件不存在）"""
        monkeypatch.chdir(tmp_path)
        create_env_file()
        env_file = tmp_path / ".env"
        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert "API_PROVIDER=openai" in content
        assert "OPENAI_API_KEY=your_openai_api_key_here" in content

    def test_create_env_file_when_exists(self, tmp_path, monkeypatch):
        """测试创建.env文件（文件已存在）"""
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("existing content", encoding="utf-8")
        create_env_file()
        # 文件应该保持不变
        assert env_file.read_text(encoding="utf-8") == "existing content"


class TestInitConfig:
    """测试init_config函数"""

    def test_init_config_with_dotenv(self, monkeypatch):
        """测试init_config加载.env文件"""
        # 创建.env文件
        env_content = """
API_PROVIDER=gemini
GEMINI_API_KEY=test_gemini_key
MODEL_MAX_TOKENS=300000
"""
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.__iter__ = lambda self: iter(
                env_content.splitlines()
            )
            mock_open.return_value.__enter__.return_value.__next__ = lambda self: next(
                iter(env_content.splitlines())
            )

            init_config(create_env_if_missing=False)

    def test_init_config_without_dotenv(self, monkeypatch):
        """测试init_config没有.env文件"""
        monkeypatch.chdir("/tmp")
        init_config(create_env_if_missing=False)


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_api_provider_constant(self):
        """测试API_PROVIDER常量"""
        assert isinstance(API_PROVIDER, str)
        assert API_PROVIDER in SUPPORTED_API_PROVIDERS

    def test_api_key_wrapper(self):
        """测试API_KEY包装器"""
        assert isinstance(API_KEY, _APIKeyWrapper)

    def test_get_txt_file(self):
        """测试get_txt_file函数"""
        from config import get_txt_file

        assert get_txt_file() == get_processing_config().default_txt_file

    def test_get_output_dir(self):
        """测试get_output_dir函数"""
        from config import get_output_dir

        assert get_output_dir() == get_processing_config().output_dir

    def test_get_progress_file(self):
        """测试get_progress_file函数"""
        from config import get_progress_file

        assert get_progress_file() == get_processing_config().progress_file

    def test_get_encodings(self):
        """测试get_encodings函数"""
        from config import get_encodings

        assert get_encodings() == get_processing_config().encodings
