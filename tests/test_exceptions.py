"""
测试自定义异常类模块
"""

import pytest

from exceptions import (
    APIError,
    APIKeyError,
    ConfigurationError,
    EncodingError,
    FileValidationError,
    NovelOutlineError,
    ProcessingError,
    RateLimitError,
    TokenLimitError,
)


class TestNovelOutlineError:
    """测试NovelOutlineError基础异常类"""

    def test_novel_outline_error_with_message(self):
        """测试创建带有消息的NovelOutlineError"""
        error = NovelOutlineError("测试错误消息")
        assert str(error) == "测试错误消息"
        assert error.message == "测试错误消息"
        assert error.details is None

    def test_novel_outline_error_with_message_and_details(self):
        """测试创建带有消息和详情的NovelOutlineError"""
        error = NovelOutlineError("测试错误消息", details="详细错误信息")
        assert str(error) == "测试错误消息"
        assert error.message == "测试错误消息"
        assert error.details == "详细错误信息"

    def test_novel_outline_error_with_none_details(self):
        """测试创建带有None详情的NovelOutlineError"""
        error = NovelOutlineError("测试错误消息", details=None)
        assert error.details is None

    def test_novel_outline_error_is_exception(self):
        """测试NovelOutlineError是Exception的子类"""
        error = NovelOutlineError("测试")
        assert isinstance(error, Exception)

    def test_novel_outline_error_can_be_raised_and_caught(self):
        """测试NovelOutlineError可以被抛出和捕获"""
        with pytest.raises(NovelOutlineError) as exc_info:
            raise NovelOutlineError("测试错误")

        assert str(exc_info.value) == "测试错误"
        assert exc_info.value.message == "测试错误"


class TestAPIKeyError:
    """测试APIKeyError异常类"""

    def test_api_key_error_inheritance(self):
        """测试APIKeyError继承自NovelOutlineError"""
        error = APIKeyError("API密钥错误")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_api_key_error_message(self):
        """测试APIKeyError的消息"""
        error = APIKeyError("API密钥未配置")
        assert str(error) == "API密钥未配置"
        assert error.message == "API密钥未配置"

    def test_api_key_error_with_details(self):
        """测试APIKeyError带有详情"""
        error = APIKeyError("API密钥错误", details="请检查.env文件")
        assert error.details == "请检查.env文件"

    def test_api_key_error_can_be_raised_and_caught(self):
        """测试APIKeyError可以被抛出和捕获"""
        with pytest.raises(APIKeyError) as exc_info:
            raise APIKeyError("API密钥错误")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestConfigurationError:
    """测试ConfigurationError异常类"""

    def test_configuration_error_inheritance(self):
        """测试ConfigurationError继承自NovelOutlineError"""
        error = ConfigurationError("配置错误")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_configuration_error_message(self):
        """测试ConfigurationError的消息"""
        error = ConfigurationError("配置无效")
        assert str(error) == "配置无效"
        assert error.message == "配置无效"

    def test_configuration_error_with_details(self):
        """测试ConfigurationError带有详情"""
        error = ConfigurationError("配置错误", details="MODEL_MAX_TOKENS必须大于0")
        assert error.details == "MODEL_MAX_TOKENS必须大于0"

    def test_configuration_error_can_be_raised_and_caught(self):
        """测试ConfigurationError可以被抛出和捕获"""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("配置错误")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestFileValidationError:
    """测试FileValidationError异常类"""

    def test_file_validation_error_inheritance(self):
        """测试FileValidationError继承自NovelOutlineError"""
        error = FileValidationError("文件验证失败")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_file_validation_error_message(self):
        """测试FileValidationError的消息"""
        error = FileValidationError("文件格式不正确")
        assert str(error) == "文件格式不正确"
        assert error.message == "文件格式不正确"

    def test_file_validation_error_with_details(self):
        """测试FileValidationError带有详情"""
        error = FileValidationError("文件验证失败", details="不支持的文件格式")
        assert error.details == "不支持的文件格式"

    def test_file_validation_error_can_be_raised_and_caught(self):
        """测试FileValidationError可以被抛出和捕获"""
        with pytest.raises(FileValidationError) as exc_info:
            raise FileValidationError("文件验证失败")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestProcessingError:
    """测试ProcessingError异常类"""

    def test_processing_error_inheritance(self):
        """测试ProcessingError继承自NovelOutlineError"""
        error = ProcessingError("处理失败")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_processing_error_message(self):
        """测试ProcessingError的消息"""
        error = ProcessingError("处理过程中发生错误")
        assert str(error) == "处理过程中发生错误"
        assert error.message == "处理过程中发生错误"

    def test_processing_error_with_details(self):
        """测试ProcessingError带有详情"""
        error = ProcessingError("处理失败", details="块ID: 5")
        assert error.details == "块ID: 5"

    def test_processing_error_can_be_raised_and_caught(self):
        """测试ProcessingError可以被抛出和捕获"""
        with pytest.raises(ProcessingError) as exc_info:
            raise ProcessingError("处理失败")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestAPIError:
    """测试APIError异常类"""

    def test_api_error_inheritance(self):
        """测试APIError继承自NovelOutlineError"""
        error = APIError("API调用失败")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_api_error_message(self):
        """测试APIError的消息"""
        error = APIError("API调用失败")
        assert str(error) == "API调用失败"
        assert error.message == "API调用失败"

    def test_api_error_with_error_code(self):
        """测试APIError带有错误代码"""
        error = APIError("API调用失败", error_code="rate_limit_exceeded")
        assert error.error_code == "rate_limit_exceeded"

    def test_api_error_with_is_retryable(self):
        """测试APIError带有is_retryable标志"""
        error = APIError("API调用失败", is_retryable=True)
        assert error.is_retryable is True

    def test_api_error_default_values(self):
        """测试APIError的默认值"""
        error = APIError("API调用失败")
        assert error.error_code is None
        assert error.is_retryable is False

    def test_api_error_with_all_parameters(self):
        """测试APIError带有所有参数"""
        error = APIError("API调用失败", error_code="401", is_retryable=False)
        assert error.message == "API调用失败"
        assert error.error_code == "401"
        assert error.is_retryable is False

    def test_api_error_can_be_raised_and_caught(self):
        """测试APIError可以被抛出和捕获"""
        with pytest.raises(APIError) as exc_info:
            raise APIError("API调用失败")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestRateLimitError:
    """测试RateLimitError异常类"""

    def test_rate_limit_error_inheritance(self):
        """测试RateLimitError继承自APIError"""
        error = RateLimitError("速率限制")
        assert isinstance(error, APIError)
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_rate_limit_error_message(self):
        """测试RateLimitError的消息"""
        error = RateLimitError("速率限制")
        assert str(error) == "速率限制"
        assert error.message == "速率限制"

    def test_rate_limit_error_with_retry_after(self):
        """测试RateLimitError带有retry_after"""
        error = RateLimitError("速率限制", retry_after=60)
        assert error.retry_after == 60

    def test_rate_limit_error_default_retry_after(self):
        """测试RateLimitError的默认retry_after"""
        error = RateLimitError("速率限制")
        assert error.retry_after is None

    def test_rate_limit_error_is_retryable(self):
        """测试RateLimitError的is_retryable默认为True"""
        error = RateLimitError("速率限制")
        assert error.is_retryable is True

    def test_rate_limit_error_can_be_raised_and_caught(self):
        """测试RateLimitError可以被抛出和捕获"""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError("速率限制")

        assert isinstance(exc_info.value, APIError)
        assert isinstance(exc_info.value, NovelOutlineError)


class TestTokenLimitError:
    """测试TokenLimitError异常类"""

    def test_token_limit_error_inheritance(self):
        """测试TokenLimitError继承自NovelOutlineError"""
        error = TokenLimitError("Token限制")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_token_limit_error_message(self):
        """测试TokenLimitError的消息"""
        error = TokenLimitError("Token超出限制")
        assert str(error) == "Token超出限制"
        assert error.message == "Token超出限制"

    def test_token_limit_error_with_details(self):
        """测试TokenLimitError带有详情"""
        error = TokenLimitError("Token限制", details="最大token数: 200000")
        assert error.details == "最大token数: 200000"

    def test_token_limit_error_can_be_raised_and_caught(self):
        """测试TokenLimitError可以被抛出和捕获"""
        with pytest.raises(TokenLimitError) as exc_info:
            raise TokenLimitError("Token超出限制")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestEncodingError:
    """测试EncodingError异常类"""

    def test_encoding_error_inheritance(self):
        """测试EncodingError继承自NovelOutlineError"""
        error = EncodingError("编码错误")
        assert isinstance(error, NovelOutlineError)
        assert isinstance(error, Exception)

    def test_encoding_error_message(self):
        """测试EncodingError的消息"""
        error = EncodingError("编码不支持")
        assert str(error) == "编码不支持"
        assert error.message == "编码不支持"

    def test_encoding_error_with_details(self):
        """测试EncodingError带有详情"""
        error = EncodingError("编码错误", details="尝试的编码: gbk, utf-8")
        assert error.details == "尝试的编码: gbk, utf-8"

    def test_encoding_error_can_be_raised_and_caught(self):
        """测试EncodingError可以被抛出和捕获"""
        with pytest.raises(EncodingError) as exc_info:
            raise EncodingError("编码错误")

        assert isinstance(exc_info.value, NovelOutlineError)


class TestExceptionHierarchy:
    """测试异常类层次结构"""

    def test_all_custom_exceptions_inherit_from_novel_outline_error(self):
        """测试所有自定义异常都继承自NovelOutlineError"""
        exceptions = [
            APIKeyError,
            ConfigurationError,
            FileValidationError,
            ProcessingError,
            APIError,
            RateLimitError,
            TokenLimitError,
            EncodingError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, NovelOutlineError)

    def test_api_error_subclasses_inherit_from_api_error(self):
        """测试APIError的子类继承自APIError"""
        assert issubclass(RateLimitError, APIError)

    def test_catching_novel_outline_error_catches_all_custom_exceptions(self):
        """测试捕获NovelOutlineError可以捕获所有自定义异常"""
        exceptions_to_raise = [
            (APIKeyError, "测试"),
            (ConfigurationError, "测试"),
            (FileValidationError, "测试"),
            (ProcessingError, "测试"),
            (APIError, "测试"),
            (RateLimitError, "测试"),
            (TokenLimitError, "测试"),
            (EncodingError, "测试"),
        ]

        for exc_class, message in exceptions_to_raise:
            with pytest.raises(NovelOutlineError):
                raise exc_class(message)

    def test_specific_exceptions_can_be_caught_individually(self):
        """测试特定异常可以单独捕获"""
        with pytest.raises(APIKeyError):
            raise APIKeyError("测试")

        with pytest.raises(ConfigurationError):
            raise ConfigurationError("测试")

        with pytest.raises(RateLimitError):
            raise RateLimitError("测试")
