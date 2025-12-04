"""
自定义异常类模块
定义项目中使用的各种异常类型
"""
from typing import Optional


class NovelOutlineError(Exception):
    """基础异常类，所有项目相关的异常都应继承此类"""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class APIKeyError(NovelOutlineError):
    """API密钥相关错误"""
    pass


class ConfigurationError(NovelOutlineError):
    """配置相关错误"""
    pass


class FileValidationError(NovelOutlineError):
    """文件验证错误"""
    pass


class ProcessingError(NovelOutlineError):
    """处理过程中的错误"""
    pass


class APIError(NovelOutlineError):
    """API调用错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, is_retryable: bool = False):
        self.error_code = error_code
        self.is_retryable = is_retryable
        super().__init__(message)


class RateLimitError(APIError):
    """API速率限制错误"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(message, is_retryable=True)


class TokenLimitError(NovelOutlineError):
    """Token限制相关错误"""
    pass


class EncodingError(NovelOutlineError):
    """文本编码错误"""
    pass