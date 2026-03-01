"""
自定义异常类模块
定义项目中使用的各种异常类型
"""


class NovelOutlineError(Exception):
    """基础异常类，所有项目相关的异常都应继承此类"""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


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

    def __init__(self, message: str, error_code: str | None = None, is_retryable: bool = False):
        self.error_code = error_code
        self.is_retryable = is_retryable
        super().__init__(message)

    def __str__(self) -> str:
        parts = []
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.is_retryable:
            parts.append("Retryable")

        base_msg = super().__str__()
        if parts:
            return f"{base_msg} [{', '.join(parts)}]"
        return base_msg


class RateLimitError(APIError):
    """API速率限制错误"""

    def __init__(self, message: str, retry_after: int | None = None):
        self.retry_after = retry_after
        super().__init__(message, is_retryable=True)

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.retry_after is not None:
            return f"{base_msg} (Retry after: {self.retry_after}s)"
        return base_msg


class TokenLimitError(NovelOutlineError):
    """Token限制相关错误"""

    pass


class EncodingError(NovelOutlineError):
    """文本编码错误"""

    pass
