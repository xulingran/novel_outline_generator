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
    """API调用错误

    使用示例:
        # 简单的API错误
        raise APIError("请求超时")

        # 带错误代码和重试标志的错误
        raise APIError(
            message="服务暂不可用",
            error_code="SERVICE_UNAVAILABLE",
            is_retryable=True
        )

        # 带详细信息的错误
        raise APIError(
            message="请求失败",
            error_code="INVALID_REQUEST",
            details="请求参数格式不正确"
        )
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        is_retryable: bool = False,
        details: str | None = None,
    ):
        self.error_code = error_code
        self.is_retryable = is_retryable
        super().__init__(message, details=details)

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

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        error_code: str | None = None,
        details: str | None = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, error_code=error_code, is_retryable=True, details=details)

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
