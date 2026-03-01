"""
用户友好的错误消息模块

将技术性错误转换为用户易于理解的提示信息。
"""

from dataclasses import dataclass


@dataclass
class UserErrorMessage:
    """用户错误消息"""

    title: str
    message: str
    suggestion: str


ERROR_MESSAGE_MAP: dict[str, UserErrorMessage] = {
    "authentication": UserErrorMessage(
        title="API 密钥无效",
        message="您的 API 密钥无效或已过期。",
        suggestion="请前往配置页面检查 API 密钥设置。",
    ),
    "connection": UserErrorMessage(
        title="网络连接失败",
        message="无法连接到 API 服务器。",
        suggestion="请检查网络连接，或尝试配置代理。",
    ),
    "timeout": UserErrorMessage(
        title="请求超时",
        message="API 请求超时，服务器响应时间过长。",
        suggestion="请稍后重试，或检查网络连接质量。",
    ),
    "rate_limit": UserErrorMessage(
        title="请求频率超限",
        message="API 请求频率超过限制。",
        suggestion="请稍后重试，或在配置中降低并发限制。",
    ),
    "file_not_found": UserErrorMessage(
        title="文件不存在",
        message="指定的文件不存在或无法访问。",
        suggestion="请确认文件路径正确，文件未被删除或移动。",
    ),
    "file_read": UserErrorMessage(
        title="文件读取失败",
        message="无法读取文件内容。",
        suggestion="请确认文件格式正确，且有读取权限。",
    ),
    "file_write": UserErrorMessage(
        title="文件保存失败",
        message="无法保存处理结果。",
        suggestion="请确认输出目录有写入权限，磁盘空间充足。",
    ),
    "invalid_response": UserErrorMessage(
        title="响应格式错误",
        message="API 返回了无效的响应格式。",
        suggestion="请检查日志获取详细信息，或尝试更换模型。",
    ),
    "token_limit": UserErrorMessage(
        title="内容超长",
        message="文本内容超过了模型的处理限制。",
        suggestion="请尝试减小分块大小，或使用支持更长上下文的模型。",
    ),
    "cancelled": UserErrorMessage(
        title="已取消",
        message="处理任务已被取消。",
        suggestion="已完成的进度已保存，可随时继续处理。",
    ),
}

DEFAULT_ERROR = UserErrorMessage(
    title="处理失败",
    message="处理过程中发生错误。",
    suggestion="请查看日志获取详细信息。",
)


def get_user_error_message(error: BaseException) -> UserErrorMessage:
    """
    根据异常类型获取用户友好的错误消息

    Args:
        error: 异常对象

    Returns:
        UserErrorMessage: 用户友好的错误消息
    """
    error_type = type(error).__name__.lower()
    error_message = str(error).lower()

    if "auth" in error_type or "auth" in error_message or "api key" in error_message:
        return ERROR_MESSAGE_MAP["authentication"]

    if "connection" in error_type or "connection" in error_message:
        return ERROR_MESSAGE_MAP["connection"]

    if "timeout" in error_type or "timeout" in error_message:
        return ERROR_MESSAGE_MAP["timeout"]

    if "rate" in error_message or "limit" in error_message:
        return ERROR_MESSAGE_MAP["rate_limit"]

    if "filenotfound" in error_type or "file not found" in error_message:
        return ERROR_MESSAGE_MAP["file_not_found"]

    if "permission" in error_type or "permission" in error_message:
        return ERROR_MESSAGE_MAP["file_read"]

    if "cancelled" in error_type or "cancel" in error_message:
        return ERROR_MESSAGE_MAP["cancelled"]

    if "token" in error_message or "length" in error_message:
        return ERROR_MESSAGE_MAP["token_limit"]

    return DEFAULT_ERROR


def format_error_dialog(error: BaseException) -> tuple[str, str]:
    """
    格式化错误对话框内容

    Args:
        error: 异常对象

    Returns:
        tuple[str, str]: (标题, 详细消息)
    """
    user_error = get_user_error_message(error)
    detail = f"{user_error.message}\n\n💡 建议：{user_error.suggestion}"
    return user_error.title, detail
