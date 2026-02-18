"""
错误消息模块测试
"""

from gui.utils.error_messages import (
    DEFAULT_ERROR,
    ERROR_MESSAGE_MAP,
    UserErrorMessage,
    format_error_dialog,
    get_user_error_message,
)


class TestUserErrorMessage:
    """用户错误消息数据类测试"""

    def test_user_error_message_creation(self):
        """测试创建用户错误消息"""
        error = UserErrorMessage(
            title="测试标题",
            message="测试消息",
            suggestion="测试建议",
        )
        assert error.title == "测试标题"
        assert error.message == "测试消息"
        assert error.suggestion == "测试建议"

    def test_default_error(self):
        """测试默认错误消息"""
        assert DEFAULT_ERROR.title == "处理失败"
        assert DEFAULT_ERROR.message == "处理过程中发生错误。"
        assert DEFAULT_ERROR.suggestion == "请查看日志获取详细信息。"


class TestErrorMessageMap:
    """错误消息映射测试"""

    def test_all_error_types_exist(self):
        """测试所有错误类型都存在"""
        expected_keys = [
            "authentication",
            "connection",
            "timeout",
            "rate_limit",
            "file_not_found",
            "file_read",
            "file_write",
            "invalid_response",
            "token_limit",
            "cancelled",
        ]
        for key in expected_keys:
            assert key in ERROR_MESSAGE_MAP, f"Missing error type: {key}"

    def test_all_errors_have_required_fields(self):
        """测试所有错误都有必需字段"""
        for key, error in ERROR_MESSAGE_MAP.items():
            assert isinstance(error, UserErrorMessage), f"{key} is not UserErrorMessage"
            assert error.title, f"{key} missing title"
            assert error.message, f"{key} missing message"
            assert error.suggestion, f"{key} missing suggestion"


class TestGetUserErrorMessage:
    """获取用户错误消息函数测试"""

    def test_authentication_error_from_type(self):
        """测试从类型识别认证错误"""

        class AuthenticationError(Exception):
            pass

        error = AuthenticationError("Some auth error")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["authentication"]

    def test_authentication_error_from_message(self):
        """测试从消息识别认证错误"""
        error = Exception("Invalid API key provided")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["authentication"]

    def test_connection_error_from_type(self):
        """测试从类型识别连接错误"""

        class ConnectionError(Exception):
            pass

        error = ConnectionError("Connection failed")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["connection"]

    def test_connection_error_from_message(self):
        """测试从消息识别连接错误"""
        error = Exception("Connection refused")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["connection"]

    def test_timeout_error_from_type(self):
        """测试从类型识别超时错误"""

        class TimeoutError(Exception):
            pass

        error = TimeoutError("Request timed out")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["timeout"]

    def test_timeout_error_from_message(self):
        """测试从消息识别超时错误"""
        error = Exception("The operation timeout exceeded")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["timeout"]

    def test_rate_limit_error(self):
        """测试识别速率限制错误"""
        error = Exception("Rate limit exceeded")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["rate_limit"]

    def test_file_not_found_from_type(self):
        """测试从类型识别文件不存在错误"""

        class FileNotFoundError(Exception):
            pass

        error = FileNotFoundError("File not found")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["file_not_found"]

    def test_file_not_found_from_message(self):
        """测试从消息识别文件不存在错误"""
        error = Exception("The file not found in path")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["file_not_found"]

    def test_permission_error(self):
        """测试识别权限错误"""

        class PermissionError(Exception):
            pass

        error = PermissionError("Permission denied")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["file_read"]

    def test_cancelled_error_from_type(self):
        """测试从类型识别取消错误"""

        class CancelledError(Exception):
            pass

        error = CancelledError("Operation cancelled")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["cancelled"]

    def test_cancelled_error_from_message(self):
        """测试从消息识别取消错误"""
        error = Exception("Task was cancel")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["cancelled"]

    def test_token_limit_error(self):
        """测试识别 Token 限制错误"""
        error = Exception("Token exceeded")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["token_limit"]

    def test_length_error(self):
        """测试识别长度错误"""
        error = Exception("Content length exceeded")
        result = get_user_error_message(error)
        assert result == ERROR_MESSAGE_MAP["token_limit"]

    def test_unknown_error_returns_default(self):
        """测试未知错误返回默认消息"""
        error = Exception("Some unknown error")
        result = get_user_error_message(error)
        assert result == DEFAULT_ERROR


class TestFormatErrorDialog:
    """格式化错误对话框函数测试"""

    def test_format_error_dialog_returns_tuple(self):
        """测试返回元组格式"""
        error = Exception("Invalid API key")
        result = format_error_dialog(error)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_format_error_dialog_structure(self):
        """测试返回结构"""
        error = Exception("Invalid API key")
        title, detail = format_error_dialog(error)

        assert isinstance(title, str)
        assert isinstance(detail, str)
        assert "💡 建议：" in detail

    def test_format_error_dialog_with_known_error(self):
        """测试已知错误格式化"""
        error = Exception("Request timeout occurred")
        title, detail = format_error_dialog(error)

        assert title == ERROR_MESSAGE_MAP["timeout"].title
        assert ERROR_MESSAGE_MAP["timeout"].message in detail
        assert ERROR_MESSAGE_MAP["timeout"].suggestion in detail

    def test_format_error_dialog_with_unknown_error(self):
        """测试未知错误格式化"""
        error = Exception("Some random error")
        title, detail = format_error_dialog(error)

        assert title == DEFAULT_ERROR.title
        assert DEFAULT_ERROR.message in detail
        assert DEFAULT_ERROR.suggestion in detail


class TestErrorMessageContent:
    """错误消息内容测试"""

    def test_authentication_error_content(self):
        """测试认证错误内容"""
        error = ERROR_MESSAGE_MAP["authentication"]
        assert "API 密钥" in error.message
        assert "配置页面" in error.suggestion

    def test_connection_error_content(self):
        """测试连接错误内容"""
        error = ERROR_MESSAGE_MAP["connection"]
        assert "网络" in error.message or "连接" in error.message
        assert "网络" in error.suggestion or "代理" in error.suggestion

    def test_rate_limit_error_content(self):
        """测试速率限制错误内容"""
        error = ERROR_MESSAGE_MAP["rate_limit"]
        assert "频率" in error.message or "限制" in error.message
        assert "并发" in error.suggestion or "稍后" in error.suggestion

    def test_file_errors_content(self):
        """测试文件错误内容"""
        for key in ["file_not_found", "file_read", "file_write"]:
            error = ERROR_MESSAGE_MAP[key]
            assert "文件" in error.message or "文件" in error.title

    def test_cancelled_error_content(self):
        """测试取消错误内容"""
        error = ERROR_MESSAGE_MAP["cancelled"]
        assert "取消" in error.message or "取消" in error.title
        assert "保存" in error.suggestion or "继续" in error.suggestion
