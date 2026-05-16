"""
GUI 工具模块
"""

from gui.utils.color import darken_color, hex_to_rgb, lighten_color, rgb_to_hex
from gui.utils.error_messages import (
    DEFAULT_ERROR,
    UserErrorMessage,
    format_error_dialog,
    get_user_error_message,
)

__all__ = [
    "hex_to_rgb",
    "rgb_to_hex",
    "lighten_color",
    "darken_color",
    "UserErrorMessage",
    "DEFAULT_ERROR",
    "get_user_error_message",
    "format_error_dialog",
]
