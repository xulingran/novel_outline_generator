"""颜色工具函数

提供十六进制与 RGB 颜色空间的转换，以及变亮/变暗操作。
"""


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """将十六进制颜色转换为 RGB

    Args:
        hex_color: 6位十六进制颜色，可选 '#' 前缀（如 '#ff0000' 或 'ff0000'）

    Raises:
        ValueError: 颜色格式无效
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"十六进制颜色需要6位字符，收到: '{hex_color}'")
    try:
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    except ValueError as e:
        raise ValueError(f"无效的十六进制颜色: '{hex_color}'") from e


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """将 RGB 转换为十六进制颜色"""
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten_color(hex_color: str, percent: int) -> str:
    """使颜色变亮"""
    r, g, b = hex_to_rgb(hex_color)
    r = min(255, int(r + (255 - r) * percent / 100))
    g = min(255, int(g + (255 - g) * percent / 100))
    b = min(255, int(b + (255 - b) * percent / 100))
    return rgb_to_hex(r, g, b)


def darken_color(hex_color: str, percent: int) -> str:
    """使颜色变暗"""
    r, g, b = hex_to_rgb(hex_color)
    r = max(0, int(r * (100 - percent) / 100))
    g = max(0, int(g * (100 - percent) / 100))
    b = max(0, int(b * (100 - percent) / 100))
    return rgb_to_hex(r, g, b)
