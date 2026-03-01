#!/usr/bin/env python3
"""
创建应用图标和GUI组件图标

生成：
1. 应用图标（PNG, ICO, ICNS）
2. GUI组件图标（文件、文件夹、按钮等）
"""

from pathlib import Path

from PIL import Image, ImageDraw

# Nord主题颜色
NORD_ACCENT_DARK = "#88C0D0"  # 青色（暗色主题）
NORD_ACCENT_LIGHT = "#5E81AC"  # 蓝色（亮色主题）
NORD_ERROR = "#BF616A"
NORD_SUCCESS = "#A3BE8C"
NORD_WARNING = "#EBCB8B"


def hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple:
    """将十六进制颜色转换为RGBA元组"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def create_simple_icon(size: int = 512) -> Image.Image:
    """创建简单的书本图标"""
    # 创建图像
    img = Image.new("RGBA", (size, size), (70, 130, 180, 255))  # 钢蓝色背景
    draw = ImageDraw.Draw(img)

    # 绘制书本
    margin = size // 8
    book_width = size - 2 * margin
    book_height = int(book_width * 0.7)
    x = margin
    y = (size - book_height) // 2

    # 书的轮廓
    draw.rounded_rectangle(
        [x, y, x + book_width, y + book_height],
        radius=size // 32,
        outline=(255, 255, 255, 255),
        width=size // 64,
    )

    # 书脊（中间线）
    spine_x = x + book_width // 2
    draw.line(
        [spine_x, y, spine_x, y + book_height],
        fill=(255, 255, 255, 255),
        width=size // 64,
    )

    # 绘制一些代表文字的横线
    line_margin = size // 16
    line_spacing = size // 16
    line_width = (book_width // 2) - 2 * line_margin

    for i in range(5):
        line_y = y + size // 4 + i * line_spacing
        # 左页
        draw.line(
            [x + line_margin, line_y, x + line_margin + line_width, line_y],
            fill=(255, 255, 255, 200),
            width=size // 128,
        )
        # 右页
        draw.line(
            [spine_x + line_margin, line_y, spine_x + line_margin + line_width, line_y],
            fill=(255, 255, 255, 200),
            width=size // 128,
        )

    return img


def create_gui_component_icons():
    """创建GUI组件图标"""
    icons_dir = Path(__file__).parent / "icons"
    icons_dir.mkdir(exist_ok=True)

    print("创建GUI组件图标...")

    # 需要创建的图标类型和尺寸
    sizes = [16, 24, 32]

    # 文件图标
    for size in sizes:
        _create_file_icon(icons_dir / f"file_{size}.png", size)
        _create_folder_icon(icons_dir / f"folder_{size}.png", size)

    # 按钮图标
    for size in sizes:
        _create_play_icon(icons_dir / f"play_{size}.png", size)
        _create_pause_icon(icons_dir / f"pause_{size}.png", size)
        _create_stop_icon(icons_dir / f"stop_{size}.png", size)
        _create_refresh_icon(icons_dir / f"refresh_{size}.png", size)
        _create_clear_icon(icons_dir / f"clear_{size}.png", size)

    # 状态图标
    for size in sizes:
        _create_settings_icon(icons_dir / f"settings_{size}.png", size)
        _create_success_icon(icons_dir / f"success_{size}.png", size)
        _create_error_icon(icons_dir / f"error_{size}.png", size)

    print(f"✓ GUI组件图标已保存到: {icons_dir}")


def _create_file_icon(output_path: Path, size: int):
    """创建文件图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 文件背景
    margin = size // 8
    file_width = size - 2 * margin
    file_height = int(file_width * 1.2)

    # 使用Nord accent色
    accent = hex_to_rgba(NORD_ACCENT_DARK)

    # 绘制文件轮廓
    draw.rounded_rectangle(
        [margin, margin, margin + file_width, margin + file_height],
        radius=max(1, size // 16),
        outline=accent,
        width=max(1, size // 16),
    )

    # 绘制折角
    corner_size = size // 4
    draw.polygon(
        [
            (margin + file_width - corner_size, margin),
            (margin + file_width, margin + corner_size),
            (margin + file_width, margin),
        ],
        fill=accent,
    )

    # 绘制文件内容线
    line_y = margin + size // 4
    line_height = size // 16

    for i in range(3):
        y = line_y + i * line_height * 2
        if y + line_height < margin + file_height:
            draw.rectangle(
                [
                    margin + size // 4,
                    y,
                    margin + file_width - size // 4,
                    y + line_height,
                ],
                fill=accent,
            )

    img.save(output_path, "PNG")


def _create_folder_icon(output_path: Path, size: int):
    """创建文件夹图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 8
    folder_width = size - 2 * margin
    folder_height = int(folder_width * 0.8)

    accent = hex_to_rgba(NORD_ACCENT_DARK)

    # 绘制文件夹标签
    tab_width = folder_width // 3
    tab_height = folder_height // 5
    draw.rectangle(
        [margin, margin, margin + tab_width, margin + tab_height],
        fill=accent,
    )

    # 绘制文件夹主体
    draw.rounded_rectangle(
        [margin, margin + tab_height // 2, margin + folder_width, margin + folder_height],
        radius=max(1, size // 16),
        outline=accent,
        width=max(1, size // 16),
    )

    img.save(output_path, "PNG")


def _create_play_icon(output_path: Path, size: int):
    """创建播放图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_SUCCESS)

    # 绘制三角形播放按钮
    margin = size // 6
    center_y = size // 2
    triangle_size = size - 2 * margin

    # 三角形顶点
    points = [
        (margin, center_y - triangle_size // 2),
        (margin, center_y + triangle_size // 2),
        (margin + triangle_size, center_y),
    ]

    draw.polygon(points, fill=accent)
    img.save(output_path, "PNG")


def _create_pause_icon(output_path: Path, size: int):
    """创建暂停图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_WARNING)

    bar_width = size // 5
    bar_height = size - 2 * (size // 4)

    # 两个竖条
    draw.rectangle(
        [size // 4, size // 4, size // 4 + bar_width, size // 4 + bar_height],
        fill=accent,
    )
    draw.rectangle(
        [
            size - size // 4 - bar_width,
            size // 4,
            size - size // 4,
            size // 4 + bar_height,
        ],
        fill=accent,
    )

    img.save(output_path, "PNG")


def _create_stop_icon(output_path: Path, size: int):
    """创建停止图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_ERROR)

    stop_size = size - 2 * (size // 4)
    margin = (size - stop_size) // 2

    draw.rectangle([margin, margin, margin + stop_size, margin + stop_size], fill=accent)
    img.save(output_path, "PNG")


def _create_refresh_icon(output_path: Path, size: int):
    """创建刷新图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_ACCENT_DARK)

    center_x = size // 2
    center_y = size // 2
    radius = (size - 2 * (size // 4)) // 2
    line_width = max(1, size // 16)

    # 绘制圆形（顶部开口）
    draw.arc(
        [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ],
        start=30,
        end=330,
        fill=accent,
        width=line_width,
    )

    # 箭头
    arrow_size = radius // 3
    draw.polygon(
        [
            (center_x + int(radius * 0.866), center_y - int(radius * 0.5)),
            (
                center_x + int(radius * 0.866) - arrow_size // 2,
                center_y - int(radius * 0.5) - arrow_size,
            ),
            (
                center_x + int(radius * 0.866) - arrow_size // 2,
                center_y - int(radius * 0.5) + arrow_size // 2,
            ),
        ],
        fill=accent,
    )

    img.save(output_path, "PNG")


def _create_clear_icon(output_path: Path, size: int):
    """创建清除图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_ERROR)

    margin = size // 6
    line_width = max(1, size // 16)

    # X形状
    draw.line(
        [margin, margin, size - margin, size - margin],
        fill=accent,
        width=line_width,
    )
    draw.line(
        [size - margin, margin, margin, size - margin],
        fill=accent,
        width=line_width,
    )

    img.save(output_path, "PNG")


def _create_settings_icon(output_path: Path, size: int):
    """创建设置图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_ACCENT_DARK)

    center_x = size // 2
    center_y = size // 2
    radius = (size - 2 * (size // 4)) // 2
    hole_radius = max(1, size // 10)
    line_width = max(1, size // 16)

    # 外圆（6个齿的简化表示）
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        outline=accent,
        width=line_width,
    )

    # 内圆（孔）
    draw.ellipse(
        [
            center_x - hole_radius,
            center_y - hole_radius,
            center_x + hole_radius,
            center_y + hole_radius,
        ],
        fill=accent,
    )

    img.save(output_path, "PNG")


def _create_success_icon(output_path: Path, size: int):
    """创建成功图标（勾选）"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_SUCCESS)
    line_width = max(1, size // 16)

    margin = size // 4

    # 绘制勾选标记
    draw.line(
        [margin, size // 2, size // 3, size - margin],
        fill=accent,
        width=line_width,
    )
    draw.line(
        [size // 3, size - margin, size - margin, size // 3],
        fill=accent,
        width=line_width,
    )

    img.save(output_path, "PNG")


def _create_error_icon(output_path: Path, size: int):
    """创建错误图标（感叹号）"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = hex_to_rgba(NORD_ERROR)

    center_x = size // 2

    # 绘制感叹号
    bar_width = max(1, size // 8)
    bar_height = size - 2 * (size // 3)

    draw.rectangle(
        [center_x - bar_width // 2, size // 4, center_x + bar_width // 2, size // 4 + bar_height],
        fill=accent,
    )

    # 底部圆点
    dot_size = max(1, size // 10)
    draw.ellipse(
        [
            center_x - dot_size,
            size - size // 3 - dot_size,
            center_x + dot_size,
            size - size // 3,
        ],
        fill=accent,
    )

    img.save(output_path, "PNG")


def create_png_icon(output_path: Path, size: int = 512):
    """创建 PNG 图标"""
    img = create_simple_icon(size)
    img.save(output_path, "PNG")
    print(f"✓ 创建 PNG 图标: {output_path}")


def create_ico_icon(output_path: Path):
    """创建 ICO 图标（Windows）"""
    img = create_simple_icon(512)

    # ICO 需要多个尺寸
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    icons = []

    for size in sizes:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        icons.append(resized)

    # 保存为 ICO
    icons[0].save(output_path, format="ICO", sizes=[(s[0], s[1]) for s in sizes])
    print(f"✓ 创建 ICO 图标: {output_path}")


def create_icns_icon(output_path: Path):
    """创建 ICNS 图标（macOS）"""
    # macOS 需要特殊格式，这里创建一个 PNG 作为替代
    # 实际的 ICNS 需要在 macOS 上使用 iconutil 工具创建
    img = create_simple_icon(512)
    img.save(output_path.with_suffix(".png"), "PNG")
    print(f"✓ 创建 PNG 图标（macOS 需要转换为 ICNS）: {output_path.with_suffix('.png')}")
    print("  在 macOS 上使用以下命令转换:")
    print(
        f"  mkdir icon.iconset && sips -z 16 16 {output_path.with_suffix('.png')} --out icon.iconset/icon_16x16.png"
    )
    print(f"  sips -z 32 32 {output_path.with_suffix('.png')} --out icon.iconset/icon_16x16@2x.png")
    print(f"  sips -z 32 32 {output_path.with_suffix('.png')} --out icon.iconset/icon_32x32.png")
    print(f"  sips -z 64 64 {output_path.with_suffix('.png')} --out icon.iconset/icon_32x32@2x.png")
    print(
        f"  sips -z 128 128 {output_path.with_suffix('.png')} --out icon.iconset/icon_128x128.png"
    )
    print(
        f"  sips -z 256 256 {output_path.with_suffix('.png')} --out icon.iconset/icon_128x128@2x.png"
    )
    print(
        f"  sips -z 256 256 {output_path.with_suffix('.png')} --out icon.iconset/icon_256x256.png"
    )
    print(
        f"  sips -z 512 512 {output_path.with_suffix('.png')} --out icon.iconset/icon_256x256@2x.png"
    )
    print(
        f"  sips -z 512 512 {output_path.with_suffix('.png')} --out icon.iconset/icon_512x512.png"
    )
    print("  iconutil -c icns icon.iconset")


def main():
    """主函数"""
    assets_dir = Path(__file__).parent
    print("创建应用图标...")
    print(f"输出目录: {assets_dir}")
    print()

    try:
        # PNG (Linux)
        create_png_icon(assets_dir / "icon.png", 512)

        # ICO (Windows)
        create_ico_icon(assets_dir / "icon.ico")

        # ICNS (macOS - 需要额外处理)
        create_icns_icon(assets_dir / "icon.icns")

        print()
        print("✓ 应用图标创建完成！")
        print()

        # GUI组件图标
        create_gui_component_icons()

        print()
        print("✓ 所有图标创建完成！")

    except Exception as e:
        print(f"✗ 创建图标失败: {e}")
        print()
        print("提示：请确保已安装 Pillow:")
        print("  pip install pillow")


if __name__ == "__main__":
    main()
