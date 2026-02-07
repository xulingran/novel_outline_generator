#!/usr/bin/env python3
"""
创建简单的应用图标

生成基础的占位符图标，用于开发和测试。
"""

from pathlib import Path

from PIL import Image, ImageDraw


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
        print("✓ 所有图标创建完成！")

    except Exception as e:
        print(f"✗ 创建图标失败: {e}")
        print()
        print("提示：请确保已安装 Pillow:")
        print("  pip install pillow")


if __name__ == "__main__":
    main()
