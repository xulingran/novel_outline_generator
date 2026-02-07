#!/usr/bin/env python3
"""
修复 Python 3.10+ 类型注解兼容性问题

将 `str | None` 等转换为 `Optional[str]`，支持 Python 3.9+
"""

import re
from pathlib import Path


def fix_type_annotations(file_path: Path):
    """修复单个文件的类型注解"""
    content = file_path.read_text(encoding="utf-8")
    original_content = content

    # 替换模式列表
    replacements = [
        # 基本类型联合 -> Optional
        (r': str \| None', r': Optional[str]'),
        (r': int \| None', r': Optional[int]'),
        (r': float \| None', r': Optional[float]'),
        (r': bool \| None', r': Optional[bool]'),
        (r': dict \| None', r': Optional[dict]'),
        (r': list \| None', r': Optional[list]'),
        (r': Path \| None', r': Optional[Path]'),
        (r': Callable\[\[.*?\], None\]', r': Callable[[], None]'),

        # 复杂类型联合
        (r'\| None\]', r']'),
        (r'\| None', r' | None'),  # 在 | 周围没有 ] 的情况

        # 需要导入 Optional 的情况
        (r': Optional\[str\]', ': typing.Optional[str]'),
        (r': Optional\[int\]', ': typing.Optional[int]'),
        (r': Optional\[float\]', ': typing.Optional[float]'),
        (r': Optional\[bool\]', ': typing.Optional[bool]'),
        (r': Optional\[dict\]', ': typing.Optional[dict]'),
        (r': Optional\[list\]', ': typing.Optional[list]'),
        (r': Optional\[Path\]', ': typing.Optional[Path]'),
        (r': Callable\[\[\], None\]', ': Callable[[], None]'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # 如果文件有变化，写回文件
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    """主函数"""
    # 需要修复的文件列表（从 grep 结果中提取）
    files_to_fix = [
        "gui/config_dialog.py",
        "gui/widgets/file_selector.py",
        "gui/widgets/log_viewer.py",
        "gui/widgets/progress_bar.py",
        "gui/main_window.py",
        "gui/async_worker.py",
    ]

    # 先检查是否需要添加 typing 导入
    for file_path_str in files_to_fix:
        file_path = Path(file_path_str)
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")

        # 检查是否使用 Optional 但没有导入
        needs_typing_import = False
        if 'Optional[' in content and 'from typing import' not in content:
            needs_typing_import = True

        # 检查是否使用 Callable 但没有导入
        if 'Callable[' in content and 'from typing import' not in content:
            needs_typing_import = True

        # 如果需要导入，添加到现有导入中
        if needs_typing_import:
            # 在文件开头添加或插入到现有的 from typing import 行
            if 'from collections.abc import' in content:
                # 在 collections.abc 导入后添加
                content = re.sub(
                    r'(from collections\.abc import [^\n]+)',
                    r'\1\nfrom typing import Optional, Callable',
                    content,
                    count=1
                )
            elif 'from pathlib import Path' in content:
                # 在 pathlib 导入后添加
                content = re.sub(
                    r'(from pathlib import Path[^\n]+)',
                    r'\1\nfrom typing import Optional, Callable',
                    content,
                    count=1
                )
            elif 'import logging' in content:
                # 在 logging 导入后添加
                content = re.sub(
                    r'(import logging\n)',
                    r'\1from typing import Optional, Callable\n\n',
                    content,
                    count=1
                )
            else:
                # 在文件开头添加
                content = 'from typing import Optional, Callable\n\n' + content

            file_path.write_text(content, encoding="utf-8")

        # 修复类型注解
        if fix_type_annotations(file_path):
            print(f"✓ 修复: {file_path_str}")
        else:
            print(f"- 跳过: {file_path_str} (无需修改)")


if __name__ == "__main__":
    main()
