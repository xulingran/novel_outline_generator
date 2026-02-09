"""
关于页面

居中简洁布局，显示应用信息。
"""

import logging

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class AboutPage(ctk.CTkFrame):
    """
    关于页面

    居中简洁布局，显示应用信息、版本号、开源协议等。
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        # 主容器（居中）
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True)

        # 内容容器（居中）
        content = ctk.CTkFrame(main_container, fg_color="transparent")
        content.place(relx=0.5, rely=0.5, anchor="center")

        # Logo 区域
        try:
            from gui.components.icon import Icon, IconSize

            logo_icon = Icon(content, name="rocket", size=IconSize.XXL)
            logo_icon.pack(pady=SPACING["lg"])
        except Exception:
            # 回退到文字
            logo_label = ctk.CTkLabel(
                content,
                text="🚀",
                font=ctk.CTkFont(size=64),
            )
            logo_label.pack(pady=SPACING["lg"])

        # 应用名称
        name_label = ctk.CTkLabel(
            content,
            text="Novel Outline Generator",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        name_label.pack(pady=(SPACING["sm"], SPACING["xs"]))

        # 版本号
        version_label = ctk.CTkLabel(
            content,
            text="v1.0.0",
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        version_label.pack(pady=(0, SPACING["lg"]))

        # 描述
        desc_label = ctk.CTkLabel(
            content,
            text="基于 LLM 的小说大纲自动生成工具\n\n"
            "支持多种 API 提供商，自动分块处理长篇小说，\n"
            "生成结构化大纲内容。",
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_primary", mode="auto"),
            justify="center",
        )
        desc_label.pack(pady=SPACING["lg"])

        # 信息卡片
        info_frame = ctk.CTkFrame(
            content,
            fg_color=get_color("bg_secondary", mode="auto"),
            corner_radius=12,
            width=400,
        )
        info_frame.pack(pady=SPACING["lg"])
        info_frame.pack_propagate(False)

        # 信息项
        info_items = [
            ("作者", "Novel Outline Contributors"),
            ("开源协议", "MIT License"),
            ("Python 版本", "3.12+"),
        ]

        for label, value in info_items:
            item_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
            item_frame.pack(fill="x", padx=SPACING["md"], pady=SPACING["sm"])

            key_label = ctk.CTkLabel(
                item_frame,
                text=label,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=get_color("fg_secondary", mode="auto"),
                width=100,
                anchor="w",
            )
            key_label.pack(side="left")

            value_label = ctk.CTkLabel(
                item_frame,
                text=value,
                font=ctk.CTkFont(size=12),
                text_color=get_color("fg_primary", mode="auto"),
                anchor="w",
            )
            value_label.pack(side="left", fill="x", expand=True)

        # 分隔线
        separator = ctk.CTkFrame(
            content,
            height=1,
            width=200,
            fg_color=get_color("border", mode="auto"),
        )
        separator.pack(pady=SPACING["lg"])

        # 链接按钮
        links_frame = ctk.CTkFrame(content, fg_color="transparent")
        links_frame.pack()

        from gui.components.button import Button, ButtonSize, ButtonVariant

        # GitHub 链接
        github_button = Button(
            links_frame,
            text="GitHub 仓库",
            variant=ButtonVariant.SECONDARY,
            size=ButtonSize.MD,
            command=self._open_github,
            width=140,
        )
        github_button.pack(side="left", padx=SPACING["xs"])

        # 文档链接
        docs_button = Button(
            links_frame,
            text="使用文档",
            variant=ButtonVariant.TERTIARY,
            size=ButtonSize.MD,
            command=self._open_docs,
            width=120,
        )
        docs_button.pack(side="left", padx=SPACING["xs"])

        # 版权信息
        copyright_label = ctk.CTkLabel(
            content,
            text="© 2025 Novel Outline Generator. All rights reserved.\n\n"
            "Built with ❤️ using CustomTkinter and Claude.",
            font=ctk.CTkFont(size=11),
            text_color=get_color("fg_tertiary", mode="auto"),
            justify="center",
        )
        copyright_label.pack(pady=SPACING["xl"])

    def _open_github(self):
        """打开 GitHub 仓库"""
        import webbrowser

        webbrowser.open("https://github.com/yourusername/novel-outline-generator")

    def _open_docs(self):
        """打开文档"""
        import webbrowser

        webbrowser.open(
            "https://github.com/yourusername/novel-outline-generator/blob/main/README.md"
        )
