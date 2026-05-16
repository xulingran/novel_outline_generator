"""
关于页面

展示应用信息，并提供外观模式设置入口。
"""

import logging

import customtkinter as ctk

from gui.theme_manager import SPACING, get_color, get_theme_manager

logger = logging.getLogger(__name__)


class AboutPage(ctk.CTkFrame):
    """
    关于页面

    集中展示应用介绍、外观设置和项目链接。
    """

    THEME_OPTIONS = {
        "dark": "深色",
        "light": "浅色",
        "system": "跟随系统",
    }

    def __init__(self, master, **kwargs):
        if "fg_color" not in kwargs:
            kwargs["fg_color"] = get_color("bg_primary", mode="auto")
        super().__init__(master, **kwargs)

        self._theme_manager = get_theme_manager()
        self._theme_manager.on_theme_change(self._on_theme_changed)
        self.bind("<Destroy>", self._on_destroy, add="+")

        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        from gui.components.button import Button, ButtonSize, ButtonVariant
        from gui.components.card import Card

        container = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=get_color("bg_tertiary", mode="auto"),
            scrollbar_button_hover_color=get_color("border", mode="auto"),
        )
        container.pack(fill="both", expand=True, padx=40, pady=40)

        content = ctk.CTkFrame(container, fg_color="transparent")
        content.pack(fill="x", expand=True)

        hero_card = Card(content, padding="lg")
        hero_card.pack(fill="x", pady=(0, SPACING["lg"]))

        hero_inner = ctk.CTkFrame(hero_card.content, fg_color="transparent")
        hero_inner.pack(fill="x")

        try:
            from gui.components.icon import Icon, IconSize

            logo_icon = Icon(hero_inner, name="rocket", size=IconSize.XXL)
            logo_icon.pack(anchor="center", pady=(SPACING["sm"], SPACING["md"]))
        except Exception:
            fallback_icon = ctk.CTkLabel(
                hero_inner, text="N", font=ctk.CTkFont(size=40, weight="bold")
            )
            fallback_icon.pack(anchor="center", pady=(SPACING["sm"], SPACING["md"]))

        name_label = ctk.CTkLabel(
            hero_inner,
            text="Novel Outline Generator",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
            justify="center",
        )
        name_label.pack(fill="x")

        summary_label = ctk.CTkLabel(
            hero_inner,
            text="面向长篇文本的 AI 小说大纲生成工具，支持多模型、分块处理和断点续跑。",
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_secondary", mode="auto"),
            justify="center",
        )
        summary_label.pack(fill="x", pady=(SPACING["sm"], SPACING["md"]))

        tag_row = ctk.CTkFrame(hero_inner, fg_color="transparent")
        tag_row.pack(anchor="center", pady=(0, SPACING["sm"]))

        for tag in ["Async First", "Resume Ready", "Multi Provider"]:
            chip = ctk.CTkLabel(
                tag_row,
                text=tag,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=get_color("accent", mode="auto"),
                fg_color=get_color("bg_primary", mode="auto"),
                corner_radius=999,
                padx=12,
                pady=6,
            )
            chip.pack(side="left", padx=SPACING["xs"])

        settings_card = Card(
            content,
            title="外观设置",
            subtitle="在这里切换界面主题，立即预览效果。",
            padding="lg",
        )
        settings_card.pack(fill="x", pady=(0, SPACING["lg"]))

        segmented_values = list(self.THEME_OPTIONS.values())
        self._theme_segmented = ctk.CTkSegmentedButton(
            settings_card.content,
            values=segmented_values,
            command=self._on_theme_segmented_change,
            selected_color=get_color("accent", mode="auto"),
            selected_hover_color=get_color("accent_secondary", mode="auto"),
            unselected_color=get_color("bg_tertiary", mode="auto"),
            unselected_hover_color=get_color("border", mode="auto"),
            text_color=get_color("fg_primary", mode="auto"),
            height=40,
        )
        self._theme_segmented.pack(anchor="w")

        current_theme = self._theme_manager.get_current_theme()
        self._theme_segmented.set(self.THEME_OPTIONS.get(current_theme, self.THEME_OPTIONS["dark"]))

        self._theme_hint = ctk.CTkLabel(
            settings_card.content,
            text="推荐在桌面端使用“跟随系统”，会自动适应系统明暗模式。",
            font=ctk.CTkFont(size=12),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        self._theme_hint.pack(anchor="w", pady=(SPACING["sm"], 0))

        info_card = Card(content, title="项目信息", padding="lg")
        info_card.pack(fill="x", pady=(0, SPACING["lg"]))

        info_items = [
            ("版本", "v1.0.0"),
            ("作者", "Novel Outline Contributors"),
            ("开源协议", "MIT License"),
            ("Python", "3.12+"),
        ]

        for label, value in info_items:
            row = ctk.CTkFrame(info_card.content, fg_color="transparent")
            row.pack(fill="x", pady=SPACING["xs"])

            key_label = ctk.CTkLabel(
                row,
                text=label,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=get_color("fg_secondary", mode="auto"),
                width=96,
                anchor="w",
            )
            key_label.pack(side="left")

            value_label = ctk.CTkLabel(
                row,
                text=value,
                font=ctk.CTkFont(size=12),
                text_color=get_color("fg_primary", mode="auto"),
                anchor="w",
            )
            value_label.pack(side="left", fill="x", expand=True)

        links_card = Card(content, title="相关链接", padding="lg")
        links_card.pack(fill="x")

        button_row = ctk.CTkFrame(links_card.content, fg_color="transparent")
        button_row.pack(anchor="w")

        github_button = Button(
            button_row,
            text="GitHub 仓库",
            variant=ButtonVariant.SECONDARY,
            size=ButtonSize.MD,
            command=self._open_github,
            width=140,
        )
        github_button.pack(side="left", padx=(0, SPACING["sm"]))

        docs_button = Button(
            button_row,
            text="使用文档",
            variant=ButtonVariant.TERTIARY,
            size=ButtonSize.MD,
            command=self._open_docs,
            width=120,
        )
        docs_button.pack(side="left")

        note_label = ctk.CTkLabel(
            links_card.content,
            text="界面主题切换后会立即生效，并记住你当前选择的模式。",
            font=ctk.CTkFont(size=11),
            text_color=get_color("fg_tertiary", mode="auto"),
        )
        note_label.pack(anchor="w", pady=(SPACING["md"], 0))

    def _on_theme_segmented_change(self, selected_label: str):
        """处理主题切换"""
        theme_key = next(
            (key for key, value in self.THEME_OPTIONS.items() if value == selected_label),
            None,
        )
        if theme_key is None:
            return

        self._theme_manager.set_theme(theme_key)

    def _on_theme_changed(self, theme: str):
        """主题变化回调"""
        if hasattr(self, "_theme_segmented"):
            label = self.THEME_OPTIONS.get(theme)
            if label is not None and self._theme_segmented.get() != label:
                self._theme_segmented.set(label)

    def _on_destroy(self, event):
        """销毁时取消主题订阅"""
        if event.widget is self:
            self._theme_manager.remove_callback(self._on_theme_changed)

    def _open_github(self):
        """打开 GitHub 仓库"""
        import webbrowser

        webbrowser.open("https://github.com/xulingran/novel_outline_generator")

    def _open_docs(self):
        """打开文档"""
        import webbrowser

        webbrowser.open("https://github.com/xulingran/novel_outline_generator/blob/main/README.md")
