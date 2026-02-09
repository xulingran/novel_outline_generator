"""
配置页面

单列卡片流布局，支持 API 配置、处理参数、代理配置等。
"""

import logging
from pathlib import Path

import customtkinter as ctk

from config import SUPPORTED_API_PROVIDERS, _refresh_config_cache
from gui.theme_manager import SPACING, get_color

logger = logging.getLogger(__name__)


class ConfigPage(ctk.CTkFrame):
    """
    配置页面

    单列卡片流，最大宽度 720px 居中。
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self._config_data = {}
        self._load_current_config()
        self._setup_ui()

    def _load_current_config(self):
        """加载当前配置"""
        from config import get_api_config, get_processing_config

        self.api_config = get_api_config()
        self.proc_config = get_processing_config()

    def _setup_ui(self):
        """设置 UI"""
        # 主容器（居中）
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=40, pady=40)

        # 居中容器
        center_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        center_frame.pack(fill="both", expand=True)

        # 限制最大宽度
        center_frame.grid_propagate(False)

        # 创建可滚动区域
        scrollable = ctk.CTkScrollableFrame(
            center_frame,
            fg_color="transparent",
        )
        scrollable.pack(fill="both", expand=True)
        scrollable.grid_propagate(False)

        # API 配置卡片
        self._setup_api_card(scrollable)

        # 处理参数卡片
        self._setup_processing_card(scrollable)

        # 代理配置卡片
        self._setup_proxy_card(scrollable)

        # 保存按钮区域
        self._setup_actions(scrollable)

    def _setup_api_card(self, parent):
        """设置 API 配置卡片"""
        from gui.components.card import Card

        api_card = Card(parent, title="API 配置", subtitle="选择提供商并配置密钥", padding="lg")
        api_card.pack(fill="x", pady=(0, SPACING["lg"]))

        # 提供商选择 (Segmented Control)
        provider_frame = ctk.CTkFrame(api_card.content, fg_color="transparent")
        provider_frame.pack(fill="x", pady=(0, SPACING["lg"]))

        provider_label = ctk.CTkLabel(
            provider_frame,
            text="API 提供商",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        provider_label.pack(anchor="w", pady=(0, SPACING["sm"]))

        self._provider_var = ctk.StringVar(value=self.api_config.provider)

        # 使用 Segmented Button 风格的选择器
        provider_selector = ctk.CTkFrame(
            provider_frame,
            fg_color=get_color("bg_tertiary", mode="auto"),
            corner_radius=8,
        )
        provider_selector.pack(fill="x")

        for provider in SUPPORTED_API_PROVIDERS:
            btn = ctk.CTkRadioButton(
                provider_selector,
                text=provider,
                variable=self._provider_var,
                value=provider,
                command=self._on_provider_change,
                font=ctk.CTkFont(size=12),
                text_color=get_color("fg_primary", mode="auto"),
            )
            btn.pack(side="left", padx=SPACING["sm"], pady=SPACING["xs"])

        # API 密钥配置区（折叠面板风格）
        self._api_fields_frame = ctk.CTkFrame(api_card.content, fg_color="transparent")
        self._api_fields_frame.pack(fill="x")

        # 根据当前提供商显示对应的配置字段
        self._update_api_fields()

    def _update_api_fields(self):
        """更新 API 配置字段"""
        # 清空现有字段
        for widget in self._api_fields_frame.winfo_children():
            widget.destroy()

        provider = self._provider_var.get()

        # 根据提供商创建对应字段
        match provider:
            case "OpenAI":
                self._create_api_key_field(
                    "OpenAI API Key", "openai_key", self.api_config.openai_key or ""
                )
                self._create_text_field(
                    "OpenAI Base URL",
                    "openai_base",
                    self.api_config.openai_base or "",
                    "https://api.openai.com/v1",
                )
                self._create_text_field(
                    "OpenAI 模型", "openai_model", self.api_config.openai_model, "gpt-4o-mini"
                )
            case "Gemini":
                self._create_api_key_field(
                    "Gemini API Key", "gemini_key", self.api_config.gemini_key or ""
                )
                self._create_text_field(
                    "Gemini 模型",
                    "gemini_model",
                    self.api_config.gemini_model,
                    "gemini-2.0-flash-exp",
                )
            case "Zhipu":
                self._create_api_key_field(
                    "智谱 API Key", "zhipu_key", self.api_config.zhipu_key or ""
                )
                self._create_text_field(
                    "智谱 Base URL",
                    "zhipu_base",
                    self.api_config.zhipu_base or "",
                    "https://open.bigmodel.cn/api/paas/v4",
                )
                self._create_text_field(
                    "智谱模型", "zhipu_model", self.api_config.zhipu_model, "glm-4-flash"
                )
            case "AiHubMix":
                self._create_api_key_field(
                    "AiHubMix API Key", "aihubmix_key", self.api_config.aihubmix_api_key or ""
                )
                self._create_text_field(
                    "AiHubMix Base URL",
                    "aihubmix_base",
                    self.api_config.aihubmix_api_base or "",
                    "",
                )
                self._create_text_field(
                    "AiHubMix 模型", "aihubmix_model", self.api_config.aihubmix_model, ""
                )
            case _:
                pass

    def _create_api_key_field(self, label: str, key: str, default: str):
        """创建 API 密钥字段"""
        frame = ctk.CTkFrame(self._api_fields_frame, fg_color="transparent")
        frame.pack(fill="x", pady=(0, SPACING["md"]))

        label_widget = ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_primary", mode="auto"),
            width=150,
            anchor="w",
        )
        label_widget.pack(side="left", padx=(0, SPACING["sm"]))

        var = ctk.StringVar(value=default)
        setattr(self, f"_{key}_var", var)

        entry = ctk.CTkEntry(
            frame,
            textvariable=var,
            show="*",
            placeholder_text="输入 API 密钥",
        )
        entry.pack(side="left", fill="x", expand=True)

        # 显示/隐藏按钮
        show_button = ctk.CTkButton(
            frame,
            text="显示",
            width=60,
            command=lambda: self._toggle_password(entry, show_button),
            font=ctk.CTkFont(size=11),
        )
        show_button.pack(side="left", padx=(SPACING["sm"], 0))

    def _create_text_field(self, label: str, key: str, default: str, placeholder: str = ""):
        """创建文本字段"""
        frame = ctk.CTkFrame(self._api_fields_frame, fg_color="transparent")
        frame.pack(fill="x", pady=(0, SPACING["md"]))

        label_widget = ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_primary", mode="auto"),
            width=150,
            anchor="w",
        )
        label_widget.pack(side="left", padx=(0, SPACING["sm"]))

        var = ctk.StringVar(value=default)
        setattr(self, f"_{key}_var", var)

        entry = ctk.CTkEntry(
            frame,
            textvariable=var,
            placeholder_text=placeholder,
        )
        entry.pack(side="left", fill="x", expand=True)

    def _toggle_password(self, entry: ctk.CTkEntry, button: ctk.CTkButton):
        """切换密码显示"""
        if entry.cget("show") == "*":
            entry.configure(show="")
            button.configure(text="隐藏")
        else:
            entry.configure(show="*")
            button.configure(text="显示")

    def _on_provider_change(self):
        """提供商变更事件"""
        self._update_api_fields()

    def _setup_processing_card(self, parent):
        """设置处理参数卡片"""
        from gui.components.card import Card

        proc_card = Card(parent, title="处理参数", subtitle="调整分块和并发设置", padding="lg")
        proc_card.pack(fill="x", pady=(0, SPACING["lg"]))

        # 目标分块大小 (滑块)
        chunk_frame = ctk.CTkFrame(proc_card.content, fg_color="transparent")
        chunk_frame.pack(fill="x", pady=(0, SPACING["lg"]))

        chunk_label = ctk.CTkLabel(
            chunk_frame,
            text="目标分块大小",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
        )
        chunk_label.pack(anchor="w")

        chunk_desc = ctk.CTkLabel(
            chunk_frame,
            text="每个块的目标 token 数量，影响处理速度和成功率",
            font=ctk.CTkFont(size=11),
            text_color=get_color("fg_secondary", mode="auto"),
        )
        chunk_desc.pack(anchor="w", pady=(0, SPACING["sm"]))

        chunk_slider_frame = ctk.CTkFrame(chunk_frame, fg_color="transparent")
        chunk_slider_frame.pack(fill="x")

        self._chunk_size_var = ctk.IntVar(value=self.proc_config.target_tokens_per_chunk)

        slider = ctk.CTkSlider(
            chunk_slider_frame,
            from_=1000,
            to=10000,
            variable=self._chunk_size_var,
            number_of_steps=9,
        )
        slider.pack(side="left", fill="x", expand=True)

        self._chunk_value_label = ctk.CTkLabel(
            chunk_slider_frame,
            text=f"{self._chunk_size_var.get()}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=get_color("accent", mode="auto"),
            width=60,
        )
        self._chunk_value_label.pack(side="left", padx=(SPACING["sm"], 0))

        # 绑定滑块值变化
        slider.configure(command=self._on_chunk_size_change)

        # 并发限制
        parallel_frame = ctk.CTkFrame(proc_card.content, fg_color="transparent")
        parallel_frame.pack(fill="x", pady=(0, SPACING["md"]))

        parallel_label = ctk.CTkLabel(
            parallel_frame,
            text="并发限制",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
            width=150,
            anchor="w",
        )
        parallel_label.pack(side="left")

        self._parallel_var = ctk.IntVar(value=self.proc_config.parallel_limit)

        parallel_entry = ctk.CTkEntry(
            parallel_frame,
            textvariable=self._parallel_var,
            width=100,
        )
        parallel_entry.pack(side="left", padx=(SPACING["sm"], 0))

        # 最大重试次数
        retry_frame = ctk.CTkFrame(proc_card.content, fg_color="transparent")
        retry_frame.pack(fill="x")

        retry_label = ctk.CTkLabel(
            retry_frame,
            text="最大重试次数",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
            width=150,
            anchor="w",
        )
        retry_label.pack(side="left")

        self._retry_var = ctk.IntVar(value=self.proc_config.max_retry)

        retry_entry = ctk.CTkEntry(
            retry_frame,
            textvariable=self._retry_var,
            width=100,
        )
        retry_entry.pack(side="left", padx=(SPACING["sm"], 0))

    def _on_chunk_size_change(self, value):
        """分块大小滑块变化"""
        self._chunk_value_label.configure(text=f"{int(value)}")

    def _setup_proxy_card(self, parent):
        """设置代理配置卡片"""
        from gui.components.card import Card

        proxy_card = Card(parent, title="代理配置", subtitle="配置网络代理（可选）", padding="lg")
        proxy_card.pack(fill="x", pady=(0, SPACING["lg"]))

        # 启用代理开关
        enable_frame = ctk.CTkFrame(proxy_card.content, fg_color="transparent")
        enable_frame.pack(fill="x", pady=(0, SPACING["md"]))

        self._proxy_enabled_var = ctk.BooleanVar(value=False)

        enable_switch = ctk.CTkSwitch(
            enable_frame,
            text="启用代理",
            variable=self._proxy_enabled_var,
            font=ctk.CTkFont(size=13),
            text_color=get_color("fg_primary", mode="auto"),
        )
        enable_switch.pack(anchor="w")

        # 代理地址
        url_frame = ctk.CTkFrame(proxy_card.content, fg_color="transparent")
        url_frame.pack(fill="x")

        url_label = ctk.CTkLabel(
            url_frame,
            text="代理地址",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=get_color("fg_primary", mode="auto"),
            width=150,
            anchor="w",
        )
        url_label.pack(side="left")

        self._proxy_url_var = ctk.StringVar(value="")

        url_entry = ctk.CTkEntry(
            url_frame,
            textvariable=self._proxy_url_var,
            placeholder_text="http://proxy.example.com:8080",
        )
        url_entry.pack(side="left", fill="x", expand=True, padx=(SPACING["sm"], 0))

    def _setup_actions(self, parent):
        """设置操作按钮"""
        action_frame = ctk.CTkFrame(parent, fg_color="transparent")
        action_frame.pack(fill="x", pady=(SPACING["lg"], 0))

        from gui.components.button import Button, ButtonSize, ButtonVariant

        save_button = Button(
            action_frame,
            text="保存配置",
            variant=ButtonVariant.PRIMARY,
            size=ButtonSize.MD,
            command=self._on_save,
            width=120,
        )
        save_button.pack(side="left")

        reset_button = Button(
            action_frame,
            text="重置默认",
            variant=ButtonVariant.SECONDARY,
            size=ButtonSize.MD,
            command=self._on_reset,
            width=120,
        )
        reset_button.pack(side="left", padx=(SPACING["sm"], 0))

    def _on_save(self):
        """保存配置"""
        from tkinter import messagebox

        # 收集配置
        config_lines = []

        # API 配置
        config_lines.append(f"API_PROVIDER={self._provider_var.get()}")

        # 根据提供商保存对应配置
        provider = self._provider_var.get()
        match provider:
            case "OpenAI":
                config_lines.append(f"OPENAI_API_KEY={self._openai_key_var.get()}")
                config_lines.append(f"OPENAI_API_BASE={self._openai_base_var.get()}")
                config_lines.append(f"OPENAI_MODEL={self._openai_model_var.get()}")
            case "Gemini":
                config_lines.append(f"GEMINI_API_KEY={self._gemini_key_var.get()}")
                config_lines.append(f"GEMINI_MODEL={self._gemini_model_var.get()}")
            case "Zhipu":
                config_lines.append(f"ZHIPU_API_KEY={self._zhipu_key_var.get()}")
                config_lines.append(f"ZHIPU_API_BASE={self._zhipu_base_var.get()}")
                config_lines.append(f"ZHIPU_MODEL={self._zhipu_model_var.get()}")
            case "AiHubMix":
                config_lines.append(f"AIHUBMIX_API_KEY={self._aihubmix_key_var.get()}")
                config_lines.append(f"AIHUBMIX_API_BASE={self._aihubmix_base_var.get()}")
                config_lines.append(f"AIHUBMIX_MODEL={self._aihubmix_model_var.get()}")

        # 处理配置
        config_lines.append(f"TARGET_TOKENS_PER_CHUNK={self._chunk_size_var.get()}")
        config_lines.append(f"PARALLEL_LIMIT={self._parallel_var.get()}")
        config_lines.append(f"MAX_RETRY={self._retry_var.get()}")

        # 代理配置
        if self._proxy_enabled_var.get():
            config_lines.append(f"HTTP_PROXY={self._proxy_url_var.get()}")
            config_lines.append(f"HTTPS_PROXY={self._proxy_url_var.get()}")

        # 写入 .env 文件
        env_file = Path(".env")
        try:
            if env_file.exists():
                backup_file = Path(".env.backup")
                env_file.rename(backup_file)

            env_file.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

            messagebox.showinfo("成功", "配置已保存。请重启应用以使配置生效。")
            logger.info("Configuration saved")

            _refresh_config_cache()

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            messagebox.showerror("错误", f"保存配置失败: {e}")

    def _on_reset(self):
        """重置为默认配置"""
        from tkinter import messagebox

        if messagebox.askyesno("确认", "确定要重置为默认配置吗？"):
            # TODO: 实现重置逻辑
            pass
