"""
配置对话框模块

提供可视化配置编辑界面，支持 API 配置、处理配置、代理配置等。
"""

import logging
from pathlib import Path

import customtkinter as ctk

from config import SUPPORTED_API_PROVIDERS, _refresh_config_cache

logger = logging.getLogger(__name__)


class ConfigDialog(ctk.CTkToplevel):
    """
    配置对话框

    功能：
    - API 配置（提供商、密钥、Base URL、模型名称）
    - 处理配置（分块大小、并发限制、最大重试次数）
    - 代理配置
    - 保存到 .env 文件
    - 刷新配置缓存
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.title("配置管理")
        self.geometry("700x600")

        # 使对话框模态
        self.grab_set()

        # 加载当前配置
        self._load_current_config()

        # 设置 UI
        self._setup_ui()

        logger.info("打开配置对话框")

    def _load_current_config(self):
        """加载当前配置"""
        from config import get_api_config, get_processing_config

        self.api_config = get_api_config()
        self.proc_config = get_processing_config()

    def _setup_ui(self):
        """设置 UI"""
        # 主容器
        main_frame = ctk.CTkScrollableFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # API 配置区域
        self._setup_api_config(main_frame)

        # 处理配置区域
        self._setup_processing_config(main_frame)

        # 代理配置区域
        self._setup_proxy_config(main_frame)

        # 按钮区域
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))

        save_button = ctk.CTkButton(button_frame, text="保存配置", command=self._on_save, width=120)
        save_button.pack(side="left", padx=5)

        cancel_button = ctk.CTkButton(button_frame, text="取消", command=self.destroy, width=120)
        cancel_button.pack(side="left", padx=5)

        export_button = ctk.CTkButton(
            button_frame, text="导出配置", command=self._on_export, width=120
        )
        export_button.pack(side="right", padx=5)

    def _setup_api_config(self, parent):
        """设置 API 配置区域"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 20))

        # 标题
        title_label = ctk.CTkLabel(frame, text="API 配置", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(15, 10))

        # 提供商选择
        provider_frame = ctk.CTkFrame(frame, fg_color="transparent")
        provider_frame.pack(fill="x", padx=15, pady=5)

        provider_label = ctk.CTkLabel(provider_frame, text="API 提供商:", width=120)
        provider_label.pack(side="left", padx=5)

        self.provider_var = ctk.StringVar(value=self.api_config.provider)
        provider_menu = ctk.CTkOptionMenu(
            provider_frame, values=SUPPORTED_API_PROVIDERS, variable=self.provider_var
        )
        provider_menu.pack(side="left", padx=5)

        # OpenAI 配置
        self._create_api_key_field(
            frame, "OpenAI API Key:", self.api_config.openai_key or "", "openai_key_var", show="*"
        )
        self._create_text_field(
            frame, "OpenAI Base URL:", self.api_config.openai_base or "", "openai_base_var"
        )
        self._create_text_field(
            frame, "OpenAI 模型:", self.api_config.openai_model, "openai_model_var"
        )

        # Gemini 配置
        self._create_api_key_field(
            frame, "Gemini API Key:", self.api_config.gemini_key or "", "gemini_key_var", show="*"
        )
        self._create_text_field(
            frame, "Gemini 模型:", self.api_config.gemini_model, "gemini_model_var"
        )

        # 智谱配置
        self._create_api_key_field(
            frame, "智谱 API Key:", self.api_config.zhipu_key or "", "zhipu_key_var", show="*"
        )
        self._create_text_field(
            frame, "智谱 Base URL:", self.api_config.zhipu_base or "", "zhipu_base_var"
        )
        self._create_text_field(frame, "智谱模型:", self.api_config.zhipu_model, "zhipu_model_var")

        # AiHubMix 配置
        self._create_api_key_field(
            frame,
            "AiHubMix API Key:",
            self.api_config.aihubmix_api_key or "",
            "aihubmix_key_var",
            show="*",
        )
        self._create_text_field(
            frame,
            "AiHubMix Base URL:",
            self.api_config.aihubmix_api_base or "",
            "aihubmix_base_var",
        )
        self._create_text_field(
            frame, "AiHubMix 模型:", self.api_config.aihubmix_model, "aihubmix_model_var"
        )

    def _setup_processing_config(self, parent):
        """设置处理配置区域"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 20))

        # 标题
        title_label = ctk.CTkLabel(frame, text="处理配置", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(15, 10))

        # 目标分块大小
        chunk_frame = ctk.CTkFrame(frame, fg_color="transparent")
        chunk_frame.pack(fill="x", padx=15, pady=5)

        chunk_label = ctk.CTkLabel(chunk_frame, text="目标分块大小:", width=120)
        chunk_label.pack(side="left", padx=5)

        self.chunk_size_var = ctk.StringVar(value=str(self.proc_config.target_tokens_per_chunk))
        chunk_entry = ctk.CTkEntry(chunk_frame, textvariable=self.chunk_size_var, width=150)
        chunk_entry.pack(side="left", padx=5)

        chunk_unit = ctk.CTkLabel(chunk_frame, text="tokens")
        chunk_unit.pack(side="left", padx=5)

        # 并发限制
        parallel_frame = ctk.CTkFrame(frame, fg_color="transparent")
        parallel_frame.pack(fill="x", padx=15, pady=5)

        parallel_label = ctk.CTkLabel(parallel_frame, text="并发限制:", width=120)
        parallel_label.pack(side="left", padx=5)

        self.parallel_limit_var = ctk.StringVar(value=str(self.proc_config.parallel_limit))
        parallel_entry = ctk.CTkEntry(
            parallel_frame, textvariable=self.parallel_limit_var, width=150
        )
        parallel_entry.pack(side="left", padx=5)

        # 最大重试次数
        retry_frame = ctk.CTkFrame(frame, fg_color="transparent")
        retry_frame.pack(fill="x", padx=15, pady=5)

        retry_label = ctk.CTkLabel(retry_frame, text="最大重试次数:", width=120)
        retry_label.pack(side="left", padx=5)

        self.max_retry_var = ctk.StringVar(value=str(self.proc_config.max_retry))
        retry_entry = ctk.CTkEntry(retry_frame, textvariable=self.max_retry_var, width=150)
        retry_entry.pack(side="left", padx=5)

    def _setup_proxy_config(self, parent):
        """设置代理配置区域"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 20))

        # 标题
        title_label = ctk.CTkLabel(frame, text="代理配置", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(15, 10))

        # 启用代理
        proxy_enable_frame = ctk.CTkFrame(frame, fg_color="transparent")
        proxy_enable_frame.pack(fill="x", padx=15, pady=5)

        self.proxy_enabled_var = ctk.BooleanVar(value=False)
        proxy_enable_checkbox = ctk.CTkCheckBox(
            proxy_enable_frame, text="启用代理", variable=self.proxy_enabled_var
        )
        proxy_enable_checkbox.pack(side="left", padx=5)

        # 代理地址
        self._create_text_field(
            frame, "代理地址:", "", "proxy_url_var", placeholder="http://proxy.example.com:8080"
        )

    def _create_api_key_field(
        self, parent, label: str, default_value: str, var_name: str, show: str | None = None
    ):
        """创建 API 密钥输入字段"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=5)

        label_widget = ctk.CTkLabel(frame, text=label, width=150)
        label_widget.pack(side="left", padx=5)

        var = ctk.StringVar(value=default_value)
        setattr(self, var_name, var)

        entry = ctk.CTkEntry(frame, textvariable=var, show=show, width=300)
        entry.pack(side="left", padx=5, expand=True, fill="x")

        # 显示/隐藏按钮
        if show:
            show_button: ctk.CTkButton = ctk.CTkButton(
                frame,
                text="显示",
                width=60,
                command=lambda: self._toggle_password(entry, show_button),
            )
            show_button.pack(side="left", padx=5)

    def _create_text_field(
        self, parent, label: str, default_value: str, var_name: str, placeholder: str = ""
    ):
        """创建文本输入字段"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=5)

        label_widget = ctk.CTkLabel(frame, text=label, width=150)
        label_widget.pack(side="left", padx=5)

        var = ctk.StringVar(value=default_value)
        setattr(self, var_name, var)

        entry = ctk.CTkEntry(frame, textvariable=var, placeholder_text=placeholder)
        entry.pack(side="left", padx=5, expand=True, fill="x")

    def _toggle_password(self, entry: ctk.CTkEntry, button: ctk.CTkButton):
        """切换密码显示"""
        if entry.cget("show") == "*":
            entry.configure(show="")
            button.configure(text="隐藏")
        else:
            entry.configure(show="*")
            button.configure(text="显示")

    def _on_save(self):
        """保存配置"""
        from tkinter import messagebox

        # 收集配置
        config_lines = []

        # API 配置
        config_lines.append(f"API_PROVIDER={self.provider_var.get()}")
        config_lines.append(f"OPENAI_API_KEY={self.openai_key_var.get()}")
        config_lines.append(f"OPENAI_API_BASE={self.openai_base_var.get()}")
        config_lines.append(f"OPENAI_MODEL={self.openai_model_var.get()}")
        config_lines.append(f"GEMINI_API_KEY={self.gemini_key_var.get()}")
        config_lines.append(f"GEMINI_MODEL={self.gemini_model_var.get()}")
        config_lines.append(f"ZHIPU_API_KEY={self.zhipu_key_var.get()}")
        config_lines.append(f"ZHIPU_API_BASE={self.zhipu_base_var.get()}")
        config_lines.append(f"ZHIPU_MODEL={self.zhipu_model_var.get()}")
        config_lines.append(f"AIHUBMIX_API_KEY={self.aihubmix_key_var.get()}")
        config_lines.append(f"AIHUBMIX_API_BASE={self.aihubmix_base_var.get()}")
        config_lines.append(f"AIHUBMIX_MODEL={self.aihubmix_model_var.get()}")

        # 处理配置
        config_lines.append(f"TARGET_TOKENS_PER_CHUNK={self.chunk_size_var.get()}")
        config_lines.append(f"PARALLEL_LIMIT={self.parallel_limit_var.get()}")
        config_lines.append(f"MAX_RETRY={self.max_retry_var.get()}")

        # 代理配置
        if self.proxy_enabled_var.get():
            config_lines.append(f"HTTP_PROXY={self.proxy_url_var.get()}")
            config_lines.append(f"HTTPS_PROXY={self.proxy_url_var.get()}")

        # 写入 .env 文件
        env_file = Path(".env")
        try:
            # 备份现有配置
            if env_file.exists():
                backup_file = Path(".env.backup")
                env_file.rename(backup_file)

            # 写入新配置
            env_file.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

            messagebox.showinfo("成功", "配置已保存。请重启应用以使配置生效。")
            logger.info("配置已保存到 .env 文件")

            # 刷新配置缓存
            _refresh_config_cache()

            self.destroy()

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            messagebox.showerror("错误", f"保存配置失败: {e}")

    def _on_export(self):
        """导出配置"""
        from tkinter import filedialog, messagebox

        filepath = filedialog.asksaveasfilename(
            defaultextension=".env",
            filetypes=[("环境变量文件", "*.env"), ("所有文件", "*.*")],
            title="导出配置",
        )

        if filepath:
            try:
                # 收集配置
                config_lines = []

                # API 配置
                config_lines.append(f"API_PROVIDER={self.provider_var.get()}")
                config_lines.append(f"OPENAI_API_KEY={self.openai_key_var.get()}")
                config_lines.append(f"OPENAI_API_BASE={self.openai_base_var.get()}")
                config_lines.append(f"OPENAI_MODEL={self.openai_model_var.get()}")
                config_lines.append(f"GEMINI_API_KEY={self.gemini_key_var.get()}")
                config_lines.append(f"GEMINI_MODEL={self.gemini_model_var.get()}")
                config_lines.append(f"ZHIPU_API_KEY={self.zhipu_key_var.get()}")
                config_lines.append(f"ZHIPU_API_BASE={self.zhipu_base_var.get()}")
                config_lines.append(f"ZHIPU_MODEL={self.zhipu_model_var.get()}")
                config_lines.append(f"AIHUBMIX_API_KEY={self.aihubmix_key_var.get()}")
                config_lines.append(f"AIHUBMIX_API_BASE={self.aihubmix_base_var.get()}")
                config_lines.append(f"AIHUBMIX_MODEL={self.aihubmix_model_var.get()}")

                # 处理配置
                config_lines.append(f"TARGET_TOKENS_PER_CHUNK={self.chunk_size_var.get()}")
                config_lines.append(f"PARALLEL_LIMIT={self.parallel_limit_var.get()}")
                config_lines.append(f"MAX_RETRY={self.max_retry_var.get()}")

                # 代理配置
                if self.proxy_enabled_var.get():
                    config_lines.append(f"HTTP_PROXY={self.proxy_url_var.get()}")
                    config_lines.append(f"HTTPS_PROXY={self.proxy_url_var.get()}")

                # 写入文件
                Path(filepath).write_text("\n".join(config_lines) + "\n", encoding="utf-8")

                messagebox.showinfo("成功", f"配置已导出到: {filepath}")
                logger.info(f"配置已导出到: {filepath}")

            except Exception as e:
                logger.error(f"导出配置失败: {e}")
                messagebox.showerror("错误", f"导出配置失败: {e}")
