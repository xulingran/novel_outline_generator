"""配置编辑对话框。"""

import logging
import re
from pathlib import Path
from typing import Any

import customtkinter as ctk

from config import (
    SUPPORTED_API_PROVIDERS,
    _refresh_config_cache,
    get_api_config,
    get_processing_config,
)

logger = logging.getLogger(__name__)
ENV_LINE_PATTERN = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<sep>\s*=\s*)"
    r"(?P<value>[^#\n\r]*)"
    r"(?P<comment>\s*(?:#.*)?)$"
)


class ConfigDialog(ctk.CTkToplevel):
    """编辑并保存 `.env` 配置。"""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.title("配置管理")
        self.geometry("700x600")
        self.grab_set()

        self.api_config = get_api_config()
        self.proc_config = get_processing_config()
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_frame = ctk.CTkScrollableFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self._setup_api_config(main_frame)
        self._setup_processing_config(main_frame)
        self._setup_proxy_config(main_frame)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(btn_frame, text="保存配置", command=self._on_save, width=120).pack(
            side="left", padx=5
        )
        ctk.CTkButton(btn_frame, text="取消", command=self.destroy, width=120).pack(
            side="left", padx=5
        )
        ctk.CTkButton(btn_frame, text="导出配置", command=self._on_export, width=120).pack(
            side="right", padx=5
        )

    def _setup_api_config(self, parent: Any) -> None:
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(frame, text="API 配置", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(15, 10)
        )

        provider_frame = ctk.CTkFrame(frame, fg_color="transparent")
        provider_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(provider_frame, text="API 提供商:", width=120).pack(side="left", padx=5)
        self.provider_var = ctk.StringVar(value=self.api_config.provider)
        ctk.CTkOptionMenu(
            provider_frame,
            values=SUPPORTED_API_PROVIDERS,
            variable=self.provider_var,
        ).pack(side="left", padx=5)

        self._create_key_field(
            frame, "OpenAI API Key:", self.api_config.openai_key or "", "openai_key_var"
        )
        self._create_text_field(
            frame, "OpenAI Base URL:", self.api_config.openai_base or "", "openai_base_var"
        )
        self._create_text_field(
            frame, "OpenAI 模型:", self.api_config.openai_model, "openai_model_var"
        )

        self._create_key_field(
            frame, "Gemini API Key:", self.api_config.gemini_key or "", "gemini_key_var"
        )
        self._create_text_field(
            frame, "Gemini 模型:", self.api_config.gemini_model, "gemini_model_var"
        )

        self._create_key_field(
            frame, "智谱 API Key:", self.api_config.zhipu_key or "", "zhipu_key_var"
        )
        self._create_text_field(
            frame, "智谱 Base URL:", self.api_config.zhipu_base or "", "zhipu_base_var"
        )
        self._create_text_field(frame, "智谱模型:", self.api_config.zhipu_model, "zhipu_model_var")

        self._create_key_field(
            frame,
            "AiHubMix API Key:",
            self.api_config.aihubmix_api_key or "",
            "aihubmix_key_var",
        )
        self._create_text_field(
            frame,
            "AiHubMix Base URL:",
            self.api_config.aihubmix_api_base or "",
            "aihubmix_base_var",
        )
        self._create_text_field(
            frame,
            "AiHubMix 模型:",
            self.api_config.aihubmix_model,
            "aihubmix_model_var",
        )

    def _setup_processing_config(self, parent: Any) -> None:
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(frame, text="处理配置", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(15, 10)
        )

        self.chunk_size_var = ctk.StringVar(value=str(self.proc_config.target_tokens_per_chunk))
        self.parallel_limit_var = ctk.StringVar(value=str(self.proc_config.parallel_limit))
        self.max_retry_var = ctk.StringVar(value=str(self.proc_config.max_retry))

        self._line_input(frame, "目标分块大小:", self.chunk_size_var, unit="tokens")
        self._line_input(frame, "并发限制:", self.parallel_limit_var)
        self._line_input(frame, "最大重试次数:", self.max_retry_var)

    def _setup_proxy_config(self, parent: Any) -> None:
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(frame, text="代理配置", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(15, 10)
        )

        self.proxy_enabled_var = ctk.BooleanVar(value=self.proc_config.use_proxy)
        self.proxy_url_var = ctk.StringVar(value=self.proc_config.proxy_url)

        ctk.CTkCheckBox(frame, text="启用代理", variable=self.proxy_enabled_var).pack(
            anchor="w", padx=20, pady=5
        )
        self._line_input(frame, "代理地址:", self.proxy_url_var)

    def _line_input(
        self,
        parent: Any,
        label: str,
        var: Any,
        unit: str = "",
    ) -> None:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(frame, text=label, width=120).pack(side="left", padx=5)
        ctk.CTkEntry(frame, textvariable=var, width=220).pack(side="left", padx=5)
        if unit:
            ctk.CTkLabel(frame, text=unit).pack(side="left", padx=5)

    def _create_text_field(self, parent: Any, label: str, default: str, var_name: str) -> None:
        var = ctk.StringVar(value=default)
        setattr(self, var_name, var)
        self._line_input(parent, label, var)

    def _create_key_field(self, parent: Any, label: str, default: str, var_name: str) -> None:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(frame, text=label, width=150).pack(side="left", padx=5)
        var = ctk.StringVar(value=default)
        setattr(self, var_name, var)
        entry = ctk.CTkEntry(frame, textvariable=var, show="*", width=300)
        entry.pack(side="left", padx=5, expand=True, fill="x")

        show_btn = ctk.CTkButton(frame, text="显示", width=60)
        show_btn.configure(command=lambda: self._toggle_password(entry, show_btn))
        show_btn.pack(side="left", padx=5)

    def _toggle_password(self, entry: Any, button: Any) -> None:
        """切换密钥显隐。"""
        if entry.cget("show") == "*":
            entry.configure(show="")
            button.configure(text="隐藏")
        else:
            entry.configure(show="*")
            button.configure(text="显示")

    def _collect_env_lines(self) -> list[str]:
        return [f"{key}={value}" for key, value in self._collect_env_updates().items()]

    def _collect_env_updates(self) -> dict[str, str]:
        """收集配置项键值。"""
        return {
            "API_PROVIDER": self.provider_var.get(),
            "OPENAI_API_KEY": self.openai_key_var.get(),
            "OPENAI_API_BASE": self.openai_base_var.get(),
            "OPENAI_MODEL": self.openai_model_var.get(),
            "GEMINI_API_KEY": self.gemini_key_var.get(),
            "GEMINI_MODEL": self.gemini_model_var.get(),
            "ZHIPU_API_KEY": self.zhipu_key_var.get(),
            "ZHIPU_API_BASE": self.zhipu_base_var.get(),
            "ZHIPU_MODEL": self.zhipu_model_var.get(),
            "AIHUBMIX_API_KEY": self.aihubmix_key_var.get(),
            "AIHUBMIX_API_BASE": self.aihubmix_base_var.get(),
            "AIHUBMIX_MODEL": self.aihubmix_model_var.get(),
            "TARGET_TOKENS_PER_CHUNK": self.chunk_size_var.get(),
            "PARALLEL_LIMIT": self.parallel_limit_var.get(),
            "MAX_RETRY": self.max_retry_var.get(),
            "USE_PROXY": "true" if self.proxy_enabled_var.get() else "false",
            "PROXY_URL": self.proxy_url_var.get(),
        }

    @staticmethod
    def _normalize_env_value(raw_value: str) -> str:
        """归一化 env 值用于比较。"""
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    @staticmethod
    def _format_updated_value(original_value: str, new_value: str) -> str:
        """在保留原引号风格的前提下更新值。"""
        stripped = original_value.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            quote = stripped[0]
            return f"{quote}{new_value}{quote}"
        return new_value

    def _update_env_file(self, env_file: Path, updates: dict[str, str]) -> None:
        """仅更新变更项，保留注释和未改动行。"""
        if not env_file.exists():
            env_file.write_text(
                "\n".join(f"{key}={value}" for key, value in updates.items()) + "\n",
                encoding="utf-8",
            )
            return

        lines = env_file.read_text(encoding="utf-8").splitlines()
        updated_lines: list[str] = []
        seen_keys: set[str] = set()

        for line in lines:
            match = ENV_LINE_PATTERN.match(line)
            if not match:
                updated_lines.append(line)
                continue

            key = match.group("key")
            if key not in updates:
                updated_lines.append(line)
                continue

            seen_keys.add(key)
            new_value = updates[key]
            current_value = self._normalize_env_value(match.group("value"))
            if current_value == new_value:
                updated_lines.append(line)
                continue

            replaced = (
                f"{match.group('prefix')}{key}{match.group('sep')}"
                f"{self._format_updated_value(match.group('value'), new_value)}"
                f"{self._normalize_comment_spacing(match.group('comment'))}"
            )
            updated_lines.append(replaced)

        for key, value in updates.items():
            if key not in seen_keys:
                updated_lines.append(f"{key}={value}")

        env_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _normalize_comment_spacing(comment: str) -> str:
        """确保行内注释前有一个空格。"""
        stripped = comment.lstrip()
        if not stripped:
            return comment
        if stripped.startswith("#"):
            return f" {stripped}"
        return comment

    def _on_save(self) -> None:
        """保存配置到项目根目录 `.env`。"""
        from tkinter import messagebox

        try:
            env_file = Path(".env")
            self._update_env_file(env_file, self._collect_env_updates())
            _refresh_config_cache()
            messagebox.showinfo("成功", "配置已保存。请重启应用以使配置生效。")
            self.destroy()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"保存配置失败: {exc}")
            messagebox.showerror("错误", f"保存配置失败: {exc}")

    def _on_export(self) -> None:
        """导出配置到指定路径。"""
        from tkinter import filedialog, messagebox

        filepath = filedialog.asksaveasfilename(
            defaultextension=".env",
            filetypes=[("环境变量文件", "*.env"), ("所有文件", "*.*")],
            title="导出配置",
        )
        if not filepath:
            return

        try:
            Path(filepath).write_text("\n".join(self._collect_env_lines()) + "\n", encoding="utf-8")
            messagebox.showinfo("成功", f"配置已导出到: {filepath}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"导出配置失败: {exc}")
            messagebox.showerror("错误", f"导出配置失败: {exc}")
