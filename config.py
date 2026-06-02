"""
配置管理模块
使用环境变量管理配置，提高安全性
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from exceptions import APIKeyError, ConfigurationError

logger = logging.getLogger(__name__)

# 支持的API提供商列表
SUPPORTED_API_PROVIDERS = ["openai", "gemini", "zhipu", "aihubmix"]

# 项目常量
MAX_INPUT_TOKEN_RATIO = 0.8  # 输入token占模型最大token的比例上限
RECOMMENDED_MAX_PARALLEL_LIMIT = 20  # 建议的最大并发数


def load_env_file(path: str = ".env") -> dict[str, str]:
    """读取 .env 文件键值（仅解析 KEY=VALUE 行）。"""
    env_path = os.path.abspath(path)
    if not os.path.exists(env_path):
        return {}

    data: dict[str, str] = {}
    with open(env_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def env_field(
    env_var: str,
    default: str | None = None,
    *,
    cast: type = str,
) -> Any:
    """从环境变量创建 dataclass field。

    Args:
        env_var: 环境变量名
        default: 默认值（字符串形式）
        cast: 类型转换函数（int、bool 等）
    """

    def factory():
        raw = os.getenv(env_var, default)
        if raw is None:
            return None
        if cast is bool:
            return raw.lower() == "true"
        return cast(raw) if cast is not str else raw

    return field(default_factory=factory)


@dataclass
class APIConfig:
    """API配置类"""

    provider: str = env_field("API_PROVIDER", "openai")
    openai_key: str | None = env_field("OPENAI_API_KEY")
    openai_base: str | None = env_field("OPENAI_API_BASE")
    openai_model: str = env_field("OPENAI_MODEL", "gpt-4o-mini")
    gemini_key: str | None = env_field("GEMINI_API_KEY")
    gemini_model: str = env_field("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_safety: str = env_field("GEMINI_SAFETY_SETTINGS", "BLOCK_NONE")
    zhipu_key: str | None = env_field("ZHIPU_API_KEY")
    zhipu_base: str | None = env_field("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    zhipu_model: str = env_field("ZHIPU_MODEL", "glm-4-flash")
    aihubmix_api_key: str | None = env_field("AIHUBMIX_API_KEY")
    aihubmix_model: str = env_field("AIHUBMIX_MODEL", "gpt-3.5-turbo")
    aihubmix_api_base: str | None = env_field("AIHUBMIX_API_BASE", "https://aihubmix.com/v1")
    _validated: bool = field(default=False, init=False)

    # 统一的 provider 配置映射（类变量，不参与 dataclass 机制）
    _PROVIDER_REGISTRY: ClassVar[dict[str, dict[str, str | None]]] = {
        "openai": {
            "key_field": "openai_key",
            "base_field": "openai_base",
            "model_field": "openai_model",
            "name": "OpenAI API",
            "env_var": "OPENAI_API_KEY",
            "hint": "提示：OpenAI API Key 通常以 'sk-' 开头",
        },
        "gemini": {
            "key_field": "gemini_key",
            "base_field": None,
            "model_field": "gemini_model",
            "name": "Gemini API",
            "env_var": "GEMINI_API_KEY",
            "hint": "",
        },
        "zhipu": {
            "key_field": "zhipu_key",
            "base_field": "zhipu_base",
            "model_field": "zhipu_model",
            "name": "智谱API",
            "env_var": "ZHIPU_API_KEY",
            "hint": "",
        },
        "aihubmix": {
            "key_field": "aihubmix_api_key",
            "base_field": "aihubmix_api_base",
            "model_field": "aihubmix_model",
            "name": "AiHubMix API",
            "env_var": "AIHUBMIX_API_KEY",
            "hint": "",
        },
    }

    def _validate_api_key(self, key_value: str | None, config: dict[str, str | None]) -> None:
        """验证单个 API 密钥

        Args:
            key_value: API 密钥值
            config: 提供商配置字典
        """
        if not key_value or "your_" in key_value.lower() or "here" in key_value.lower():
            name = config["name"]
            env_var = config["env_var"]
            hint = config.get("hint", "")
            msg = (
                f"使用{name}时必须设置{env_var}环境变量。\n"
                "当前值看起来像是占位符，请在 .env 文件中填入真实的 API Key"
            )
            if hint:
                msg += f"\n{hint}"
            raise ConfigurationError(msg)

    def validate(self) -> None:
        """验证配置（延迟到实际使用时）"""
        if self._validated:
            return

        if self.provider not in SUPPORTED_API_PROVIDERS:
            raise ConfigurationError(
                f"不支持的API提供商: {self.provider}. "
                f"支持的提供商: {', '.join(SUPPORTED_API_PROVIDERS)}"
            )

        # 使用配置驱动的方式验证 API 密钥
        provider_config = self._PROVIDER_REGISTRY.get(self.provider)
        if provider_config:
            key_field = provider_config["key_field"]
            if key_field is not None:
                key_value = getattr(self, key_field)
                self._validate_api_key(key_value, provider_config)

        self._validated = True

    @property
    def api_key(self) -> str:
        """获取当前API密钥"""
        self.validate()
        config = self._PROVIDER_REGISTRY[self.provider]
        key_field = config["key_field"]
        assert isinstance(key_field, str)
        value: str | None = getattr(self, key_field)
        if not value:
            raise APIKeyError(f"{config['name']}密钥未配置")
        return value

    @property
    def base_url(self) -> str | None:
        """获取API基础URL"""
        config = self._PROVIDER_REGISTRY.get(self.provider)
        if not config or not config.get("base_field"):
            return None
        base_field = config["base_field"]
        assert isinstance(base_field, str)
        result: str | None = getattr(self, base_field)
        return result

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        config = self._PROVIDER_REGISTRY[self.provider]
        model_field = config["model_field"]
        assert isinstance(model_field, str)
        return str(getattr(self, model_field))


@dataclass
class ProcessingConfig:
    """处理配置类"""

    # 文件配置
    default_txt_file: str = field(default="novel.txt")
    output_dir: str = field(default="outputs")
    progress_file: str = field(init=False)
    # 支持的文本文件扩展名
    allowed_extensions: list[str] = field(default_factory=lambda: [".txt", ".md", ".text"])
    # 上传文件大小限制（MB）
    max_upload_file_size_mb: int = env_field("MAX_UPLOAD_FILE_SIZE_MB", "100", cast=int)

    # 编码配置
    encodings: list[str] = field(
        default_factory=lambda: [
            "utf-8",
            "gbk",
            "gb2312",
            "gb18030",
            "big5",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
            "latin1",
            "cp1252",
        ]
    )

    # 处理参数
    model_max_tokens: int = env_field("MODEL_MAX_TOKENS", "200000", cast=int)
    target_tokens_per_chunk: int = env_field("TARGET_TOKENS_PER_CHUNK", "64000", cast=int)
    parallel_limit: int = env_field("PARALLEL_LIMIT", "5", cast=int)
    max_retry: int = env_field("MAX_RETRY", "5", cast=int)
    log_every: int = env_field("LOG_EVERY", "1", cast=int)
    sub_chunk_count: int = env_field("SUB_CHUNK_COUNT", "5", cast=int)
    retry_backoff_base: int = env_field("RETRY_BACKOFF_BASE", "1", cast=int)
    stream_split_threshold_mb: int = env_field("STREAM_SPLIT_THRESHOLD_MB", "20", cast=int)

    # 代理配置
    use_proxy: bool = env_field("USE_PROXY", "false", cast=bool)
    proxy_url: str = env_field("PROXY_URL", "http://127.0.0.1:7897")

    def __post_init__(self):
        """初始化后处理"""
        self.progress_file = os.path.join(self.output_dir, "progress.json")

    def validate(self) -> None:
        """验证配置"""
        if self.model_max_tokens <= 0:
            raise ConfigurationError("MODEL_MAX_TOKENS必须大于0")

        if self.target_tokens_per_chunk <= 0:
            raise ConfigurationError("TARGET_TOKENS_PER_CHUNK必须大于0")

        if self.target_tokens_per_chunk >= self.model_max_tokens:
            raise ConfigurationError("TARGET_TOKENS_PER_CHUNK必须小于MODEL_MAX_TOKENS")

        if self.parallel_limit <= 0:
            raise ConfigurationError("PARALLEL_LIMIT必须大于0")

        if self.max_retry < 0:
            raise ConfigurationError("MAX_RETRY不能小于0")

        if self.sub_chunk_count <= 0:
            raise ConfigurationError("SUB_CHUNK_COUNT必须大于0")

        if self.retry_backoff_base < 0:
            raise ConfigurationError("RETRY_BACKOFF_BASE不能小于0")

        if self.stream_split_threshold_mb <= 0:
            raise ConfigurationError("STREAM_SPLIT_THRESHOLD_MB必须大于0")


# 创建全局配置实例
_api_config = None
_processing_config = None
_config_lock = threading.Lock()


def get_api_config() -> APIConfig:
    """获取API配置单例（线程安全）"""
    global _api_config
    if _api_config is None:
        with _config_lock:
            if _api_config is None:
                _api_config = APIConfig()
    return _api_config


def get_processing_config() -> ProcessingConfig:
    """获取处理配置单例（线程安全）"""
    global _processing_config
    if _processing_config is None:
        with _config_lock:
            if _processing_config is None:
                _processing_config = ProcessingConfig()
    return _processing_config


def reset_api_config() -> None:
    """重置API配置单例（主要用于测试）"""
    global _api_config
    _api_config = None


def reset_processing_config() -> None:
    """重置处理配置单例（主要用于测试）"""
    global _processing_config
    _processing_config = None


def reset_all_configs() -> None:
    """重置所有配置单例（主要用于测试）"""
    reset_api_config()
    reset_processing_config()


def _refresh_config_cache() -> None:
    """重置配置单例缓存，使下次访问时重新加载环境变量"""
    global _api_config, _processing_config
    _api_config = None
    _processing_config = None


class _LazyAPIKey:
    """延迟验证的 API Key 包装器

    避免在导入时触发 API Key 验证，只在实际使用（转换为字符串）时才检查。
    """

    def __str__(self) -> str:
        return get_api_config().api_key

    def __repr__(self) -> str:
        return repr(str(self))


# 模块级缓存变量（延迟初始化）


class _LazyConfig:
    """延迟配置访问包装器 - 保持向后兼容

    通过 __getattr__ 实现延迟加载，只有在实际访问时才计算配置值。
    这样可以避免模块导入时的副作用，同时保持 `from config import XXX` 的用法。
    """

    def __getattr__(self, name: str) -> Any:
        """延迟加载配置值"""
        value = self._resolve_config(name)
        return value

    def _resolve_config(self, name: str) -> Any:
        """根据名称解析配置值"""
        match name:
            # 路径配置
            case "TXT_FILE":
                return get_processing_config().default_txt_file
            case "OUTPUT_DIR":
                return get_processing_config().output_dir
            case "PROGRESS_FILE":
                return get_processing_config().progress_file
            case "ENCODINGS":
                return get_processing_config().encodings

            # API 配置（延迟验证，避免导入时触发）
            case "API_PROVIDER":
                return get_api_config().provider
            case "API_BASE":
                return get_api_config().base_url
            case "MODEL_NAME":
                return get_api_config().model_name
            case "GEMINI_SAFETY_SETTINGS":
                return get_api_config().gemini_safety
            case "API_KEY":
                # 延迟验证，只在实际使用时检查
                return _LazyAPIKey()

            # 处理配置
            case "MODEL_MAX_TOKENS":
                return get_processing_config().model_max_tokens
            case "TARGET_TOKENS_PER_CHUNK":
                return get_processing_config().target_tokens_per_chunk
            case "PARALLEL_LIMIT":
                return get_processing_config().parallel_limit
            case "MAX_RETRY":
                return get_processing_config().max_retry
            case "LOG_EVERY":
                return get_processing_config().log_every
            case "USE_PROXY":
                return get_processing_config().use_proxy
            case "PROXY_URL":
                return get_processing_config().proxy_url

            case _:
                raise AttributeError(f"模块 'config' 没有属性 '{name}'")


# 创建延迟配置实例
_lazy_config = _LazyConfig()


# 为了支持 `from config import XXX` 语法，我们在模块级别定义 __getattr__
# Python 3.7+ 支持模块级别的 __getattr__
def __getattr__(name: str) -> Any:
    """模块级延迟属性访问"""
    return getattr(_lazy_config, name)


# 为了 IDE 和类型检查器能识别这些常量，我们声明它们的类型
# 实际值通过 __getattr__ 延迟加载
if TYPE_CHECKING:
    # 路径配置
    TXT_FILE: str = ""
    OUTPUT_DIR: str = ""
    PROGRESS_FILE: str = ""
    ENCODINGS: list[str] = []

    # API 配置
    API_PROVIDER: str = ""
    API_BASE: str | None = None
    MODEL_NAME: str = ""
    GEMINI_SAFETY_SETTINGS: str = ""
    API_KEY: str = ""

    # 处理配置
    MODEL_MAX_TOKENS: int = 0
    TARGET_TOKENS_PER_CHUNK: int = 0
    PARALLEL_LIMIT: int = 0
    MAX_RETRY: int = 0
    LOG_EVERY: int = 0
    USE_PROXY: bool = False
    PROXY_URL: str = ""


def create_env_file():
    """创建.env文件模板（如果不存在）"""
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("""# 小说大纲生成工具环境变量配置
# 复制此文件并填入你的API密钥

# API提供商选择: openai, gemini, zhipu 或 aihubmix
API_PROVIDER=openai

# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Gemini API配置（使用时取消注释并配置）
# GEMINI_API_KEY=your_gemini_api_key_here
# GEMINI_MODEL=gemini-2.5-flash
# GEMINI_SAFETY_SETTINGS=BLOCK_NONE

# 智谱清言API配置（使用时取消注释并配置）
# ZHIPU_API_KEY=your_zhipu_api_key_here
# ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4
# ZHIPU_MODEL=glm-4-flash

# AiHubMix API配置（使用时取消注释并配置）
# AIHUBMIX_API_KEY=your_aihubmix_api_key_here
# AIHUBMIX_MODEL=gpt-3.5-turbo
# AIHUBMIX_API_BASE=https://aihubmix.com/v1

# 处理参数（可选）
MODEL_MAX_TOKENS=200000
TARGET_TOKENS_PER_CHUNK=64000
PARALLEL_LIMIT=5
MAX_RETRY=5
LOG_EVERY=1

# 日志级别（可选）: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# 代理配置（可选）
USE_PROXY=false
PROXY_URL=http://127.0.0.1:7897

# CORS 允许的来源（可选，多个用逗号分隔）
# CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# 注意：提示词模板现已内置，使用 prompts.py 中的函数
""")
        print(f"✓ 已创建环境变量模板文件: {env_file}")
        print("  请编辑该文件并填入你的API密钥")


def init_config(create_env_if_missing: bool = True):
    """初始化配置

    此函数负责：
    1. 加载 .env 文件（使用 python-dotenv 或手动读取）
    2. 检查 API 密钥是否存在
    3. 如果缺少 API 密钥且 .env 不存在，则创建 .env 文件并打印提示（可选）

    Args:
        create_env_if_missing: 如果缺少 API 密钥且 .env 不存在，是否创建 .env 文件。
            默认为 True 以保持向后兼容。在模块导入时调用应设置为 False 以避免副作用。

    注意：此函数应在需要时显式调用，而不是在模块导入时自动执行
    """
    # 加载.env文件中的环境变量
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # 如果没有安装python-dotenv，尝试手动加载
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_data = load_env_file(env_file)
        for key, value in env_data.items():
            # 只在环境变量不存在时才设置，保持环境变量优先
            if key not in os.environ:
                os.environ[key] = value

    # 检查并创建环境变量文件
    if (
        create_env_if_missing
        and not os.getenv("OPENAI_API_KEY")
        and not os.getenv("GEMINI_API_KEY")
    ):
        if not os.path.exists(".env"):
            print("\n⚠️  警告: 未检测到API密钥环境变量")
            print("  建议使用环境变量或.env文件来管理API密钥")
            create_env_file()

    _refresh_config_cache()
