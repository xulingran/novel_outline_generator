"""
配置管理模块
使用环境变量管理配置，提高安全性
"""

import os
from dataclasses import dataclass, field

from exceptions import APIKeyError, ConfigurationError
from prompts import DEFAULT_OUTLINE_PROMPT_TEMPLATE

# 加载.env文件中的环境变量
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # 如果没有安装python-dotenv，尝试手动加载
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


@dataclass
class APIConfig:
    """API配置类"""

    provider: str = field(default_factory=lambda: os.getenv("API_PROVIDER", "openai"))
    openai_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_BASE"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    gemini_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_safety: str = field(
        default_factory=lambda: os.getenv("GEMINI_SAFETY_SETTINGS", "BLOCK_NONE")
    )
    zhipu_key: str | None = field(default_factory=lambda: os.getenv("ZHIPU_API_KEY"))
    zhipu_base: str | None = field(
        default_factory=lambda: os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    )
    zhipu_model: str = field(default_factory=lambda: os.getenv("ZHIPU_MODEL", "glm-4-flash"))
    _validated: bool = field(default=False, init=False)

    def validate(self) -> None:
        """验证配置（延迟到实际使用时）"""
        if self._validated:
            return

        if self.provider not in ["openai", "gemini", "zhipu"]:
            raise ConfigurationError(
                f"不支持的API提供商: {self.provider}. 支持的提供商: openai, gemini, zhipu"
            )

        if self.provider == "openai":
            if (
                not self.openai_key
                or "your_" in self.openai_key.lower()
                or "here" in self.openai_key.lower()
            ):
                raise ConfigurationError(
                    "使用OpenAI API时必须设置OPENAI_API_KEY环境变量。\n"
                    "当前值看起来像是占位符，请在 .env 文件中填入真实的 API Key。\n"
                    "提示：OpenAI API Key 通常以 'sk-' 开头"
                )

        if self.provider == "gemini":
            if (
                not self.gemini_key
                or "your_" in self.gemini_key.lower()
                or "here" in self.gemini_key.lower()
            ):
                raise ConfigurationError(
                    "使用Gemini API时必须设置GEMINI_API_KEY环境变量。\n"
                    "当前值看起来像是占位符，请在 .env 文件中填入真实的 API Key"
                )

        if self.provider == "zhipu":
            if (
                not self.zhipu_key
                or "your_" in self.zhipu_key.lower()
                or "here" in self.zhipu_key.lower()
            ):
                raise ConfigurationError(
                    "使用智谱API时必须设置ZHIPU_API_KEY环境变量。\n"
                    "当前值看起来像是占位符，请在 .env 文件中填入真实的 API Key"
                )

        self._validated = True

    @property
    def api_key(self) -> str:
        """获取当前API密钥"""
        # 先验证配置
        self.validate()

        if self.provider == "openai":
            if not self.openai_key:
                raise APIKeyError("OpenAI API密钥未配置")
            return self.openai_key
        elif self.provider == "gemini":
            if not self.gemini_key:
                raise APIKeyError("Gemini API密钥未配置")
            return self.gemini_key
        else:  # zhipu
            if not self.zhipu_key:
                raise APIKeyError("智谱API密钥未配置")
            return self.zhipu_key

    @property
    def base_url(self) -> str | None:
        """获取API基础URL"""
        if self.provider == "openai":
            return self.openai_base
        elif self.provider == "zhipu":
            return self.zhipu_base
        else:
            return None

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        if self.provider == "openai":
            return self.openai_model
        elif self.provider == "gemini":
            return self.gemini_model
        else:  # zhipu
            return self.zhipu_model


@dataclass
class ProcessingConfig:
    """处理配置类"""

    # 文件配置
    default_txt_file: str = field(default="novel.txt")
    output_dir: str = field(default="outputs")
    progress_file: str = field(init=False)

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
    model_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MODEL_MAX_TOKENS", "200000"))
    )
    target_tokens_per_chunk: int = field(
        default_factory=lambda: int(os.getenv("TARGET_TOKENS_PER_CHUNK", "6000"))
    )
    parallel_limit: int = field(default_factory=lambda: int(os.getenv("PARALLEL_LIMIT", "5")))
    max_retry: int = field(default_factory=lambda: int(os.getenv("MAX_RETRY", "5")))
    log_every: int = field(default_factory=lambda: int(os.getenv("LOG_EVERY", "1")))

    # 代理配置
    use_proxy: bool = field(
        default_factory=lambda: os.getenv("USE_PROXY", "false").lower() == "true"
    )
    proxy_url: str = field(default_factory=lambda: os.getenv("PROXY_URL", "http://127.0.0.1:7897"))

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


# 创建全局配置实例
_api_config = None
_processing_config = None


def get_api_config() -> APIConfig:
    """获取API配置单例"""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config


def get_processing_config() -> ProcessingConfig:
    """获取处理配置单例"""
    global _processing_config
    if _processing_config is None:
        _processing_config = ProcessingConfig()
    return _processing_config


# 为了向前兼容，保留一些常量
def get_txt_file() -> str:
    """获取默认文本文件路径"""
    return get_processing_config().default_txt_file


def get_output_dir() -> str:
    """获取输出目录"""
    return get_processing_config().output_dir


def get_progress_file() -> str:
    """获取进度文件路径"""
    return get_processing_config().progress_file


def get_encodings() -> list[str]:
    """获取支持的编码列表"""
    return get_processing_config().encodings


# 向后兼容的常量（已弃用，建议使用配置函数）
# 注意：API_KEY等需要在首次访问时验证的常量，会延迟到实际使用时才验证
# 如果需要在导入时避免验证，请直接使用配置函数
TXT_FILE = get_txt_file()
OUTPUT_DIR = get_output_dir()
PROGRESS_FILE = get_progress_file()
ENCODINGS = get_encodings()

# API相关配置 - 延迟验证，避免在导入时抛出异常
# 只有在实际使用时才会调用api_key属性（会触发验证）
try:
    _api_cfg = get_api_config()
    API_PROVIDER = _api_cfg.provider
    # 不在这里验证api_key，避免导入时异常
    # API_KEY 将通过 get_api_config().api_key 访问
    API_BASE = _api_cfg.base_url
    MODEL_NAME = _api_cfg.model_name
    GEMINI_SAFETY_SETTINGS = _api_cfg.gemini_safety
except Exception:
    # 如果配置加载失败，使用默认值（向后兼容）
    API_PROVIDER = "openai"
    API_BASE = None
    MODEL_NAME = "gpt-4o-mini"
    GEMINI_SAFETY_SETTINGS = "BLOCK_NONE"


# API_KEY延迟访问函数（向后兼容）
def _get_api_key():
    """获取API密钥（延迟验证）"""
    return get_api_config().api_key


# 为了向后兼容，创建一个对象模拟API_KEY属性
class _APIKeyWrapper:
    """API密钥包装器，延迟验证"""

    def __str__(self):
        return _get_api_key()

    def __repr__(self):
        return repr(str(self))


API_KEY = _APIKeyWrapper()

# 处理配置（这些不会在导入时验证）
_processing_cfg = get_processing_config()
MODEL_MAX_TOKENS = _processing_cfg.model_max_tokens
TARGET_TOKENS_PER_CHUNK = _processing_cfg.target_tokens_per_chunk
PARALLEL_LIMIT = _processing_cfg.parallel_limit
MAX_RETRY = _processing_cfg.max_retry
LOG_EVERY = _processing_cfg.log_every
USE_PROXY = _processing_cfg.use_proxy
PROXY_URL = _processing_cfg.proxy_url


def create_env_file():
    """创建.env文件模板（如果不存在）"""
    env_file = ".env"
    if not os.path.exists(env_file):
        outline_prompt_env = DEFAULT_OUTLINE_PROMPT_TEMPLATE.replace("\n", "\\n")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(
                """# 小说大纲生成工具环境变量配置
# 复制此文件并填入你的API密钥

# API提供商选择: openai, gemini 或 zhipu
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

# 处理参数（可选）
MODEL_MAX_TOKENS=200000
TARGET_TOKENS_PER_CHUNK=6000
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
"""
            )
        with open(env_file, "a", encoding="utf-8") as f:
            f.write("\n# 大纲提示词配置\n")
            f.write(f"OUTLINE_PROMPT_TEMPLATE={outline_prompt_env}\n")
        print(f"✓ 已创建环境变量模板文件: {env_file}")
        print("  请编辑该文件并填入你的API密钥")


# 检查并创建环境变量文件
if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    if not os.path.exists(".env"):
        print("\n⚠️  警告: 未检测到API密钥环境变量")
        print("  建议使用环境变量或.env文件来管理API密钥")
        create_env_file()
