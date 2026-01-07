"""
创建环境变量文件
用于快速设置项目配置
"""

import sys
from datetime import datetime
from pathlib import Path

from prompts import DEFAULT_OUTLINE_PROMPT_TEMPLATE


def create_env_file():
    """创建 .env 文件"""
    env_file = Path(".env")

    # 如果文件已存在，询问是否覆盖
    if env_file.exists():
        print(f"⚠️  {env_file} 已存在")
        response = input("是否覆盖？(y/n): ").strip().lower()
        if response not in ["y", "yes", "Y"]:
            print("操作已取消")
            return

    # 获取用户选择
    print("\n" + "=" * 60)
    print("小说大纲生成工具 - 初始化配置")
    print("=" * 60)

    print("\n请选择API提供商:")
    print("1. OpenAI (推荐，需要国外网络或代理)")
    print("2. 智谱清言 (国产，易用)")
    print("3. Google Gemini (需要国外网络或代理)")
    print("4. AiHubMix (国产，易用)")

    while True:
        choice = input("\n请选择 (1/2/3/4): ").strip()
        if choice == "1":
            api_provider = "openai"
            api_key_var = "OPENAI_API_KEY"
            api_model = "gpt-4o-mini"
            api_base_comment = "OPENAI_API_BASE=https://api.openai.com/v1"
            break
        elif choice == "2":
            api_provider = "zhipu"
            api_key_var = "ZHIPU_API_KEY"
            api_model = "glm-4-flash"
            api_base_comment = "ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4"
            break
        elif choice == "3":
            api_provider = "gemini"
            api_key_var = "GEMINI_API_KEY"
            api_model = "gemini-2.5-flash"
            api_base_comment = "# GEMINI_API_BASE=https://generativelanguage.googleapis.com"
            break
        elif choice == "4":
            api_provider = "aihubmix"
            api_key_var = "AIHUBMIX_API_KEY"
            api_model = "gpt-3.5-turbo"
            api_base_comment = "AIHUBMIX_API_BASE=https://aihubmix.com/v1"
            break
        else:
            print("无效选择，请输入 1、2、3 或 4")

    # 获取API密钥
    print(f"\n请输入你的 {api_provider.upper()} API 密钥:")
    print("(留空则稍后手动配置)")
    api_key = input("API密钥: ").strip()

    # 询问是否使用代理
    use_proxy = input("\n是否使用代理？(y/n，默认n): ").strip().lower()
    use_proxy = use_proxy in ["y", "yes", "Y"]

    proxy_url = ""
    if use_proxy:
        proxy_url = input("代理地址 (例: http://127.0.0.1:7890): ").strip()

    # 生成 .env 内容
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_content = f"""# 小说大纲生成工具配置文件
# 自动生成于 {date_str}

# API提供商
API_PROVIDER={api_provider}

"""

    # 添加对应的API配置
    if api_provider == "openai":
        env_content += f"""# OpenAI API配置
{api_key_var}={api_key if api_key else 'your_openai_api_key_here'}
{api_base_comment}
OPENAI_MODEL={api_model}

"""
    elif api_provider == "gemini":
        env_content += f"""# Google Gemini API配置
{api_key_var}={api_key if api_key else 'your_gemini_api_key_here'}
GEMINI_MODEL={api_model}
GEMINI_SAFETY_SETTINGS=BLOCK_ONLY_HIGH

"""
    elif api_provider == "zhipu":
        env_content += f"""# 智谱清言 API配置
{api_key_var}={api_key if api_key else 'your_zhipu_api_key_here'}
{api_base_comment}
ZHIPU_MODEL={api_model}

"""
    elif api_provider == "aihubmix":
        env_content += f"""# AiHubMix API配置
{api_key_var}={api_key if api_key else 'your_aihubmix_api_key_here'}
{api_base_comment}
AIHUBMIX_MODEL={api_model}

"""

    # 添加代理配置
    if use_proxy and proxy_url:
        env_content += f"""
# 代理配置
USE_PROXY=true
PROXY_URL={proxy_url}
"""
    else:
        env_content += """
# 代理配置
USE_PROXY=false
# PROXY_URL=http://127.0.0.1:7890
"""

    # 添加通用配置
    env_content += """
# 处理参数（一般无需修改）
MODEL_MAX_TOKENS=200000
TARGET_TOKENS_PER_CHUNK=6000
PARALLEL_LIMIT=5
MAX_RETRY=5
LOG_EVERY=1
"""

    outline_prompt_env = DEFAULT_OUTLINE_PROMPT_TEMPLATE.replace("\n", "\\n")
    env_content += f"""
# 大纲提示词配置（使用 {{chunk}}/{{idx}} 占位符，\n 表示换行）
OUTLINE_PROMPT_TEMPLATE={outline_prompt_env}
"""

    # 写入文件
    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_content)
        print(f"\n✅ 配置文件已创建: {env_file}")

        if not api_key:
            print(f"\n⚠️  请编辑 {env_file} 文件，填入你的API密钥")
            print(f"   {api_key_var}=your_api_key_here")

        # 提供下一步指引
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行程序: python main.py")

    except Exception as e:
        print(f"\n❌ 创建文件失败: {e}")


def show_help():
    """显示帮助信息"""
    print(
        """
小说大纲生成工具 - 配置助手

用法:
  python create_env.py     创建 .env 配置文件

支持的API提供商:
  1. OpenAI - 需要国外网络或代理
     获取密钥: https://platform.openai.com/api-keys

  2. 智谱清言 - 国内可直接使用
     获取密钥: https://open.bigmodel.cn/

  3. Google Gemini - 需要国外网络或代理
     获取密钥: https://makersuite.google.com/app/apikey

  4. AiHubMix - 国内可直接使用
     获取密钥: https://aihubmix.com/

配置文件说明:
  .env - 环境变量配置文件（不要提交到版本控制）

示例:
  # 使用智谱API
  API_PROVIDER=zhipu
  ZHIPU_API_KEY=your_api_key_here

  # 使用OpenAI + 代理
  API_PROVIDER=openai
  OPENAI_API_KEY=your_api_key_here
  USE_PROXY=true
  PROXY_URL=http://127.0.0.1:7890

  # 使用AiHubMix
  API_PROVIDER=aihubmix
  AIHUBMIX_API_KEY=your_api_key_here
"""
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
    else:
        create_env_file()
