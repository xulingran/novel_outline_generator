# DeepSeek 和小米 MiMo API 使用指南

## 一、快速开始

### 1.1 DeepSeek API 使用

#### 方式一：使用专用配置（推荐）

```bash
# .env 文件配置
API_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

#### 方式二：使用 OpenAI 配置（无需修改代码）

```bash
# .env 文件配置
API_PROVIDER=openai
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

### 1.2 小米 MiMo API 使用

```bash
# .env 文件配置
API_PROVIDER=mimo
MIMO_API_KEY=your_mimo_api_key
MIMO_API_BASE=https://api.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2-flash
```

## 二、代码集成示例

### 2.1 基本使用

```python
from config import init_config, get_api_config
from services.llm_service import create_llm_service

# 初始化配置
init_config()

# 创建 LLM 服务
llm_service = create_llm_service()

# 打印配置信息
api_config = get_api_config()
print(f"当前使用: {api_config.provider}")
print(f"模型: {api_config.model_name}")
print(f"API 地址: {api_config.base_url}")

# 调用 LLM
async def main():
    response = await llm_service.call("你好，请介绍一下你自己")
    print(f"响应内容: {response.content}")
    print(f"Token 使用: {response.token_usage}")

# 运行
import asyncio
asyncio.run(main())
```

### 2.2 在小说大纲生成项目中使用

```python
import asyncio
from services.novel_processing_service import NovelProcessingService

async def generate_outline():
    # 配置已在环境变量中设置，会自动选择对应的 LLM 提供商
    service = NovelProcessingService()
    
    # 处理小说文件
    result = await service.process_novel(
        file_path="novel.txt",
        resume=False
    )
    
    print(f"处理完成!")
    print(f"生成大纲数: {result.get('chunk_count', 0)}")
    print(f"输出目录: {result.get('output_dir', '')}")

# 运行
asyncio.run(generate_outline())
```

## 三、获取 API Key

### 3.1 DeepSeek API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册并登录账号
3. 进入 API Keys 页面创建新的 API Key
4. 复制生成的 Key 并配置到环境变量

### 3.2 小米 MiMo API Key

1. 访问 [MiMo API 开放平台](https://platform.xiaomimimo.com/)
2. 注册并登录账号（目前限时免费）
3. 进入 API Keys 页面创建新的 API Key
4. 复制生成的 Key 并配置到环境变量

## 四、参数配置对比

| 参数 | DeepSeek | 小米 MiMo | OpenAI |
|------|----------|-----------|--------|
| API 地址 | `https://api.deepseek.com/v1` | `https://api.xiaomimimo.com/v1` | `https://api.openai.com/v1` |
| Token 限制参数 | `max_tokens` | `max_completion_tokens` | `max_tokens` |
| 默认模型 | `deepseek-chat` | `mimo-v2-flash` | `gpt-4o-mini` |
| 上下文窗口 | 32,768 tokens | 256,000 tokens | 128,000 tokens |
| 特殊功能 | 支持 Reasoning | 支持 Thinking 模式 | 标准功能 |

## 五、注意事项

### 5.1 错误处理

项目内置了完善的错误处理机制：

- **APIKeyError**: API 密钥无效或缺失
- **RateLimitError**: 请求频率超限（会自动重试）
- **APIError**: 其他 API 错误
- **ContentFilterError**: 内容被安全过滤器阻止（不触发熔断器）

### 5.2 代理配置

如果需要通过代理访问 API：

```bash
USE_PROXY=true
PROXY_URL=http://127.0.0.1:7897
```

### 5.3 并发控制

项目使用连接池管理 HTTP 连接，支持高并发请求：

```python
from services.connection_pool import HTTPConnectionPool

# 自定义连接池配置
pool = HTTPConnectionPool(max_connections=20)
service = create_llm_service()
```

## 六、性能对比

### 6.1 成本对比

| 提供商 | 输入价格 | 输出价格 | 免费额度 |
|--------|----------|----------|----------|
| DeepSeek | ¥1/M tokens | ¥2/M tokens | 有 |
| 小米 MiMo | ¥0.7/M tokens | ¥2.1/M tokens | 限时免费 |
| OpenAI | $2.5/M tokens | $10/M tokens | 无 |

### 6.2 性能特点

- **DeepSeek**: 性价比高，适合大多数场景
- **小米 MiMo**: 支持超长上下文（256K），适合长文本处理
- **OpenAI**: 生态成熟，稳定性好

## 七、测试验证

### 7.1 验证配置是否正确

```python
from config import get_api_config, init_config

# 初始化
init_config()

# 获取配置
config = get_api_config()
print(f"Provider: {config.provider}")
print(f"Model: {config.model_name}")
print(f"Base URL: {config.base_url}")

# 验证 API Key
try:
    key = config.api_key
    print(f"API Key: {key[:8]}...{key[-4:]}")  # 部分显示
    print("✅ 配置验证通过")
except Exception as e:
    print(f"❌ 配置错误: {e}")
```

### 7.2 测试 API 连接

```python
import asyncio
from services.llm_service import create_llm_service

async def test_connection():
    try:
        service = create_llm_service()
        response = await service.call("请回复 OK")
        if "ok" in response.content.lower() or "OK" in response.content:
            print("✅ API 连接测试成功")
            return True
        else:
            print(f"⚠️ API 响应异常: {response.content}")
            return False
    except Exception as e:
        print(f"❌ API 连接失败: {e}")
        return False

# 运行测试
asyncio.run(test_connection())
```

## 八、故障排除

### 8.1 常见问题

**Q: 提示 "不支持的API提供商"**
A: 检查 `API_PROVIDER` 环境变量是否正确设置，当前支持: `openai`, `deepseek`, `gemini`, `zhipu`, `aihubmix`, `mimo`

**Q: API Key 验证失败**
A: 确认 API Key 正确且未过期，检查是否包含占位符文字（如 "your_key_here"）

**Q: 请求超时**
A: 尝试开启代理或检查网络连接

**Q: 速率限制**
A: 项目内置重试机制，如持续遇到可降低 `PARALLEL_LIMIT`

### 8.2 日志调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 运行时查看日志输出
```

## 九、完整配置示例

### 9.1 DeepSeek 配置

```bash
# .env
API_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# 可选参数
PARALLEL_LIMIT=5
MAX_RETRY=5
USE_PROXY=false
```

### 9.2 MiMo 配置

```bash
# .env
API_PROVIDER=mimo
MIMO_API_KEY=your_mimo_api_key_here
MIMO_API_BASE=https://api.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2-flash

# 可选参数
PARALLEL_LIMIT=5
MAX_RETRY=5
USE_PROXY=false
```

### 9.3 多提供商切换示例

```python
import os
from config import init_config

# 切换到 DeepSeek
os.environ['API_PROVIDER'] = 'deepseek'
init_config()

# 验证切换
from config import get_api_config
config = get_api_config()
print(f"当前: {config.provider} - {config.model_name}")
```

## 十、总结

该项目已成功集成 **DeepSeek** 和 **小米 MiMo** 两个新的 LLM 提供商：

- ✅ **DeepSeek**: 完全兼容，通过 `OpenAIService` 实现，开箱即用
- ✅ **小米 MiMo**: 需要适配，已添加 `MiMoService` 类处理参数差异

两个提供商都具有较高的性价比，特别是小米 MiMo 支持超长上下文，非常适合处理大型小说文件。
