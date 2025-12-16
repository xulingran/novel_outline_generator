# API 配置指南

## 问题：API Key 配置错误

如果您看到类似以下的错误：

```
Error code: 401 - {'error': {'message': 'Incorrect API key provided: your_ope************here. ...
```

这说明您的 API Key 还没有正确配置。

## 解决方案

### 方法 1: 使用 create_env.py 工具（推荐）

1. 运行配置工具:
   ```bash
   python create_env.py
   ```

2. 按照提示选择 API 提供商

3. 输入您的 API Key

4. 工具会自动创建 `.env` 文件

5. 启动服务

### 方法 2: 手动编辑 .env 文件

1. 在项目根目录找到 `.env` 文件（如果没有，运行 `python create_env.py` 创建）

2. 打开 `.env` 文件，找到对应的配置项：

   ```ini
   # 示例：使用 OpenAI API
   API_PROVIDER=openai
   OPENAI_API_KEY=sk-your-actual-api-key-here
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4o-mini
   ```

3. 将 `your_openai_api_key_here` 替换为您的真实 API Key

4. 保存文件

5. 重启服务使配置生效

> **注意**: Web UI 中的"环境配置"部分仅用于查看当前配置，不支持直接修改。请使用上述方法进行配置修改。

## 使用中转 API

如果您使用的是中转 API（如 aihubmix），需要同时配置：

```ini
API_PROVIDER=openai
OPENAI_API_KEY=你的中转API密钥
OPENAI_API_BASE=https://api.aihubmix.com/v1
OPENAI_MODEL=gpt-4o-mini
```

**注意事项：**
1. 确保 API Key 是真实有效的，不是占位符文本
2. 中转 API 的 Key 通常也是以 `sk-` 开头
3. 修改配置后必须重启服务才能生效

## 获取 API Key

### OpenAI
- 官方: https://platform.openai.com/api-keys
- 中转服务: 根据您选择的中转商获取

### Google Gemini
- https://makersuite.google.com/app/apikey

### 智谱清言
- https://open.bigmodel.cn/

## 验证配置

配置完成后，可以通过以下方式验证：

1. 启动服务时查看控制台输出：
   ```
   ? OpenAI API客户端初始化成功 (模型: gpt-4o-mini)
   ```

2. 在 Web UI 的"模型配置"部分应该能看到正确的信息

3. 上传文件并点击"预测 Token"按钮测试连接

## 常见问题

### Q: 为什么保存配置后还是显示错误的 API Key？
A: 修改 `.env` 文件或通过 Web UI 保存配置后，必须重启后端服务才能使新配置生效。

### Q: 如何知道我的配置是否正确？
A: 启动服务时会有明确的提示信息。如果配置错误，会显示详细的错误信息。

### Q: 中转 API 和官方 API 有什么区别？
A: 中转 API 只需要修改 `OPENAI_API_BASE` 的值，其他使用方式相同。确保使用中转商提供的正确 base URL。

## 安全提示

- **不要**将 `.env` 文件提交到 Git 仓库
- **不要**在公开场合分享您的 API Key
- 建议定期更换 API Key
- 使用中转服务时，确保服务商可信
