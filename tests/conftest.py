"""
Pytest 配置文件
为 CI 环境提供测试所需的环境变量
"""
import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """为所有测试设置必要的环境变量"""
    # 设置 API 提供商为 openai（默认）
    monkeypatch.setenv('API_PROVIDER', 'openai')

    # 设置测试用的 API Key（不需要真实的 key，测试使用 mock）
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-for-ci')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o-mini')

    # 设置其他可选的 API Key
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('ZHIPU_API_KEY', 'test-zhipu-key')


@pytest.fixture(autouse=True)
def disable_dotenv():
    """禁用 .env 文件加载，避免 CI 环境中缺少文件导致错误"""
    # 这个 fixture 在 setup_test_env 之后运行，确保环境变量已设置
    pass
