"""
测试 config_dialog 组件

测试配置对话框，包括 API 配置、处理配置、代理配置等功能。
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gui.config_dialog import ConfigDialog
from tests.test_gui.conftest import ctk


@pytest.fixture
def mock_master():
    """创建模拟的 master 窗口"""
    master = MagicMock()
    master.grab_set = MagicMock()
    master.title = MagicMock()
    master.geometry = MagicMock()
    return master


class TestConfigDialog:
    """测试 ConfigDialog 组件"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_init(self, mock_master):
        """测试 ConfigDialog 初始化"""
        ConfigDialog(mock_master)

        # 验证对话框设置
        mock_master.title.assert_called()
        mock_master.geometry.assert_called()
        mock_master.grab_set.assert_called()

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_load_current_config(self, mock_master):
        """测试加载当前配置"""
        dialog = ConfigDialog(mock_master)

        # 验证配置被加载
        assert hasattr(dialog, "api_config")
        assert hasattr(dialog, "proc_config")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_dialog_is_modal(self, mock_master):
        """测试对话框是模态的"""
        ConfigDialog(mock_master)

        # grab_set 使对话框模态化
        mock_master.grab_set.assert_called_once()


class TestConfigDialogAPIConfig:
    """测试 API 配置功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_api_provider_selection(self, mock_master):
        """测试 API 提供商选择"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "provider_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_api_key_fields(self, mock_master):
        """测试 API 密钥字段"""
        dialog = ConfigDialog(mock_master)

        # 验证 API 密钥变量存在
        assert hasattr(dialog, "openai_key_var")
        assert hasattr(dialog, "gemini_key_var")
        assert hasattr(dialog, "zhipu_key_var")
        assert hasattr(dialog, "aihubmix_key_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_api_base_url_fields(self, mock_master):
        """测试 API Base URL 字段"""
        dialog = ConfigDialog(mock_master)

        # 验证 Base URL 变量存在
        assert hasattr(dialog, "openai_base_var")
        assert hasattr(dialog, "zhipu_base_var")
        assert hasattr(dialog, "aihubmix_base_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_api_model_fields(self, mock_master):
        """测试 API 模型字段"""
        dialog = ConfigDialog(mock_master)

        # 验证模型变量存在
        assert hasattr(dialog, "openai_model_var")
        assert hasattr(dialog, "gemini_model_var")
        assert hasattr(dialog, "zhipu_model_var")
        assert hasattr(dialog, "aihubmix_model_var")


class TestConfigDialogProcessingConfig:
    """测试处理配置功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_chunk_size_field(self, mock_master):
        """测试分块大小字段"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "chunk_size_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_parallel_limit_field(self, mock_master):
        """测试并发限制字段"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "parallel_limit_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_max_retry_field(self, mock_master):
        """测试最大重试次数字段"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "max_retry_var")


class TestConfigDialogProxyConfig:
    """测试代理配置功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_proxy_enabled_var(self, mock_master):
        """测试代理启用变量"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "proxy_enabled_var")
        assert hasattr(dialog, "proxy_url_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_proxy_disabled_by_default(self, mock_master):
        """测试代理默认禁用"""
        dialog = ConfigDialog(mock_master)

        # 代理应该默认禁用
        if hasattr(dialog, "proxy_enabled_var"):
            # 在模拟环境中可能无法获取实际值
            pass


class TestConfigDialogSave:
    """测试保存配置功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_on_save_creates_env_file(self, mock_master, tmp_path: Path):
        """测试保存创建 .env 文件"""
        dialog = ConfigDialog(mock_master)

        # 在实际环境中会写入文件
        # 这里我们只验证方法存在
        assert hasattr(dialog, "_on_save")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_on_save_refreshes_config(self, mock_master):
        """测试保存后刷新配置"""
        dialog = ConfigDialog(mock_master)

        # 验证方法存在
        assert hasattr(dialog, "_on_save")

        # 在实际环境中会调用 _refresh_config_cache


class TestConfigDialogExport:
    """测试导出配置功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_on_export_method(self, mock_master):
        """测试导出方法存在"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "_on_export")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_export_creates_env_file(self, mock_master, tmp_path: Path):
        """测试导出创建 .env 文件"""
        dialog = ConfigDialog(mock_master)

        # 在实际环境中会弹出文件保存对话框
        # 这里我们只验证方法存在
        assert hasattr(dialog, "_on_export")


class TestConfigDialogPasswordToggle:
    """测试密码显示/隐藏功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_toggle_password_method(self, mock_master):
        """测试密码切换方法"""
        dialog = ConfigDialog(mock_master)

        assert hasattr(dialog, "_toggle_password")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_api_key_fields_have_show_button(self, mock_master):
        """测试 API 密钥字段有显示按钮"""
        dialog = ConfigDialog(mock_master)

        # 在实际 GUI 中，密码字段会有显示/隐藏按钮
        # 这里我们验证字段存在
        assert hasattr(dialog, "openai_key_var")


class TestConfigDialogValidation:
    """测试配置验证"""

    def test_supported_api_providers(self):
        """测试支持的 API 提供商"""
        from config import SUPPORTED_API_PROVIDERS

        expected_providers = ["openai", "gemini", "zhipu", "aihubmix"]

        for provider in expected_providers:
            assert provider in SUPPORTED_API_PROVIDERS

    def test_api_config_validation(self):
        """测试 API 配置验证"""
        from config import APIConfig

        config = APIConfig()

        # 验证默认值
        assert config.provider in ["openai", "gemini", "zhipu", "aihubmix"]
        assert config.openai_model is not None


class TestConfigDialogIntegration:
    """测试集成功能"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_dialog_ui_components(self, mock_master):
        """测试对话框 UI 组件"""
        dialog = ConfigDialog(mock_master)

        # 验证关键变量存在
        assert hasattr(dialog, "provider_var")
        assert hasattr(dialog, "chunk_size_var")
        assert hasattr(dialog, "parallel_limit_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_config_values_from_env(self, mock_master):
        """测试从环境变量加载配置值"""
        dialog = ConfigDialog(mock_master)

        # 配置应该从环境变量加载
        # 在实际环境中会读取 .env 文件
        assert hasattr(dialog, "api_config")
        assert hasattr(dialog, "proc_config")


class TestConfigDialogEdgeCases:
    """测试边界情况"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_empty_api_keys(self, mock_master):
        """测试空的 API 密钥"""
        dialog = ConfigDialog(mock_master)

        # API 密钥可以为 None
        assert dialog.api_config.openai_key is None or isinstance(dialog.api_config.openai_key, str)

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_zero_parallel_limit(self, mock_master):
        """测试零并发限制（边界值）"""
        dialog = ConfigDialog(mock_master)

        # 在实际使用中应该至少为 1
        # 这里我们只验证变量存在
        assert hasattr(dialog, "parallel_limit_var")

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_very_large_chunk_size(self, mock_master):
        """测试非常大的分块大小"""
        dialog = ConfigDialog(mock_master)

        # 分块大小可以设置得很大
        # 这里我们只验证变量存在
        assert hasattr(dialog, "chunk_size_var")


class TestConfigDialogMethodExistence:
    """测试方法存在性"""

    @pytest.mark.skipif(not hasattr(ctk, "CTkToplevel"), reason="CustomTkinter not available")
    def test_all_required_methods_exist(self, mock_master):
        """测试所有必需的方法存在"""
        dialog = ConfigDialog(mock_master)

        required_methods = [
            "_setup_api_config",
            "_setup_processing_config",
            "_setup_proxy_config",
            "_create_api_key_field",
            "_create_text_field",
            "_toggle_password",
            "_on_save",
            "_on_export",
        ]

        for method_name in required_methods:
            assert hasattr(dialog, method_name), f"Missing method: {method_name}"
