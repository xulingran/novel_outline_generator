"""
小说大纲生成工具 - 主程序
重构后的版本，使用新的服务架构
"""

import asyncio
import logging
import webbrowser
from pathlib import Path

from config import get_api_config, get_processing_config, get_txt_file, init_config
from exceptions import APIKeyError, ConfigurationError, NovelOutlineError
from services.file_service import FileService
from services.novel_processing_service import NovelProcessingService
from services.progress_service import ProgressService
from utils import setup_logging
from validators import validate_file_path

setup_logging()

logger = logging.getLogger(__name__)


class NovelOutlineApp:
    """小说大纲生成应用主类"""

    def __init__(self):
        self.processing_service = NovelProcessingService()
        self.progress_service = ProgressService()
        self.file_service = FileService()
        self.processing_config = get_processing_config()

    async def run(self) -> None:
        """运行主程序"""
        try:
            # 显示欢迎信息
            self._print_welcome()

            # 选择模式
            mode = self._select_mode()

            if mode == "process":
                await self._process_novel_mode()
            elif mode == "web_ui":
                self._start_web_ui()

        except KeyboardInterrupt:
            print("\n用户中断操作")
        except (APIKeyError, ConfigurationError) as e:
            print(f"\n❌ 配置错误: {e}")
            print("\n💡 请检查环境变量或.env文件中的配置")
        except NovelOutlineError as e:
            print(f"\n❌ 处理错误: {e}")
        except Exception as e:
            logger.exception("未预期的错误")
            print(f"\n❌ 发生未知错误: {e}")
            print("请查看日志文件 novel_outline.log 获取详细信息")
            print("\n提示：如需使用 Web UI，请确认已安装 fastapi/uvicorn 等依赖，并选择模式 2。")

    def _print_welcome(self) -> None:
        """打印欢迎信息"""
        print("\n" + "=" * 60)
        print("📝 小说大纲生成工具 v2.0")
        print("=" * 60)
        api_cfg = get_api_config()
        print(f"🔧 API提供商: {api_cfg.provider.upper()}")
        print(f"📊 并发限制: {self.processing_config.parallel_limit}")
        print(f"🎯 目标块大小: {self.processing_config.target_tokens_per_chunk} tokens")
        print("=" * 60 + "\n")

    def _select_mode(self) -> str:
        """选择运行模式"""
        while True:
            print("请选择模式：")
            print("  1. 启用 Web UI（需要 uvicorn / fastapi 支持）")
            print("  2. 处理新文件（分析小说并生成大纲）")

            choice = input("\n请输入选项 (1/2，直接回车默认 Web UI): ").strip()
            if not choice:
                return "web_ui"
            if choice == "1":
                return "web_ui"
            elif choice == "2":
                return "process"
            else:
                print("❌ 无效选项，请输入 1 或 2\n")

    async def _process_novel_mode(self) -> None:
        """处理小说模式"""
        # 1. 获取文件路径
        file_path = self._get_input_file_path()

        # 2. 显示文件信息
        await self._show_file_info(file_path)

        # 3. 预测token使用量
        await self._predict_tokens(file_path)

        # 4. 询问是否恢复进度
        resume = self._ask_resume_progress(file_path)

        # 5. 开始处理
        print("\n🚀 开始处理...")
        result = await self.processing_service.process_novel(file_path=file_path, resume=resume)

        # 6. 显示结果
        self._show_results(result)

    def _start_web_ui(self) -> None:
        """启动 Web UI（FastAPI + uvicorn）"""
        try:
            import uvicorn

            print("\n🚀 正在启动 Web UI（http://localhost:8000）...")
            print("   若需自定义端口，请直接运行：uvicorn web_api:app --reload --port 8000")

            # 尝试自动在浏览器打开前端页面（本地文件引用后端 API）
            ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
            if ui_path.exists():
                try:
                    webbrowser.open(ui_path.as_uri())
                    print(f"   已尝试打开前端: {ui_path}")
                except Exception as open_err:
                    print(f"⚠️ 自动打开浏览器失败: {open_err}")
            else:
                print(f"⚠️ 未找到前端文件: {ui_path}")

            uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=True)
        except ImportError:
            print("❌ 启动失败：未安装 fastapi/uvicorn。请先运行: pip install -r requirements.txt")
        except Exception as e:
            print(f"❌ 启动 Web UI 失败: {e}")

    def _get_input_file_path(self) -> str:
        """获取输入文件路径"""
        default_file = get_txt_file()

        while True:
            user_input = input("\n请输入要分析的txt文件名（直接回车使用默认值）: ").strip()

            if not user_input:
                file_path = default_file
                print(f"使用默认文件: {file_path}")
            else:
                file_path = user_input

            try:
                # 验证文件
                validated_path = validate_file_path(
                    file_path, allowed_extensions=[".txt", ".md"], max_size_mb=100
                )
                return str(validated_path)
            except Exception as e:
                print(f"❌ 文件错误: {e}")
                retry = input("是否重新输入？(y/n，默认y): ").strip().lower()
                if retry and retry not in ["y", "yes"]:
                    print("使用默认文件")
                    return default_file

    async def _show_file_info(self, file_path: str) -> None:
        """显示文件信息"""
        file_info = self.file_service.get_file_info(file_path)
        if file_info["exists"]:
            print("\n📄 文件信息:")
            print(f"   路径: {file_path}")
            print(f"   大小: {file_info['size_formatted']}")
            print(f"   修改时间: {file_info['modified']}")

            # 估算块数
            text, _ = self.file_service.read_text_file(file_path)
            from splitter import get_splitter

            splitter = get_splitter()
            estimated_chunks = splitter.estimate_chunk_count(text)
            print(f"   预估块数: {estimated_chunks}")

    async def _predict_tokens(self, file_path: str) -> None:
        """预测token使用量"""
        text, _ = self.file_service.read_text_file(file_path)
        from splitter import get_splitter
        from tokenizer import count_tokens

        total_tokens = count_tokens(text)
        splitter = get_splitter()
        chunks = splitter.split_text(text)

        # 估算token消耗
        chunk_tokens = sum(count_tokens(chunk) for chunk in chunks)
        chunk_responses = int(chunk_tokens * 0.3)  # 估算响应
        merge_tokens = count_tokens(text) * 0.1  # 估算合并消耗

        total_estimated = chunk_tokens + chunk_responses + merge_tokens

        print("\n📊 Token使用预测:")
        print(f"   原始文本: {total_tokens:,} tokens")
        print(f"   分块处理: {chunk_tokens:,} tokens (输入) + {chunk_responses:,} tokens (输出)")
        print(f"   合并处理: {merge_tokens:,} tokens")
        print(f"   总计预计: {total_estimated:,} tokens")

        # 确认继续
        while True:
            user_input = input("\n是否继续处理？(y/n): ").strip().lower()
            if user_input in ["y", "yes"]:
                return
            elif user_input in ["n", "no"]:
                print("操作已取消")
                exit(0)
            else:
                print("请输入 y/yes 或 n/no")

    def _ask_resume_progress(self, file_path: str) -> bool:
        """询问是否恢复进度"""
        progress_data = self.progress_service.load_progress()
        if not progress_data:
            return False

        # 检查进度是否匹配
        # 简单的文件路径匹配检查，详细的hash验证在process_novel中进行
        if progress_data.txt_file != file_path:
            return False

        # 显示进度信息
        summary = self.progress_service.get_progress_summary(progress_data)
        print("\n📋 发现未完成的进度:")
        print(f"   文件: {summary['file']}")
        print(
            f"   进度: {summary['completed_chunks']}/{summary['total_chunks']} ({summary['completion_rate']})"
        )
        print(f"   最后更新: {summary['last_update']}")

        while True:
            user_input = input("\n是否恢复进度？(y/n): ").strip().lower()
            if user_input in ["y", "yes"]:
                return True
            elif user_input in ["n", "no"]:
                self.progress_service.clear_progress()
                return False
            else:
                print("请输入 y/yes 或 n/no")

    def _show_results(self, result: dict) -> None:
        """显示处理结果"""
        print("\n" + "=" * 60)
        print("🎉 处理完成！")
        print("=" * 60)
        print(f"✅ 成功处理: {result['chunk_count']} 个文本块")
        print(f"⏱️  处理时间: {result['processing_time']:.1f} 秒")
        print(f"📁 输出目录: {result['output_dir']}")
        print("\n生成文件:")
        print("   📄 chunk_outlines.json - 分块大纲")
        print("   📄 final_outline.txt - 最终大纲")
        print("   📄 processing_metadata.json - 处理元数据")


async def main():
    """主入口函数"""
    # 初始化配置（加载 .env 文件并检查 API 密钥）
    init_config()
    app = NovelOutlineApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
