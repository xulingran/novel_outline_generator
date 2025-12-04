import json
from prompts import merge_prompt
from exceptions import NovelOutlineError

# extractor.py 已被弃用，其功能已整合到 async_llm.py
# 如果需要使用同步调用，请使用新的服务层


def merge_all(chunks):
    """合并所有块（已弃用：请使用新的服务层）"""
    raise NotImplementedError(
        "merge_all 函数已弃用。请使用 novel_processing_service.py 中的 NovelProcessingService.merge_outlines_recursive"
    )


def build_character_profiles(chunks):
    """构建人物配置文件（已弃用：请使用新的服务层）"""
    raise NotImplementedError(
        "build_character_profiles 函数已弃用。请使用 visualization_service.py 中的服务"
    )
