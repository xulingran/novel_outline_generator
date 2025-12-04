"""
LLM提示词模板模块
定义用于生成和处理大纲的各种提示词
"""
import os
from typing import List, Dict, Any


DEFAULT_OUTLINE_PROMPT_TEMPLATE = """你是专业文学编辑。请对下面这段小说内容生成结构化剧情大纲。

要求：
1. 用条理清晰的结构概括主要事件
2. 标出人物关系变化
3. 标出冲突与伏笔
4. 适当压缩，突出关键信息
5. 最终大纲用 JSON 格式输出，必须包含以下字段：
   - "events": 主要事件列表
   - "characters": 本段出现的主要人物列表
   - "relationships": 人物关系列表，格式为 [["人物A", "人物B", "关系描述"], ...]
   - "conflicts": 冲突与伏笔

文本编号: {idx}

内容如下：
{chunk}
"""


def _load_template_from_env() -> str:
    """获取由环境变量 OUTLINE_PROMPT_TEMPLATE 配置的模板，支持使用 \n 表示换行。"""
    template = os.getenv("OUTLINE_PROMPT_TEMPLATE")
    if not template:
        return DEFAULT_OUTLINE_PROMPT_TEMPLATE
    return template.replace("\\n", "\n")


def _apply_template(template: str, chunk: str, idx: int) -> str:
    """填充模板中的占位符，确保原文内容被插入。"""
    filled = template
    replacements = {
        "{chunk}": chunk,
        "{chunk_content}": chunk,
        "{idx}": str(idx),
        "{chunk_id}": str(idx),
    }

    has_chunk_placeholder = False
    for key, value in replacements.items():
        if key in filled:
            filled = filled.replace(key, value)
            if key in ("{chunk}", "{chunk_content}"):
                has_chunk_placeholder = True

    if not has_chunk_placeholder:
        filled = f"{filled}\n\n内容如下：\n{chunk}"

    return filled


def chunk_prompt(chunk: str, idx: int) -> str:
    """
    生成处理单个文本块的提示词
    """
    template = _load_template_from_env()
    return _apply_template(template, chunk, idx)


def merge_prompt(json_blocks: str) -> str:
    """
    生成合并JSON格式大纲的提示词
    """
    return f"""你有一整部小说的分段剧情信息（JSON 列表）。请你：

- 整合所有剧情段落
- 总结主线、支线、人物发展弧光、核心冲突、高潮转折点
- 给出一个完整、清晰、结构化的总剧情大纲

以下是JSON数据：
{json_blocks}
"""


def merge_text_prompt(text_blocks: List[str]) -> str:
    """
    生成合并文本格式大纲的提示词
    """
    if not text_blocks:
        return """以下是小说不同部分的大纲，请你将它们整合成一个完整、连贯的总剧情大纲：

- 整合所有剧情段落，确保时间线和逻辑连贯
- 总结主线、支线、人物发展弧光、核心冲突、高潮转折点
- 给出一个完整、清晰、结构化的总剧情大纲
"""

    blocks_text = ""
    for i, block in enumerate(text_blocks, 1):
        blocks_text += f"\n\n--- 分段大纲 {i} ---\n\n{block}\n"

    return f"""以下是小说不同部分的大纲，请你将它们整合成一个完整、连贯的总剧情大纲：

- 整合所有剧情段落，确保时间线和逻辑连贯
- 总结主线、支线、人物发展弧光、核心冲突、高潮转折点
- 给出一个完整、清晰、结构化的总剧情大纲

{blocks_text}
"""


def character_extraction_prompt(text: str) -> str:
    """
    生成提取人物信息的提示词
    """
    return f"""请从以下小说文本中提取人物信息，并以JSON格式输出：

要求：
1. 识别所有出现的人物角色
2. 提取人物之间的关系
3. 判断人物的重要性（主角、配角、路人等）
4. 输出格式：
{{
    "characters": [
        {{
            "name": "人物姓名",
            "aliases": ["别名1", "别名2"],
            "importance": "主角/配角/路人",
            "description": "人物描述"
        }}
    ],
    "relationships": [
        {{
            "character_a": "人物A",
            "character_b": "人物B",
            "relationship": "关系描述",
            "importance": "重要/次要"
        }}
    ]
}}

文本内容：
{text}
"""


def outline_validation_prompt(outline: str, chunk: str) -> str:
    """
    生成验证大纲质量的提示词
    """
    return f"""请验证以下大纲是否准确概括了原始文本内容，并进行必要的修正：

任务：
1. 检查大纲是否遗漏重要事件
2. 验证人物识别是否准确
3. 确认关系描述是否正确
4. 修正任何错误或遗漏

原始文本：
{chunk}

当前大纲：
{outline}

请输出修正后的大纲（JSON格式）：
"""


def style_adjustment_prompt(outline: str, style: str) -> str:
    """
    生成调整大纲风格的提示词
    """
    style_instructions: Dict[str, str] = {
        "concise": "请将以下大纲改写得更简洁，只保留最重要的信息",
        "detailed": "请将以下大纲扩展得更详细，增加更多细节和分析",
        "academic": "请以学术分析的风格重写以下大纲",
        "casual": "请以轻松易懂的风格重写以下大纲",
    }

    instruction = style_instructions.get(style, style_instructions["concise"])

    return f"""{instruction}：

原始大纲：
{outline}

请输出调整后的大纲：
"""
