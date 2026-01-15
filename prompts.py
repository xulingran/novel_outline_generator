"""
LLM提示词模板模块
定义用于生成和处理大纲的各种提示词
"""


def chunk_prompt(chunk: str, idx: int | str) -> str:
    """
    生成处理单个文本块的提示词（提取人物、关系、剧情）
    """
    return f"""请从以下小说文本中提取主要人物、人物关系和剧情要点：

要求：
1. 识别所有主要人物角色（主角、重要配角）
2. 提取人物之间的关系（朋友、敌人、师徒、亲人等）
3. 按时间顺序提取剧情要点（关键事件、转折点、冲突发展）
4. 输出JSON格式

输出格式：
{{
  "characters": ["人物A", "人物B"],
  "relationships": [
    ["人物A", "人物B", "关系描述"],
    ["人物C", "人物D", "关系描述"]
  ],
  "plot": [
    "情节点1：某某事件发生",
    "情节点2：某某冲突展开"
  ]
}}

文本内容：
{chunk}"""


def merge_prompt(json_blocks: str) -> str:
    """
    生成合并JSON格式大纲的提示词（从头讲到尾）
    """
    return f"""你有一整部小说的分段人物和关系信息（JSON 列表）。请你：

- 整合所有剧情段落，确保时间线和逻辑连贯
- 给出一个完整、清晰、结构化的总剧情大纲
- **输出纯文本格式，不要使用markdown格式符号（如#、*、-等标题或列表标记）**

以下是JSON数据：
{json_blocks}
"""


def merge_text_prompt(text_blocks: list[str]) -> str:
    """
    生成合并文本格式大纲的提示词（从头讲到尾）
    """
    blocks_text = ""
    for i, block in enumerate(text_blocks, 1):
        blocks_text += f"\n\n--- 分段大纲 {i} ---\n\n{block}\n"

    return f"""以下是小说不同部分的大纲，请你将它们整合成一个完整、连贯的总剧情大纲：

- 整合所有剧情段落，确保时间线和逻辑连贯
- 给出一个完整、清晰、结构化的总剧情大纲
- **输出纯文本格式，不要使用markdown格式符号（如#、*、-等标题或列表标记）**

{blocks_text}
"""
