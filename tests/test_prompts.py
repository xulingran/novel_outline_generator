"""
测试LLM提示词模板模块
"""

import pytest

from prompts import chunk_prompt, merge_prompt, merge_text_prompt


class TestChunkPrompt:
    """测试chunk_prompt函数"""

    def test_chunk_prompt_basic(self):
        """测试基本的chunk_prompt生成"""
        chunk_text = "这是一个测试文本片段。"
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        assert "请从以下小说文本中提取主要人物、人物关系和剧情要点" in result
        assert "characters" in result
        assert "relationships" in result
        assert "plot" in result
        assert chunk_text in result
        assert "JSON格式" in result

    def test_chunk_prompt_with_string_idx(self):
        """测试使用字符串索引的chunk_prompt"""
        chunk_text = "测试文本"
        idx = "test"

        result = chunk_prompt(chunk_text, idx)

        assert chunk_text in result
        assert "请从以下小说文本中提取主要人物、人物关系和剧情要点" in result

    def test_chunk_prompt_with_integer_idx(self):
        """测试使用整数索引的chunk_prompt"""
        chunk_text = "测试文本"
        idx = 5

        result = chunk_prompt(chunk_text, idx)

        assert chunk_text in result
        assert "请从以下小说文本中提取主要人物、人物关系和剧情要点" in result

    def test_chunk_prompt_with_long_text(self):
        """测试处理长文本的chunk_prompt"""
        chunk_text = "这是一个很长的文本片段。" * 100
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        assert chunk_text in result
        assert "请从以下小说文本中提取主要人物、人物关系和剧情要点" in result

    def test_chunk_prompt_with_special_characters(self):
        """测试包含特殊字符的chunk_prompt"""
        chunk_text = "测试文本！@#$%^&*()_+"
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        assert chunk_text in result

    def test_chunk_prompt_with_chinese(self):
        """测试包含中文的chunk_prompt"""
        chunk_text = "这是一个中文测试文本。"
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        assert chunk_text in result
        assert "人物" in result
        assert "剧情" in result

    def test_chunk_prompt_structure(self):
        """测试chunk_prompt的结构"""
        chunk_text = "测试文本"
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        # 检查包含必要的说明
        assert "识别所有主要人物角色" in result
        assert "提取人物之间的关系" in result
        assert "按时间顺序提取剧情要点" in result
        assert "输出JSON格式" in result

    def test_chunk_prompt_json_format_example(self):
        """测试chunk_prompt中的JSON格式示例"""
        chunk_text = "测试文本"
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        # 检查JSON格式示例
        assert '"characters"' in result
        assert '"relationships"' in result
        assert '"plot"' in result

    def test_chunk_prompt_empty_chunk(self):
        """测试空文本的chunk_prompt"""
        chunk_text = ""
        idx = 1

        result = chunk_prompt(chunk_text, idx)

        assert "请从以下小说文本中提取主要人物、人物关系和剧情要点" in result
        assert "文本内容：" in result


class TestMergePrompt:
    """测试merge_prompt函数"""

    def test_merge_prompt_basic(self):
        """测试基本的merge_prompt生成"""
        json_blocks = '[{"chunk_id": 1, "plot": ["事件1"]}]'

        result = merge_prompt(json_blocks)

        assert "你有一整部小说的分段人物和关系信息" in result
        assert "整合所有剧情段落" in result
        assert "时间线和逻辑连贯" in result
        assert "完整、清晰、结构化的总剧情大纲" in result
        assert json_blocks in result
        assert "纯文本格式" in result

    def test_merge_prompt_with_multiple_blocks(self):
        """测试包含多个块的merge_prompt"""
        json_blocks = '[{"chunk_id": 1, "plot": ["事件1"]}, {"chunk_id": 2, "plot": ["事件2"]}]'

        result = merge_prompt(json_blocks)

        assert json_blocks in result
        assert "整合所有剧情段落" in result

    def test_merge_prompt_structure(self):
        """测试merge_prompt的结构"""
        json_blocks = '{"chunk_id": 1, "plot": ["事件1"]}'

        result = merge_prompt(json_blocks)

        # 检查包含必要的说明
        assert "整合所有剧情段落，确保时间线和逻辑连贯" in result
        assert "给出一个完整、清晰、结构化的总剧情大纲" in result
        assert "输出纯文本格式" in result
        assert "不要使用markdown格式符号" in result

    def test_merge_prompt_no_markdown_warning(self):
        """测试merge_prompt中的markdown警告"""
        json_blocks = '{"chunk_id": 1, "plot": ["事件1"]}'

        result = merge_prompt(json_blocks)

        assert "不要使用markdown格式符号" in result
        assert "#" in result or "*" in result or "-" in result  # 提到这些符号

    def test_merge_prompt_with_empty_json(self):
        """测试空JSON的merge_prompt"""
        json_blocks = "[]"

        result = merge_prompt(json_blocks)

        assert json_blocks in result
        assert "整合所有剧情段落" in result

    def test_merge_prompt_with_complex_json(self):
        """测试复杂JSON的merge_prompt"""
        json_blocks = '''[
            {
                "chunk_id": 1,
                "plot": ["事件1", "事件2"],
                "characters": ["人物A", "人物B"],
                "relationships": [["人物A", "人物B", "朋友"]]
            }
        ]'''

        result = merge_prompt(json_blocks)

        assert json_blocks in result


class TestMergeTextPrompt:
    """测试merge_text_prompt函数"""

    def test_merge_text_prompt_basic(self):
        """测试基本的merge_text_prompt生成"""
        text_blocks = ["这是第一段大纲。", "这是第二段大纲。"]

        result = merge_text_prompt(text_blocks)

        assert "以下是小说不同部分的大纲" in result
        assert "整合成一个完整、连贯的总剧情大纲" in result
        assert "时间线和逻辑连贯" in result
        assert "完整、清晰、结构化的总剧情大纲" in result
        assert "这是第一段大纲。" in result
        assert "这是第二段大纲。" in result
        assert "纯文本格式" in result

    def test_merge_text_prompt_with_multiple_blocks(self):
        """测试包含多个块的merge_text_prompt"""
        text_blocks = ["第一段", "第二段", "第三段", "第四段", "第五段"]

        result = merge_text_prompt(text_blocks)

        # 检查所有块都包含在结果中
        for block in text_blocks:
            assert block in result

        # 检查分段标记
        assert "--- 分段大纲 1 ---" in result
        assert "--- 分段大纲 2 ---" in result
        assert "--- 分段大纲 3 ---" in result
        assert "--- 分段大纲 4 ---" in result
        assert "--- 分段大纲 5 ---" in result

    def test_merge_text_prompt_structure(self):
        """测试merge_text_prompt的结构"""
        text_blocks = ["测试大纲"]

        result = merge_text_prompt(text_blocks)

        # 检查包含必要的说明
        assert "整合所有剧情段落，确保时间线和逻辑连贯" in result
        assert "给出一个完整、清晰、结构化的总剧情大纲" in result
        assert "输出纯文本格式" in result
        assert "不要使用markdown格式符号" in result

    def test_merge_text_prompt_no_markdown_warning(self):
        """测试merge_text_prompt中的markdown警告"""
        text_blocks = ["测试大纲"]

        result = merge_text_prompt(text_blocks)

        assert "不要使用markdown格式符号" in result
        assert "#" in result or "*" in result or "-" in result  # 提到这些符号

    def test_merge_text_prompt_with_empty_blocks(self):
        """测试空块列表的merge_text_prompt"""
        text_blocks = []

        result = merge_text_prompt(text_blocks)

        assert "以下是小说不同部分的大纲" in result
        assert "整合成一个完整、连贯的总剧情大纲" in result

    def test_merge_text_prompt_with_single_block(self):
        """测试单个块的merge_text_prompt"""
        text_blocks = ["这是唯一的一段大纲。"]

        result = merge_text_prompt(text_blocks)

        assert "--- 分段大纲 1 ---" in result
        assert "这是唯一的一段大纲。" in result

    def test_merge_text_prompt_with_long_text(self):
        """测试包含长文本的merge_text_prompt"""
        text_blocks = ["这是一段很长的文本。" * 100]

        result = merge_text_prompt(text_blocks)

        assert "这是一段很长的文本。" * 100 in result

    def test_merge_text_prompt_with_special_characters(self):
        """测试包含特殊字符的merge_text_prompt"""
        text_blocks = ["测试文本！@#$%^&*()_+"]

        result = merge_text_prompt(text_blocks)

        assert "测试文本！@#$%^&*()_+" in result

    def test_merge_text_prompt_with_chinese(self):
        """测试包含中文的merge_text_prompt"""
        text_blocks = ["这是一段中文大纲。"]

        result = merge_text_prompt(text_blocks)

        assert "这是一段中文大纲。" in result
        assert "整合成一个完整、连贯的总剧情大纲" in result

    def test_merge_text_prompt_block_numbering(self):
        """测试块编号的正确性"""
        text_blocks = ["块1", "块2", "块3"]

        result = merge_text_prompt(text_blocks)

        assert "--- 分段大纲 1 ---" in result
        assert "--- 分段大纲 2 ---" in result
        assert "--- 分段大纲 3 ---" in result

    def test_merge_text_prompt_formatting(self):
        """测试merge_text_prompt的格式"""
        text_blocks = ["测试大纲"]

        result = merge_text_prompt(text_blocks)

        # 检查分段格式
        assert "--- 分段大纲" in result
        # 检查前后有换行
        assert "\n\n--- 分段大纲 1 ---\n\n" in result


class TestPromptFunctions:
    """测试提示词函数的通用特性"""

    def test_all_functions_return_string(self):
        """测试所有函数都返回字符串"""
        chunk_text = "测试文本"
        json_blocks = '{"chunk_id": 1, "plot": ["事件1"]}'
        text_blocks = ["测试大纲"]

        chunk_result = chunk_prompt(chunk_text, 1)
        merge_result = merge_prompt(json_blocks)
        merge_text_result = merge_text_prompt(text_blocks)

        assert isinstance(chunk_result, str)
        assert isinstance(merge_result, str)
        assert isinstance(merge_text_result, str)

    def test_all_functions_non_empty(self):
        """测试所有函数都返回非空字符串"""
        chunk_text = "测试文本"
        json_blocks = '{"chunk_id": 1, "plot": ["事件1"]}'
        text_blocks = ["测试大纲"]

        chunk_result = chunk_prompt(chunk_text, 1)
        merge_result = merge_prompt(json_blocks)
        merge_text_result = merge_text_prompt(text_blocks)

        assert len(chunk_result) > 0
        assert len(merge_result) > 0
        assert len(merge_text_result) > 0

    def test_chunk_prompt_different_from_merge_prompts(self):
        """测试chunk_prompt与merge_prompt不同"""
        chunk_text = "测试文本"
        json_blocks = '{"chunk_id": 1, "plot": ["事件1"]}'
        text_blocks = ["测试大纲"]

        chunk_result = chunk_prompt(chunk_text, 1)
        merge_result = merge_prompt(json_blocks)
        merge_text_result = merge_text_prompt(text_blocks)

        # 这些函数应该返回不同的提示词
        assert chunk_result != merge_result
        assert chunk_result != merge_text_result
        # merge_prompt和merge_text_prompt也可能不同
        # 但它们的目的相似，所以这个断言可能会失败
        # assert merge_result != merge_text_result