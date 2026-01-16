"""
测试大纲相关的数据模型
"""

import json
from datetime import datetime

import pytest

from models.outline import OutlineData, TextChunk


class TestTextChunk:
    """测试TextChunk类"""

    def test_text_chunk_initialization(self):
        """测试TextChunk初始化"""
        chunk = TextChunk(
            id=1,
            content="测试内容",
            token_count=100,
            start_position=0,
            end_position=100,
            chapter_title="第一章"
        )
        assert chunk.id == 1
        assert chunk.content == "测试内容"
        assert chunk.token_count == 100
        assert chunk.start_position == 0
        assert chunk.end_position == 100
        assert chunk.chapter_title == "第一章"

    def test_text_chunk_without_chapter_title(self):
        """测试没有章节标题的TextChunk"""
        chunk = TextChunk(
            id=1,
            content="测试内容",
            token_count=100,
            start_position=0,
            end_position=100
        )
        assert chunk.chapter_title is None

    def test_text_chunk_str_representation(self):
        """测试TextChunk的字符串表示"""
        chunk = TextChunk(
            id=1,
            content="测试内容",
            token_count=100,
            start_position=0,
            end_position=100
        )
        assert str(chunk) == "TextChunk(id=1, tokens=100)"


class TestOutlineData:
    """测试OutlineData类"""

    def test_outline_data_initialization(self):
        """测试OutlineData初始化"""
        outline = OutlineData(
            chunk_id=1,
            plot=["剧情1", "剧情2"],
            characters=["人物1", "人物2"],
            relationships=[["人物1", "人物2", "朋友"]],
            raw_response="测试响应",
            processing_time=1.5
        )
        assert outline.chunk_id == 1
        assert outline.plot == ["剧情1", "剧情2"]
        assert outline.characters == ["人物1", "人物2"]
        assert outline.relationships == [["人物1", "人物2", "朋友"]]
        assert outline.raw_response == "测试响应"
        assert outline.processing_time == 1.5
        assert isinstance(outline.created_at, datetime)

    def test_outline_data_with_defaults(self):
        """测试使用默认值的OutlineData"""
        outline = OutlineData(chunk_id=1)
        assert outline.chunk_id == 1
        assert outline.plot == []
        assert outline.characters == []
        assert outline.relationships == []
        assert outline.raw_response is None
        assert outline.processing_time is None
        assert isinstance(outline.created_at, datetime)

    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime(2024, 1, 1, 12, 0, 0)
        outline = OutlineData(
            chunk_id=1,
            plot=["剧情1"],
            characters=["人物1"],
            relationships=[["人物1", "人物2", "朋友"]],
            raw_response="响应",
            processing_time=1.5,
            created_at=now
        )
        result = outline.to_dict()
        assert result["chunk_id"] == 1
        assert result["plot"] == ["剧情1"]
        assert result["events"] == ["剧情1"]  # 向后兼容字段
        assert result["characters"] == ["人物1"]
        assert result["relationships"] == [["人物1", "人物2", "朋友"]]
        assert result["processing_time"] == 1.5
        assert result["created_at"] == "2024-01-01T12:00:00"
        # to_dict不包含raw_response字段
        assert "raw_response" not in result

    def test_to_dict_with_none_created_at(self):
        """测试created_at为None时的to_dict"""
        outline = OutlineData(chunk_id=1)
        outline.created_at = None
        result = outline.to_dict()
        assert result["created_at"] is None

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "chunk_id": 1,
            "plot": ["剧情1"],
            "characters": ["人物1"],
            "relationships": [["人物1", "人物2", "朋友"]],
            "processing_time": 1.5,
            "created_at": "2024-01-01T12:00:00"
        }
        outline = OutlineData.from_dict(data)
        assert outline.chunk_id == 1
        assert outline.plot == ["剧情1"]
        assert outline.characters == ["人物1"]
        assert outline.relationships == [["人物1", "人物2", "朋友"]]
        assert outline.processing_time == 1.5
        assert outline.created_at == datetime(2024, 1, 1, 12, 0, 0)

    def test_from_dict_with_events_backward_compatibility(self):
        """测试从字典创建（使用events字段向后兼容）"""
        data = {
            "chunk_id": 1,
            "events": ["剧情1", "剧情2"],  # 使用旧字段名
            "characters": ["人物1"],
            "relationships": []
        }
        outline = OutlineData.from_dict(data)
        assert outline.plot == ["剧情1", "剧情2"]
        assert outline.characters == ["人物1"]

    def test_from_dict_with_missing_fields(self):
        """测试从字典创建（缺少字段）"""
        data = {"chunk_id": 1}
        outline = OutlineData.from_dict(data)
        assert outline.chunk_id == 1
        assert outline.plot == []
        assert outline.characters == []
        assert outline.relationships == []
        assert outline.processing_time is None

    def test_from_dict_with_none_created_at(self):
        """测试从字典创建（created_at为None）"""
        data = {
            "chunk_id": 1,
            "created_at": None
        }
        outline = OutlineData.from_dict(data)
        # from_dict方法会检查created_at是否在data中且不为None
        # 如果created_at为None，它会保持datetime.now()的默认值
        # 因为if "created_at" in data and data["created_at"]: 这个条件不会满足
        assert isinstance(outline.created_at, datetime)

    def test_from_dict_with_missing_created_at(self):
        """测试从字典创建（缺少created_at字段）"""
        data = {"chunk_id": 1}
        outline = OutlineData.from_dict(data)
        assert isinstance(outline.created_at, datetime)

    def test_validate_complete_outline(self):
        """测试验证完整大纲"""
        outline = OutlineData(
            chunk_id=1,
            plot=["剧情1"],
            characters=["人物1"]
        )
        assert outline.validate(allow_partial=False) is True

    def test_validate_with_only_plot(self):
        """测试验证只有剧情的大纲"""
        outline = OutlineData(
            chunk_id=1,
            plot=["剧情1"]
        )
        assert outline.validate(allow_partial=False) is True

    def test_validate_with_only_characters(self):
        """测试验证只有人物的大纲"""
        outline = OutlineData(
            chunk_id=1,
            characters=["人物1"]
        )
        assert outline.validate(allow_partial=False) is True

    def test_validate_empty_outline(self):
        """测试验证空大纲"""
        outline = OutlineData(chunk_id=1)
        assert outline.validate(allow_partial=False) is False

    def test_validate_with_negative_chunk_id(self):
        """测试验证负数chunk_id"""
        outline = OutlineData(chunk_id=-1, plot=["剧情1"])
        assert outline.validate(allow_partial=False) is False

    def test_validate_partial_with_plot(self):
        """测试验证部分完成大纲（有剧情）"""
        outline = OutlineData(
            chunk_id=1,
            plot=["剧情1"]
        )
        assert outline.validate(allow_partial=True) is True

    def test_validate_partial_with_characters(self):
        """测试验证部分完成大纲（有人物）"""
        outline = OutlineData(
            chunk_id=1,
            characters=["人物1"]
        )
        assert outline.validate(allow_partial=True) is True

    def test_validate_partial_with_relationships(self):
        """测试验证部分完成大纲（有关系）"""
        outline = OutlineData(
            chunk_id=1,
            relationships=[["人物1", "人物2", "朋友"]]
        )
        assert outline.validate(allow_partial=True) is True

    def test_validate_partial_empty(self):
        """测试验证空部分完成大纲"""
        outline = OutlineData(chunk_id=1)
        assert outline.validate(allow_partial=True) is False

    def test_validate_partial_with_negative_chunk_id(self):
        """测试验证部分完成大纲（负数chunk_id）"""
        outline = OutlineData(
            chunk_id=-1,
            plot=["剧情1"]
        )
        assert outline.validate(allow_partial=True) is False

    def test_character_count_property(self):
        """测试character_count属性"""
        outline = OutlineData(
            chunk_id=1,
            characters=["人物1", "人物2", "人物3"]
        )
        assert outline.character_count == 3

    def test_character_count_empty(self):
        """测试空人物列表的character_count"""
        outline = OutlineData(chunk_id=1)
        assert outline.character_count == 0

    def test_relationship_count_property(self):
        """测试relationship_count属性"""
        outline = OutlineData(
            chunk_id=1,
            relationships=[
                ["人物1", "人物2", "朋友"],
                ["人物2", "人物3", "敌人"]
            ]
        )
        assert outline.relationship_count == 2

    def test_relationship_count_empty(self):
        """测试空关系列表的relationship_count"""
        outline = OutlineData(chunk_id=1)
        assert outline.relationship_count == 0

    def test_merge_with_same_chunk_id(self):
        """测试合并相同chunk_id的大纲"""
        outline1 = OutlineData(
            chunk_id=1,
            plot=["剧情1"],
            characters=["人物1"],
            relationships=[["人物1", "人物2", "朋友"]]
        )
        outline2 = OutlineData(
            chunk_id=1,
            plot=["剧情2"],
            characters=["人物2"],
            relationships=[["人物2", "人物3", "敌人"]]
        )
        outline1.merge_with(outline2)
        assert outline1.plot == ["剧情1", "剧情2"]
        assert outline1.characters == ["人物1", "人物2"]
        assert outline1.relationships == [["人物1", "人物2", "朋友"], ["人物2", "人物3", "敌人"]]

    def test_merge_with_duplicate_characters(self):
        """测试合并时去重人物"""
        outline1 = OutlineData(
            chunk_id=1,
            characters=["人物1", "人物2"]
        )
        outline2 = OutlineData(
            chunk_id=1,
            characters=["人物2", "人物3"]
        )
        outline1.merge_with(outline2)
        assert outline1.characters == ["人物1", "人物2", "人物3"]

    def test_merge_with_duplicate_relationships(self):
        """测试合并时去重关系"""
        outline1 = OutlineData(
            chunk_id=1,
            relationships=[["人物1", "人物2", "朋友"]]
        )
        outline2 = OutlineData(
            chunk_id=1,
            relationships=[["人物1", "人物2", "朋友"], ["人物2", "人物3", "敌人"]]
        )
        outline1.merge_with(outline2)
        assert outline1.relationships == [["人物1", "人物2", "朋友"], ["人物2", "人物3", "敌人"]]

    def test_merge_with_different_chunk_id_raises_error(self):
        """测试合并不同chunk_id的大纲应抛出异常"""
        outline1 = OutlineData(chunk_id=1, plot=["剧情1"])
        outline2 = OutlineData(chunk_id=2, plot=["剧情2"])
        with pytest.raises(ValueError, match="Cannot merge outlines with different chunk IDs"):
            outline1.merge_with(outline2)

    def test_merge_with_raw_response(self):
        """测试合并时raw_response的处理"""
        outline1 = OutlineData(
            chunk_id=1,
            raw_response="响应1"
        )
        outline2 = OutlineData(
            chunk_id=1,
            raw_response="响应2"
        )
        outline1.merge_with(outline2)
        assert outline1.raw_response == "响应1"  # 保留原始响应

    def test_merge_with_none_raw_response(self):
        """测试合并时raw_response为None的处理"""
        outline1 = OutlineData(
            chunk_id=1,
            raw_response=None
        )
        outline2 = OutlineData(
            chunk_id=1,
            raw_response="响应2"
        )
        outline1.merge_with(outline2)
        assert outline1.raw_response == "响应2"  # 使用新响应

    def test_merge_with_processing_time(self):
        """测试合并时processing_time的处理"""
        outline1 = OutlineData(
            chunk_id=1,
            processing_time=1.0
        )
        outline2 = OutlineData(
            chunk_id=1,
            processing_time=2.0
        )
        outline1.merge_with(outline2)
        # processing_time不会更新
        assert outline1.processing_time == 1.0

    def test_json_serialization_roundtrip(self):
        """测试JSON序列化和反序列化往返"""
        original = OutlineData(
            chunk_id=1,
            plot=["剧情1", "剧情2"],
            characters=["人物1", "人物2"],
            relationships=[["人物1", "人物2", "朋友"]],
            processing_time=1.5
        )
        # 序列化
        json_str = json.dumps(original.to_dict(), ensure_ascii=False)
        # 反序列化
        data = json.loads(json_str)
        restored = OutlineData.from_dict(data)
        # 验证
        assert restored.chunk_id == original.chunk_id
        assert restored.plot == original.plot
        assert restored.characters == original.characters
        assert restored.relationships == original.relationships
        assert restored.processing_time == original.processing_time

    def test_json_serialization_backward_compatibility(self):
        """测试使用旧字段名events的JSON序列化"""
        data = {
            "chunk_id": 1,
            "events": ["剧情1", "剧情2"],  # 使用旧字段名
            "characters": ["人物1"]
        }
        json_str = json.dumps(data, ensure_ascii=False)
        restored_data = json.loads(json_str)
        outline = OutlineData.from_dict(restored_data)
        assert outline.plot == ["剧情1", "剧情2"]
        assert outline.characters == ["人物1"]

    def test_outline_data_with_zero_chunk_id(self):
        """测试chunk_id为0的情况"""
        outline = OutlineData(chunk_id=0, plot=["剧情1"])
        assert outline.validate(allow_partial=False) is True

    def test_outline_data_with_large_chunk_id(self):
        """测试大chunk_id"""
        outline = OutlineData(chunk_id=999999, plot=["剧情1"])
        assert outline.chunk_id == 999999
        assert outline.validate(allow_partial=False) is True

    def test_outline_data_with_empty_plot(self):
        """测试空剧情列表"""
        outline = OutlineData(chunk_id=1, plot=[], characters=["人物1"])
        assert outline.validate(allow_partial=False) is True

    def test_outline_data_with_empty_characters(self):
        """测试空人物列表"""
        outline = OutlineData(chunk_id=1, plot=["剧情1"], characters=[])
        assert outline.validate(allow_partial=False) is True

    def test_outline_data_with_empty_relationships(self):
        """测试空关系列表"""
        outline = OutlineData(
            chunk_id=1,
            plot=["剧情1"],
            relationships=[]
        )
        assert outline.relationship_count == 0

    def test_outline_data_created_at_auto_generated(self):
        """测试created_at自动生成"""
        before = datetime.now()
        outline = OutlineData(chunk_id=1)
        after = datetime.now()
        assert before <= outline.created_at <= after

    def test_outline_data_with_custom_created_at(self):
        """测试自定义created_at"""
        custom_time = datetime(2024, 6, 15, 10, 30, 0)
        outline = OutlineData(chunk_id=1, created_at=custom_time)
        assert outline.created_at == custom_time