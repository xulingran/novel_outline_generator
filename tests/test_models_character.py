"""
测试人物相关的数据模型
"""

import pytest

from models.character import Character, Relationship


class TestRelationship:
    """测试Relationship类"""

    def test_relationship_initialization(self):
        """测试Relationship初始化"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友",
            weight=2,
            contexts=["第一章", "第二章"]
        )
        assert rel.character_a == "人物1"
        assert rel.character_b == "人物2"
        assert rel.description == "朋友"
        assert rel.weight == 2
        assert rel.contexts == ["第一章", "第二章"]

    def test_relationship_with_defaults(self):
        """测试使用默认值的Relationship"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        assert rel.weight == 1
        assert rel.contexts == []

    def test_relationship_with_same_characters_raises_error(self):
        """测试创建两端相同人物的Relationship应抛出异常"""
        with pytest.raises(ValueError, match="人物关系的两端不能是同一个人"):
            Relationship(
                character_a="人物1",
                character_b="人物1",
                description="自恋"
            )

    def test_relationship_str_representation(self):
        """测试Relationship的字符串表示"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        assert str(rel) == "人物1 - 人物2: 朋友"

    def test_add_context(self):
        """测试添加关系上下文"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        rel.add_context("第一章")
        assert rel.contexts == ["第一章"]
        rel.add_context("第二章")
        assert rel.contexts == ["第一章", "第二章"]

    def test_add_context_duplicate(self):
        """测试添加重复的上下文"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        rel.add_context("第一章")
        rel.add_context("第一章")  # 重复添加
        assert rel.contexts == ["第一章"]

    def test_add_context_empty_string(self):
        """测试添加空字符串上下文"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        rel.add_context("")
        assert rel.contexts == []

    def test_add_context_none(self):
        """测试添加None上下文"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        rel.add_context(None)
        assert rel.contexts == []

    def test_pair_property(self):
        """测试pair属性"""
        rel = Relationship(
            character_a="人物2",  # 字母顺序在后面
            character_b="人物1",  # 字母顺序在前面
            description="朋友"
        )
        pair = rel.pair
        assert pair == ("人物1", "人物2")

    def test_pair_property_already_sorted(self):
        """测试已经排序的pair属性"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友"
        )
        pair = rel.pair
        assert pair == ("人物1", "人物2")

    def test_pair_property_with_numbers(self):
        """测试包含数字的字符"""
        rel = Relationship(
            character_a="人物2",
            character_b="人物10",
            description="朋友"
        )
        pair = rel.pair
        # 字符串排序，"10" < "2"
        assert pair == ("人物10", "人物2")

    def test_relationship_weight(self):
        """测试关系权重"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="朋友",
            weight=5
        )
        assert rel.weight == 5

    def test_relationship_description(self):
        """测试关系描述"""
        rel = Relationship(
            character_a="人物1",
            character_b="人物2",
            description="最好的朋友"
        )
        assert rel.description == "最好的朋友"


class TestCharacter:
    """测试Character类"""

    def test_character_initialization(self):
        """测试Character初始化"""
        char = Character(
            name="人物1",
            aliases=["别名1", "别名2"],
            descriptions=["描述1", "描述2"],
            appearances=[1, 2, 3],
            first_appearance=1,
            relationship_count=2,
            importance_score=0.8
        )
        assert char.name == "人物1"
        assert char.aliases == ["别名1", "别名2"]
        assert char.descriptions == ["描述1", "描述2"]
        assert char.appearances == [1, 2, 3]
        assert char.first_appearance == 1
        assert char.relationship_count == 2
        assert char.importance_score == 0.8

    def test_character_with_defaults(self):
        """测试使用默认值的Character"""
        char = Character(name="人物1")
        assert char.name == "人物1"
        assert char.aliases == []
        assert char.descriptions == []
        assert char.appearances == []
        assert char.first_appearance is None
        assert char.relationship_count == 0
        assert char.importance_score == 0.0

    def test_character_str_representation(self):
        """测试Character的字符串表示"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3]
        )
        assert str(char) == "Character(人物1, appearances=3)"

    def test_character_str_representation_no_appearances(self):
        """测试没有出现记录的Character字符串表示"""
        char = Character(name="人物1")
        assert str(char) == "Character(人物1, appearances=0)"

    def test_add_appearance(self):
        """测试添加出现记录"""
        char = Character(name="人物1")
        char.add_appearance(1)
        assert char.appearances == [1]
        char.add_appearance(2)
        assert char.appearances == [1, 2]
        char.add_appearance(3)
        assert char.appearances == [1, 2, 3]

    def test_add_appearance_sorted(self):
        """测试出现记录自动排序"""
        char = Character(name="人物1")
        char.add_appearance(3)
        char.add_appearance(1)
        char.add_appearance(2)
        assert char.appearances == [1, 2, 3]

    def test_add_appearance_duplicate(self):
        """测试添加重复的出现记录"""
        char = Character(name="人物1")
        char.add_appearance(1)
        char.add_appearance(1)  # 重复添加
        assert char.appearances == [1]

    def test_add_appearance_updates_first_appearance(self):
        """测试添加出现记录更新首次出现位置"""
        char = Character(name="人物1")
        assert char.first_appearance is None
        char.add_appearance(5)
        assert char.first_appearance == 5
        char.add_appearance(3)
        assert char.first_appearance == 3
        char.add_appearance(7)
        assert char.first_appearance == 3  # 不应该更新

    def test_add_appearance_with_zero_chunk_id(self):
        """测试添加chunk_id为0的出现记录"""
        char = Character(name="人物1")
        char.add_appearance(0)
        assert char.appearances == [0]
        assert char.first_appearance == 0

    def test_add_appearance_with_negative_chunk_id(self):
        """测试添加负数chunk_id的出现记录"""
        char = Character(name="人物1")
        char.add_appearance(-1)
        assert char.appearances == [-1]
        assert char.first_appearance == -1

    def test_add_alias(self):
        """测试添加别名"""
        char = Character(name="人物1")
        char.add_alias("别名1")
        assert char.aliases == ["别名1"]
        char.add_alias("别名2")
        assert char.aliases == ["别名1", "别名2"]

    def test_add_alias_duplicate(self):
        """测试添加重复的别名"""
        char = Character(name="人物1")
        char.add_alias("别名1")
        char.add_alias("别名1")  # 重复添加
        assert char.aliases == ["别名1"]

    def test_add_alias_same_as_name(self):
        """测试添加与名字相同的别名"""
        char = Character(name="人物1")
        char.add_alias("人物1")
        assert char.aliases == []

    def test_add_alias_empty_string(self):
        """测试添加空字符串别名"""
        char = Character(name="人物1")
        char.add_alias("")
        assert char.aliases == []

    def test_add_alias_none(self):
        """测试添加None别名"""
        char = Character(name="人物1")
        char.add_alias(None)
        assert char.aliases == []

    def test_add_description(self):
        """测试添加描述"""
        char = Character(name="人物1")
        char.add_description("描述1")
        assert char.descriptions == ["描述1"]
        char.add_description("描述2")
        assert char.descriptions == ["描述1", "描述2"]

    def test_add_description_duplicate(self):
        """测试添加重复的描述"""
        char = Character(name="人物1")
        char.add_description("描述1")
        char.add_description("描述1")  # 重复添加
        assert char.descriptions == ["描述1"]

    def test_add_description_empty_string(self):
        """测试添加空字符串描述"""
        char = Character(name="人物1")
        char.add_description("")
        assert char.descriptions == []

    def test_add_description_none(self):
        """测试添加None描述"""
        char = Character(name="人物1")
        char.add_description(None)
        assert char.descriptions == []

    def test_total_appearances_property(self):
        """测试total_appearances属性"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3, 4, 5]
        )
        assert char.total_appearances == 5

    def test_total_appearances_empty(self):
        """测试空出现记录的total_appearances"""
        char = Character(name="人物1")
        assert char.total_appearances == 0

    def test_calculate_importance_with_zero_total_chunks(self):
        """测试total_chunks为0时的重要性计算"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3],
            relationship_count=2,
            descriptions=["描述1", "描述2"]
        )
        score = char.calculate_importance(0)
        assert score == 0.0
        assert char.importance_score == 0.0

    def test_calculate_importance_high_frequency(self):
        """测试高频率出现的重要性计算"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            relationship_count=5,
            descriptions=["描述1", "描述2", "描述3"]
        )
        # 出现在10个块中的10个，频率=1.0
        # 关系数5，关系因子=min(5/10, 1.0)=0.5
        # 描述数3，描述因子=min(3/5, 1.0)=0.6
        # 综合评分 = 1.0*0.5 + 0.5*0.3 + 0.6*0.2 = 0.5 + 0.15 + 0.12 = 0.77
        score = char.calculate_importance(10)
        assert score == 0.77
        assert char.importance_score == 0.77

    def test_calculate_importance_low_frequency(self):
        """测试低频率出现的重要性计算"""
        char = Character(
            name="人物1",
            appearances=[5],  # 只出现1次
            relationship_count=0,
            descriptions=["描述1"]
        )
        # 出现在10个块中的1个，频率=0.1
        # 关系数0，关系因子=0.0
        # 描述数1，描述因子=min(1/5, 1.0)=0.2
        # 综合评分 = 0.1*0.5 + 0.0*0.3 + 0.2*0.2 = 0.05 + 0.0 + 0.04 = 0.09
        score = char.calculate_importance(10)
        assert score == pytest.approx(0.09)
        assert char.importance_score == pytest.approx(0.09)

    def test_calculate_importance_medium_frequency(self):
        """测试中等频率出现的重要性计算"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3, 4, 5],  # 出现在10个块中的5个
            relationship_count=3,
            descriptions=["描述1", "描述2"]
        )
        # 频率=5/10=0.5
        # 关系因子=min(3/10, 1.0)=0.3
        # 描述因子=min(2/5, 1.0)=0.4
        # 综合评分 = 0.5*0.5 + 0.3*0.3 + 0.4*0.2 = 0.25 + 0.09 + 0.08 = 0.42
        score = char.calculate_importance(10)
        assert score == 0.42
        assert char.importance_score == 0.42

    def test_calculate_importance_max_relationships(self):
        """测试最大关系数的重要性计算"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3],
            relationship_count=15,  # 超过10
            descriptions=["描述1", "描述2", "描述3"]
        )
        # 关系因子=min(15/10, 1.0)=1.0
        score = char.calculate_importance(10)
        # 频率=3/10=0.3，描述因子=min(3/5, 1.0)=0.6
        # 综合评分 = 0.3*0.5 + 1.0*0.3 + 0.6*0.2 = 0.15 + 0.3 + 0.12 = 0.57
        assert score == 0.57

    def test_calculate_importance_max_descriptions(self):
        """测试最大描述数的重要性计算"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3],
            relationship_count=2,
            descriptions=["描述1", "描述2", "描述3", "描述4", "描述5", "描述6", "描述7"]  # 超过5
        )
        # 描述因子=min(7/5, 1.0)=1.0
        score = char.calculate_importance(10)
        # 频率=3/10=0.3，关系因子=min(2/10, 1.0)=0.2
        # 综合评分 = 0.3*0.5 + 0.2*0.3 + 1.0*0.2 = 0.15 + 0.06 + 0.2 = 0.41
        assert score == pytest.approx(0.41)

    def test_calculate_importance_updates_score(self):
        """测试calculate_importance更新importance_score"""
        char = Character(
            name="人物1",
            appearances=[1, 2, 3]
        )
        assert char.importance_score == 0.0
        char.calculate_importance(10)
        assert char.importance_score > 0.0

    def test_character_with_large_appearances(self):
        """测试大量出现记录"""
        char = Character(name="人物1")
        for i in range(100):
            char.add_appearance(i)
        assert char.total_appearances == 100
        assert char.appearances == list(range(100))

    def test_character_with_many_aliases(self):
        """测试多个别名"""
        char = Character(name="人物1")
        for i in range(20):
            char.add_alias(f"别名{i}")
        assert len(char.aliases) == 20

    def test_character_with_many_descriptions(self):
        """测试多个描述"""
        char = Character(name="人物1")
        for i in range(20):
            char.add_description(f"描述{i}")
        assert len(char.descriptions) == 20

    def test_character_name_property(self):
        """测试人物名字属性"""
        char = Character(name="张三")
        assert char.name == "张三"

    def test_character_with_special_characters_in_name(self):
        """测试名字中包含特殊字符"""
        char = Character(name="人物A-1")
        assert char.name == "人物A-1"

    def test_character_with_unicode_name(self):
        """测试Unicode名字"""
        char = Character(name="李明")
        assert char.name == "李明"

    def test_character_relationship_count_property(self):
        """测试关系数量属性"""
        char = Character(name="人物1", relationship_count=5)
        assert char.relationship_count == 5

    def test_character_importance_score_property(self):
        """测试重要性分数属性"""
        char = Character(name="人物1", importance_score=0.95)
        assert char.importance_score == 0.95

    def test_character_appearances_property(self):
        """测试出现记录属性"""
        char = Character(name="人物1", appearances=[1, 3, 5])
        assert char.appearances == [1, 3, 5]

    def test_character_first_appearance_property(self):
        """测试首次出现位置属性"""
        char = Character(name="人物1", first_appearance=10)
        assert char.first_appearance == 10

    def test_character_aliases_property(self):
        """测试别名属性"""
        char = Character(name="人物1", aliases=["别名A", "别名B"])
        assert char.aliases == ["别名A", "别名B"]

    def test_character_descriptions_property(self):
        """测试描述属性"""
        char = Character(name="人物1", descriptions=["描述A", "描述B"])
        assert char.descriptions == ["描述A", "描述B"]