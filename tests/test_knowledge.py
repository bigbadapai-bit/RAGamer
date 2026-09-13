"""知识库元数据：建库、读回来，以及两种「用不了」的区分。

写入侧（界面）与读取侧（导入端点）共用这一份形状，所以这里两头都验：
存进去的键名读得回来、坏了的数据报得出来。
"""

from __future__ import annotations

import pytest

from ragamer.knowledge import (
    KB_COLLECTION,
    BrokenKnowledgeBase,
    KnowledgeBase,
    UnknownKnowledgeBase,
    create_knowledge_base,
    list_knowledge_bases,
    vocabulary_of,
)
from ragamer.stores.memory import InMemoryDocStore
from ragamer.tagging import SubjectType

GAME = "black_myth"


@pytest.fixture
def docs() -> InMemoryDocStore:
    return InMemoryDocStore()


# --- 建库 ---


def test_建好的库能原样读回来(docs):
    create_knowledge_base(
        docs,
        KnowledgeBase.new(GAME, "黑神话·悟空", (SubjectType.CHARACTER, SubjectType.ITEM)),
    )

    stored = docs.get(KB_COLLECTION, GAME)
    read_back = KnowledgeBase.from_payload(GAME, stored)

    assert read_back.game_id == GAME
    assert read_back.name == "黑神话·悟空"
    assert read_back.vocabulary.subject_types == (SubjectType.CHARACTER, SubjectType.ITEM)
    assert read_back.problem == ""


def test_显示名留空时回落到游戏_id(docs):
    create_knowledge_base(docs, KnowledgeBase.new(GAME, "  "))

    assert list_knowledge_bases(docs)[0].name == GAME


def test_一个主体类型都没启用时报错而不是默认全开(docs):
    """漏勾和多勾一样容易发生；静默全开会让标签多出一批人以为自己关掉了的。"""
    with pytest.raises(ValueError, match="至少要启用一个主体类型"):
        create_knowledge_base(docs, KnowledgeBase.new(GAME, "", ()))

    assert docs.get(KB_COLLECTION, GAME) is None


@pytest.mark.parametrize("game_id", ["黑神话", "a-b", "", "_bad space"])
def test_游戏_id_不合法时报错(docs, game_id):
    """它同时是 Milvus 的 collection 名，规则与那边共用一套（ADR-0002）。"""
    with pytest.raises(ValueError, match="游戏 id 不合法"):
        create_knowledge_base(docs, KnowledgeBase.new(game_id))


def test_同_id_再建一次报错且不动已有那份(docs):
    """覆盖会连着术语映射一起换掉，而界面上看起来只是「又建了一个」。"""
    original = KnowledgeBase.new(GAME, "黑神话·悟空", (SubjectType.CHARACTER,))
    create_knowledge_base(docs, original)
    before = docs.get(KB_COLLECTION, GAME)

    with pytest.raises(ValueError, match="已经有一个 id 为"):
        create_knowledge_base(docs, KnowledgeBase.new(GAME, "换个名字", (SubjectType.ITEM,)))

    assert docs.get(KB_COLLECTION, GAME) == before


# --- 读库 ---


def test_配置读不了的库仍然列出来并带上原因(docs):
    """藏起来的话，id 又被占着——人既建不了同 id 的新库，也不知道该去修哪个。"""
    docs.put(KB_COLLECTION, GAME, {"name": "坏掉的库", "subject_types": ["这不是类目"]})

    (broken,) = list_knowledge_bases(docs)

    assert (broken.game_id, broken.name) == (GAME, "坏掉的库")
    assert "这不是类目" in broken.problem


def test_库不存在与配置读不了是两种错(docs):
    with pytest.raises(UnknownKnowledgeBase, match="不存在"):
        vocabulary_of(docs, GAME)

    docs.put(KB_COLLECTION, GAME, {"subject_types": ["这不是类目"]})

    with pytest.raises(BrokenKnowledgeBase, match="配置读不了") as caught:
        vocabulary_of(docs, GAME)
    assert "这不是类目" in str(caught.value)


def test_没配主体类型的库按默认全开降级(docs):
    """用户自定义库没配映射时走的就是这条：标签稀疏，但不会漏（架构文档 §2.3）。"""
    docs.put(KB_COLLECTION, GAME, {"name": "某款游戏"})

    vocabulary = vocabulary_of(docs, GAME)

    assert vocabulary.subject_types == tuple(SubjectType)
    assert vocabulary.term_mapping == {}


def test_列表按游戏_id_排列(docs):
    for game_id in ("zelda", "black_myth", "ghost"):
        create_knowledge_base(docs, KnowledgeBase.new(game_id))

    assert [base.game_id for base in list_knowledge_bases(docs)] == [
        "black_myth",
        "ghost",
        "zelda",
    ]
