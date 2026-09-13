"""知识库元数据：建库、改库、读回来、删干净，以及两种「用不了」的区分。

写入侧（界面）与读取侧（导入端点）共用这一份形状，所以这里两头都验：
存进去的键名读得回来、坏了的数据报得出来。

删库那几条同时钉住「配置最后才删」这一条：配置是这个库还在的凭据，
先删它，剩下的切片与原图就成了看不见的孤儿。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ragamer.knowledge import (
    KB_COLLECTION,
    BrokenKnowledgeBase,
    KnowledgeBase,
    PurgeError,
    UnknownKnowledgeBase,
    create_knowledge_base,
    list_knowledge_bases,
    purge_inventory,
    purge_knowledge_base,
    update_knowledge_base,
    vocabulary_of,
)
from ragamer.stores.base import UNVERSIONED, image_key, image_prefix
from ragamer.stores.memory import InMemoryDocStore, InMemoryObjectStore
from ragamer.tagging import SubjectType, TagVocabulary

from .conftest import BrokenChunkStore, make_chunk, make_container

GAME = "black_myth"


@pytest.fixture
def docs() -> InMemoryDocStore:
    return InMemoryDocStore()


@pytest.fixture
def container():
    return make_container()


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


# --- 当前生效版本 ---


def test_当前生效版本存得进去也读得回来(docs):
    create_knowledge_base(docs, replace(KnowledgeBase.new(GAME), version="2.0"))

    stored = docs.get(KB_COLLECTION, GAME)
    assert KnowledgeBase.from_payload(GAME, stored).version == "2.0"


def test_没配版本的库读回来是未标注版本(docs):
    """老库里没有这个键。默认不是「随便挑一个版本」，而是未标注版本那条路。"""
    docs.put(KB_COLLECTION, GAME, {"name": "某款游戏"})

    assert list_knowledge_bases(docs)[0].version == UNVERSIONED


# --- 改库 ---


def test_改库之后读回来是新的(docs):
    """名称、启用的类目、术语映射、当前生效版本一起换掉，**保存后立即生效**。"""
    create_knowledge_base(docs, KnowledgeBase.new(GAME, "旧名", (SubjectType.CHARACTER,)))

    update_knowledge_base(
        docs,
        KnowledgeBase(
            game_id=GAME,
            name="新名",
            vocabulary=TagVocabulary((SubjectType.ITEM,), {"根器": SubjectType.ITEM}),
            version="2.0",
        ),
    )

    read_back = list_knowledge_bases(docs)[0]
    assert read_back.name == "新名"
    assert read_back.vocabulary.subject_types == (SubjectType.ITEM,)
    assert read_back.vocabulary.term_mapping == {"根器": SubjectType.ITEM}
    assert read_back.version == "2.0"


def test_改一个不存在的库时报错而不是凭空建一个(docs):
    """界面上点的是「保存」：id 敲错时冒出一个新库，不是它要的结果。"""
    with pytest.raises(UnknownKnowledgeBase, match="不存在"):
        update_knowledge_base(docs, KnowledgeBase.new(GAME))

    assert docs.get(KB_COLLECTION, GAME) is None


# --- 删库 ---


def _stocked(container) -> None:
    """一个建好、导过资料、存过原图的库。"""
    create_knowledge_base(container.docs, KnowledgeBase.new(GAME, "黑神话·悟空"))
    container.chunks.upsert(GAME, [make_chunk(1), make_chunk(2)])
    container.objects.put(image_key(GAME, "a1b2", "立绘.png"), b"PNG")


def test_数一遍这个库在各处占着多少东西(container):
    """确认页照着它列「将要清理什么」，所以它自己必须先能数准，而且只读。"""
    _stocked(container)

    inventory = purge_inventory(container.chunks, container.objects, GAME)

    assert (inventory.chunk_count, inventory.image_count) == (2, 1)
    # 数一遍不动任何数据
    assert container.chunks.count(GAME) == 2
    assert container.docs.get(KB_COLLECTION, GAME) is not None


def test_删库把切片_原图与配置一并清掉(container):
    _stocked(container)

    inventory = purge_knowledge_base(container.chunks, container.docs, container.objects, GAME)

    assert (inventory.chunk_count, inventory.image_count) == (2, 1)
    assert container.chunks.count(GAME) == 0
    assert container.objects.list_keys(image_prefix(GAME)) == []
    assert container.docs.get(KB_COLLECTION, GAME) is None


def test_一处清不掉时不跳过其余_并且配置留着(container):
    """配置一删，库里剩下的数据就再也看不见了。留着它，界面上还能再点一次。"""
    _stocked(container)

    with pytest.raises(PurgeError, match="向量库") as caught:
        purge_knowledge_base(BrokenChunkStore(), container.docs, container.objects, GAME)

    # 一次看清还差什么：向量库没清掉，原图照清
    assert container.objects.list_keys(image_prefix(GAME)) == []
    # 库还在列表上，可以重来
    assert container.docs.get(KB_COLLECTION, GAME) is not None
    assert GAME in str(caught.value)


def test_重来一次能把没清掉的补上(container):
    """上一处失败留下的状态是可重入的：已经清掉的那几处再清一次不出错。"""
    _stocked(container)
    with pytest.raises(PurgeError):
        purge_knowledge_base(BrokenChunkStore(), container.docs, container.objects, GAME)

    inventory = purge_knowledge_base(container.chunks, container.docs, container.objects, GAME)

    # 原图上一轮已经清掉了，这一轮报 0；切片这一轮才清掉
    assert (inventory.chunk_count, inventory.image_count) == (2, 0)
    assert container.docs.get(KB_COLLECTION, GAME) is None


class ExplodingObjectStore(InMemoryObjectStore):
    """删前缀时炸的不是存储错误。三个适配器只把「连不上」包成 `StoreError`，
    连上之后操作失败漏出来的是供应商自己的异常类型——那种失败同样得保住配置。"""

    def delete_prefix(self, prefix: str) -> int:
        raise RuntimeError("MinIO 返回的删除结果缺了一段")


def test_不是存储错误的那种失败也保住配置(container):
    _stocked(container)

    with pytest.raises(PurgeError, match="对象存储"):
        purge_knowledge_base(container.chunks, container.docs, ExplodingObjectStore(), GAME)

    assert container.docs.get(KB_COLLECTION, GAME) is not None
