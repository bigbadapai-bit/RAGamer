"""知识库元数据：建库、改库、读回来、删干净，以及两种「用不了」的区分。

写入侧（界面）与读取侧（导入端点）共用这一份形状，所以这里两头都验：
存进去的键名读得回来、坏了的数据报得出来。

删库那几条同时钉住「配置最后才删」这一条：配置是这个库还在的凭据，
先删它，剩下的切片与原图就成了看不见的孤儿。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ragamer.caching import CachedAnswer, CacheUnavailableError, InMemoryAnswerCache, cache_key
from ragamer.conversations import CONVERSATIONS
from ragamer.knowledge import (
    KB_COLLECTION,
    BrokenKnowledgeBase,
    KnowledgeBase,
    PurgeDocumentError,
    PurgeError,
    PurgeInventory,
    UnknownKnowledgeBase,
    create_knowledge_base,
    document_inventory,
    list_knowledge_bases,
    purge_document,
    purge_inventory,
    purge_knowledge_base,
    remove_term,
    set_term,
    update_knowledge_base,
    vocabulary_of,
)
from ragamer.stores.base import UNVERSIONED, StoreUnavailableError, image_key, image_prefix
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


# --- 现行版本 ---


def test_现行版本存得进去也读得回来(docs):
    create_knowledge_base(docs, replace(KnowledgeBase.new(GAME), version="2.0"))

    stored = docs.get(KB_COLLECTION, GAME)
    assert KnowledgeBase.from_payload(GAME, stored).version == "2.0"


def test_没配版本的库读回来是未标注版本(docs):
    """老库里没有这个键。默认不是「随便挑一个版本」，而是未标注版本那条路。"""
    docs.put(KB_COLLECTION, GAME, {"name": "某款游戏"})

    assert list_knowledge_bases(docs)[0].version == UNVERSIONED


# --- 改库 ---


def test_改库之后读回来是新的(docs):
    """名称、启用的类目、术语映射、现行版本一起换掉，**保存后立即生效**。"""
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


#: 删库要清四处，会话那一处存在哪个集合由调用方给（见 `ragamer.knowledge` 的模块说明）
SESSIONS = CONVERSATIONS
#: 一条缓存，用来验证「删库把缓存也清了」
CACHED_KEY = cache_key(GAME, "1.0", "二郎神怎么打")


def _stocked(container) -> None:
    """一个建好、导过资料、存过原图、聊过一句、缓存里有东西的库。"""
    create_knowledge_base(container.docs, KnowledgeBase.new(GAME, "黑神话·悟空"))
    container.chunks.upsert(GAME, [make_chunk(1), make_chunk(2)])
    container.objects.put(image_key(GAME, "a1b2", "立绘.png"), b"PNG")
    container.docs.put(SESSIONS, "s1", {"game_id": GAME, "title": "二郎神怎么打"})
    container.cache.set(CACHED_KEY, CachedAnswer("先定身再贴身输出[1]。"))
    container.cache.record_question(GAME, "二郎神怎么打")


def _purge(container, chunks=None, objects=None, cache=None) -> PurgeInventory:
    """按四处清一遍。默认都用容器里那套，用例只覆盖自己关心的那一个。"""
    return purge_knowledge_base(
        chunks if chunks is not None else container.chunks,
        container.docs,
        objects if objects is not None else container.objects,
        cache if cache is not None else container.cache,
        GAME,
        sessions=SESSIONS,
    )


def test_数一遍这个库在各处占着多少东西(container):
    """确认页照着它列「将要清理什么」，所以它自己必须先能数准，而且只读。"""
    _stocked(container)

    inventory = purge_inventory(
        container.chunks, container.docs, container.objects, GAME, sessions=SESSIONS
    )

    assert (inventory.chunk_count, inventory.image_count, inventory.session_count) == (2, 1, 1)
    # 数一遍不动任何数据
    assert container.chunks.count(GAME) == 2
    assert container.docs.get(KB_COLLECTION, GAME) is not None
    assert container.docs.find(SESSIONS, {"game_id": GAME}) != []


def test_删库把四处一并清掉(container):
    """切片、原图、会话、缓存、配置——**一处都不能留**。

    会话与缓存那两处漏掉的话，用同一个 id 重建库会把旧会话放回左栏、把旧的热门问题
    顶回来，而引用的来源早就删了。
    """
    _stocked(container)

    inventory = _purge(container)

    assert (inventory.chunk_count, inventory.image_count, inventory.session_count) == (2, 1, 1)
    assert container.chunks.count(GAME) == 0
    assert container.objects.list_keys(image_prefix(GAME)) == []
    assert container.docs.find(SESSIONS, {"game_id": GAME}) == []
    assert container.cache.get(CACHED_KEY) is None
    # **提问计数也要清**：它不在缓存前缀里，只清答案那半截会留下旧库的热门问题
    assert container.cache.top_questions(GAME) == ()
    assert container.docs.get(KB_COLLECTION, GAME) is None


def test_删一个库不牵连别的库的会话与缓存(container):
    """与「清前缀时不碰 id 是它前缀的另一个库」同一条道理，只是换到会话与缓存上。"""
    sibling = f"{GAME}_2"
    _stocked(container)
    container.docs.put(SESSIONS, "s9", {"game_id": sibling, "title": "别个库的"})
    other_key = cache_key(sibling, "1.0", "今汐怎么养")
    container.cache.set(other_key, CachedAnswer("先堆暴击[1]。"))
    container.cache.record_question(sibling, "今汐怎么养")

    _purge(container)

    assert [item["_id"] for item in container.docs.find(SESSIONS, {"game_id": sibling})] == ["s9"]
    assert container.cache.get(other_key) is not None
    assert container.cache.top_questions(sibling) != ()


def test_一处清不掉时不跳过其余_并且配置留着(container):
    """配置一删，库里剩下的数据就再也看不见了。留着它，界面上还能再点一次。"""
    _stocked(container)

    with pytest.raises(PurgeError, match="向量库") as caught:
        _purge(container, chunks=BrokenChunkStore())

    # 一次看清还差什么：向量库没清掉，原图、会话、缓存照清
    assert container.objects.list_keys(image_prefix(GAME)) == []
    assert container.docs.find(SESSIONS, {"game_id": GAME}) == []
    assert container.cache.get(CACHED_KEY) is None
    # 库还在列表上，可以重来
    assert container.docs.get(KB_COLLECTION, GAME) is not None
    assert GAME in str(caught.value)


def test_重来一次能把没清掉的补上(container):
    """上一处失败留下的状态是可重入的：已经清掉的那几处再清一次不出错。"""
    _stocked(container)
    with pytest.raises(PurgeError):
        _purge(container, chunks=BrokenChunkStore())

    inventory = _purge(container)

    # 原图与会话上一轮已经清掉了，这一轮报 0；切片这一轮才清掉
    assert (inventory.chunk_count, inventory.image_count, inventory.session_count) == (2, 0, 0)
    assert container.docs.get(KB_COLLECTION, GAME) is None


class ExplodingObjectStore(InMemoryObjectStore):
    """删前缀时炸的不是存储错误。三个适配器只把「连不上」包成 `StoreError`，
    连上之后操作失败漏出来的是供应商自己的异常类型——那种失败同样得保住配置。"""

    def delete_prefix(self, prefix: str) -> int:
        raise RuntimeError("MinIO 返回的删除结果缺了一段")


def test_不是存储错误的那种失败也保住配置(container):
    _stocked(container)

    with pytest.raises(PurgeError, match="对象存储"):
        _purge(container, objects=ExplodingObjectStore())

    assert container.docs.get(KB_COLLECTION, GAME) is not None


class ExplodingCache(InMemoryAnswerCache):
    """删缓存时炸的缓存。缓存连不上本该降级，但**删库这一路不能静默跳过**：
    漏掉的那一处会让重建出来的库顶着旧库的答案与热门问题。"""

    def drop(self, game_id: str) -> int:
        raise CacheUnavailableError("Redis", "redis.test:6379", 5.0, "连接被拒绝")


def test_缓存清不掉也保住配置(container):
    _stocked(container)

    with pytest.raises(PurgeError, match="缓存"):
        _purge(container, cache=ExplodingCache())

    assert container.docs.get(KB_COLLECTION, GAME) is not None


def test_清前缀时不碰_id_是它前缀的另一个库(container):
    """`delete_prefix` 比的是字符串前缀，不是目录：按 `images/black_myth` 去删，
    `black_myth_2` 这个库的原图会被一并收走，而且不报错。"""
    sibling = f"{GAME}_2"
    _stocked(container)
    container.objects.put(image_key(sibling, "a1b2", "它的立绘.png"), b"PNG")
    container.chunks.upsert(sibling, [make_chunk(9, game_id=sibling)])

    inventory = purge_inventory(
        container.chunks, container.docs, container.objects, GAME, sessions=SESSIONS
    )
    _purge(container)

    assert inventory.image_count == 1  # 只数自己那一张
    assert container.objects.list_keys(image_prefix(sibling)) == [
        image_key(sibling, "a1b2", "它的立绘.png")
    ]
    assert container.chunks.count(sibling) == 1


# --- 删一份资料（`purge_document`） ---


#: 这一份自己的图
OWN_IMAGE = image_key(GAME, "a1b2", "立绘.png")
#: 两份资料都引用着的图：同一个来源被导成两份标题不同的文档时就是这样
SHARED_IMAGE = image_key(GAME, "c3d4", "地图.png")


def _two_documents(container) -> None:
    """一个建好的库，里面两份资料：共用一张图，各自还有自己的一张。"""
    create_knowledge_base(container.docs, KnowledgeBase.new(GAME, "黑神话·悟空"))
    container.chunks.upsert(
        GAME,
        [
            make_chunk(1, doc_title="二郎神", image_urls=(OWN_IMAGE, SHARED_IMAGE)),
            make_chunk(
                2,
                doc_title="白骨精",
                image_urls=(SHARED_IMAGE, image_key(GAME, "e5f6", "三阶段.png")),
            ),
        ],
    )
    for key in (OWN_IMAGE, SHARED_IMAGE, image_key(GAME, "e5f6", "三阶段.png")):
        container.objects.put(key, b"PNG")


def _purge_document(container, objects=None, cache=None):
    return purge_document(
        container.chunks,
        objects if objects is not None else container.objects,
        cache if cache is not None else container.cache,
        GAME,
        doc_title="二郎神",
        version="1.0",
    )


def test_删一份资料只动它自己的那一片(container):
    _two_documents(container)

    inventory = _purge_document(container)

    assert (inventory.chunk_count, inventory.image_count) == (1, 1)
    assert container.chunks.fetch_document(GAME, "二郎神", version="1.0") == []
    remaining = container.chunks.fetch_document(GAME, "白骨精", version="1.0")
    assert [chunk.doc_title for chunk in remaining] == ["白骨精"]
    # 库本身还在：删的是资料，不是库
    assert container.docs.get(KB_COLLECTION, GAME) is not None


def test_别的资料还在引用的原图不删(container):
    """对象 key 里那层摘要来自来源本身：同一个来源被导成两份标题不同的文档时，两份指着
    同一批 key——照单删下去，另一份的图会变成打不开的空图，而且不报错。"""
    _two_documents(container)

    _purge_document(container)

    keys = set(container.objects.list_keys())
    assert SHARED_IMAGE in keys
    assert image_key(GAME, "e5f6", "三阶段.png") in keys
    assert OWN_IMAGE not in keys  # 只有这一份在用的那张，跟着走


def test_确认页数的与真删的是同一批(container):
    """页面上写「8 个原图」而真删掉 20 个，那份确认就成了摆设。"""
    _two_documents(container)

    counted = document_inventory(container.chunks, GAME, doc_title="二郎神", version="1.0")
    deleted = _purge_document(container)

    assert (counted.chunk_count, counted.image_count) == (deleted.chunk_count, deleted.image_count)
    assert (counted.chunk_count, counted.image_count) == (1, 1)


def test_数一遍不动任何数据(container):
    _two_documents(container)

    document_inventory(container.chunks, GAME, doc_title="二郎神", version="1.0")

    assert container.chunks.count(GAME) == 2
    assert len(container.objects.list_keys()) == 3


def test_删一份资料清缓存但留着提问计数与会话(container):
    """库还在，热门问题该留着；会话是历史，删库那条路才会连它一起清。"""
    _two_documents(container)
    container.docs.put(SESSIONS, "s1", {"game_id": GAME, "title": "二郎神怎么打"})
    container.cache.set(CACHED_KEY, CachedAnswer("先定身再贴身输出[1]。"))
    container.cache.record_question(GAME, "二郎神怎么打")

    _purge_document(container)

    assert container.cache.get(CACHED_KEY) is None
    assert container.cache.top_questions(GAME) != ()
    assert container.docs.find(SESSIONS, {"game_id": GAME}) != []


def test_只删对象_key_外链不会被拿去删(container):
    """切分摘掉图片地址那次修复之前入库的切片里还留着外链——拿一条网址去删对象，
    只会把那一步整条弄失败（而切片本来是该删掉的）。"""
    create_knowledge_base(container.docs, KnowledgeBase.new(GAME, "黑神话·悟空"))
    container.chunks.upsert(
        GAME,
        [make_chunk(1, doc_title="二郎神", image_urls=("https://img.test/18px-图标-衣甲.png",))],
    )

    inventory = _purge_document(container)

    assert inventory.image_count == 0
    assert container.chunks.fetch_document(GAME, "二郎神", version="1.0") == []


class BrokenObjectStore(InMemoryObjectStore):
    """删对象时炸的存储。"""

    def delete(self, key: str) -> None:
        raise StoreUnavailableError("MinIO", "minio.test:9000", 5.0, "连接被拒绝")


def test_原图删不掉时报出来_已经删掉的切片不回头(container):
    """一处失败不跳过其余，也不假装成功：缓存照清，报出还差哪一处。"""
    _two_documents(container)
    container.cache.set(CACHED_KEY, CachedAnswer("先定身[1]。"))

    with pytest.raises(PurgeDocumentError, match="对象存储"):
        _purge_document(container, objects=BrokenObjectStore())

    assert container.chunks.fetch_document(GAME, "二郎神", version="1.0") == []
    assert container.cache.get(CACHED_KEY) is None


class ExplodingDocumentCache(InMemoryAnswerCache):
    """清缓存时炸的缓存。这一路不能静默跳过：漏掉那一处，再问同一个问题还会命中
    基于旧语料的答案，而它带着已删资料的引用、看起来完全正常。"""

    def invalidate(self, game_id: str) -> int:
        raise CacheUnavailableError("Redis", "redis.test:6379", 5.0, "连接被拒绝")


def test_缓存清不掉也算没清干净(container):
    _two_documents(container)

    with pytest.raises(PurgeDocumentError, match="缓存"):
        _purge_document(container, cache=ExplodingDocumentCache())


# --- 术语映射的增删 ---


def test_加一条映射之后读词表就有它了(docs):
    create_knowledge_base(docs, KnowledgeBase.new(GAME, "", (SubjectType.CHARACTER,)))

    set_term(docs, GAME, "妖王", SubjectType.CHARACTER)

    assert vocabulary_of(docs, GAME).resolve("妖王") is SubjectType.CHARACTER
    # 启用的类目原样带着，没被这次改动冲掉
    assert vocabulary_of(docs, GAME).subject_types == (SubjectType.CHARACTER,)


def test_同一个叫法再加一次是改归类不是加两条(docs):
    create_knowledge_base(
        docs, KnowledgeBase.new(GAME, "", (SubjectType.CHARACTER, SubjectType.ITEM))
    )

    set_term(docs, GAME, "妖王", SubjectType.CHARACTER)
    set_term(docs, GAME, "妖王", SubjectType.ITEM)

    assert vocabulary_of(docs, GAME).term_mapping == {"妖王": SubjectType.ITEM}


def test_去掉一条映射之后这个词就只剩模型兜底了(docs):
    create_knowledge_base(
        docs,
        KnowledgeBase(
            GAME, "", TagVocabulary((SubjectType.CHARACTER,), {"妖王": SubjectType.CHARACTER})
        ),
    )

    remove_term(docs, GAME, "妖王")

    assert vocabulary_of(docs, GAME).term_mapping == {}
    # 删一个表里没有的叫法不出错
    remove_term(docs, GAME, "没配过的叫法")


def test_配置读不了的库不让改映射(docs):
    """那份映射本来就没读出来，照着默认值写回去等于把它悄悄清空。"""
    broken = {"subject_types": ["这不是类目"], "term_mapping": {"妖王": "character"}}
    docs.put(KB_COLLECTION, GAME, broken)

    with pytest.raises(BrokenKnowledgeBase, match="配置读不了"):
        set_term(docs, GAME, "心法", SubjectType.SKILL)

    assert docs.get(KB_COLLECTION, GAME) == broken
