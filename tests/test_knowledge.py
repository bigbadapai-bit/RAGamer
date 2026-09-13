"""知识库元数据的读取：库里有哪些游戏、每个库叫什么、现行版本是哪个。

这份集合的 id 同时是 Milvus 的 collection 名与检索的入口，所以读到的 id 直接能用；
显示名只是给人看的，缺了就退回 id。**缺一个键不等于配置坏了**——建库时没填名字、
老库里没有 `version` 键，都是常事，该走默认值而不是把整个库判成坏的。
"""

from __future__ import annotations

from ragamer.knowledge import KB_COLLECTION, knowledge_base, knowledge_bases
from ragamer.stores.base import UNVERSIONED
from ragamer.stores.memory import InMemoryDocStore

GAME = "black_myth"


def docs_with(**payloads: dict) -> InMemoryDocStore:
    docs = InMemoryDocStore()
    for game_id, payload in payloads.items():
        docs.put(KB_COLLECTION, game_id, payload)
    return docs


def test_列出全部知识库并带上显示名与现行版本():
    docs = docs_with(
        black_myth={"name": "黑神话·悟空", "version": "2.0"},
        yanyun={"name": "燕云十六声", "version": "1.0"},
    )

    bases = knowledge_bases(docs)

    assert [(base.game_id, base.name, base.version) for base in bases] == [
        ("black_myth", "黑神话·悟空", "2.0"),
        ("yanyun", "燕云十六声", "1.0"),
    ]


def test_没填名字时显示名退回游戏_id():
    """建库时名字留空是常事：界面上显示 `black_myth` 好过显示一片空白。"""
    docs = docs_with(black_myth={"version": "1.0"})

    assert knowledge_bases(docs)[0].name == GAME


def test_老库里没有版本键时是未标注版本():
    """现行的库里没有这个键。回落到未标注版本，不是随手挑一个，也不是判成坏配置。"""
    docs = docs_with(black_myth={"name": "黑神话·悟空"})

    assert knowledge_bases(docs)[0].version == UNVERSIONED


def test_读一个不存在的库返回_None_而不是抛异常():
    """库不存在与配置读不了要分得开：前者是游戏选错了，后者是库自己的数据坏了。

    这里用 `None` 表达前者，让 HTTP 那层去决定报什么状态码——判断只在一处做。
    """
    assert knowledge_base(InMemoryDocStore(), GAME) is None


def test_一个字都没建的库读出空():
    assert knowledge_bases(InMemoryDocStore()) == ()
