"""主检索路：混合检索取候选、精排、按分数落差截断。

截断这一段全是纯函数，直接摆分断言——**它是这一层的核心**：取固定前 K 条是凑数，
凑进来的那几条会把上下文稀释掉，而稀释不报错，只让答案悄悄变差。
两个口径（绝对落差、相对落差）与上界各有一条边界用例。

取候选与精排那一段接内存假件：本层要验的是接线（两路是否同一次产出、精排吃的是
正文还是元数据、过滤条件有没有透传），不是语义相似度。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from ragamer.retrieval import (
    CANDIDATE_LIMIT,
    MAX_CHUNKS,
    MAX_PARENT_CHARS,
    aggregate_parents,
    cliff_cut,
    retrieve,
)
from ragamer.stores.base import UNVERSIONED, ChunkFilter, ChunkHit
from ragamer.stores.memory import InMemoryChunkStore
from ragamer.vectors.base import Embedding, ModelOutputError
from ragamer.vectors.fake import FakeEmbedder

from .conftest import RecordingChunkStore, ScriptedReranker, chunk_store, make_chunk

GAME = "black_myth"


def hit(chunk_id: int, score: float, **overrides: object) -> ChunkHit:
    """一条命中。`overrides` 透传给 `make_chunk`——聚合要按命中切片的 `doc_title`
    回查，命中那条与库里那条对不上号时，用例验的就不是真实链路了。"""
    return ChunkHit(chunk=make_chunk(chunk_id, **overrides), score=score)


def scores_of(hits: Sequence[ChunkHit]) -> list[float]:
    return [item.score for item in hits]


class ShortReranker:
    """不管候选几条，只回固定个数的分数。"""

    def __init__(self, count: int) -> None:
        self.count = count

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        return [0.5] * self.count


class SilentEmbedder:
    """一条向量都产不出的假件。"""

    def embed(self, texts: Sequence[str]) -> Embedding:
        return Embedding(dense=(), sparse=())


# --- 断崖截断 ---


def test_绝对落差大的地方截断():
    """相邻分差超过 0.3 就不再往下取——后面那几条是另一档的相关度。"""
    cut = cliff_cut([hit(1, 0.9), hit(2, 0.85), hit(3, 0.8), hit(4, 0.4), hit(5, 0.38)])

    assert scores_of(cut) == [0.9, 0.85, 0.8]


def test_相对落差过半也截断():
    """0.5 → 0.2 的绝对落差正好 0.3，卡在绝对口径的阈值上；但分数掉了一半以上。"""
    cut = cliff_cut([hit(1, 0.5), hit(2, 0.2)])

    assert scores_of(cut) == [0.5]


def test_阈值是大于而不是大于等于():
    """正好掉一半、正好差 0.29 都不算断崖。

    绝对口径那边不摆「正好 0.3」：0.8 - 0.5 在浮点里比 0.3 大一丝（0.30000000000000004），
    拿它当用例等于把浮点表示当判据。要紧的是口径是「大于」，不是边界上那一格。
    """
    assert scores_of(cliff_cut([hit(1, 0.4), hit(2, 0.2)])) == [0.4, 0.2]
    assert scores_of(cliff_cut([hit(1, 0.9), hit(2, 0.61)])) == [0.9, 0.61]


def test_分数为负时不拿它算相对落差():
    """相对落差要除以「前一条的分数」。前一条是 0 或负数时定义不了，
    只能退回绝对落差——否则要么除零，要么把负数除法算成一个假的断崖。"""
    assert scores_of(cliff_cut([hit(1, 0.0), hit(2, 0.0)])) == [0.0, 0.0]
    assert scores_of(cliff_cut([hit(1, 0.0), hit(2, -0.4)])) == [0.0]


def test_分数不在零点一到一的量纲里时留痕(caplog):
    """断崖的 0.3 / 0.5 是按精排分的 [0, 1] 标定的（真实精排 sigmoid 之后）。

    换上一个返回 logits 的精排，绝对落差几乎必然命中（下例 5.0 → 4.0 一上来就切）、
    相对落差则被除以「前一条分数」那一步整个关掉，两条都不报错——截断位置于是悄悄换了
    个依据。所以这里要留痕，断言的也正是这条痕。
    """
    with caplog.at_level(logging.WARNING):
        cut = cliff_cut([hit(1, 5.0), hit(2, 4.0)])

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("量纲" in message for message in warnings)
    assert len(cut) == 1


def test_上界是有上限的():
    """一条断崖都没有时由数量定，但绝不因为「候选多」就多给。"""
    cut = cliff_cut([hit(index, 0.9 - index * 0.01) for index in range(1, 16)])

    assert len(cut) == MAX_CHUNKS


def test_上界不超过候选数():
    """候选本来就少，不该凭空凑到上界。"""
    cut = cliff_cut([hit(1, 0.9), hit(2, 0.89), hit(3, 0.88)])

    assert scores_of(cut) == [0.9, 0.89, 0.88]


def test_没有候选时是空():
    assert cliff_cut([]) == ()


# --- 取候选与精排 ---


def test_检索把稠密与稀疏两路一起交给存储():
    """两路必须出自同一次向量化：分两次调，两批文本对不上号也不报错。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()

    retrieve(
        "二郎神怎么打",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({"正文1": 0.9}),
    )

    assert embedder.calls == [["二郎神怎么打"]]
    search = store.searches[0]
    assert search["sparse"] is not None
    assert search["dense"] == embedder.embed(["二郎神怎么打"]).dense[0]
    assert search["limit"] == CANDIDATE_LIMIT


def test_候选池比截断上界大():
    """池子不大于上界，上界就会把每条候选都保下来，断崖等于没生效。"""
    assert CANDIDATE_LIMIT > MAX_CHUNKS


def test_精排吃正文不吃附加文本():
    """`content_meta` 按设计不参与向量化，打分同理——表格的长文本列会把分数带偏。"""
    store = chunk_store(GAME, make_chunk(1, content_meta="| 说明 | 一长串表格里的说明 |"))
    reranker = ScriptedReranker({"正文1": 0.9})

    retrieve("二郎神", game_id=GAME, chunks=store, embedder=FakeEmbedder(), reranker=reranker)

    assert reranker.calls[0][1] == ["正文1"]


def test_精排按分数排出顺序():
    """存储回来的顺序由它自己定，交出去的顺序由精排定。"""
    store = chunk_store(GAME, make_chunk(1), make_chunk(2), make_chunk(3))

    found = retrieve(
        "二郎神",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.5, "正文2": 0.9, "正文3": 0.7}),
    )

    assert [item.chunk.chunk_id for item in found] == [2, 3, 1]


def test_同分时按切片序号定序():
    """同分的两条每次都要排出同一个顺序，否则截断位置会在它们之间随机挪。"""
    store = chunk_store(GAME, make_chunk(7), make_chunk(3))

    found = retrieve(
        "二郎神",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文7": 0.5, "正文3": 0.5}),
    )

    assert [item.chunk.chunk_id for item in found] == [3, 7]


def test_过滤条件透传给存储():
    """过滤是存储的事，检索这一层只负责把条件原样带下去。"""
    recording = RecordingChunkStore()
    recording.upsert(GAME, [make_chunk(1)])
    where = ChunkFilter(version="1.0")

    retrieve(
        "二郎神",
        game_id=GAME,
        chunks=recording,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
        where=where,
    )

    assert recording.searches[0]["where"] is where


def test_没有候选时不调精排():
    """精排是模型调用，空候选上白调一次。"""
    reranker = ScriptedReranker({})

    found = retrieve(
        "二郎神",
        game_id=GAME,
        chunks=InMemoryChunkStore(),
        embedder=FakeEmbedder(),
        reranker=reranker,
    )

    assert found == ()
    assert reranker.calls == []


def test_精排分数与候选条数对不上就报错():
    """按短的一边截齐会得到一个静默错位的排序——查不出、也不报错。"""
    store = chunk_store(GAME, make_chunk(1), make_chunk(2))

    with pytest.raises(ModelOutputError):
        retrieve(
            "二郎神",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ShortReranker(1),
        )


def test_向量化一条都没产出就报错():
    store = chunk_store(GAME, make_chunk(1))

    with pytest.raises(ModelOutputError):
        retrieve(
            "二郎神",
            game_id=GAME,
            chunks=store,
            embedder=SilentEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
        )


# --- 聚合父块 ---


def test_父块含该文档的全部切片且按源文档顺序():
    """命中一条细粒度切片，交给生成的是整页：同文档的兄弟切片按 `chunk_index` 升序拼齐。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="打法", chunk_index=0),
        make_chunk(2, content="掉落", chunk_index=1),
        make_chunk(3, content="获取方式", chunk_index=2),
    )

    blocks = aggregate_parents([hit(2, 0.9)], game_id=GAME, chunks=store)

    assert [block.doc_title for block in blocks] == ["二郎神"]
    assert [chunk.content for chunk in blocks[0].chunks] == ["打法", "掉落", "获取方式"]
    # 文档不长，整个父块就是这一篇；收敛只发生在超长文档上
    assert blocks[0].ancestor_path == ""


def test_不同文档各出一个父块按命中的先后():
    """父块的顺序就是命中的顺序（精排分降序），引用顺序跟着它走。"""
    store = chunk_store(GAME, make_chunk(1), make_chunk(2, doc_title="世界观"))

    blocks = aggregate_parents(
        [hit(2, 0.9, doc_title="世界观"), hit(1, 0.8)], game_id=GAME, chunks=store
    )

    assert [block.doc_title for block in blocks] == ["世界观", "二郎神"]


def test_同一个文档只回查一次():
    """同一文档命中多条时，回查不该按命中条数各来一遍。"""
    recording = RecordingChunkStore()
    recording.upsert(GAME, [make_chunk(1), make_chunk(2)])

    aggregate_parents([hit(1, 0.9), hit(2, 0.8)], game_id=GAME, chunks=recording)

    assert recording.fetched == ["二郎神"]


def test_回查带同一个版本过滤():
    """预置同一文档的两个版本切片，拼出来的父块不混版本（ADR-0004）。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="1.0 的正文", version="1.0", chunk_index=0),
        make_chunk(2, content="2.0 的正文", version="2.0", chunk_index=1),
    )

    blocks = aggregate_parents(
        [hit(1, 0.9)], game_id=GAME, chunks=store, where=ChunkFilter(version="1.0")
    )

    assert [chunk.content for chunk in blocks[0].chunks] == ["1.0 的正文"]


def test_未标注版本的兄弟切片一并拼进来():
    """回查的过滤条件与检索同一条：「该版本**或**未标注版本」（ADR-0004）。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="标了版本", version="1.0", chunk_index=0),
        make_chunk(2, content="没标版本", version=UNVERSIONED, chunk_index=1),
    )

    blocks = aggregate_parents(
        [hit(1, 0.9)], game_id=GAME, chunks=store, where=ChunkFilter(version="1.0")
    )

    assert [chunk.content for chunk in blocks[0].chunks] == ["标了版本", "没标版本"]


def test_问题与知识库都给不出版本时聚合也不过滤():
    """`version_filter` 在两处都判不出版本时有意收窄成不过滤，聚合跟随同一个口径——
    另立一套的话，同一个问题会在检索与聚合两步各按一个版本口径判一次。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="1.0 的正文", version="1.0", chunk_index=0),
        make_chunk(2, content="2.0 的正文", version="2.0", chunk_index=1),
    )

    blocks = aggregate_parents([hit(1, 0.9)], game_id=GAME, chunks=store)

    assert len(blocks[0].chunks) == 2


def test_超长文档收敛到命中切片所在的小节():
    """整页超过上限就不再整页喂，只留命中切片所在的那一节（§2.5 第三条约束）。"""
    filler = "长" * MAX_PARENT_CHARS
    store = chunk_store(
        GAME,
        make_chunk(1, content=filler, ancestor_path="二郎神 › 背景", chunk_index=0),
        make_chunk(2, content=filler, ancestor_path="二郎神 › 打法", chunk_index=1),
    )

    blocks = aggregate_parents([hit(2, 0.9)], game_id=GAME, chunks=store)

    assert blocks[0].ancestor_path == "二郎神 › 打法"
    assert [chunk.chunk_index for chunk in blocks[0].chunks] == [1]


def test_正好到上限的文档不算超长():
    """判定是「大于」：整页正好等于上限仍算一页，不因为卡在边界上就切走半篇。"""
    half = "长" * (MAX_PARENT_CHARS // 2)
    store = chunk_store(
        GAME,
        make_chunk(1, content=half, ancestor_path="二郎神 › 背景", chunk_index=0),
        make_chunk(2, content=half, ancestor_path="二郎神 › 打法", chunk_index=1),
    )

    blocks = aggregate_parents([hit(2, 0.9)], game_id=GAME, chunks=store)

    assert len(blocks[0].chunks) == 2


def test_重导之后父块跟着变不需要同步步骤():
    """父块是查出来的：覆盖同一批 `chunk_id` 之后它就变了，中间没有「同步父块」这一步。"""
    store = chunk_store(GAME, make_chunk(1, content="旧正文"))

    store.upsert(GAME, [make_chunk(1, content="新正文")])
    blocks = aggregate_parents([hit(1, 0.9)], game_id=GAME, chunks=store)

    assert [chunk.content for chunk in blocks[0].chunks] == ["新正文"]


def test_删库之后父块不留孤儿():
    """父块不占独立存储：删库删掉的就是它赖以存在的那些切片，没有第二份要清理。"""
    store = chunk_store(GAME, make_chunk(1))
    aggregate_parents([hit(1, 0.9)], game_id=GAME, chunks=store)

    store.drop(GAME)

    assert aggregate_parents([hit(1, 0.9)], game_id=GAME, chunks=store) == ()


def test_回查不到兄弟切片时不出一个空父块(caplog):
    """命中切片按 `doc_title` 回查一定查得到——它就是照这个条件检出来的。
    查不到说明索引与数据对不上；空父块交给生成等于一条空资料，宁可不出这个块。"""
    with caplog.at_level(logging.WARNING):
        blocks = aggregate_parents([hit(1, 0.9)], game_id=GAME, chunks=InMemoryChunkStore())

    assert blocks == ()
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("对不上" in message for message in warnings)
