"""多路召回、融合、精排与按分数落差截断。

截断这一段全是纯函数，直接摆分断言——**它是这一层的核心**：取固定前 K 条是凑数，
凑进来的那几条会把上下文稀释掉，而稀释不报错，只让答案悄悄变差。
两个口径（绝对落差、相对落差）与上界各有一条边界用例。

取候选、融合与精排那一段接内存假件：本层要验的是接线（各路是否同一次向量化、精排吃的是
正文还是元数据、过滤条件有没有透传到该到的那一路、没接上的路有没有被跳过），
不是语义相似度。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pytest

from ragamer import retrieval
from ragamer.llm import FakeLlm, LlmTimeout
from ragamer.retrieval import (
    CANDIDATE_LIMIT,
    MAX_CHUNKS,
    MAX_PARENT_CHARS,
    RRF_K,
    WEB_RESULTS,
    aggregate_parents,
    cliff_cut,
    retrieve,
    rrf,
)
from ragamer.routing import RecallPath, Route
from ragamer.stores.base import (
    UNVERSIONED,
    ChunkFilter,
    ChunkHit,
    StoreError,
    StoreUnavailableError,
)
from ragamer.stores.memory import InMemoryChunkStore
from ragamer.tagging import ContentNature, SubjectType
from ragamer.vectors.base import Embedding, ModelOutputError
from ragamer.vectors.fake import FakeEmbedder
from ragamer.websearch import (
    FakeWebSearch,
    WebResult,
    WebSearchRejected,
    WebSearchUnavailable,
)

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


class BrokenStore(InMemoryChunkStore):
    """前几次检索失败、之后照常的切片存储。

    用来摆出「某一路失败、另一路照常」的局面。路是按路由表里的先后依次检索的，
    所以「第几次」就对得上「哪一路」——存储那边看不见是第几路在调它，
    而那正是这一层要证的：它不挑是哪一路挂的。
    """

    def __init__(self, *, failing_times: int) -> None:
        super().__init__()
        self._failing = failing_times
        self.attempts = 0

    def search(
        self,
        game_id: str,
        *,
        dense: Sequence[float],
        sparse: Mapping[int, float] | None = None,
        where: ChunkFilter | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        self.attempts += 1
        if self.attempts <= self._failing:
            raise StoreUnavailableError("Milvus", "milvus.test:19530", 2.5, "连接被拒绝")
        return super().search(game_id, dense=dense, sparse=sparse, where=where, limit=limit)


def broken_store(*, failing_times: int) -> BrokenStore:
    store = BrokenStore(failing_times=failing_times)
    store.upsert(GAME, [make_chunk(1)])
    return store


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

    assert [item.chunk.chunk_id for item in found.hits] == [2, 3, 1]


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

    assert [item.chunk.chunk_id for item in found.hits] == [3, 7]


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

    assert found.hits == ()
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


# --- 多路召回 ---


def test_不给路由时只走主检索路():
    """选路是 `ragamer.routing` 的事：不给组合就只走主检索，这一层不替调用方决定。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])

    retrieve(
        "二郎神",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
    )

    assert len(store.searches) == 1
    assert store.searches[0]["sparse"] is not None


def test_各路共用同一次向量化():
    """两条路检索的必须是同一个问题。各调一次向量化，两路问的就不是一回事了。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()

    retrieve(
        "二郎神血量多少",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.MAIN, RecallPath.METADATA)),
    )

    assert len(store.searches) == 2
    assert embedder.calls == [["二郎神血量多少"]]


def test_元数据过滤路是单路检索():
    """它要的是「按标签直接取候选」：再叠一路稀疏，会把标签之外的近义内容也捞进来，
    正好把这次过滤抵消掉。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])

    retrieve(
        "二郎神血量多少",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.MAIN, RecallPath.METADATA)),
    )

    main, metadata = store.searches
    assert main["sparse"] is not None
    assert metadata["sparse"] is None
    # 两路取的是同一批数据上的同一个问题，稠密那一路因此完全一样
    assert metadata["dense"] == main["dense"]


def test_元数据过滤路按主体类型与内容性质取回候选():
    """第二路召回的立身之本：按标签直接取，不靠向量相似度。

    过滤是存储的事，这里断言的是**条件原样透传**——透传错了的表现是候选里混进别的
    主体类型或别的性质，而它不报错，只是答案的依据悄悄换了。
    """
    store = RecordingChunkStore()
    store.upsert(
        GAME,
        [
            make_chunk(1, subject_type=("character",), content_nature=("stats",)),
            make_chunk(2, subject_type=("item",), content_nature=("stats",)),
            make_chunk(3, subject_type=("character",), content_nature=("intro",)),
        ],
    )

    found = retrieve(
        "二郎神血量多少",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9, "正文2": 0.5, "正文3": 0.5}),
        route=Route(
            (RecallPath.METADATA,),
            content_natures=(ContentNature.STATS,),
            subject_types=(SubjectType.CHARACTER,),
        ),
    )

    where = store.searches[0]["where"]
    assert where.subject_types == (SubjectType.CHARACTER,)
    assert where.content_natures == (ContentNature.STATS,)
    # 三条都在库里，过滤之后只剩一条——透传对了才有的结果
    assert [item.chunk.chunk_id for item in found.hits] == [1]


def test_元数据过滤路的过滤条件不影响主检索路():
    """两路各自过滤，条件混用会让主检索路也少掉一批候选，而它不报错。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1, content_nature=("stats",))])

    retrieve(
        "二郎神血量多少",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
        where=ChunkFilter(version="1.0"),
        route=Route((RecallPath.MAIN, RecallPath.METADATA), content_natures=(ContentNature.STATS,)),
    )

    main, metadata = store.searches
    assert main["where"].version == "1.0"
    assert main["where"].content_natures == ()
    assert metadata["where"].version == "1.0"
    assert metadata["where"].content_natures == (ContentNature.STATS,)


def test_元数据过滤路也带上版本条件():
    """这一路另立一套版本口径，等于把版本判错两次——而错的那次是静默的（ADR-0004）。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])

    retrieve(
        "二郎神血量多少",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
        where=ChunkFilter(version="1.0"),
        route=Route((RecallPath.METADATA,), content_natures=(ContentNature.STATS,)),
    )

    assert store.searches[0]["where"].version == "1.0"


def test_两路都取到的切片只精排一次():
    """同一个切片在多路里出现只算一条：按两条送进精排，上下文里就多一份重复内容。"""
    store = chunk_store(GAME, make_chunk(1))
    reranker = ScriptedReranker({"正文1": 0.9})

    found = retrieve(
        "二郎神血量多少",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=reranker,
        route=Route((RecallPath.MAIN, RecallPath.METADATA), content_natures=()),
    )

    assert reranker.calls[0][1] == ["正文1"]
    assert [item.chunk.chunk_id for item in found.hits] == [1]


def test_多路融合后的最终引用顺序():
    """多路输入下交出去的那一批：合成一份引用，**顺序仍由精排说了算**。

    三路一起跑。主检索路与多查询改写路问的是同一个问题、带着同一套过滤条件，取回的是
    同一批候选（只有问法不同），元数据路取回的是另一批（它把性质换成路由给的那个）。
    这一条因此同时钉住三件事：

    - **去重**：2 号在两条路里都出现，精排只该看到它一次——重复的那份会在上下文里变成
      两段一模一样的内容；
    - **互补**：1 号只有元数据路取得到，少了那一路它就进不了引用；
    - **顺序**：融合分高的那条（两路都取到的 2 号）并不是交出去的第一条。融合分出去
      之前就被精排整个换掉了，最终顺序只认精排分。
    """
    store = chunk_store(
        GAME,
        make_chunk(1, content_nature=("stats",)),
        make_chunk(2, content_nature=("intro",)),
    )
    where = ChunkFilter(content_natures=(ContentNature.INTRO,))
    # 精排分与融合分**反着排**：融合说 2 号在前，精排说 1 号在前。落差要落在断崖以内
    # （≤ 0.3 绝对、≤ 0.5 相对），否则后面那条会被切掉——断崖的边界在别处验
    reranker = ScriptedReranker({"正文1": 0.8, "正文2": 0.6})

    def recalled(*paths: RecallPath) -> tuple[list[int], list[str]]:
        found = retrieve(
            "二郎神血量多少",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=reranker,
            where=where,
            route=Route(paths, content_natures=(ContentNature.STATS,)),
            llm=FakeLlm({"queries": ["二郎神还有多少血"]}),
        )
        return [item.chunk.chunk_id for item in found.hits], reranker.calls[-1][1]

    # 单跑：两条路各取得到一条，谁也替不了谁
    assert recalled(RecallPath.MAIN)[0] == [2]
    assert recalled(RecallPath.METADATA)[0] == [1]

    order, reranked = recalled(RecallPath.MAIN, RecallPath.MULTI_QUERY, RecallPath.METADATA)

    # 2 号被两条路取到，进精排的却只有一条
    assert sorted(reranked) == ["正文1", "正文2"]
    # 顺序是精排给的，不是融合分的
    assert order == [1, 2]


def test_没配依赖的路跳过并留痕(caplog):
    """静默跳过会让「这条路这次跑不了」与「路由表配错了」在日志里长得一模一样，
    而两者的处理方式相反（去补配置 / 现在去改路由表）。"""
    store = chunk_store(GAME, make_chunk(1))

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "二郎神怎么打",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            # 没给语言模型：多查询改写与 HyDE 这两路这次跑不了
            route=Route((RecallPath.MAIN, RecallPath.HYDE)),
        )

    assert [item.chunk.chunk_id for item in found.hits] == [1]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("hyde" in message for message in warnings)


def test_还没接上的路跳过并留痕(caplog, monkeypatch):
    """六条现在全接上了，这条拦截拦的是**新加的那条**：往词表里添一条路、却忘了在
    检索层实现它，这时该跳过并说清楚，而不是落进默认那一支偷偷用原问检索一遍。"""
    monkeypatch.setattr(retrieval, "WIRED_PATHS", frozenset({RecallPath.MAIN}))
    store = chunk_store(GAME, make_chunk(1))

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "二郎神怎么打",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            route=Route((RecallPath.MAIN, RecallPath.HYDE)),
            llm=FakeLlm({"queries": ["二郎神 打法"]}),
        )

    assert [item.chunk.chunk_id for item in found.hits] == [1]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("还没接上" in message for message in warnings)


def test_选中的路一条都跑不了时退回主检索(caplog):
    """真按空组合跑，这一类问题会一条候选都取不到，对外只说一句「知识库里没有找到
    相关资料」——把一次配置事故说成了语料问题。"""
    store = chunk_store(GAME, make_chunk(1))

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "寒江雪的属性",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            # 只选了 HyDE 却没给语言模型：这一条路跑不了，退回主检索
            route=Route((RecallPath.HYDE,)),
        )

    assert [item.chunk.chunk_id for item in found.hits] == [1]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("退回主检索" in message for message in warnings)


# --- 多查询改写路 ---


def test_多查询改写把扩出来的每一条各检索一遍():
    """这一路的意义就在「各检索一遍」：合并成一次检索，扩出来的说法就白扩了。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()

    found = retrieve(
        "二郎神怎么打",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.MULTI_QUERY,)),
        llm=FakeLlm({"queries": ["二郎神 打法", "二郎神 怎么打"]}),
    )

    assert embedder.calls == [["二郎神 打法", "二郎神 怎么打"]]
    assert len(store.searches) == 2
    assert [item.chunk.chunk_id for item in found.hits] == [1]


def test_多查询改写只检索扩出来的问法():
    """**原问由主检索路负责**——默认表里这两路总是成对出现，扩写那边也照这个前提去重
    （与原问同形的会被丢掉）。所以这里断言的是：它自己不再查一遍原问。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()

    retrieve(
        "二郎神怎么打",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.MULTI_QUERY,)),
        llm=FakeLlm({"queries": ["二郎神 打法"]}),
    )

    assert embedder.calls == [["二郎神 打法"]]


def test_主检索与多查询改写之间也去重():
    """模型偶尔会把原问原样吐回来。去重之后它不会变成两次一模一样的检索——
    那不只是白跑一趟，还会让原问在 RRF 里投出两票。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()

    retrieve(
        "二郎神怎么打",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.MAIN, RecallPath.MULTI_QUERY)),
        llm=FakeLlm({"queries": ["二郎神怎么打", "二郎神 打法"]}),
    )

    assert embedder.calls == [["二郎神怎么打", "二郎神 打法"]]
    assert len(store.searches) == 2


# --- HyDE 路 ---


def test_HyDE拿假想答案去检索():
    """这一路的全部要点：检索用的不是原问，是模型写的那段假想资料。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()
    written = "二郎神是隐藏 BOSS，血量 8000，二阶段会分身。"

    retrieve(
        "那个很难的 BOSS 怎么过",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.HYDE,)),
        llm=FakeLlm(written),
    )

    assert embedder.calls == [[written]]


def test_HyDE的假想答案是模型编的不进交给生成的内容():
    """把它当资料交出去，就是让模型照着自己编的东西回答——而且看起来与真答案一样。
    所以进精排、进上下文的只能是检索回来的切片。"""
    store = chunk_store(GAME, make_chunk(1))
    reranker = ScriptedReranker({"正文1": 0.9})

    found = retrieve(
        "那个很难的 BOSS 怎么过",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=reranker,
        route=Route((RecallPath.HYDE,)),
        llm=FakeLlm("二郎神是隐藏 BOSS，血量 8000。"),
    )

    assert reranker.calls[0][1] == ["正文1"]
    assert [item.chunk.chunk_id for item in found.hits] == [1]


def test_假想答案一个字都没写出来时这一路不检索():
    """空串拿去向量化会查出任意一批切片，与空问题的毛病是同一个。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    embedder = FakeEmbedder()

    found = retrieve(
        "那个很难的 BOSS 怎么过",
        game_id=GAME,
        chunks=store,
        embedder=embedder,
        reranker=ScriptedReranker({}),
        route=Route((RecallPath.HYDE,)),
        llm=FakeLlm("   "),
    )

    assert embedder.calls == []
    assert store.searches == []
    assert found.hits == ()


# --- 结构化表格路 ---


def test_表格路只取表格切片():
    """数值类问题的答案常在表里，而表格正文是排版过的行（`| 属性 | 值 |`），
    与问句的字面重合度低，靠向量相似度排不上来——所以要单独去取它。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1), make_chunk(2, chunk_type="table")])

    found = retrieve(
        "寒江雪的属性",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文2": 0.9}),
        route=Route((RecallPath.TABLE,)),
    )

    assert store.searches[0]["where"].chunk_type == "table"
    assert [item.chunk.chunk_id for item in found.hits] == [2]


def test_表格路也带上版本条件():
    """版本那一条沿用检索的：这一路另立一套口径等于把版本判错两次（ADR-0004）。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1, chunk_type="table")])

    retrieve(
        "寒江雪的属性",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
        where=ChunkFilter(version="1.0"),
        route=Route((RecallPath.TABLE,)),
    )

    assert store.searches[0]["where"].version == "1.0"


# --- 联网兜底路 ---


def web(*results: WebResult) -> FakeWebSearch:
    """**一次**检索的脚本：这次搜回来这几条。要排「这一路失败」用 `FakeWebSearch(异常)`。"""
    return FakeWebSearch(list(results))


def test_联网那批单独给不进融合也不精排():
    """它不是语料里的切片：没有父块可回查，也不该被当成语料参与精排与断崖。
    合成的「切片」硬塞进候选池，只会在那两处各留一个特例。"""
    store = RecordingChunkStore()
    store.upsert(GAME, [make_chunk(1)])
    reranker = ScriptedReranker({"正文1": 0.9})
    result = WebResult("1.1 版本更新公告", "https://example.com/p", "金箍棒改了。", "2026-01-01")

    found = retrieve(
        "这版本改了什么",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=reranker,
        route=Route((RecallPath.MAIN, RecallPath.WEB)),
        search=web(result),
    )

    assert [item.chunk.chunk_id for item in found.hits] == [1]
    assert found.web == (result,)
    assert reranker.calls[0][1] == ["正文1"]  # 网络那批没进精排


def test_联网按原问去搜():
    """扩展那两路是给本地检索扩的；联网这一路问的就是用户那一句——
    拿假想答案去搜外面，搜回来的东西与问题隔了一层。"""
    store = chunk_store(GAME, make_chunk(1))
    search = web(WebResult("公告", "https://example.com/p", "正文"))

    retrieve(
        "这版本改了什么",
        game_id=GAME,
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({"正文1": 0.9}),
        route=Route((RecallPath.WEB,)),
        search=search,
    )

    assert search.calls == [("这版本改了什么", WEB_RESULTS)]


def test_只有网络来源时也算检索到了内容():
    """本地一条都没有、网上有：这一路的意义就在这里，别按「没找到」处理。"""
    found = retrieve(
        "这版本改了什么",
        game_id=GAME,
        chunks=InMemoryChunkStore(),
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({}),
        route=Route((RecallPath.WEB,)),
        search=web(WebResult("公告", "https://example.com/p", "正文")),
    )

    assert found.hits == ()
    assert bool(found)


# --- 任一路失败不拖垮整体 ---


def test_多查询改写失败不拖垮主检索(caplog):
    """扩展那一路挂了是它自己的事：已经取回来的候选照常交出去。"""
    store = chunk_store(GAME, make_chunk(1))

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "二郎神怎么打",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            route=Route((RecallPath.MAIN, RecallPath.MULTI_QUERY)),
            llm=FakeLlm(LlmTimeout("模型超时")),
        )

    assert [item.chunk.chunk_id for item in found.hits] == [1]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("multi_query" in message for message in warnings)


def test_联网失败不拖垮本地那几路(caplog):
    """兜底那一路连不上，不该让本地语料的结果一起没了。"""
    store = chunk_store(GAME, make_chunk(1))

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "这版本改了什么",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            route=Route((RecallPath.MAIN, RecallPath.WEB)),
            search=FakeWebSearch(WebSearchUnavailable("连不上")),
        )

    assert [item.chunk.chunk_id for item in found.hits] == [1]
    assert found.web == ()
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("web" in message for message in warnings)


def test_密钥或余额被拒按_ERROR_留痕(caplog):
    """那之后每一次提问都会栽在这一路，只留一条 WARNING 会让人以为是偶发。"""
    store = chunk_store(GAME, make_chunk(1))

    with caplog.at_level(logging.ERROR, logger="ragamer.retrieval"):
        retrieve(
            "这版本改了什么",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            route=Route((RecallPath.MAIN, RecallPath.WEB)),
            search=FakeWebSearch(WebSearchRejected("HTTP 401")),
        )

    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_所有选中的路都失败时把失败抛出去():
    """一条路都没跑成不是「某一路的问题」：报成「知识库里没有找到相关资料」
    等于把一次故障说成了语料问题（与空候选是两回事）。"""
    store = broken_store(failing_times=2)

    with pytest.raises(StoreError):
        retrieve(
            "二郎神怎么打",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({}),
            route=Route((RecallPath.MAIN, RecallPath.METADATA)),
        )


def test_有一路跑成了就不抛(caplog):
    """主检索挂了但元数据路成了：有东西可用就继续答，失败的那一路留在日志里。"""
    store = broken_store(failing_times=1)

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "二郎神怎么打",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({"正文1": 0.9}),
            route=Route((RecallPath.MAIN, RecallPath.METADATA)),
        )

    assert [item.chunk.chunk_id for item in found.hits] == [1]


def test_有一路查空另一路失败时也抛(caplog):
    """判据是**最终有没有东西交得出去**，不是「是不是每一路都失败了」。

    元数据路跑成了但查空、主检索那一路失败：结局同样是「没东西可答，而且有故障」，
    报成「知识库里没有找到相关资料」就把一次故障说成了语料问题。
    """
    store = broken_store(failing_times=1)
    store.drop(GAME)  # 元数据路查得到、但库里什么都没有

    with pytest.raises(StoreError):
        retrieve(
            "二郎神怎么打",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({}),
            route=Route((RecallPath.MAIN, RecallPath.METADATA)),
        )


def test_没有失败时查空就是查空():
    """一路都没出错、就是没查到——这时「没找到资料」是实话，不该报成故障。"""
    found = retrieve(
        "二郎神怎么打",
        game_id=GAME,
        chunks=InMemoryChunkStore(),
        embedder=FakeEmbedder(),
        reranker=ScriptedReranker({}),
        route=Route((RecallPath.MAIN, RecallPath.METADATA)),
    )

    assert not found


def test_本地失败但联网搜到了就不抛(caplog):
    """有东西交得出去时，失败的那一路只留在日志里——这一轮照样答得出。"""
    store = broken_store(failing_times=1)

    with caplog.at_level(logging.WARNING, logger="ragamer.retrieval"):
        found = retrieve(
            "这版本改了什么",
            game_id=GAME,
            chunks=store,
            embedder=FakeEmbedder(),
            reranker=ScriptedReranker({}),
            route=Route((RecallPath.MAIN, RecallPath.WEB)),
            search=web(WebResult("公告", "https://example.com/p", "改了")),
        )

    assert found.hits == ()
    assert len(found.web) == 1


# --- RRF 融合 ---


def test_融合只看名次不看分数():
    """两路的分数是两套量纲，放在一起比大小等于让量纲决定谁进上下文。

    融合分整个由名次算出来：把分数换成一万倍，排出来的顺序一模一样。
    """
    assert [item.chunk.chunk_id for item in rrf([[hit(1, 0.9), hit(2, 0.8)]])] == [1, 2]
    assert [item.chunk.chunk_id for item in rrf([[hit(1, 900.0), hit(2, 800.0)]])] == [1, 2]


def test_多路都取到的切片排在只被一路取到的前面():
    """两路都排得上名次，说明两边都认为它相关——这条信号只有融合看得到，
    单看哪一路的分数都看不出来。"""
    fused = rrf([[hit(1, 0.9), hit(2, 0.1)], [hit(2, 0.1)]])

    assert [item.chunk.chunk_id for item in fused] == [2, 1]


def test_融合的分数就是_k_加名次的倒数和():
    """逐点是 `1 / (k + 名次)`，`k` 默认 60（§3.3）。它调的是「头部名次值多少」，
    改它要先有评测集。"""
    fused = rrf([[hit(1, 0.9), hit(2, 0.8)], [hit(2, 0.7)]])

    scores = {item.chunk.chunk_id: item.score for item in fused}
    assert scores[1] == pytest.approx(1 / (RRF_K + 1))
    assert scores[2] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))


def test_融合常数默认是六十且可以改写():
    """常数不写死在公式里：默认 60，换一个值就把名次的权重整个换掉。

    它得是个能改的旋钮——评测阶段要扫的就是它（§11：没有评测集时不该动它，
    有了评测集就得动得了）。
    """
    assert RRF_K == 60
    assert rrf([[hit(1, 0.9)]])[0].score == pytest.approx(1 / (RRF_K + 1))
    assert rrf([[hit(1, 0.9)]], k=1)[0].score == pytest.approx(1 / 2)


def test_融合常数越小头部名次的优势越大():
    """同一条名次差，`k` 越小分差越大（坑 #11）。这条钉的是那个参数的**作用**，
    不只是它能被传进去。"""
    big = scores_of(rrf([[hit(1, 0.9), hit(2, 0.8)]], k=100))
    small = scores_of(rrf([[hit(1, 0.9), hit(2, 0.8)]], k=1))

    assert small[0] - small[1] > big[0] - big[1]


def test_融合常数给成负数当场报错():
    """负的 k 会让 `k + 名次` 落到零或负数上：除以零，或者算出一个负的融合分把名次
    倒过来。与其在检索中途炸一个看不懂的除零，不如在入口处说清楚是哪个参数不对。"""
    with pytest.raises(ValueError, match="不能是负数"):
        rrf([[hit(1, 0.9)]], k=-1)


def test_融合之后同分按切片序号定序():
    """名次剖面一样的两条会被融合成同一个分数：顺序不定，截断位置就会在它们之间挪。"""
    fused = rrf([[hit(1, 0.9), hit(2, 0.8)], [hit(2, 0.9), hit(1, 0.8)]])

    assert [item.chunk.chunk_id for item in fused] == [1, 2]


def test_同一片在两路里各投一票融合分相加():
    """同一片出现在两路里只出一条，但两路的名次都算数——一票当两票，正是融合在做的
    那件事。只留一条是另一件事：两条会在上下文里变成两份重复内容。"""
    fused = rrf([[hit(1, 0.9)], [hit(1, 0.8)]])

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 1))


def test_多路都取到的切片保留第一次出现的那份():
    """同一个切片序号在两路里带着不同的主体信息回来时，留**第一次**见到的那份（坑 #10）。

    两路带回来的本该是同一份切片数据，留哪份都一样——但两路带回来的不是同一份时
    （索引与数据对不上、两路过滤条件不同），留哪份就有区别了，而这件事不报错。
    取第一次见到的，是因为「哪一路先见到它」在调试时是个有用的信号。
    """
    first = hit(1, 0.9)
    later = hit(1, 0.8, subject_name="灌江口二郎", ancestor_path="二郎神 › 其他")

    fused = rrf([[first], [later]])

    assert len(fused) == 1
    assert fused[0].chunk.subject_name == first.chunk.subject_name
    assert fused[0].chunk.ancestor_path == first.chunk.ancestor_path


def test_只有一路时融合不扰动原有顺序():
    """一路进来的顺序就是它自己的名次顺序，融合原样交出去。

    融合是给多路用的；只有一路时它若按别的东西重排一遍（这里两条的原始分同为 0.5），
    「单路」与「多路」两条链路的候选顺序就对不上了，而截断位置跟着走。
    """
    one = [hit(3, 0.9), hit(1, 0.5), hit(2, 0.5)]

    assert [item.chunk.chunk_id for item in rrf([one])] == [3, 1, 2]


def test_没有候选时融合出空():
    assert rrf([]) == []
    assert rrf([[], []]) == []


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


def test_收敛之后仍然超长时只留命中那一条(caplog):
    """整篇只有一节、没有更细的粒度可收敛时也不能把整页塞进去——上下文预算是硬约束，
    宁可不带上下文。这种情况要留痕：它多半说明切分没切出结构，是数据侧该修的事。"""
    filler = "长" * MAX_PARENT_CHARS
    store = chunk_store(
        GAME,
        make_chunk(1, content=f"{filler}甲", ancestor_path="二郎神", chunk_index=0),
        make_chunk(2, content=f"{filler}乙", ancestor_path="二郎神", chunk_index=1),
    )

    with caplog.at_level(logging.WARNING):
        blocks = aggregate_parents(
            [hit(2, 0.9, ancestor_path="二郎神")], game_id=GAME, chunks=store
        )

    assert [chunk.chunk_id for chunk in blocks[0].chunks] == [2]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("没有更细的粒度" in message for message in warnings)


def test_没有祖先标题路径的超长文档也只留命中那一条():
    """扁平文档压根没有小节：路径为空串时同样退到只剩命中那一条，不是退回整篇。"""
    filler = "长" * MAX_PARENT_CHARS
    store = chunk_store(
        GAME,
        make_chunk(1, content=filler, ancestor_path="", chunk_index=0),
        make_chunk(2, content=filler, ancestor_path="", chunk_index=1),
    )

    blocks = aggregate_parents([hit(2, 0.9, ancestor_path="")], game_id=GAME, chunks=store)

    assert [chunk.chunk_id for chunk in blocks[0].chunks] == [2]


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
