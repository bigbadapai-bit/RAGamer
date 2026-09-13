"""答案缓存与热门问题：缓存挡在主检索路前，整条读取侧的内存版跑一遍。

六组用例：**命中与不命中**（第二次提问不再检索与生成）、**命中不降级**（引用、图片、
逐字流式一个不少）、**键**（游戏与版本进键、按改写后的问法归一）、**失效**（导入新资料
按游戏前缀批量删、过期自走）、**降级**（缓存不可用时作答照常）、**热门问题**。

模型接 `FakeLlm`：脚本排空后再被调用会当场炸，「第二次没有再生成」这条断言就落在它上面。
检索那一段接内存假件与确定性假件，一行云端代码都不碰。
"""

from __future__ import annotations

import logging

import pytest

from ragamer.answering import NOT_FOUND, Answerer
from ragamer.caching import (
    TTL_SECONDS,
    CachedAnswer,
    CachedAnswerer,
    CacheUnavailableError,
    InMemoryAnswerCache,
    cache_key,
    replay,
)
from ragamer.importing import Importer
from ragamer.llm import FakeLlm
from ragamer.sources import SourceDocument
from ragamer.stores.memory import InMemoryChunkStore
from ragamer.vectors.fake import FakeEmbedder, FakeReranker

from .conftest import chunk_store, make_chunk

GAME = "black_myth"
OTHER = "wuthering_waves"
QUESTION = "二郎神怎么打"
#: 一句话答案，编号指的是提示词里那批切片的编号。
REPLY = "先定身再贴身输出[1]，二阶段躲开红光。"
#: 两条句子的答案：缓存命中时要分片吐出来，用它才看得出分了片。
LONG_REPLY = "先定身再贴身输出[1]。二阶段躲开红光，等它收招再上。"

#: 一份能切出多片的资料。导入那条路上用它。
ARTICLE = "# 二郎神\n\n二郎神是隐藏 BOSS，需要三阶段打完。\n\n## 打法\n\n先定身，再贴身输出。\n"

#: 正文里带图片的切片：图片地址随答案一起交回，缓存里也得有。
WITH_IMAGE = "二郎神怎么打：先定身\n![打法](images/black_myth/boss.jpg)"


def reader(store, llm, *, cache=None, embedder=None, reranker=None) -> CachedAnswerer:
    """读取侧的内存版：缓存 + 主检索路，两截都换成假件。

    `embedder` / `reranker` 由调用方传进来，是为了能数「这一层有没有真的走检索」。
    """
    return CachedAnswerer(
        answers=Answerer(
            chunks=store,
            embedder=embedder if embedder is not None else FakeEmbedder(),
            reranker=reranker if reranker is not None else FakeReranker(),
            llm=llm,
        ),
        cache=cache if cache is not None else InMemoryAnswerCache(),
    )


def boss_chunks() -> InMemoryChunkStore:
    """本游戏的一批切片，外加另一款游戏的一条——跨游戏的键不该互相命中。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content=WITH_IMAGE, doc_title="二郎神", ancestor_path="二郎神 › 打法"),
        make_chunk(
            2, content="二郎神的获取方式", doc_title="二郎神", ancestor_path="二郎神 › 获取方式"
        ),
    )
    store.upsert(OTHER, [make_chunk(3, content=QUESTION)])
    return store


class BrokenCache:
    """一个连不上的缓存：每一次操作都抛「不可达」，与真实适配器同一个异常类型。"""

    def _fail(self) -> None:
        raise CacheUnavailableError("Redis", "redis.test:6379", 5.0, "连接被拒绝")

    def get(self, key: str) -> CachedAnswer | None:
        self._fail()

    def set(self, key: str, answer: CachedAnswer, *, ttl: int = TTL_SECONDS) -> None:
        self._fail()

    def invalidate(self, game_id: str) -> int:
        self._fail()

    def record_question(self, game_id: str, rewritten_query: str) -> None:
        self._fail()

    def top_questions(self, game_id: str, limit: int = 10) -> tuple[tuple[str, int], ...]:
        self._fail()


# --- 命中与不命中 ---


def test_同一个问题第二次提问不再走检索与生成():
    """脚本只排了一条：第二次真去生成的话，假件会当场炸。"""
    store = boss_chunks()
    embedder = FakeEmbedder()
    llm = FakeLlm(REPLY)
    cached = reader(store, llm, embedder=embedder)

    first = cached.answer(QUESTION, game_id=GAME, version="1.0")
    second = cached.answer(QUESTION, game_id=GAME, version="1.0")

    assert first == second
    assert len(llm.calls) == 1
    assert len(embedder.calls) == 1  # 检索那一路也没再走


def test_命中时不再碰切片存储():
    """把库清空再问一次：还能答出来，就说明答案来自缓存而不是检索。"""
    store = boss_chunks()
    cached = reader(store, FakeLlm(REPLY))
    first = cached.answer(QUESTION, game_id=GAME, version="1.0")

    store.drop(GAME)
    second = cached.answer(QUESTION, game_id=GAME, version="1.0")

    assert second == first


def test_一次提问只写一条缓存():
    cache = InMemoryAnswerCache()
    cached = reader(boss_chunks(), FakeLlm(REPLY), cache=cache)

    cached.answer(QUESTION, game_id=GAME, version="1.0")

    assert cache.get(cache_key(GAME, "1.0", QUESTION)) is not None


# --- 命中不降级：引用、图片、流式一个不少 ---


def test_命中时引用与图片一并回来():
    """只缓存文本的话，命中之后引用与图片就没了——那条路径会静默地比未命中时少东西。"""
    store = boss_chunks()
    cached = reader(store, FakeLlm(REPLY))
    first = cached.answer(QUESTION, game_id=GAME, version="1.0")
    store.drop(GAME)

    second = cached.answer(QUESTION, game_id=GAME, version="1.0")

    assert first.images == ("images/black_myth/boss.jpg",)
    assert second.citations == first.citations
    assert second.images == first.images
    assert [citation.label for citation in second.citations] == ["二郎神"]


def test_命中时仍然逐字流式():
    """缓存里的答案是一整段，一次吐出去就是「流式」在缓存路径上静默失效。
    **粒度与引用都要与未命中那条路一样**——命中反而是最常走的那条路。"""
    cached = reader(boss_chunks(), FakeLlm(LONG_REPLY, LONG_REPLY))
    missed = cached.stream(QUESTION, game_id=GAME, version="1.0")
    hit = cached.stream(QUESTION, game_id=GAME, version="1.0")

    assert list(hit.deltas) == list(missed.deltas)  # 假模型逐字，重放也逐字
    assert len(list(replay(LONG_REPLY))) == len(LONG_REPLY)
    assert hit.citations == missed.citations
    assert hit.images == missed.images


def test_未命中时边走边吐并把整段写回缓存():
    llm = FakeLlm(LONG_REPLY)
    cache = InMemoryAnswerCache()
    cached = reader(boss_chunks(), llm, cache=cache)

    pieces = list(cached.stream(QUESTION, game_id=GAME, version="1.0").deltas)

    assert "".join(pieces) == LONG_REPLY
    # 逐字：假模型默认一片一个字
    assert len(pieces) == len(LONG_REPLY)
    stored = cache.get(cache_key(GAME, "1.0", QUESTION))
    assert stored is not None and stored.text == LONG_REPLY


def test_流没吐完时不写缓存():
    """生成中途断开：半截答案进缓存比不缓存更糟——下次命中它，用户拿到一段断掉的话。"""
    cache = InMemoryAnswerCache()
    cached = reader(boss_chunks(), FakeLlm(LONG_REPLY), cache=cache)

    streamed = cached.stream(QUESTION, game_id=GAME, version="1.0")
    next(iter(streamed.deltas))

    assert cache.get(cache_key(GAME, "1.0", QUESTION)) is None


def test_没检索到内容的结果不入缓存():
    """「暂时没有」不该被钉成 7 天的「没有」：新资料进来它就该能答上了。"""
    cache = InMemoryAnswerCache()
    llm = FakeLlm()  # 一条脚本都没排：真被调用到会当场炸
    cached = reader(InMemoryChunkStore(), llm, cache=cache)

    answer = cached.answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.text == NOT_FOUND
    assert cache.get(cache_key(GAME, "1.0", QUESTION)) is None
    assert llm.calls == []


# --- 键 ---


def test_键含游戏与版本_换版本或换游戏都不命中():
    store = boss_chunks()
    store.upsert(GAME, [make_chunk(4, content=QUESTION, version="2.0")])
    llm = FakeLlm(REPLY, REPLY, REPLY)
    cached = reader(store, llm)

    cached.answer(QUESTION, game_id=GAME, version="1.0")
    cached.answer(QUESTION, game_id=GAME, version="2.0")
    cached.answer(QUESTION, game_id=OTHER, version="1.0")

    assert len(llm.calls) == 3


def test_版本没点名时按知识库的现行版本算键():
    """问题里没点名时，实际生效的是知识库标的那个——键得跟着它，否则换了现行版本
    仍会命中旧版本的答案。"""
    store = boss_chunks()
    store.upsert(GAME, [make_chunk(4, content=QUESTION, version="2.0")])
    llm = FakeLlm(REPLY, REPLY)
    cached = reader(store, llm)

    cached.answer(QUESTION, game_id=GAME, current_version="1.0")
    cached.answer(QUESTION, game_id=GAME, current_version="2.0")
    cached.answer(QUESTION, game_id=GAME, current_version="2.0")

    assert len(llm.calls) == 2


def test_改写后的问法算归一_换个问法仍命中同一条():
    """「那它怎么打」改写出来就是「二郎神怎么打」，精确匹配因此命得中（§4）。"""
    llm = FakeLlm(REPLY)
    cached = reader(boss_chunks(), llm)

    cached.answer(QUESTION, game_id=GAME, version="1.0", rewritten_query="二郎神怎么打")
    again = cached.answer(
        " 那它怎么打 ", game_id=GAME, version="1.0", rewritten_query="二郎神怎么打"
    )

    assert again.text == REPLY
    assert len(llm.calls) == 1


def test_改写为空时按原问题算键():
    """改写那一步降级时按原问题算键。落到空串上就会让所有降级提问共用一个键，
    一个问题的答案发给了另一个问题——而且不报错。"""
    llm = FakeLlm(REPLY, REPLY)
    cached = reader(boss_chunks(), llm)

    cached.answer(QUESTION, game_id=GAME, version="1.0")
    cached.answer("二郎神在哪", game_id=GAME, version="1.0")

    assert len(llm.calls) == 2
    assert cache_key(GAME, "1.0", "") != cache_key(GAME, "1.0", QUESTION)


def test_空问题既不入缓存也不报缓存相关的错():
    llm = FakeLlm()
    cached = reader(boss_chunks(), llm)

    with pytest.raises(ValueError):
        cached.answer("   ", game_id=GAME)


# --- 失效 ---


def test_导入新资料后该游戏的缓存按前缀批量删():
    """语料变了，基于旧语料的答案不该再命中；另一款游戏的缓存不该被牵连。"""
    cache = InMemoryAnswerCache()
    cached = reader(boss_chunks(), FakeLlm(REPLY, REPLY), cache=cache)
    cached.answer(QUESTION, game_id=GAME, version="1.0")
    cached.answer(QUESTION, game_id=OTHER, version="1.0")

    Importer(chunks=InMemoryChunkStore(), embedder=FakeEmbedder(), cache=cache).batch(
        [SourceDocument(filename="二郎神.md", data=ARTICLE.encode("utf-8"))], game_id=GAME
    )

    assert cache.get(cache_key(GAME, "1.0", QUESTION)) is None
    assert cache.get(cache_key(OTHER, "1.0", QUESTION)) is not None


def test_过期之后不再命中():
    """长过期是兜底：没人来删的条目自己走掉。"""
    clock = [0.0]
    cache = InMemoryAnswerCache(now=lambda: clock[0])
    llm = FakeLlm(REPLY, REPLY)
    cached = reader(boss_chunks(), llm, cache=cache)
    cached.answer(QUESTION, game_id=GAME, version="1.0")

    clock[0] += TTL_SECONDS
    cached.answer(QUESTION, game_id=GAME, version="1.0")

    assert len(llm.calls) == 2


# --- 降级 ---


def test_缓存连不上时作答照常(caplog):
    """缓存是加速器，不是链路的一环：读、写、计数哪一步失败都只留一条日志。"""
    llm = FakeLlm(REPLY, REPLY)
    cached = reader(boss_chunks(), llm, cache=BrokenCache())

    with caplog.at_level(logging.WARNING):
        answer = cached.answer(QUESTION, game_id=GAME, version="1.0")
        streamed = cached.stream(QUESTION, game_id=GAME, version="1.0")

    assert answer.text == REPLY
    assert "".join(streamed.deltas) == REPLY
    # 两次都真的走了生成——缓存一次也没用上
    assert len(llm.calls) == 2
    assert "不可达" in caplog.text


def test_缓存连不上时热门问题返回空而不是报错():
    cached = reader(boss_chunks(), FakeLlm(REPLY), cache=BrokenCache())

    assert cached.top_questions(GAME) == ()


# --- 热门问题 ---


def test_提问频次按次数排序_命中与否都计数():
    cached = reader(boss_chunks(), FakeLlm(REPLY, REPLY, REPLY))

    for _ in range(3):
        cached.answer("二郎神怎么打", game_id=GAME, version="1.0")
    for _ in range(2):
        cached.answer("二郎神在哪", game_id=GAME, version="1.0")
    cached.answer("二郎神掉什么", game_id=GAME, version="1.0")

    assert cached.top_questions(GAME) == (
        ("二郎神怎么打", 3),
        ("二郎神在哪", 2),
        ("二郎神掉什么", 1),
    )


def test_热门问题只数本游戏的():
    cached = reader(boss_chunks(), FakeLlm(REPLY))

    cached.answer(QUESTION, game_id=GAME, version="1.0")
    cached.top_questions(OTHER)

    assert cached.top_questions(OTHER) == ()


# --- 伪装流式的粒度 ---


def test_重放逐字_拼回去等于原文():
    """粒度与未命中那条路取齐（那边吐的是模型的 delta），不是按短句成片地给。"""
    text = "先定身[1]。二阶段躲开红光，等它收招再上[2]。"
    pieces = list(replay(text))

    assert "".join(pieces) == text
    assert all(len(piece) == 1 for piece in pieces)
    assert len(pieces) == len(text)


def test_空答案不吐任何字():
    assert list(replay("")) == []
