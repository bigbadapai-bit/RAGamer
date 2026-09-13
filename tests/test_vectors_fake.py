"""确定性假件：不加载真实模型也能验证接线。

断言只看形状、顺序与确定性——假件的取值跟语义无关，拿它比相似度是比不出东西的。
真实模型那一半在 `tests/test_vectors_bge.py`（注入加载器）与
`tests/test_vectors_integration.py`（真跑）里。
"""

from __future__ import annotations

import math

import pytest

from ragamer.vectors import (
    DENSE_DIM,
    Embedder,
    Embedding,
    FakeEmbedder,
    FakeReranker,
    ModelOutputError,
    Reranker,
)

from .conftest import make_chunk


def test_两个假件满足各自的协议():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeReranker(), Reranker)


def test_一批文本一次调用出两路():
    embedder = FakeEmbedder()

    embedding = embedder.embed(["二郎神怎么打", "寒江雪的属性"])

    # 调用只发生一次，两路条数都与文本一一对应
    assert embedder.calls == [["二郎神怎么打", "寒江雪的属性"]]
    assert len(embedding) == 2
    assert len(embedding.dense) == len(embedding.sparse) == 2


def test_稠密向量已归一化():
    """归一化是接口的后置条件：配 Milvus 的 IP 度量等价余弦（坑 #2）。"""
    vector = FakeEmbedder().embed(["二郎神怎么打"]).dense[0]

    assert len(vector) == DENSE_DIM
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_同一段文本永远得到同一个向量():
    """导入时算过的向量得与提问时算出来的一致，否则检索测试自己就会抖。"""
    first = FakeEmbedder().embed(["二郎神怎么打"])
    second = FakeEmbedder().embed(["二郎神怎么打"])

    assert first.dense == second.dense
    assert first.sparse == second.sparse


def test_不同的文本得到不同的向量():
    embedding = FakeEmbedder().embed(["二郎神怎么打", "寒江雪的属性"])

    assert embedding.dense[0] != embedding.dense[1]
    assert embedding.sparse[0] != embedding.sparse[1]


def test_稀疏向量的键是_int_权重是词频():
    """Milvus 的稀疏向量只认 int 键；真实适配器那边也把字符串键转成了 int。"""
    sparse = FakeEmbedder().embed(["神神"]).sparse[0]

    assert len(sparse) == 1
    assert all(isinstance(term, int) for term in sparse)
    assert set(sparse.values()) == {2.0}


def test_空批次返回空的嵌入():
    assert len(FakeEmbedder().embed([])) == 0


def test_稠密与稀疏条数对不上时当场报错():
    with pytest.raises(ModelOutputError):
        Embedding(dense=((0.0,) * DENSE_DIM,), sparse=())


def test_维度与_collection_对不上时当场报错():
    """真让它漏到写入那一步，报的就是 Milvus 自己的话，而且是在云端才发现。"""
    with pytest.raises(ModelOutputError):
        Embedding(dense=((0.0, 0.0),), sparse=({},))


def test_没归一化的稠密向量当场报错():
    """归一化没做的话，IP 度量就不再等价于余弦——分数悄悄变了意思，不报错。"""
    with pytest.raises(ModelOutputError) as excinfo:
        Embedding(dense=((1.0,) * DENSE_DIM,), sparse=({},))

    assert "没有归一化" in str(excinfo.value)


def test_精排分数与候选一一对应且同序():
    docs = [
        "二郎神怎么打：先躲技能再反击",
        "寒江雪的属性面板",
        "二郎神",
    ]

    scores = FakeReranker().rerank("二郎神怎么打", docs)

    # 问题切出来是「二 郎 神 怎 么 打」六个字：全中 1.0、一个不中 0.0、中三个 0.5
    assert scores == [1.0, 0.0, 0.5]


def test_精排分数落在_0_到_1_之间():
    """下游的断崖截断按 0.3 / 0.5 判，假件的量纲必须与真实精排一致。"""
    scores = FakeReranker().rerank("二郎神", ["二郎神", "寒江雪", ""])

    assert scores == [1.0, 0.0, 0.0]


def test_精排收到的是完整候选():
    """不截断、不摘要——假件这边表现为：交给它的就是原文，一个字符不少。"""
    reranker = FakeReranker()
    long_doc = "二郎神" * 5000

    reranker.rerank("二郎神怎么打", [long_doc])

    query, received = reranker.calls[0]
    assert query == "二郎神怎么打"
    assert received == [long_doc]


def test_精排空候选返回空分数():
    assert FakeReranker().rerank("二郎神", []) == []


def test_内存版的组合根里换的就是这两个假件(memory_container):
    """组合根换得掉，整条链路就能在测试里跑起来——两个模型同样是这一条。"""
    assert isinstance(memory_container.embedder, Embedder)
    assert isinstance(memory_container.reranker, Reranker)

    assert len(memory_container.embedder.embed(["二郎神怎么打"])) == 1
    assert memory_container.reranker.rerank("二郎神", ["二郎神"]) == [1.0]


def test_两路向量直接放得进切片并检索得回来(memory_container):
    """向量化与入库之间的接缝。

    `Chunk` 要的正是「归一化的稠密 tuple + int 键的稀疏 Mapping」，对不上的话
    要么在这里炸、要么在写 Milvus 时炸；后者得等到真跑一次云端才知道。
    """
    embedding = memory_container.embedder.embed(["二郎神怎么打"])
    chunk = make_chunk(
        1,
        content="二郎神怎么打：先躲技能再反击",
        dense_vector=embedding.dense[0],
        sparse_vector=embedding.sparse[0],
    )

    memory_container.chunks.upsert("black_myth", [chunk])
    hits = memory_container.chunks.search("black_myth", dense=embedding.dense[0])

    assert [hit.chunk.chunk_id for hit in hits] == [1]
