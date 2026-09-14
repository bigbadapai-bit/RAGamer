"""真实模型适配器：换了加载器就能把「只加载一次」「不截断」「两路同源」钉死。

跑这里**不需要 torch**——适配器只在真的要用模型时才去 import FlagEmbedding，
测试注入一个假加载器就绕开了它。真实模型本身的效果在
`tests/test_vectors_integration.py`，那部分默认不跑。
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ragamer.config import EmbedSettings, ModelSettings, RerankSettings
from ragamer.vectors import (
    DENSE_DIM,
    BgeM3Embedder,
    BgeReranker,
    ModelOutputError,
    ModelUnavailableError,
)
from ragamer.vectors.bge import _looks_like_path, _require_local_dir, load_bge_m3, load_bge_reranker


class _Loader:
    """记下加载了几次、每次拿到什么配置。返回值就是被"加载"出来的假模型。"""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls: list[tuple[Any, ModelSettings]] = []

    def __call__(self, config: Any, shared: ModelSettings) -> Any:
        self.calls.append((config, shared))
        return self.model


class _StubM3:
    """假 BGE-M3：返回固定形状的两路向量，并记下每一次调用拿到了什么。

    稠密向量按真实模型的样子归一化（`Embedding` 会核这一条），稀疏向量的键**照
    FlagEmbedding 的原样用字符串**——真实那边就是 `str(token_id)`，适配器必须转成 int。
    """

    def __init__(self, dim: int = DENSE_DIM) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append({"sentences": list(sentences), **kwargs})
        return {
            "dense_vecs": [[1.0 / math.sqrt(self.dim)] * self.dim for _ in sentences],
            "lexical_weights": [{"12": 0.5, "34": 0.25} for _ in sentences],
        }


class _StubReranker:
    """假 reranker：分数按脚本给，默认按传入顺序递增，便于断言顺序。"""

    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def compute_score(
        self, sentence_pairs: Sequence[tuple[str, str]], **kwargs: Any
    ) -> list[float]:
        self.calls.append({"pairs": list(sentence_pairs), **kwargs})
        if self.scores is not None:
            return list(self.scores)
        return [float(index) for index, _ in enumerate(sentence_pairs)]


def _embedder(
    stub: _StubM3 | None = None,
    *,
    config: EmbedSettings | None = None,
    shared: ModelSettings | None = None,
) -> tuple[BgeM3Embedder, _Loader, _StubM3]:
    model = stub if stub is not None else _StubM3()
    loader = _Loader(model)
    return (
        BgeM3Embedder(config or EmbedSettings(), shared or ModelSettings(), loader=loader),
        loader,
        model,
    )


def _reranker(
    stub: _StubReranker | None = None,
    *,
    config: RerankSettings | None = None,
    shared: ModelSettings | None = None,
) -> tuple[BgeReranker, _Loader, _StubReranker]:
    model = stub if stub is not None else _StubReranker()
    loader = _Loader(model)
    return (
        BgeReranker(config or RerankSettings(), shared or ModelSettings(), loader=loader),
        loader,
        model,
    )


def test_构造组合根时不加载模型():
    """启动自检要的是"配置对不对"，不该卡在几个 G 的权重上。"""
    _, loader, _ = _embedder()

    assert loader.calls == []


def test_模型只加载一次并复用():
    embedder, loader, _ = _embedder()

    for _ in range(5):
        embedder.embed(["二郎神怎么打"])

    assert len(loader.calls) == 1


def test_并发首次调用也只加载一次():
    """界面后端会把同步端点丢进线程池，并发首次调用会同时看见"还没加载"。"""
    loads: list[EmbedSettings] = []

    def slow_loader(config: EmbedSettings, shared: ModelSettings) -> _StubM3:
        loads.append(config)
        # 把加载拉长，放大"两个线程同时看见 None"的那个窗口
        time.sleep(0.05)
        return _StubM3()

    embedder = BgeM3Embedder(EmbedSettings(), ModelSettings(), loader=slow_loader)
    threads = [threading.Thread(target=embedder.embed, args=(["二郎神怎么打"],)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(loads) == 1


def test_加载参数原样来自配置():
    config = EmbedSettings(model="BAAI/bge-m3")
    shared = ModelSettings(device="cuda:2", fp16=True)
    embedder, loader, _ = _embedder(config=config, shared=shared)

    embedder.embed(["二郎神怎么打"])

    assert loader.calls == [(config, shared)]


def test_空批次不去加载模型():
    embedder, loader, _ = _embedder()

    assert len(embedder.embed([])) == 0
    assert loader.calls == []


def test_一次调用同时取回两路():
    """混合检索的两路必须同源，否则稠密与稀疏说的可能不是同一段文本。"""
    embedder, _, stub = _embedder()

    embedding = embedder.embed(["二郎神怎么打", "寒江雪的属性"])

    assert len(stub.calls) == 1
    assert stub.calls[0]["sentences"] == ["二郎神怎么打", "寒江雪的属性"]
    assert stub.calls[0]["return_dense"] is True
    assert stub.calls[0]["return_sparse"] is True
    assert len(embedding.dense) == len(embedding.sparse) == 2


def test_稀疏向量的字符串键被转成_int():
    """FlagEmbedding 用 `str(token_id)` 当字典键，Milvus 的稀疏向量要 int。"""
    embedder, _, _ = _embedder()

    sparse = embedder.embed(["二郎神怎么打"]).sparse[0]

    assert sparse == {12: 0.5, 34: 0.25}


def test_向量化按配置的上下文上限走():
    """库自己给的默认值是 512 —— 照默认值用就是把静默截断搬回来。"""
    config = EmbedSettings(max_length=4096, batch_size=16)
    embedder, _, stub = _embedder(config=config)

    embedder.embed(["二郎神怎么打"])

    assert stub.calls[0]["max_length"] == 4096
    assert stub.calls[0]["batch_size"] == 16


def test_长文本整段进模型不截断():
    embedder, _, stub = _embedder()
    long_text = "二郎神" * 4000

    embedder.embed([long_text])

    assert stub.calls[0]["sentences"] == [long_text]


def test_条数对不上时报错而不是按短的截齐():
    class _DropsOne(_StubM3):
        def encode(self, sentences, **kwargs):
            result = super().encode(sentences, **kwargs)
            return {**result, "dense_vecs": result["dense_vecs"][:1]}

    embedder, _, _ = _embedder(_DropsOne())

    with pytest.raises(ModelOutputError) as excinfo:
        embedder.embed(["一", "二"])

    assert "稠密向量有 1 条" in str(excinfo.value)
    assert "2 条" in str(excinfo.value)


def test_没归一化的稠密向量当场报错():
    """归一化一丢，IP 度量就不再等价于余弦——分数悄悄变了意思，检索侧毫无察觉。"""

    class _NotNormalized(_StubM3):
        def encode(self, sentences, **kwargs):
            result = super().encode(sentences, **kwargs)
            return {**result, "dense_vecs": [[0.5] * self.dim for _ in sentences]}

    embedder, _, _ = _embedder(_NotNormalized())

    with pytest.raises(ModelOutputError) as excinfo:
        embedder.embed(["二郎神怎么打"])

    assert "没有归一化" in str(excinfo.value)


def test_没装_FlagEmbedding_时说清怎么装(monkeypatch):
    """真实模型在可选的 models 组里；缺了要给出装法，而不是一个光秃秃的 ImportError。"""
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    embedder = BgeM3Embedder(EmbedSettings(), ModelSettings())

    with pytest.raises(ModelUnavailableError) as excinfo:
        embedder.embed(["二郎神怎么打"])

    assert "uv sync --extra models" in str(excinfo.value)


def test_精排分数与候选同序():
    reranker, _, stub = _reranker(_StubReranker(scores=[0.9, 0.1, 0.5]))

    assert reranker.rerank("二郎神怎么打", ["甲", "乙", "丙"]) == [0.9, 0.1, 0.5]
    assert stub.calls[0]["pairs"] == [
        ("二郎神怎么打", "甲"),
        ("二郎神怎么打", "乙"),
        ("二郎神怎么打", "丙"),
    ]


def test_精排按配置的长上下文走而不是库默认的_512():
    """512 正是原项目那条「超长就摘要压缩再重试」路径的来源。"""
    config = RerankSettings(max_length=8192)
    reranker, _, stub = _reranker(config=config)

    reranker.rerank("二郎神怎么打", ["甲"])

    assert stub.calls[0]["max_length"] == 8192
    # sigmoid 之后的取值，与断崖截断的 0.3 / 0.5 同一量纲
    assert stub.calls[0]["normalize"] is True


def test_精排整段吃下长候选():
    """没有截断、没有摘要：交给模型的就是候选原文。"""
    reranker, _, stub = _reranker()
    long_doc = "二郎神" * 4000 + "：先躲技能，再打第三只眼"

    reranker.rerank("二郎神怎么打", [long_doc])

    assert stub.calls[0]["pairs"] == [("二郎神怎么打", long_doc)]


def test_精排分数条数对不上时报错():
    reranker, _, _ = _reranker(_StubReranker(scores=[0.5]))

    with pytest.raises(ModelOutputError) as excinfo:
        reranker.rerank("二郎神怎么打", ["甲", "乙"])

    assert "1 个分数" in str(excinfo.value)
    assert "2 个" in str(excinfo.value)


def test_精排空候选既不加载模型也不报错():
    reranker, loader, _ = _reranker()

    assert reranker.rerank("二郎神怎么打", []) == []
    assert loader.calls == []


def test_精排只加载一次并复用():
    reranker, loader, _ = _reranker()

    for _ in range(5):
        reranker.rerank("二郎神怎么打", ["甲"])

    assert len(loader.calls) == 1


@pytest.mark.parametrize("value", ["BAAI/bge-m3", "bge-m3", "BAAI/bge-reranker-v2-m3"])
def test_模型名不算路径(value: str):
    """仓库名带斜杠，所以判据不能是「含斜杠」——否则模型名会被当成路径拦下来。"""
    assert not _looks_like_path(value)


@pytest.mark.parametrize(
    "value",
    [
        r"D:\models\bge-m3",
        "D:/models/bge-m3",
        "/opt/models/bge-m3",
        "~/models/bge-m3",
        "./models/bge-m3",
        "../models/bge-m3",
        r"\\host\share\bge-m3",
    ],
)
def test_这些形态算路径(value: str):
    assert _looks_like_path(value)


def test_目录不在时当场说清而不是去下载():
    """不让它拿着路径串去当 HuggingFace 仓库名下——那句报错与「目录不在」毫无关系。"""
    missing = "D:/这个目录不存在/bge-m3"

    with pytest.raises(ModelUnavailableError, match="目录不存在"):
        load_bge_m3(EmbedSettings(model=missing), ModelSettings())
    with pytest.raises(ModelUnavailableError, match="目录不存在"):
        load_bge_reranker(RerankSettings(model=missing), ModelSettings())


def test_模型名不拦():
    """填模型名时本来就该联网下，这道判别不该插手。

    只调判别本身，不走 `load_bge_m3`——那会真的去下载，那是集成测试的事。
    """
    _require_local_dir("BAAI/bge-m3", "向量化模型")
    _require_local_dir("BAAI/bge-reranker-v2-m3", "精排模型")


def test_目录在时不拦(tmp_path: Path):
    _require_local_dir(str(tmp_path), "向量化模型")
