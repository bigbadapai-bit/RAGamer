"""确定性假件：与两个真实适配器实现同一组协议，一个权重、一次网络都不碰。

本层要验证的是**接线**（两路是否同源、分数是否与候选同序、组合根换不换得掉），
不是语义相似度——哈希假件完全够用，而且确定：同一段文本永远得到同一个向量。
这一条比看上去重要——「导入时算过的向量」与「提问时算出来的向量」在测试里必然一致，
否则检索测试会因为假件自己抖动而时红时绿。

假件的取值不追求与 BGE-M3 同尺度（真实的稀疏权重落在 (0, 1] 附近，这里是词频），
所以拿它做的断言只该看形状、顺序与确定性，不该看分数的大小。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from ragamer.vectors.base import DENSE_DIM, Embedding, normalize_dense

#: 假词表的大小。取大一点，让不同的词基本不会撞到同一个 term id 上。
SPARSE_VOCAB = 1 << 20

#: 切词只求确定与可解释：ASCII 词整体切，CJK 逐字切——中文没有空格，逐字才切得开。
_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]")

#: 一个 64 位无符号整数的最大值，用来把哈希铺进区间。
_UINT64_MAX = (1 << 64) - 1


class FakeEmbedder:
    """哈希假向量：同一段文本永远得到同一个稠密向量与同一个稀疏向量。

    稠密向量是均匀铺开再归一化的，所以它与文本的相关性只体现在「一样／不一样」上；
    需要**有意义的排序**时用 :class:`FakeReranker`，它按词重合度打分。
    """

    def __init__(self) -> None:
        #: 每一次调用收到的文本，供断言「两路出自同一次调用」。
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> Embedding:
        self.calls.append(list(texts))
        return Embedding(
            dense=tuple(normalize_dense(_dense(text)) for text in texts),
            sparse=tuple(_sparse(text) for text in texts),
        )

    def warm(self) -> None:
        """没有权重要加载。**接口上有它，是因为真实适配器有**——
        启动预热那条路不该因为换成了假件就走不通。"""


class FakeReranker:
    """按词重合度打分的假精排。

    分数 = 问题里的词有多大比例出现在候选里，落在 [0, 1]，越大越相关。

    它不跟嵌入假件一样纯哈希，是因为下游要拿它做**有意义的排序**：融合、断崖截断、
    父块聚合这些测试需要「相关的排在前面」这件事成立，否则每个用例都得手工摆分数。
    分数与真实精排同处 [0, 1]（真实那边是 sigmoid 之后的取值），
    所以断崖截断的 0.3 / 0.5 在假件上也是同一量纲。
    """

    def __init__(self) -> None:
        #: 每一次调用收到的完整问题与候选，供断言「候选没有被截断或摘要」。
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        candidates = list(docs)
        self.calls.append((query, candidates))
        wanted = set(_tokens(query))
        if not wanted:
            return [0.0 for _ in candidates]
        return [len(wanted & set(_tokens(candidate))) / len(wanted) for candidate in candidates]

    def warm(self) -> None:
        """没有权重要加载。理由见 :meth:`FakeEmbedder.warm`。"""


def _dense(text: str) -> tuple[float, ...]:
    """文本 → 稠密向量。

    用 shake_128 直接铺出 `DENSE_DIM` 个 64 位整数再映到 [-1, 1)，不经过 `random`：
    少一层「换个 Python 版本数值就变」的可能。
    """
    raw = hashlib.shake_128(text.encode("utf-8")).digest(8 * DENSE_DIM)
    return tuple(
        int.from_bytes(raw[offset : offset + 8], "big") / _UINT64_MAX * 2.0 - 1.0
        for offset in range(0, 8 * DENSE_DIM, 8)
    )


def _sparse(text: str) -> Mapping[int, float]:
    """文本 → 词袋式的稀疏向量：词的哈希当 term id，出现次数当权重。

    key 一律是 int——Milvus 的稀疏向量要 int 键，真实适配器那边也把
    FlagEmbedding 的字符串键转成了 int，两边在这一点上必须一致。
    """
    weights: dict[int, float] = {}
    for token in _tokens(text):
        term = _term_id(token)
        weights[term] = weights.get(term, 0.0) + 1.0
    return weights


def _term_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SPARSE_VOCAB


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
