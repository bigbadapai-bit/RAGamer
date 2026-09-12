"""集成测试：跑真实模型。默认不跑。

    uv sync --extra models && uv run pytest -m integration

需要可选的 `models` 组、几个 G 的权重，以及一次 HuggingFace 下载（默认缓存到
`~/.cache/huggingface`，要换位置就设 `HF_HOME`）。

假件覆盖不到的是**效果**：中文语料上稠密向量是不是真的按语义靠近、长候选是不是真的
整段被读进去。形状与接线那些事在默认测试里已经钉过了。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ragamer.config import EmbedSettings, ModelSettings, RerankSettings
from ragamer.vectors import DENSE_DIM, BgeM3Embedder, BgeReranker

pytestmark = pytest.mark.integration

#: 一段明显长于 512 token 的正文（中文大致一字一 token）。原项目就是因为 512 的
#: 上限才逼出「超长就摘要压缩再重试」，这里用它验证那条路径确实不需要存在。
_LONG_FILLER = "这一节讲的是地图上的杂项，与问题无关。" * 120


@pytest.fixture(scope="module")
def embedder() -> Iterator[BgeM3Embedder]:
    pytest.importorskip(
        "FlagEmbedding", reason="真实模型在可选的 models 组里：uv sync --extra models"
    )
    yield BgeM3Embedder(EmbedSettings(), ModelSettings())


@pytest.fixture(scope="module")
def reranker() -> Iterator[BgeReranker]:
    pytest.importorskip(
        "FlagEmbedding", reason="真实模型在可选的 models 组里：uv sync --extra models"
    )
    yield BgeReranker(RerankSettings(), ModelSettings())


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_稠密向量的维度与归一化(embedder: BgeM3Embedder):
    """归一化之后配 IP 度量等价余弦（坑 #2），所以模长必须是 1。"""
    embedding = embedder.embed(["二郎神怎么打"])

    vector = embedding.dense[0]
    assert len(vector) == DENSE_DIM
    assert _cosine(vector, vector) == pytest.approx(1.0, abs=1e-4)


def test_一次调用同时给出两路(embedder: BgeM3Embedder):
    embedding = embedder.embed(["二郎神怎么打", "寒江雪的属性"])

    assert len(embedding.dense) == len(embedding.sparse) == 2
    # 稀疏一路按词给权重，空的话说明 return_sparse 没生效
    assert all(embedding.sparse), "稀疏向量是空的，两路里少了一路"
    assert all(isinstance(term, int) for term in embedding.sparse[0])


def test_稠密向量按语义靠近(embedder: BgeM3Embedder):
    """假件只能验证形状；「意思近的排前面」只有真模型跑得出来。"""
    query, related, unrelated = embedder.embed(
        [
            "二郎神怎么打",
            "二郎神的打法：先躲开他的突进，再打第三只眼",
            "寒江雪是燕云十六声里的一名剑客，擅长用剑",
        ]
    ).dense

    assert _cosine(query, related) > _cosine(query, unrelated)


def test_精排把相关的排在前面(reranker: BgeReranker):
    query = "二郎神怎么打"
    relevant = "二郎神的打法：先躲开他的突进，再打第三只眼"
    irrelevant = "寒江雪是燕云十六声里的一名剑客，擅长用剑"

    scores = reranker.rerank(query, [relevant, irrelevant])

    assert scores[0] > scores[1]
    # sigmoid 之后的取值，与断崖截断的 0.3 / 0.5 同一量纲
    assert 0.0 <= scores[1] < scores[0] <= 1.0


def test_精排读得进长候选不截断(reranker: BgeReranker):
    """答案埋在 512 token 之后。

    截断到 512 的话，模型看到的全是与问题无关的填充，分数会掉到跟不相关候选一个水平。
    这条是「不需要摘要压缩那条路径」的回归测试。
    """
    query = "二郎神的弱点是什么"
    long_relevant = _LONG_FILLER + "二郎神的弱点是第三只眼，打那里伤害翻倍。"
    irrelevant = "寒江雪是燕云十六声里的一名剑客。"

    long_score, irrelevant_score = reranker.rerank(query, [long_relevant, irrelevant])

    assert long_score > irrelevant_score
