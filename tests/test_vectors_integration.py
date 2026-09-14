"""集成测试：跑真实模型。默认不跑。

    uv sync --extra models && uv run pytest -m integration

**跑的是 `.env` 里配的那个模型**，不是写死的默认模型名：配了本地权重目录就直接读盘
（那几个 G 不用再下一次），配的是 HuggingFace 上的名字才去下载（默认缓存到
`~/.cache/huggingface`，要换位置就设 `HF_HOME`）。写死默认值会让下面几条在本地明明
有权重时仍然联网下几个 G，而下的还不是应用实际会加载的那个模型。

假件覆盖不到的是**效果**：中文语料上稠密向量是不是真的按语义靠近、长候选是不是真的
整段被读进去。形状与接线那些事在默认测试里已经钉过了。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from ragamer.config import Settings, get_settings
from ragamer.vectors import DENSE_DIM, BgeM3Embedder, BgeReranker

pytestmark = pytest.mark.integration

#: 一段明显长于 512 token 的正文（中文大致一字一 token）。原项目就是因为 512 的
#: 上限才逼出「超长就摘要压缩再重试」，这里用它验证那条路径确实不需要存在。
_LONG_FILLER = "这一节讲的是地图上的杂项，与问题无关。" * 120


@pytest.fixture(scope="module")
def settings() -> Settings:
    """应用自己那份配置。

    `get_settings` 是带缓存的，而它读的是相对工作目录的 `.env`——集成测试从仓库根跑，
    拿到的就是使用者配好的那份。`settings_env` 那类 fixture 用完会清缓存并还原环境变量，
    所以这里不会读到测试用的假值。
    """
    return get_settings()


@pytest.fixture(scope="module")
def embedder(settings: Settings) -> Iterator[BgeM3Embedder]:
    pytest.importorskip(
        "FlagEmbedding", reason="真实模型在可选的 models 组里：uv sync --extra models"
    )
    yield BgeM3Embedder(settings.embed, settings.models)


@pytest.fixture(scope="module")
def reranker(settings: Settings) -> Iterator[BgeReranker]:
    pytest.importorskip(
        "FlagEmbedding", reason="真实模型在可选的 models 组里：uv sync --extra models"
    )
    yield BgeReranker(settings.rerank, settings.models)


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


def test_本地目录存在时不去联网下载(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    """配了本地权重目录就该直接读盘，**一个字节都不该再下**。

    这条钉的是「本地优先」本身，而不只是 FlagEmbedding 当下的内部行为：那个判据是
    `os.path.exists(model_name_or_path)`，写在它自己的源码里，改了这里就会红。

    没配本地目录时跳过——那种配置本来就要下载，这条没有可验的东西。
    """
    if not os.path.isdir(settings.embed.model):
        pytest.skip(f"配置的向量化模型不是本地目录（{settings.embed.model}），这条无从验证")

    module = pytest.importorskip("FlagEmbedding.finetune.embedder.encoder_only.m3.runner")

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"本地目录已在（{settings.embed.model}），却仍走了下载：{args} {kwargs}"
        )

    monkeypatch.setattr(module, "snapshot_download", refuse)
    embedding = BgeM3Embedder(settings.embed, settings.models).embed(["二郎神怎么打"])

    assert embedding.dense, "本地权重应当直接读得出来"
