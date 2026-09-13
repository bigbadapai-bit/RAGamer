"""共享 fixture。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from ragamer.caching.memory import InMemoryAnswerCache
from ragamer.config import get_settings
from ragamer.container import Container
from ragamer.llm import FakeLlm
from ragamer.stores.base import Chunk, ChunkFilter, ChunkHit, StoreUnavailableError
from ragamer.stores.memory import InMemoryChunkStore, InMemoryDocStore, InMemoryObjectStore
from ragamer.vectors.fake import FakeEmbedder, FakeReranker

#: 一组完整、合法的配置。每个键给不同的值，以便断言"读到的正是这个键"。
#: 键名与模型的对应关系由 `tests/test_config.py::test_env_keys_列出模型读取的全部键` 兜住。
COMPLETE_ENV: dict[str, str] = {
    "RAGAMER_LOG_LEVEL": "DEBUG",
    "RAGAMER_STORE_TIMEOUT_SECONDS": "2.5",
    "RAGAMER_MILVUS_URI": "http://milvus.test:19530",
    "RAGAMER_MILVUS_TOKEN": "test-milvus-token",
    "RAGAMER_MILVUS_DB": "ragamer-test",
    "RAGAMER_MONGO_URI": "mongodb://mongo.test:27017/?authSource=admin",
    "RAGAMER_MONGO_DB": "ragamer-test",
    "RAGAMER_MINIO_ENDPOINT": "minio.test:9000",
    "RAGAMER_MINIO_ACCESS_KEY": "test-access-key",
    "RAGAMER_MINIO_SECRET_KEY": "test-secret-key",
    "RAGAMER_MINIO_BUCKET": "ragamer-test",
    "RAGAMER_MINIO_SECURE": "true",
    "RAGAMER_REDIS_URL": "redis://redis.test:6379/0",
    "RAGAMER_REDIS_PREFIX": "ragamer-test",
    "RAGAMER_LLM_BASE_URL": "https://llm.test/v1",
    "RAGAMER_LLM_API_KEY": "test-llm-api-key",
    "RAGAMER_LLM_MODEL": "test-model",
    "RAGAMER_LLM_TIMEOUT": "12.5",
    "RAGAMER_LLM_MAX_ATTEMPTS": "5",
    "RAGAMER_LLM_BACKOFF_BASE": "0.25",
    "RAGAMER_LLM_BACKOFF_MAX": "4",
    "RAGAMER_MODELS_DEVICE": "cuda:1",
    "RAGAMER_MODELS_FP16": "true",
    "RAGAMER_EMBED_MODEL": "test-embed-model",
    "RAGAMER_EMBED_BATCH_SIZE": "16",
    "RAGAMER_EMBED_MAX_LENGTH": "4096",
    "RAGAMER_RERANK_MODEL": "test-rerank-model",
    "RAGAMER_RERANK_BATCH_SIZE": "32",
    "RAGAMER_RERANK_MAX_LENGTH": "2048",
}


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, str]]:
    """完整配置进环境变量，工作目录里没有 `.env`，进程内配置缓存为空。

    切到 tmp_path 是因为默认装载路径是相对于工作目录的 `.env`——
    开发者本机的 `.env` 不该影响测试。
    """
    for key, value in COMPLETE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield COMPLETE_ENV
    get_settings.cache_clear()


def make_container(
    chunks=None, docs=None, objects=None, embedder=None, reranker=None, llm=None, cache=None
) -> Container:
    """造一个容器：七个依赖默认都是假件，测试只覆盖自己关心的那几个。

    内存假件与真实实现实现的是同一组协议，所以"应用跑起来"的测试都可以从它起步。
    默认的语言模型一条脚本都没排：真被调用到就会当场炸，而不是静默返回空串。
    """
    return Container(
        chunks=chunks if chunks is not None else InMemoryChunkStore(),
        docs=docs if docs is not None else InMemoryDocStore(),
        objects=objects if objects is not None else InMemoryObjectStore(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
        reranker=reranker if reranker is not None else FakeReranker(),
        llm=llm if llm is not None else FakeLlm(),
        cache=cache if cache is not None else InMemoryAnswerCache(),
    )


@pytest.fixture
def memory_container() -> Container:
    """整条链路的内存版：三个客户端与两个模型全换成假件，一行云端代码、一个权重都不碰。

    需要"应用跑起来"的测试（启动自检、将来的 HTTP 缝）都从这里拿容器。
    """
    return make_container()


def fake_vector(seed: int, dim: int = 4) -> tuple[float, ...]:
    """确定性的假向量。本层要验证的是接线，不是语义相似度。"""
    return tuple(round(((seed * 31 + index * 17) % 100) / 100, 4) for index in range(dim))


class FailingStore:
    """一个连不上的服务。自检相关的测试用它造失败项，不必真去连一个不存在的地址。"""

    def __init__(self, name: str, address: str, reason: str = "连接被拒绝") -> None:
        self.name = name
        self.address = address
        self.reason = reason

    def check(self) -> None:
        raise StoreUnavailableError(self.name, self.address, 2.5, self.reason)


def make_chunk(chunk_id: int, **overrides: object) -> Chunk:
    """造一个切片：缺省值都合法且已向量化，测试只覆盖自己关心的那几个字段。"""
    defaults: dict[str, object] = {
        "content": f"正文{chunk_id}",
        "ancestor_path": "二郎神 › 打法",
        "chunk_index": chunk_id,
        "subject_name": "二郎神",
        "game_id": "black_myth",
        "version": "1.0",
        "doc_title": "二郎神",
        "chunk_type": "text",
        "dense_vector": fake_vector(chunk_id),
        "sparse_vector": {chunk_id: 1.0},
    }
    return Chunk(chunk_id=chunk_id, **{**defaults, **overrides})


def chunk_store(game_id: str, *chunks: Chunk) -> InMemoryChunkStore:
    """一个已经装好这批切片的内存切片存储。"""
    store = InMemoryChunkStore()
    store.upsert(game_id, list(chunks))
    return store


class RecordingChunkStore(InMemoryChunkStore):
    """记下每次检索与按文档回查收到的参数，其余行为与内存假件一致。

    两条链路各要一个凭据：检索那侧看的是「过滤条件透传了没有」，聚合那侧看的是
    「哪些文档被回查了」——后者是「聚合发生在截断之后」唯一能从外面看见的证据。
    """

    def __init__(self) -> None:
        super().__init__()
        self.searches: list[dict[str, object]] = []
        self.fetched: list[str] = []

    def search(
        self,
        game_id: str,
        *,
        dense: Sequence[float],
        sparse: Mapping[int, float] | None = None,
        where: ChunkFilter | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        self.searches.append(
            {"game_id": game_id, "dense": dense, "sparse": sparse, "where": where, "limit": limit}
        )
        return super().search(game_id, dense=dense, sparse=sparse, where=where, limit=limit)

    def fetch_document(self, game_id: str, doc_title: str, *, version: str | None) -> list[Chunk]:
        self.fetched.append(doc_title)
        return super().fetch_document(game_id, doc_title, version=version)


class ScriptedReranker:
    """按预置分数打分：候选正文 → 分数。

    截断要的是**摆好的落差**，而 `FakeReranker` 按词重合度打分、给不出指定的分差，
    所以这里直接排分数。分数按正文对号入座、不按位置——存储回来的顺序由它自己定，
    按位置给分等于把用例的意图押在存储的实现细节上。少配了一条会当场 KeyError。
    """

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        self.calls.append((query, list(docs)))
        return [self.scores[doc] for doc in docs]
