"""共享 fixture。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ragamer.config import get_settings
from ragamer.container import Container
from ragamer.llm import FakeLlm
from ragamer.sources import MarkdownParser, ParserRouter
from ragamer.stores.base import Chunk, StoreUnavailableError
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
    "RAGAMER_LLM_BASE_URL": "https://llm.test/v1",
    "RAGAMER_LLM_API_KEY": "test-llm-api-key",
    "RAGAMER_LLM_MODEL": "test-model",
    "RAGAMER_LLM_TIMEOUT": "12.5",
    "RAGAMER_LLM_MAX_ATTEMPTS": "5",
    "RAGAMER_LLM_BACKOFF_BASE": "0.25",
    "RAGAMER_LLM_BACKOFF_MAX": "4",
    "RAGAMER_VISION_BASE_URL": "https://vision.test/v1",
    "RAGAMER_VISION_API_KEY": "test-vision-api-key",
    "RAGAMER_VISION_MODEL": "test-vision-model",
    "RAGAMER_VISION_TIMEOUT": "30",
    "RAGAMER_VISION_MAX_ATTEMPTS": "2",
    "RAGAMER_VISION_BACKOFF_BASE": "0.5",
    "RAGAMER_VISION_BACKOFF_MAX": "2",
    "RAGAMER_MINERU_BASE_URL": "https://mineru.test",
    "RAGAMER_MINERU_API_KEY": "test-mineru-api-key",
    "RAGAMER_MINERU_MODEL_VERSION": "pipeline",
    "RAGAMER_MINERU_POLL_INTERVAL_SECONDS": "0.5",
    "RAGAMER_MINERU_POLL_TIMEOUT_SECONDS": "30",
    "RAGAMER_MINERU_REQUEST_TIMEOUT_SECONDS": "12.5",
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
    chunks=None,
    docs=None,
    objects=None,
    embedder=None,
    reranker=None,
    llm=None,
    parser=None,
    vision=None,
    ocr=None,
) -> Container:
    """造一个容器：依赖默认都是假件，测试只覆盖自己关心的那几个。

    内存假件与真实实现实现的是同一组协议，所以"应用跑起来"的测试都可以从它起步。
    默认的语言模型一条脚本都没排：真被调用到就会当场炸，而不是静默返回空串。
    默认的解析器也只有 md／txt 那条路——真正接上 MinerU 的是组合根，
    这里换掉就等于把那份资料交给假件。
    """
    return Container(
        chunks=chunks if chunks is not None else InMemoryChunkStore(),
        docs=docs if docs is not None else InMemoryDocStore(),
        objects=objects if objects is not None else InMemoryObjectStore(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
        reranker=reranker if reranker is not None else FakeReranker(),
        llm=llm if llm is not None else FakeLlm(),
        # 默认不接视觉模型：没配时组合根给的就是 None（补图只做二次 OCR）
        vision=vision,
        # 默认的 OCR 引擎一被调用就炸——排了脚本的测试才该走到它
        ocr=ocr if ocr is not None else FailingOcr(),
        parser=parser if parser is not None else ParserRouter((MarkdownParser(),)),
    )


class FailingOcr:
    """一调就炸的 OCR。默认的二次 OCR：真被用到说明这个测试接线接错了。"""

    def read(self, data: bytes) -> str:
        raise AssertionError("这个测试没排 OCR：补图那一层不该走到这里")


class FakeOcr:
    """按图逐张回话的假 OCR。记下每一张喂进来的字节。

    脚本里也可以排异常（`OcrError` / `OcrUnavailable`），用来验失败那两条路。
    脚本排空之后再被调用会当场炸——测试少排了一条时立刻看得见，不是静默给空串。
    """

    def __init__(self, *texts: Any) -> None:
        self.texts = list(texts)
        self.images: list[bytes] = []

    def read(self, data: bytes) -> str:
        self.images.append(data)
        if not self.texts:
            raise AssertionError("假 OCR 没有更多脚本回复了 —— 测试少排了一条")
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return text


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


class BrokenChunkStore(InMemoryChunkStore):
    """连不上的向量库：数不出来也删不掉。

    删库那几条要用两种形态：一直坏（清不掉时配置得留着），以及坏一次之后好起来
    （重来一次能补上）——`recover()` 管后者。`upsert` 照常可用，先得让库里有东西。
    """

    def __init__(self) -> None:
        super().__init__()
        self.broken = True

    def recover(self) -> None:
        self.broken = False

    def count(self, game_id: str) -> int:
        self._refuse()
        return super().count(game_id)

    def drop(self, game_id: str) -> None:
        self._refuse()

    def _refuse(self) -> None:
        if self.broken:
            raise StoreUnavailableError("Milvus", "milvus.test:19530", 2.5, "连接被拒绝")


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
