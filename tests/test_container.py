"""组合根与启动自检。

自检的判据是"哪里不通、说清楚没有"，不是"连没连上"——云端连不连得上属于集成测试。
"""

from __future__ import annotations

import pytest

from ragamer.config import load_settings
from ragamer.container import build_container
from ragamer.llm import LlmClient
from ragamer.stores import (
    ChunkStore,
    DocStore,
    ObjectStore,
    StoreCheckError,
)
from ragamer.vectors import Embedder, Reranker

from .conftest import FailingStore, make_container


class RecordingStore:
    """记下 `check` 被调过几次。"""

    def __init__(self, name: str = "内存假件", address: str = "内存") -> None:
        self.name = name
        self.address = address
        self.checks = 0

    def check(self) -> None:
        self.checks += 1


def test_组合根按配置构造全部外部依赖(settings_env):
    """造得出来这一条本身就是断言：真实模型是懒加载的，这里不该去碰几个 G 的权重。

    跑测试的环境没有装可选的 models 组，所以只要组合根在构造时碰了模型，
    这里就会以 ModelUnavailableError 炸掉。
    """
    container = build_container(load_settings(env_file=None))

    assert isinstance(container.chunks, ChunkStore)
    assert isinstance(container.docs, DocStore)
    assert isinstance(container.objects, ObjectStore)
    assert isinstance(container.embedder, Embedder)
    assert isinstance(container.reranker, Reranker)
    assert isinstance(container.llm, LlmClient)
    # 出错信息里出现的地址已经抹掉凭据
    assert container.chunks.address == "http://milvus.test:19530"
    assert container.objects.address == "minio.test:9000"
    assert container.docs.address == "mongodb://mongo.test:27017/"


def test_自检把三个服务都查一遍(settings_env):
    """命名空间各由自己的 check 确保（Milvus 的库、MinIO 的桶），容器只管跑一遍。

    两个模型不在自检里：它们的权重几个 G，等第一次真的要用时才加载。
    """
    stores = [RecordingStore(f"服务{index}") for index in range(3)]
    container = make_container(chunks=stores[0], docs=stores[1], objects=stores[2])

    container.check()

    assert [store.checks for store in stores] == [1, 1, 1]


def test_自检一次报出全部不通的服务(settings_env):
    """启动时一次看清全部问题，而不是修一个重启一次。"""
    container = make_container(
        chunks=FailingStore("Milvus", "milvus.test:19530"),
        docs=FailingStore("MongoDB", "mongo.test:27017"),
    )

    with pytest.raises(StoreCheckError) as excinfo:
        container.check()

    message = str(excinfo.value)
    assert len(excinfo.value.failures) == 2
    assert "Milvus 不可达（地址 milvus.test:19530" in message
    assert "MongoDB 不可达（地址 mongo.test:27017" in message
    assert "超时 2.5 秒" in message


def test_只有对象存储不通时也报出来(settings_env):
    """桶建不出来（没权限、卷属主不对）就是在这一条上暴露。"""
    container = make_container(objects=FailingStore("MinIO", "minio.test:9000", "Access Denied"))

    with pytest.raises(StoreCheckError) as excinfo:
        container.check()

    assert "Access Denied" in str(excinfo.value)
