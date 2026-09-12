"""文档存储（MongoDB）。

连接参数与请求形状用假客户端断言；真的连上 Mongo 属于云端行为，留给集成测试。
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from ragamer.config import MongoSettings, load_settings
from ragamer.stores import documents
from ragamer.stores.base import StoreUnavailableError
from ragamer.stores.documents import MongoDocStore


class FakeCollection:
    """记录调用、返回预置文档的假集合。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.document: dict[str, Any] | None = None

    def find_one(self, filter: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append(("find_one", filter))
        return None if self.document is None else dict(self.document)

    def replace_one(
        self, filter: dict[str, Any], payload: dict[str, Any], upsert: bool = False
    ) -> None:
        self.calls.append(("replace_one", {"filter": filter, "payload": payload, "upsert": upsert}))

    def delete_one(self, filter: dict[str, Any]) -> None:
        self.calls.append(("delete_one", filter))

    def find(self, filter: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append(("find", filter))
        return [{"_id": "b"}, {"_id": "a"}]


class FakeDatabase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.collections: dict[str, FakeCollection] = {}

    def __getitem__(self, collection: str) -> FakeCollection:
        return self.collections.setdefault(collection, FakeCollection())


class FakeMongoClient:
    """替代 `pymongo.MongoClient`：记录构造参数与调用，不连服务。"""

    instances: ClassVar[list[FakeMongoClient]] = []

    def __init__(self, uri: str, **kwargs: Any) -> None:
        self.uri = uri
        self.init_kwargs = kwargs
        self.databases: dict[str, FakeDatabase] = {}
        self.pings = 0
        FakeMongoClient.instances.append(self)

    @property
    def admin(self) -> FakeMongoClient:
        return self

    def command(self, name: str) -> dict[str, Any]:
        self.pings += 1
        return {"ok": 1, "command": name}

    def __getitem__(self, database: str) -> FakeDatabase:
        return self.databases.setdefault(database, FakeDatabase(database))


@pytest.fixture
def mongo(monkeypatch: pytest.MonkeyPatch) -> type[FakeMongoClient]:
    FakeMongoClient.instances = []
    monkeypatch.setattr(documents, "MongoClient", FakeMongoClient)
    return FakeMongoClient


@pytest.fixture
def store(mongo: type[FakeMongoClient], settings_env) -> MongoDocStore:
    return MongoDocStore(load_settings(env_file=None).mongo, timeout=2.5)


def _client(mongo: type[FakeMongoClient]) -> FakeMongoClient:
    assert mongo.instances, "还没有连过"
    return mongo.instances[0]


def test_连接串与超时按配置传给_Mongo(store, mongo):
    """坑 #4：远端不可达时要快速失败，serverSelectionTimeoutMS 必给。"""
    store.check()

    client = _client(mongo)
    assert client.uri == "mongodb://mongo.test:27017/?authSource=admin"
    assert client.init_kwargs["serverSelectionTimeoutMS"] == 2500
    assert client.init_kwargs["connectTimeoutMS"] == 2500


def test_自检就是_ping_一次(store, mongo):
    store.check()

    assert _client(mongo).pings == 1


def test_构造时不连服务(store, mongo):
    assert mongo.instances == []


def test_取文档时去掉内部的_id(store, mongo):
    store.check()
    collection = _client(mongo)["ragamer-test"]["knowledge_bases"]
    collection.document = {"_id": "black_myth", "current_version": "1.0"}

    assert store.get("knowledge_bases", "black_myth") == {"current_version": "1.0"}
    assert collection.calls == [("find_one", {"_id": "black_myth"})]


def test_没有这份文档时返回_None(store):
    assert store.get("knowledge_bases", "black_myth") is None


def test_写入是覆盖式的_upsert(store, mongo):
    store.put("knowledge_bases", "black_myth", {"current_version": "2.0"})

    collection = _client(mongo)["ragamer-test"]["knowledge_bases"]
    assert collection.calls == [
        (
            "replace_one",
            {
                "filter": {"_id": "black_myth"},
                # id 是参数不是载荷：混进 `_id` 会被 Mongo 判为不可改字段
                "payload": {"current_version": "2.0"},
                "upsert": True,
            },
        )
    ]


def test_载荷里的_id_被丢掉(store, mongo):
    store.put("knowledge_bases", "black_myth", {"_id": "别的", "current_version": "2.0"})

    collection = _client(mongo)["ragamer-test"]["knowledge_bases"]
    assert collection.calls[0][1]["payload"] == {"current_version": "2.0"}


def test_删除与列出文档_id(store, mongo):
    store.delete("knowledge_bases", "black_myth")

    collection = _client(mongo)["ragamer-test"]["knowledge_bases"]
    assert collection.calls == [("delete_one", {"_id": "black_myth"})]
    assert store.list_ids("knowledge_bases") == ["a", "b"]


def test_远端不可达时在超时内失败并点名服务与地址():
    settings = MongoSettings(uri="mongodb://127.0.0.1:1/?authSource=admin", db="ragamer_test")
    store = MongoDocStore(settings, timeout=1.0)

    start = time.monotonic()
    with pytest.raises(StoreUnavailableError) as excinfo:
        store.check()
    elapsed = time.monotonic() - start

    assert elapsed < 5, f"没有在设定的超时内失败：{elapsed:.1f} 秒"
    assert "MongoDB" in str(excinfo.value)
    assert "127.0.0.1:1" in str(excinfo.value)


def test_不可达时的报错里没有账号密码():
    settings = MongoSettings(uri="mongodb://root:MONGO-PW@127.0.0.1:1/", db="ragamer_test")
    store = MongoDocStore(settings, timeout=1.0)

    with pytest.raises(StoreUnavailableError) as excinfo:
        store.check()

    assert "MONGO-PW" not in str(excinfo.value)
    assert "127.0.0.1:1" in str(excinfo.value)
