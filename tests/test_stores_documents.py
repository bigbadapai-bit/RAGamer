"""文档存储（MongoDB）。

连接参数与请求形状用假客户端断言；真的连上 Mongo 属于云端行为，留给集成测试。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from ragamer.config import MongoSettings, load_settings
from ragamer.stores import documents
from ragamer.stores.base import StoreUnavailableError
from ragamer.stores.documents import MongoDocStore


class FakeCursor:
    """记录 `sort` / `limit` 的假游标，本身可迭代——真游标就是这么用的。

    `sort` / `limit` 的调用记回集合那一份 `calls` 里：**顺序是有意义的**——
    先投影、再排序、最后截断，与真实那边发出去的命令必须是这个次序。
    """

    def __init__(self, documents: list[dict[str, Any]], calls: list[tuple[str, Any]]) -> None:
        self._documents = documents
        self._calls = calls

    def sort(self, key: Any) -> FakeCursor:
        # 原样记下来：真客户端两种都收（单键或一组键值对），这里不替它归一
        self._calls.append(("sort", key))
        return self

    def limit(self, count: int) -> FakeCursor:
        self._calls.append(("limit", count))
        return self

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._documents)


class FakeCollection:
    """记录调用、返回预置文档的假集合。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.document: dict[str, Any] | None = None
        self.created_indexes: list[Any] = []
        self.deleted_count = 0

    def create_index(self, keys: Any) -> None:
        self.created_indexes.append(keys)

    def find_one(self, filter: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append(("find_one", filter))
        return None if self.document is None else dict(self.document)

    def replace_one(
        self, filter: dict[str, Any], payload: dict[str, Any], upsert: bool = False
    ) -> None:
        self.calls.append(("replace_one", {"filter": filter, "payload": payload, "upsert": upsert}))

    def delete_one(self, filter: dict[str, Any]) -> None:
        self.calls.append(("delete_one", filter))

    def delete_many(self, filter: dict[str, Any]) -> Any:
        self.calls.append(("delete_many", filter))
        # 真客户端回一个带 `deleted_count` 的结果对象，适配器读的就是它
        return SimpleNamespace(deleted_count=self.deleted_count)

    def find(self, filter: dict[str, Any], projection: dict[str, Any] | None = None) -> FakeCursor:
        self.calls.append(("find", {"filter": filter, "projection": projection}))
        return FakeCursor([{"_id": "b"}, {"_id": "a"}], self.calls)


class FakeDatabase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.collections: dict[str, FakeCollection] = {}
        self.listed = 0

    def __getitem__(self, collection: str) -> FakeCollection:
        return self.collections.setdefault(collection, FakeCollection())

    def list_collection_names(self) -> list[str]:
        self.listed += 1
        return sorted(self.collections)


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


def test_自检探的是本项目的库(store, mongo):
    """`ping` 匿名连接也能成功，试不出鉴权；列本库的集合才能证明权限落在了自己库上。"""
    store.check()

    client = _client(mongo)
    assert client.pings == 0
    assert client["ragamer-test"].listed == 1


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


def test_按字段取一批文档(store, mongo):
    """列表那一类查询：等值匹配、投影、排序、截断，**一次发出去、按这个次序**。"""
    store.find(
        "conversations",
        {"game_id": "black_myth"},
        fields=("title", "updated_at"),
        order_by="updated_at",
        descending=True,
        limit=20,
    )

    collection = _client(mongo)["ragamer-test"]["conversations"]
    assert collection.calls == [
        (
            "find",
            {
                "filter": {"game_id": "black_myth"},
                "projection": {"title": 1, "updated_at": 1},
            },
        ),
        # 次键恒为 `_id`：翻页的游标落在 (排序键, id) 上，排序少了它就与游标对不上
        ("sort", [("updated_at", -1), ("_id", -1)]),
        ("limit", 20),
    ]


def test_不排序不截断时只发一次_find(store, mongo):
    """没给的就不发——默认值不该变成一条多余的命令。"""
    found = store.find("conversations")

    collection = _client(mongo)["ragamer-test"]["conversations"]
    assert collection.calls == [("find", {"filter": {}, "projection": None})]
    # `get` 把 `_id` 摘掉是因为 id 是调用方给的；批量取反过来，调用方靠它认人
    assert [document["_id"] for document in found] == ["b", "a"]


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


def test_翻页游标落成严格排在之后那个条件(store, mongo):
    """游标必须落在 `(排序键, _id)` 上：单键在并列值上会漏条或重条。

    `$and` 那个形状是「过滤条件还在，再加上游标那一条」；少了它，翻页会把别的库的
    会话也捞进来。
    """
    store.find(
        "conversations",
        {"game_id": "black_myth"},
        order_by="updated_at",
        descending=True,
        limit=20,
        after=("2026-09-13T02:00:00+00:00", "s3"),
    )

    collection = _client(mongo)["ragamer-test"]["conversations"]
    assert collection.calls[0] == (
        "find",
        {
            "filter": {
                "$and": [
                    {"game_id": "black_myth"},
                    {
                        "$or": [
                            {"updated_at": {"$lt": "2026-09-13T02:00:00+00:00"}},
                            {
                                "updated_at": "2026-09-13T02:00:00+00:00",
                                "_id": {"$lt": "s3"},
                            },
                        ]
                    },
                ]
            },
            "projection": None,
        },
    )
    # 排序也要带上次键，否则与游标算的不是同一个次序
    assert collection.calls[1] == ("sort", [("updated_at", -1), ("_id", -1)])


def test_翻页游标要跟排序键一起给(store):
    with pytest.raises(ValueError):
        store.find("conversations", after=("x", "s1"))


def test_空条件不许批量删(store, mongo):
    """真 Mongo 的 `delete_many({})` 会清空整个集合——这一条要在发出去之前拦住。"""
    with pytest.raises(ValueError, match="必须给条件"):
        store.delete_where("conversations", {})

    store.check()  # 先连上才拿得到假集合
    assert _client(mongo)["ragamer-test"]["conversations"].calls == []


def test_建索引按复合键的顺序落下去(store, mongo):
    """键的顺序就是查询的顺序：先按库过滤，再按最后活跃倒序。"""
    store.ensure_indexes("conversations", (("game_id", 1), ("updated_at", -1)))

    collection = _client(mongo)["ragamer-test"]["conversations"]
    assert collection.created_indexes == [[("game_id", 1), ("updated_at", -1)]]


def test_按条件批量删走一次_delete_many(store, mongo):
    """一次发出去，条数由存储数回来——确认页数的与真删掉的是同一批。"""
    store.check()  # 先连上，才拿得到那个假集合
    collection = _client(mongo)["ragamer-test"]["conversations"]
    collection.deleted_count = 1

    assert store.delete_where("conversations", {"game_id": "black_myth"}) == 1

    assert collection.calls == [("delete_many", {"game_id": "black_myth"})]
