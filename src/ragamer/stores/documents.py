"""文档存储的 MongoDB 实现。

`serverSelectionTimeoutMS` 按原项目标定过的 5 秒（坑 #4）：远端不可达时要快速失败，
不能让一个连不上的 Mongo 把启动挂在那里。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from ragamer.config import MongoSettings
from ragamer.redaction import redact_address
from ragamer.stores.base import MONGO, unavailable

#: 连接串内嵌账号密码是常事，报错信息里只出现抹过的地址。
_FAILURES = (PyMongoError, OSError, ValueError)


class MongoDocStore:
    """MongoDB 上的文档存储。

    构造不连服务：`MongoClient` 本身是懒连的，第一次操作才真的去连。
    """

    def __init__(self, settings: MongoSettings, *, timeout: float) -> None:
        self.name = MONGO
        # 连接串按密钥对待：进日志之前先抹掉账号密码与查询串
        self._uri = settings.uri.get_secret_value()
        self.address = redact_address(self._uri)
        self._db = settings.db
        self._timeout = timeout
        self._client: MongoClient | None = None

    def check(self) -> None:
        """连通性自检：在本项目的库上做一次只读探测。

        🔴 用 `ping` 不行：服务端开了鉴权时，**匿名连接 ping 照样成功**——
        自检放行，等真去读写才报 Unauthorized。列一次本库的集合就能试出到底有没有
        权限落到自己的库上（库不存在也会正常返回空列表，不会平白建一个出来）。
        """
        try:
            self._connect()[self._db].list_collection_names()
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        document = self._collection(collection).find_one({"_id": doc_id})
        if document is None:
            return None
        document.pop("_id")
        return document

    def put(self, collection: str, doc_id: str, document: Mapping[str, Any]) -> None:
        # id 是参数，不是载荷的一部分：载荷里混进 `_id` 会被 Mongo 判为不可改字段而整条写不进
        payload = {key: value for key, value in document.items() if key != "_id"}
        self._collection(collection).replace_one({"_id": doc_id}, payload, upsert=True)

    def delete(self, collection: str, doc_id: str) -> None:
        self._collection(collection).delete_one({"_id": doc_id})

    def list_ids(self, collection: str) -> list[str]:
        return sorted(str(document["_id"]) for document in self._collection(collection).find({}))

    def _connect(self) -> MongoClient:
        if self._client is None:
            timeout_ms = int(self._timeout * 1000)
            self._client = MongoClient(
                self._uri,
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
            )
        return self._client

    def _collection(self, collection: str) -> Collection:
        return self._connect()[self._db][collection]
