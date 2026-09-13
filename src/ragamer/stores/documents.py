"""文档存储的 MongoDB 实现。

`serverSelectionTimeoutMS` 按原项目标定过的 5 秒（坑 #4）：远端不可达时要快速失败，
不能让一个连不上的 Mongo 把启动挂在那里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from ragamer.config import MongoSettings
from ragamer.redaction import redact_address
from ragamer.stores.base import MONGO, require_where, unavailable

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

    def find(
        self,
        collection: str,
        where: Mapping[str, Any] | None = None,
        *,
        fields: Sequence[str] = (),
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
        after: tuple[Any, str] | None = None,
    ) -> list[dict[str, Any]]:
        if after is not None and order_by is None:
            raise ValueError("翻页游标要跟 order_by 一起给：没有排序键就无从比较「之后」")
        # 投影留空即整份返回。`_id` 不用显式要：Mongo 默认就带，而调用方正靠它认人
        projection = {field: 1 for field in fields} if fields else None
        query = dict(where or {})
        direction = -1 if descending else 1
        keys: list[tuple[str, int]] = []
        if order_by is not None:
            # 次键恒为 `_id`：游标落在 (排序键, id) 上，排序少了它两边就对不上
            keys = [(order_by, direction), ("_id", direction)]
        if after is not None:
            value, last_id = after
            op = "$lt" if descending else "$gt"
            # 严格排在游标之后：排序键更靠后的那些，以及**排序键相同但 id 更靠后**的那些。
            # 第二个分支拿 `last_id` 比，不是拿排序键那个值比——写成同一个值就永远比不出东西，
            # 并列的那几条会被整批跳过（而会话列表恰恰常常并列）。
            cursor_filter: dict[str, Any] = {
                "$or": [{order_by: {op: value}}, {order_by: value, "_id": {op: last_id}}]
            }
            query = {"$and": [query, cursor_filter]} if query else cursor_filter
        cursor = self._collection(collection).find(query, projection)
        if keys:
            cursor = cursor.sort(keys)
        if limit is not None:
            cursor = cursor.limit(limit)
        return [dict(document) for document in cursor]

    def ensure_indexes(self, collection: str, fields: Sequence[tuple[str, int]]) -> None:
        self._collection(collection).create_index(list(fields))

    def delete_where(self, collection: str, where: Mapping[str, Any]) -> int:
        require_where(where)
        return int(self._collection(collection).delete_many(dict(where)).deleted_count)

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
