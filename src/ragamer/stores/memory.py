"""内存假件：与三个真实客户端实现同一组协议。

测试里把整条链路换到这三个上，一行云端代码都不碰（`tests/test_stores_memory.py`）。
过滤语义走 `base.matches`、collection 名走 `base.collection_name`——与真实适配器同一份规则，
所以在内存上跑过的行为，接线到云端仍然成立。

打分只走稠密一路：本层要验证的是**接线**（过滤条件是否生效、顺序、引用组装），
不是语义相似度。哈希假向量完全够用，而且确定。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragamer.stores.base import (
    UNVERSIONED,
    Chunk,
    ChunkFilter,
    ChunkHit,
    StoreError,
    collection_name,
    matches,
    matches_where,
    normalize_prefix,
    require_vectors,
    require_where,
)

#: 假件的名字与地址：它们永远不会报"连不上"，这两个字段只为凑齐协议。
NAME = "内存假件"
ADDRESS = "内存"


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """内积。真实那边稠密向量配 IP 度量（向量化时归一化过），口径一致。"""
    return sum(a * b for a, b in zip(left, right, strict=False))


class InMemoryChunkStore:
    """内存里的切片存储。"""

    name = NAME
    address = ADDRESS

    def __init__(self) -> None:
        self._collections: dict[str, dict[int, Chunk]] = {}

    def check(self) -> None:
        """内存里没有可检查的东西。"""

    def ensure_collection(self, game_id: str) -> None:
        self._collection(collection_name(game_id))

    def upsert(self, game_id: str, chunks: Sequence[Chunk]) -> None:
        rows = self._collection(collection_name(game_id))
        for chunk in chunks:
            require_vectors(chunk)
            rows[chunk.chunk_id] = chunk

    def search(
        self,
        game_id: str,
        *,
        dense: Sequence[float],
        sparse: Mapping[int, float] | None = None,
        where: ChunkFilter | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        """按稠密内积排序；稀疏一路不参与打分（见模块说明）。"""
        hits = [
            ChunkHit(chunk=chunk, score=_dot(dense, chunk.dense_vector or ()))
            for chunk in self._collection(collection_name(game_id)).values()
            if matches(chunk, where)
        ]
        # 分数相同时按 chunk_id 定序，结果与写入顺序无关
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits[:limit]

    def versions(self, game_id: str) -> tuple[str, ...]:
        """与真实适配器同一套口径：未标注版本不算一个可选的版本，去重后按字面升序。"""
        return tuple(
            sorted(
                {
                    chunk.version
                    for chunk in self._collection(collection_name(game_id)).values()
                    if chunk.version != UNVERSIONED
                }
            )
        )

    def fetch_document(self, game_id: str, doc_title: str, *, version: str | None) -> list[Chunk]:
        where = ChunkFilter(doc_title=doc_title, version=version)
        chunks = [
            chunk
            for chunk in self._collection(collection_name(game_id)).values()
            if matches(chunk, where)
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def delete_document(self, game_id: str, doc_title: str, *, version: str) -> None:
        """与真实适配器同一套走法：先按文档查，再只删版本精确对上的那些。"""
        rows = self._collection(collection_name(game_id))
        for chunk in self.fetch_document(game_id, doc_title, version=version):
            if chunk.version == version:
                del rows[chunk.chunk_id]

    def count(self, game_id: str) -> int:
        return len(self._collections.get(collection_name(game_id), {}))

    def drop(self, game_id: str) -> None:
        self._collections.pop(collection_name(game_id), None)

    def _collection(self, game_id: str) -> dict[int, Chunk]:
        return self._collections.setdefault(game_id, {})


class InMemoryDocStore:
    """内存里的文档存储。"""

    name = NAME
    address = ADDRESS

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}

    def check(self) -> None:
        """内存里没有可检查的东西。"""

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        document = self._collections.get(collection, {}).get(doc_id)
        return None if document is None else dict(document)

    def put(self, collection: str, doc_id: str, document: Mapping[str, Any]) -> None:
        # 真客户端会丢掉载荷里的 `_id`（id 是参数不是载荷），这里照做，行为对齐
        payload = {key: value for key, value in document.items() if key != "_id"}
        self._collections.setdefault(collection, {})[doc_id] = payload

    def delete(self, collection: str, doc_id: str) -> None:
        self._collections.get(collection, {}).pop(doc_id, None)

    def delete_where(self, collection: str, where: Mapping[str, Any]) -> int:
        require_where(where)
        rows = self._collections.get(collection, {})
        stale = [doc_id for doc_id, document in rows.items() if matches_where(document, where)]
        for doc_id in stale:
            del rows[doc_id]
        return len(stale)

    def list_ids(self, collection: str) -> list[str]:
        return sorted(self._collections.get(collection, {}))

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
        """等值匹配 + 排序 + 翻页 + 截断，与真实那边同一套语义。

        匹配走 `base.matches_where`，与真实适配器同一份规则（见模块说明）。
        **排序键与翻页游标都落在 `(order_by 字段, _id)` 上**，与真实那边的复合排序一致——
        少了次键，同分的排法两边不一样，而游标正好落在并列值上，那种不一致会漏条。
        缺 `order_by` 那个字段的按空串算，与真实那边的排法不一致，`find` 的契约里写了这一条。
        """
        if after is not None and order_by is None:
            raise ValueError("翻页游标要跟 order_by 一起给：没有排序键就无从比较「之后」")
        found = [
            {"_id": doc_id, **document}
            for doc_id, document in sorted(self._collections.get(collection, {}).items())
            if matches_where(document, where)
        ]
        if order_by is not None:
            found.sort(key=lambda document: _key_of(document, order_by), reverse=descending)
        if after is not None:
            found = [
                document
                for document in found
                if _beyond(_key_of(document, order_by), after, descending)
            ]
        if fields:
            found = [
                {"_id": document["_id"], **{f: document[f] for f in fields if f in document}}
                for document in found
            ]
        return found[:limit] if limit is not None else found

    def ensure_indexes(self, collection: str, fields: Sequence[tuple[str, int]]) -> None:
        """内存里没有索引这回事：`find` 的次序本身就定得下来（见它的说明）。"""


def _key_of(document: Mapping[str, Any], order_by: str) -> tuple[Any, str]:
    """排序与翻页共用的那对键：`(排序字段, 文档 id)`。

    两者必须**同一个取法**：排序按一对键、翻页按另一对，同分的那些就会漏条或重条。
    缺 `order_by` 那个字段的按空串算（真实那边当 null，`find` 的契约里写了这条出入）。
    """
    return (document.get(order_by, ""), str(document["_id"]))


def _beyond(here: tuple[Any, str], cursor: tuple[Any, str], descending: bool) -> bool:
    """这一条是不是严格排在游标那一条之后。倒序往下翻时，「之后」是更小的那一边。"""
    return here < cursor if descending else here > cursor


class InMemoryObjectStore:
    """内存里的对象存储。桶是隐含的：有对象就算有桶。"""

    name = NAME
    address = ADDRESS

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def check(self) -> None:
        """内存里没有可检查的东西。"""

    def ensure_bucket(self) -> None:
        """内存里没有桶这个概念。"""

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        self._objects[normalize_prefix(key)] = data

    def get(self, key: str) -> bytes:
        normalized = normalize_prefix(key)
        if normalized not in self._objects:
            # 与真实客户端同一个异常类型：缝里跑过的分支，接到云端还是同一条
            raise StoreError(f"{self.name} 上没有这个对象：{key}")
        return self._objects[normalized]

    def delete(self, key: str) -> None:
        self._objects.pop(normalize_prefix(key), None)

    def list_keys(self, prefix: str = "") -> list[str]:
        start = normalize_prefix(prefix)
        return sorted(key for key in self._objects if key.startswith(start))

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_keys(prefix)
        for key in keys:
            del self._objects[key]
        return len(keys)
