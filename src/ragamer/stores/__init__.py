"""存储适配器：三个协议、三个真实实现、三个内存假件。

业务层 `from ragamer.stores import ChunkStore` 拿协议，实现由组合根注入。
"""

from __future__ import annotations

from ragamer.stores.base import (
    UNVERSIONED,
    Chunk,
    ChunkFilter,
    ChunkHit,
    ChunkStore,
    ChunkType,
    DocStore,
    ObjectStore,
    Store,
    StoreCheckError,
    StoreError,
    StoreUnavailableError,
    collection_name,
    matches,
    normalize_prefix,
    require_vectors,
)
from ragamer.stores.chunks import MilvusChunkStore
from ragamer.stores.documents import MongoDocStore
from ragamer.stores.memory import InMemoryChunkStore, InMemoryDocStore, InMemoryObjectStore
from ragamer.stores.objects import MinioObjectStore

__all__ = [
    "UNVERSIONED",
    "Chunk",
    "ChunkFilter",
    "ChunkHit",
    "ChunkStore",
    "ChunkType",
    "DocStore",
    "InMemoryChunkStore",
    "InMemoryDocStore",
    "InMemoryObjectStore",
    "MilvusChunkStore",
    "MinioObjectStore",
    "MongoDocStore",
    "ObjectStore",
    "Store",
    "StoreCheckError",
    "StoreError",
    "StoreUnavailableError",
    "collection_name",
    "matches",
    "normalize_prefix",
    "require_vectors",
]
