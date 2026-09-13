"""答案缓存与热门问题：协议、Redis 实现、内存假件、读取侧入口。

业务层 `from ragamer.caching import AnswerCache` 拿协议，实现由组合根注入。
"""

from __future__ import annotations

from ragamer.caching.answer import CachedAnswerer, replay
from ragamer.caching.base import (
    CACHE_ROOT,
    HOT_ROOT,
    TOP_QUESTIONS,
    TTL_SECONDS,
    AnswerCache,
    CachedAnswer,
    CacheError,
    CacheUnavailableError,
    cache_key,
    game_prefix,
    hot_key,
    unavailable,
)
from ragamer.caching.memory import InMemoryAnswerCache
from ragamer.caching.redis import RedisAnswerCache

__all__ = [
    "CACHE_ROOT",
    "HOT_ROOT",
    "TOP_QUESTIONS",
    "TTL_SECONDS",
    "AnswerCache",
    "CacheError",
    "CacheUnavailableError",
    "CachedAnswer",
    "CachedAnswerer",
    "InMemoryAnswerCache",
    "RedisAnswerCache",
    "cache_key",
    "game_prefix",
    "hot_key",
    "replay",
    "unavailable",
]
