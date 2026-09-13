"""内存假件：与 `ragamer.caching.redis` 实现同一组协议。

测试里把缓存换成它，一行云端代码都不碰（`tests/test_caching.py`）。存的是**同一个
JSON 字符串**——编解码走 `base.CachedAnswer` 那一对 `to_json` / `from_json`，
所以「读不回来的条目按没命中处理」这类行为在内存上验证过，接到 Redis 上仍然成立。

时间可以注入：过期是缓存的核心行为之一，靠真的等一晚上去验一条 TTL 不现实。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

from ragamer.caching.base import (
    TOP_QUESTIONS,
    TTL_SECONDS,
    CachedAnswer,
    counted_question,
    game_prefix,
    hot_key,
    rank_questions,
)


class InMemoryAnswerCache:
    """内存里的答案缓存。键带上命名空间，与真实那份一样。"""

    def __init__(self, *, now: Callable[[], float] = time.monotonic) -> None:
        self._entries: dict[str, tuple[float, str]] = {}
        self._counts: dict[str, Counter[str]] = {}
        self._now = now

    def get(self, key: str) -> CachedAnswer | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at <= self._now():
            # 过期即当作没有，顺手清掉：真实那边由 Redis 自己回收
            del self._entries[key]
            return None
        return CachedAnswer.from_json(payload)

    def set(self, key: str, answer: CachedAnswer, *, ttl: int = TTL_SECONDS) -> None:
        self._entries[key] = (self._now() + ttl, answer.to_json())

    def invalidate(self, game_id: str) -> int:
        prefix = game_prefix(game_id)
        keys = [key for key in self._entries if key.startswith(prefix)]
        for key in keys:
            del self._entries[key]
        return len(keys)

    def record_question(self, game_id: str, rewritten_query: str) -> None:
        asked = counted_question(rewritten_query)
        if asked is None:
            return
        self._counts.setdefault(hot_key(game_id), Counter())[asked] += 1

    def top_questions(
        self, game_id: str, limit: int = TOP_QUESTIONS
    ) -> tuple[tuple[str, int], ...]:
        counts = self._counts.get(hot_key(game_id), Counter())
        return rank_questions(counts.most_common(limit))
