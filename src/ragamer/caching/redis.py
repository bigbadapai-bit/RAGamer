"""答案缓存的 Redis 实现（docs/ARCHITECTURE.md §4）。

后端在云端、多用户共用一套 Redis，所以两件事是必须的：

- **键前缀**。本项目与原项目共用实例时，前缀之外没有任何东西把两边的键隔开——
  而按前缀批量失效**是真的在删键**，撞上了就是删别人的数据。
- **按前缀删用 `SCAN` 而不是 `KEYS`**。`KEYS` 会阻塞整个实例，共用一套 Redis 时
  这个代价落在别人的请求上。

构造不连服务，第一次操作才连：缓存不通不该拦住进程起来（`ragamer.container` 的自检
里没有它，理由见那里）。连接失败与被拒一律翻成 `CacheUnavailableError`，
业务调用兜住即降级——**缓存不可用时作答照常**。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from redis import Redis
from redis.exceptions import RedisError

from ragamer.caching.base import (
    TOP_QUESTIONS,
    TTL_SECONDS,
    CachedAnswer,
    counted_question,
    game_prefix,
    hot_key,
    rank_questions,
    unavailable,
)
from ragamer.config import RedisSettings
from ragamer.logging import get_logger
from ragamer.redaction import redact_address

logger = get_logger(__name__)

#: 服务名，用于出错信息。
REDIS = "Redis"

#: 一次 `SCAN` 建议扫多少条。**只是建议**：Redis 只保证不重复返回、不保证一次给够，
#: 所以删的时候不能按它切批次之后就假定扫完了。
SCAN_COUNT = 500

#: 一次 `DEL` 删多少条。攒够一批再删，少发几个往返。
DELETE_BATCH = 500

#: 连不上时 redis-py 抛的就是这些。
_FAILURES = (RedisError, OSError)

T = TypeVar("T")


class RedisAnswerCache:
    """Redis 上的答案缓存。构造不连服务，第一次操作才连。"""

    def __init__(self, settings: RedisSettings, *, timeout: float) -> None:
        self.name = REDIS
        self.address = redact_address(settings.url.get_secret_value())
        self._url = settings.url.get_secret_value()
        self._prefix = settings.prefix
        self._timeout = timeout
        self._client: Redis | None = None

    def get(self, key: str) -> CachedAnswer | None:
        payload = self._run(lambda client: client.get(self._name(key)))
        if payload is None:
            return None
        answer = CachedAnswer.from_json(payload)
        if answer is None:
            # 格式改过、被人手改过、写了一半断电——都读不回来。当作没命中，
            # 之后照常写回新的一条。让它抛出去会让一条提问栽在一个坏条目上。
            logger.warning("缓存 %s 读不回来，按没命中处理", self._name(key))
        return answer

    def set(self, key: str, answer: CachedAnswer, *, ttl: int = TTL_SECONDS) -> None:
        self._run(lambda client: client.setex(self._name(key), ttl, answer.to_json()))

    def invalidate(self, game_id: str) -> int:
        """按前缀批量删。`SCAN` 游标扫，边扫边按批删，返回删掉的条数。

        提问计数那个 ZSET 不在这里的前缀里（`base.HOT_ROOT` 是另一个根），
        所以一次导入不会把「大家都在问什么」一起抹掉。删库那一路见 `drop`。
        """
        return self._run(lambda client: self._sweep(client, game_prefix(game_id)))

    def drop(self, game_id: str) -> int:
        """答案连提问计数一起删。**只在删库时调**，理由见协议里的说明。"""
        return self._run(
            lambda client: (
                self._sweep(client, game_prefix(game_id))
                + client.delete(self._name(hot_key(game_id)))
            )
        )

    def _sweep(self, client: Redis, prefix: str) -> int:
        """按前缀扫着删。`SCAN` 游标扫，边扫边按批删，返回删掉的条数。

        用 `SCAN` 而不是 `KEYS`：后者会把整个实例阻塞住，而缓存里可能就是几万条。
        """
        pattern = f"{self._name(prefix)}*"
        deleted = 0
        batch: list[str] = []
        for key in client.scan_iter(match=pattern, count=SCAN_COUNT):
            batch.append(key)
            if len(batch) >= DELETE_BATCH:
                deleted += client.delete(*batch)
                batch.clear()
        if batch:
            deleted += client.delete(*batch)
        return deleted

    def record_question(self, game_id: str, rewritten_query: str) -> None:
        asked = counted_question(rewritten_query)
        if asked is None:
            return
        self._run(lambda client: client.zincrby(self._name(hot_key(game_id)), 1, asked))

    def top_questions(
        self, game_id: str, limit: int = TOP_QUESTIONS
    ) -> tuple[tuple[str, int], ...]:
        """问得最多的问法：先按分数取前 `limit` 条，再按共用的口径重排一遍
        （并列时 Redis 给的是成员倒序，而顺序是给人看的）。"""
        pairs = self._run(
            lambda client: client.zrange(
                self._name(hot_key(game_id)), 0, limit - 1, desc=True, withscores=True
            )
        )
        return rank_questions([(str(member), int(score)) for member, score in pairs])

    def _name(self, key: str) -> str:
        """带上命名空间的键。读写与按前缀删都过这里，三者不会各拼一遍。"""
        return f"{self._prefix}:{key}"

    def _run(self, operation: Callable[[Redis], T]) -> T:
        try:
            return operation(self._connect())
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def _connect(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(
                self._url,
                # 远端不可达时要在这一点时间内失败，而不是挂在一次提问上
                socket_timeout=self._timeout,
                socket_connect_timeout=self._timeout,
                # 键与成员都是文本：解码在这里做一次，取回来就是 str
                decode_responses=True,
            )
        return self._client
