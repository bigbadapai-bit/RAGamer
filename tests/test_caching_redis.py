"""答案缓存的 Redis 实现。

键的形状、存活时间、按前缀批量删、提问计数用假客户端断言；真的连上 Redis 属于云端行为，
留给集成测试。假件与真实客户端的同名方法一一对应——**测的是这个适配器翻译得对不对**，
不是 Redis 本身。
"""

from __future__ import annotations

import fnmatch
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr
from redis.exceptions import ConnectionError as RedisConnectionError

from ragamer.caching import (
    TTL_SECONDS,
    CachedAnswer,
    CacheUnavailableError,
    RedisAnswerCache,
    cache_key,
    game_prefix,
)
from ragamer.caching import redis as caching_redis
from ragamer.config import RedisSettings
from ragamer.query import normalize_query

GAME = "black_myth"
OTHER = "wuthering_waves"
KEY = cache_key(GAME, "1.0", "二郎神怎么打")
ANSWER = CachedAnswer(text="先定身再贴身输出。", images=("images/black_myth/boss.jpg",))


class FakeRedis:
    """替代 `redis.Redis`：记下构造参数与每一次调用，不连服务。"""

    instances: ClassVar[list[FakeRedis]] = []

    def __init__(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self.init_kwargs = kwargs
        self.strings: dict[str, str] = {}
        self.expiry: dict[str, int] = {}
        self.rankings: dict[str, dict[str, float]] = {}
        self.scans: list[tuple[str, int]] = []
        #: 设上它，之后的每一次调用都按「连不上」炸
        self.failure: Exception | None = None

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> FakeRedis:
        client = cls(url, **kwargs)
        cls.instances.append(client)
        return client

    def get(self, key: str) -> str | None:
        self._check()
        return self.strings.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._check()
        self.strings[key] = value
        self.expiry[key] = ttl

    def scan_iter(self, match: str, count: int):
        self._check()
        self.scans.append((match, count))
        return iter(sorted(name for name in self.strings if fnmatch.fnmatch(name, match)))

    def delete(self, *keys: str) -> int:
        self._check()
        removed = 0
        for key in keys:
            if key in self.strings:
                del self.strings[key]
                self.expiry.pop(key, None)
                removed += 1
            if key in self.rankings:
                del self.rankings[key]
                removed += 1
        return removed

    def zincrby(self, key: str, amount: int, member: str) -> float:
        self._check()
        ranking = self.rankings.setdefault(key, {})
        ranking[member] = ranking.get(member, 0.0) + amount
        return ranking[member]

    def zrange(self, key: str, start: int, end: int, *, desc: bool, withscores: bool):
        self._check()
        items = sorted(self.rankings.get(key, {}).items(), key=lambda item: item[1], reverse=desc)
        window = items[start : end + 1]
        # 真实的 ZREVRANGE 并列时按成员倒序；适配器应当自己重排，这里照实模拟
        return [(member, score) for member, score in window]

    def _check(self) -> None:
        if self.failure is not None:
            raise self.failure


@pytest.fixture
def redis_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeRedis]:
    FakeRedis.instances.clear()
    monkeypatch.setattr(caching_redis, "Redis", FakeRedis)
    return FakeRedis


@pytest.fixture
def cache(redis_client: type[FakeRedis], settings_env) -> RedisAnswerCache:
    settings = RedisSettings(url=SecretStr("redis://redis.test:6379/0"), prefix="ragamer-test")
    return RedisAnswerCache(settings, timeout=5.0)


def client(redis_client: type[FakeRedis]) -> FakeRedis:
    assert redis_client.instances, "适配器还没连过"
    return redis_client.instances[-1]


# --- 键与写 ---


def test_写的键带命名空间与存活时间(cache, redis_client):
    cache.set(KEY, ANSWER)

    stored = client(redis_client)
    assert f"ragamer-test:{KEY}" in stored.strings
    assert stored.expiry[f"ragamer-test:{KEY}"] == TTL_SECONDS
    assert stored.init_kwargs["decode_responses"] is True
    assert stored.init_kwargs["socket_timeout"] == 5.0


def test_取回来的是整份答案_答案与图片都在(cache, redis_client):
    """只缓存文本的话，命中之后引用与图片就没了。"""
    answer = CachedAnswer(
        text="先定身[1]。", citations=ANSWER.citations, images=("images/black_myth/boss.jpg",)
    )

    cache.set(KEY, answer)

    assert cache.get(KEY) == answer


def test_键不存在时返回空(cache, redis_client):
    assert cache.get(cache_key(GAME, "1.0", "没问过的问题")) is None


def test_存进去的问法按归一算键(cache, redis_client):
    """问法只压平空白、不动字面：首尾多打的空白与换行仍命中同一条缓存。"""
    cache.set(cache_key(GAME, "1.0", "  二郎神怎么打\n"), ANSWER)

    assert cache.get(cache_key(GAME, "1.0", "二郎神怎么打")) is not None


# --- 坏条目与失败 ---


def test_读不回来的条目按没命中处理(cache, redis_client, caplog):
    """格式改过、被人手改过、写了一半断电——都不该让一条提问栽在一个坏条目上。"""
    cache.set(KEY, ANSWER)  # 先连上，再把它改成读不回来的样子
    client(redis_client).strings[f"ragamer-test:{KEY}"] = "{这不是 JSON"

    assert cache.get(KEY) is None

    assert "读不回来" in caplog.text


def test_连不上时抛不可达(cache, redis_client):
    cache.get(KEY)  # 先连上，再让之后的每一次调用都失败
    client(redis_client).failure = RedisConnectionError("连接被拒绝")

    with pytest.raises(CacheUnavailableError) as excinfo:
        cache.get(KEY)

    assert "Redis 不可达" in str(excinfo.value)
    assert "连接被拒绝" in str(excinfo.value)


def test_连接串里的密码不进错误信息(redis_client, settings_env):
    store = RedisAnswerCache(
        RedisSettings(url=SecretStr("redis://:REDIS-PW@redis.test:6379/0"), prefix="ragamer-test"),
        timeout=5.0,
    )

    assert "REDIS-PW" not in store.address
    assert "redis.test:6379" in store.address


# --- 按前缀批量删 ---


def test_按前缀删只删这个游戏的缓存(cache, redis_client):
    cache.set(cache_key(GAME, "1.0", "二郎神怎么打"), ANSWER)
    cache.set(cache_key(GAME, "2.0", "二郎神怎么打"), ANSWER)
    cache.set(cache_key(OTHER, "1.0", "今汐怎么养"), ANSWER)
    cache.record_question(GAME, "二郎神怎么打")

    deleted = cache.invalidate(GAME)

    stored = client(redis_client)
    assert deleted == 2
    assert stored.scans[-1][0] == f"ragamer-test:{game_prefix(GAME)}*"
    assert not [name for name in stored.strings if GAME in name]
    assert [name for name in stored.strings if OTHER in name]
    # 提问计数的 ZSET 不在缓存前缀里，一次导入不该把热门问题一起抹掉
    assert stored.rankings


def test_前缀带冒号_游戏_a_不牵连_ab(cache, redis_client):
    """少了那个冒号，一次导入会把另一款游戏的缓存清空，而且不报错。"""
    cache.set(cache_key("a", "1.0", "问题"), ANSWER)
    cache.set(cache_key("ab", "1.0", "问题"), ANSWER)

    assert cache.invalidate("a") == 1

    assert cache.get(cache_key("ab", "1.0", "问题")) is not None


def test_分批发删除(cache, redis_client, monkeypatch):
    """条数超过一批时也要删干净，且批次之间不留尾巴。"""
    monkeypatch.setattr(caching_redis, "DELETE_BATCH", 2)
    for index in range(5):
        cache.set(cache_key(GAME, "1.0", f"问题{index}"), ANSWER)

    assert cache.invalidate(GAME) == 5

    assert client(redis_client).strings == {}


# --- 提问计数与热门问题 ---


def test_热门问法按次数排序(cache, redis_client):
    for _ in range(3):
        cache.record_question(GAME, "二郎神怎么打")
    for _ in range(2):
        cache.record_question(GAME, " 二郎神在哪 ")
    cache.record_question(GAME, "二郎神掉什么")

    assert cache.top_questions(GAME) == (
        ("二郎神怎么打", 3),
        ("二郎神在哪", 2),
        ("二郎神掉什么", 1),
    )


def test_次数并列时按问法字典序(cache, redis_client):
    """并列的次序由存储定（Redis 是倒序），这一层自己重排，顺序才不随后端变。"""
    cache.record_question(GAME, "问题二")
    cache.record_question(GAME, "问题一")

    assert cache.top_questions(GAME) == (("问题一", 1), ("问题二", 1))


def test_计数写在另一个前缀下(cache, redis_client):
    cache.record_question(GAME, "二郎神怎么打")

    stored = client(redis_client)
    assert list(stored.rankings) == [f"ragamer-test:hot:{GAME}"]
    assert normalize_query("二郎神怎么打") in stored.rankings[f"ragamer-test:hot:{GAME}"]


def test_空问法不计数(cache, redis_client):
    cache.get(KEY)  # 先连上：要断言的是「什么都没写」，不是「还没连」

    cache.record_question(GAME, "   ")

    assert client(redis_client).rankings == {}
