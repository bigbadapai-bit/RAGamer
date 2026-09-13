"""答案缓存的共同约定：键、值、协议与错误（docs/ARCHITECTURE.md §4）。

**key 的设计是全部要点所在**：

```
{前缀}:cache:{游戏}:{版本}:{sha1(改写后的问题)}
```

- **复用改写后的问题做归一化**（`ragamer.query.understand`）。它已经把口语问法归一了
  （「那它怎么走」→「二郎神怎么走」），所以精确匹配就有很高的命中率——游戏攻略的
  查询分布极度倾斜，热门问法是同一句话。多一层向量相似度就是多一层要标定的阈值，
  v1 不做语义缓存（规格里明确列在 Out of Scope）。
- **版本进 key**：换版本天然不命中，不需要主动失效（ADR-0004）。
- **游戏进 key**：一个游戏一个库（ADR-0002），按前缀批量失效照游戏这一段匹配。

**缓存的是整份结果，不是一段文本**：只存文本的话，命中之后引用与图片都没了——
那条路径会静默地比未命中时少东西，而界面上看不出区别。值因此是
:class:`CachedAnswer`：答案 + 引用 + 图片地址。

真实实现（`ragamer.caching.redis`）与内存假件（`ragamer.caching.memory`）实现同一组
协议，编解码共用这里的 `to_json` / `from_json`——删库、失效、过期这些行为在内存上
跑过的，接到 Redis 上仍然成立。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ragamer.answering import Citation
from ragamer.query import normalize_query

#: 缓存键的根。提问计数那个 ZSET 另起一个根，**刻意不落在缓存前缀里**：
#: 它记的是提问频次，不是某条答案，按前缀批量失效不该把它一起抹掉。
CACHE_ROOT = "cache"
HOT_ROOT = "hot"

#: 缓存的存活时间（秒）。长过期是兜底而不是主要失效手段——导入新资料时按游戏前缀
#: 批量删（见 `ragamer.importing`），版本变了 key 自然不同。7 天是「足够热的问题一直
#: 命中，冷下来的条目自己走掉」的一个占位取值，没有评测集能证明它最优（§11）。
TTL_SECONDS = 7 * 24 * 60 * 60

#: 一个 ZSET 里最多记多少条热门问法。计数本身不设上限会长到没人看，
#: 取前 N 条才是这个功能的产出。
TOP_QUESTIONS = 10


class CacheError(Exception):
    """缓存相关的失败。业务调用按这个基类兜，兜住即降级（见 `ragamer.caching.answer`）。"""


class CacheUnavailableError(CacheError):
    """缓存服务连不上：超时、地址不通、凭据被拒。

    与 `ragamer.stores.base.StoreUnavailableError` 分开，是因为后果不同：存储不通是
    「做不了这件事」，缓存不通只是「这次没缓存可用」，**作答照常**。所以它不在启动自检里，
    也不该让任何一条提问失败。
    """

    def __init__(self, service: str, address: str, timeout: float, reason: str) -> None:
        self.service = service
        self.address = address
        self.timeout = timeout
        self.reason = reason
        super().__init__(f"{service} 不可达（地址 {address}，超时 {timeout:g} 秒）：{reason}")


def unavailable(
    service: str, address: str, timeout: float, exc: Exception
) -> CacheUnavailableError:
    """把一个连接失败翻成 :class:`CacheUnavailableError`。措辞与存储那侧一致。"""
    return CacheUnavailableError(service, address, timeout, str(exc) or type(exc).__name__)


def cache_key(game_id: str, version: str, rewritten_query: str) -> str:
    """一次提问的缓存键。**纯函数**：同一个问题、同一个游戏、同一个版本必然同一个键。

    问法按 `ragamer.query.normalize_query` 归一后再算摘要——只压平空白，不动字面。
    「怎么走」与「走法」合并成一件事属于语义缓存，要先标定阈值，v1 不做。

    游戏与版本**原样进键**而不是掺进摘要：按前缀批量失效要能只靠游戏那一段匹配，
    而人排查问题时也该一眼看出这条缓存属于谁。
    """
    digest = hashlib.sha1(normalize_query(rewritten_query).encode("utf-8")).hexdigest()
    return f"{CACHE_ROOT}:{game_id}:{version}:{digest}"


def game_prefix(game_id: str) -> str:
    """该游戏的缓存键前缀。**按前缀批量失效照它匹配**（`invalidate`）。

    末尾那个冒号是必须的：少了它，游戏 `a` 的前缀会连 `ab` 的键一起匹配上，
    一次导入就把另一款游戏的缓存清空——而且不报错。游戏 id 本身不含冒号
    （`ragamer.stores.base.collection_name` 只放行标识符），所以这一层隔得开。
    """
    return f"{CACHE_ROOT}:{game_id}:"


def hot_key(game_id: str) -> str:
    """该游戏提问计数的 ZSET 键。成员是归一后的问法，分值是次数。"""
    return f"{HOT_ROOT}:{game_id}"


@dataclass(frozen=True)
class CachedAnswer:
    """缓存里的一条答案。**整份结果**，不是一段文本。

    引用与图片都跟着答案走：命中之后要能把它们一并交回去，否则缓存路径就比未命中时
    少东西——用户看到的是同一段答案，却点不开原图、也核不了来源。
    """

    text: str
    citations: tuple[Citation, ...] = ()
    #: 这条答案引到的图片地址，按首次出现的顺序去重。
    images: tuple[str, ...] = ()

    @property
    def not_found(self) -> bool:
        """没有引用即「没检索到内容」（与 `ragamer.answering.NOT_FOUND` 同一件事）。
        **这种结果不入缓存**：它随时会因为新资料而改变，存下来等于把一次「暂时没有」
        钉成 7 天的「没有」。"""
        return not self.citations

    def to_json(self) -> str:
        """落进 Redis 的形态。字段名写全，不依赖位置——将来加字段时旧条目仍读得回来。"""
        return json.dumps(
            {
                "text": self.text,
                "citations": [
                    {
                        "index": citation.index,
                        "doc_title": citation.doc_title,
                        "ancestor_path": citation.ancestor_path,
                    }
                    for citation in self.citations
                ],
                "images": list(self.images),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(payload: str) -> CachedAnswer | None:
        """从 Redis 读回来的形态。**读不回来返回 `None`**，由调用方按没命中处理。

        缓存里出现读不懂的东西不该让一条提问失败：格式改过、被人手改过、写了一半
        断电，都可能。返回 `None` 的代价只是这次没命中，之后照常写回新的一条。
        """
        try:
            data = json.loads(payload)
            return CachedAnswer(
                text=str(data["text"]),
                citations=tuple(_citation(item) for item in data["citations"]),
                images=tuple(str(url) for url in data["images"]),
            )
        except (ValueError, TypeError, KeyError):
            return None


def _citation(item: object) -> Citation:
    """一条引用。字段对不上时抛 `KeyError` / `TypeError`，由上面兜成「读不回来」。"""
    if not isinstance(item, dict):
        raise TypeError(f"引用不是对象：{item!r}")
    return Citation(
        index=int(item["index"]),
        doc_title=str(item["doc_title"]),
        ancestor_path=str(item["ancestor_path"]),
    )


@runtime_checkable
class AnswerCache(Protocol):
    """答案缓存的全部对外能力（架构文档模块 10）。

    接口刻意做小：取、存、按游戏批量失效、记一次提问、列热门问法。**没有删除单条**——
    单条过期由 TTL 管，要动就是「这个游戏的语料变了」，那是前缀批量失效那一条。
    """

    def get(self, key: str) -> CachedAnswer | None:
        """取一条缓存；没有、过期、或读不回来时返回 `None`。"""
        ...

    def set(self, key: str, answer: CachedAnswer, *, ttl: int = TTL_SECONDS) -> None:
        """写一条缓存，带存活时间。按 key 覆盖。"""
        ...

    def invalidate(self, game_id: str) -> int:
        """按前缀批量删掉这个游戏的缓存，返回删掉的条数。

        导入完成时调它（`ragamer.importing`）：语料变了，基于旧语料的答案不该再命中。
        提问计数的 ZSET **不在这个前缀里**，不受影响。
        """
        ...

    def record_question(self, game_id: str, rewritten_query: str) -> None:
        """给这条问法计一次数，产出热门问题用。**命中与否都要记**——
        热门问题问的是「大家在问什么」，不是「什么被缓存了」。"""
        ...

    def top_questions(
        self, game_id: str, limit: int = TOP_QUESTIONS
    ) -> tuple[tuple[str, int], ...]:
        """问得最多的问法，次数多的在前。次数相同按问法字典序，顺序才是确定的。"""
        ...
