"""联网兜底：时效型问题去外部检索。

本地语料是**导进来那一刻**的快照，网游的版本更新、活动公告天然滞后于现实。这一路补的
就是这个——「这版本改了什么」这类问题，语料里根本没有答案可依据。

四件事在这里定死：

- **只取搜索服务给的摘要，不抓页面。** 抓正文是导入侧那套爬虫的活（robots.txt、编码、
  正文提取都在那里），检索期再抓一遍等于把那套逻辑复制一份，还会把一次提问变成 N 个
  外部请求——一路兜底不该拖着整条问答等它。搜索服务返回的 `summary` 本来就是为喂给
  模型准备的。
- **结果带着来源地址回来**。这不是锦上添花：网络内容与本地语料混在一份答案里时，
  读的人必须分得清哪句是知识库里查到的、哪句是网上搜来的（`ragamer.answering` 那边
  会把它标出来）。
- **失败分得开**。密钥无效、余额不足这类失败**重试无用**，之后每一次提问都会栽在这里，
  按 :class:`WebSearchRejected` 报出来；网络抖动、5xx 按 :class:`WebSearchUnavailable`。
  两者的处置不同——前者要去改配置，后者等下一次提问自己就好了。
- **没配就是不配**。`ragamer.container` 在没密钥时根本不构造这一路（`None`），
  调用方据此跳过它——而不是造一个返回空结果的实现，那种东西会让「没配」与
  「搜了但没有结果」在日志与答案里长得一模一样。

适配器与业务层之间只有 :class:`WebSearch` 协议这一层：换一家搜索服务是加一个适配器，
改组合根一行，调用方不动。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from ragamer.config import SearchSettings
from ragamer.logging import get_logger

logger = get_logger(__name__)

#: 博查返回结果的位置：`data.webPages.value[]`。写死在这里是因为换别家就是换适配器。
_RESULTS_PATH = ("data", "webPages", "value")

#: HTTP 状态码 → 这次失败算哪一类。401／403 是密钥或余额的问题，重试无用。
_REJECTED = {401, 403}

#: 博查一次最多收 50 条。超出去会被服务端拒掉整条请求，而调用方那边看只是「搜失败了」——
#: 所以在这里收口，别把「参数给大了」说成「服务不好使」。
MAX_RESULTS = 50


@dataclass(frozen=True, slots=True)
class WebResult:
    """一条外部检索结果。

    `text` 是**搜索服务给的摘要**，不是我们自己抓的正文——理由见模块说明。
    它可能为空（有的页面没有摘要），那种结果照样交出去：标题与地址本身还算一条线索，
    而且要不要用它由生成那一步判断。
    """

    title: str
    url: str
    text: str = ""
    #: 发布时间，服务给什么就是什么（原样字符串）。时效型问题里这一栏有额外价值：
    #: 答案说得出「这条是什么时候的」才算真的回答了时效。
    published_at: str = ""


class WebSearchError(Exception):
    """外部检索失败。调用方按这一路失败处理：记下来，用其余路继续作答。"""


class WebSearchRejected(WebSearchError):
    """被服务拒绝：密钥无效、余额不足。**重试无用**，之后每一次都会这样。

    与 :class:`WebSearchUnavailable` 分开，是因为处置方式相反——这个要去改配置，
    那个等下一次提问自己就好了。按 ERROR 报出来，别让它淹没在 WARNING 里。
    """


class WebSearchUnavailable(WebSearchError):
    """服务不可达、超时、5xx、被限流。换一次提问多半就好了。"""


@runtime_checkable
class WebSearch(Protocol):
    """外部检索服务的协议。业务层只认它。"""

    def search(self, query: str, *, limit: int) -> list[WebResult]:
        """搜一次，按服务的相关性排序返回至多 `limit` 条。

        :raises WebSearchRejected: 被服务拒绝（密钥、余额）。重试无用。
        :raises WebSearchUnavailable: 这次没成，下次可能就好。
        """
        ...


class BochaWebSearch:
    """博查（Bocha）Web Search 接口。地址、密钥、条数、超时全部来自配置。

    接口形状取自博查的 Web Search 文档：`POST` 一个 JSON，`Authorization: Bearer`，
    正文收 `{query, count, summary, freshness}`，结果在 `data.webPages.value[]` 里，
    每条的字段名是 `name` / `url` / `summary` / `snippet` / `datePublished`。

    两处按文档明确的注意事项办：

    - **开 `summary`**：不开就只有一两行的 `snippet`，那是给搜索结果页看的，喂给模型
      太薄。`summary` 才是长摘要。
    - **`freshness` 用 `noLimit`**：自己限定时间范围会让「范围内无结果」变成常态，
      而这一路问的正是时效——搜不到比搜到旧的更糟。要收紧时间也是问答侧的事。
    """

    def __init__(self, config: SearchSettings, *, client: httpx.Client | None = None) -> None:
        if config.api_key is None:
            # 组合根据此不构造这一路。真走到这里说明接线错了，当场报出来
            raise ValueError("没有配检索服务的密钥，不该构造这一路")
        self._config = config
        self._client = client if client is not None else httpx.Client()
        self._endpoint = config.base_url
        self._headers = {
            "Authorization": f"Bearer {config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def search(self, query: str, *, limit: int) -> list[WebResult]:
        """:raises WebSearchRejected: 密钥或余额的问题。
        :raises WebSearchUnavailable: 这次没成。
        """
        payload = {
            "query": query,
            "count": _count(limit),
            "summary": True,
            "freshness": "noLimit",
        }
        try:
            response = self._client.post(
                self._endpoint,
                json=payload,
                headers=self._headers,
                timeout=self._config.timeout,
            )
        except httpx.HTTPError as exc:
            raise WebSearchUnavailable(f"检索服务不可达：{exc}") from exc
        if response.status_code in _REJECTED:
            raise WebSearchRejected(
                f"检索服务拒绝了请求（HTTP {response.status_code}）：{_one_line(response.text)}"
            )
        if response.status_code >= 400:
            raise WebSearchUnavailable(
                f"检索服务返回 HTTP {response.status_code}：{_one_line(response.text)}"
            )
        return _results(response.json())

    def __repr__(self) -> str:
        """地址可以落日志，密钥不行——把密钥挡在日志之外。"""
        return f"BochaWebSearch({self._endpoint!r})"


class FakeWebSearch:
    """按脚本回结果的假件：一次网络都不发。

    与 `ragamer.llm.FakeLlm` 同一个用法——脚本排完就炸，而不是静默返回空列表：
    静默返回空会让「这一路没被调用」与「调用了但没有结果」分不开。
    """

    def __init__(self, *replies: Sequence[WebResult] | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int) -> list[WebResult]:
        self.calls.append((query, limit))
        if not self._replies:
            raise AssertionError(f"假件没排到这里：{query!r}。排几条 `WebResult`，或排一个异常")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return list(reply)


def _count(limit: int) -> int:
    """要几条。至少 1 条——要 0 条等于发一次请求白花钱；上限见 :data:`MAX_RESULTS`。"""
    return max(1, min(limit, MAX_RESULTS))


def _results(payload: Any) -> list[WebResult]:
    """响应体 → 结果。

    形状不对时返回空列表而不是抛异常：那是服务端换了版本，不是这一路该失败的理由——
    抛出去会被当成「这次搜失败了」，而实际发生的是「搜成功了但一条也没解析出来」，
    后者要修的是适配器。留一条 warning，别静默。
    """
    if not isinstance(payload, dict):
        logger.warning("检索服务返回的不是一个对象：%s", type(payload).__name__)
        return []
    value: Any = payload
    for key in _RESULTS_PATH:
        value = value.get(key) if isinstance(value, dict) else None
    if not isinstance(value, list):
        logger.warning("检索服务的响应里没有 %s：得改的是适配器", ".".join(_RESULTS_PATH))
        return []
    return [_result(item) for item in value if isinstance(item, dict)]


def _result(item: dict[str, Any]) -> WebResult:
    """一条结果。摘要取 `summary`，没有就退到 `snippet`；两个字段都缺就是空串。

    发布时间取 `datePublished` **不取 `dateLastCrawled`**：博查文档写明后者的 `Z`
    结尾实际是 UTC+8（历史命名问题），照 ISO 读会差 8 小时，而这一路恰恰是回答
    「什么时候的事」的。
    """
    return WebResult(
        title=str(item.get("name") or ""),
        url=str(item.get("url") or ""),
        text=str(item.get("summary") or item.get("snippet") or ""),
        published_at=str(item.get("datePublished") or ""),
    )


def _one_line(text: str, limit: int = 200) -> str:
    """错误信息里带上服务端的原话，但要压成一行——响应体可能是一整页 HTML。"""
    return " ".join(text.split())[:limit]
