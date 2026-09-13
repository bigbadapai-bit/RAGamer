"""联网兜底：博查接口的请求形状、响应解析与失败分类。

一次网络都不发——传输层换成 `httpx.MockTransport`，请求与解析走的仍是真实那条路径，
所以「发出去的 JSON 长什么样」「回来的 JSON 怎么读」这两件事是真的被验过的。

这一路最容易出的错是**解析不出东西却不报错**：搜索服务改了响应结构，代码返回空列表，
对外看起来与「确实没搜到」一模一样。所以形状不对时留痕那一条也要断。
密钥与余额被拒另算一类——那之后每一次都会失败，处置方式与「这次没连上」相反。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx
import pytest

from ragamer.config import SearchSettings
from ragamer.websearch import (
    BochaWebSearch,
    FakeWebSearch,
    WebResult,
    WebSearchRejected,
    WebSearchUnavailable,
)

QUERY = "这版本改了什么"

#: 一条正常结果的响应体，形状照博查文档。
PAYLOAD = {
    "code": 200,
    "log_id": "abc",
    "data": {
        "webPages": {
            "value": [
                {
                    "name": "1.1 版本更新公告",
                    "url": "https://example.com/patch",
                    "snippet": "短摘要",
                    "summary": "金箍棒的基础伤害下调，新增两件套装。",
                    "datePublished": "2026-01-02T10:00:00+08:00",
                }
            ]
        }
    },
}


def settings(**overrides: object) -> SearchSettings:
    return SearchSettings(api_key="test-key", **overrides)


def searching(handler: Callable[[httpx.Request], httpx.Response]) -> BochaWebSearch:
    """把传输层换成假件：请求照发，只是不发出去。"""
    return BochaWebSearch(settings(), client=httpx.Client(transport=httpx.MockTransport(handler)))


def replying(response: httpx.Response) -> BochaWebSearch:
    return searching(lambda request: response)


def ok(**overrides: object) -> BochaWebSearch:
    return replying(httpx.Response(200, json={**PAYLOAD, **overrides}))


# --- 请求形状 ---


def test_按博查的形状发请求():
    """地址、鉴权头、正文三个字段一处不对就是查不出东西，而它多半不报错。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=PAYLOAD)

    BochaWebSearch(settings(), client=httpx.Client(transport=httpx.MockTransport(handler))).search(
        QUERY, limit=5
    )

    assert seen["url"] == "https://api.bocha.cn/v1/web-search"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == {
        "query": QUERY,
        "count": 5,
        "summary": True,
        "freshness": "noLimit",
    }


def test_开着长摘要():
    """不开 `summary` 只有一两行的 `snippet`，那是给搜索结果页看的，喂给模型太薄。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=PAYLOAD)

    BochaWebSearch(settings(), client=httpx.Client(transport=httpx.MockTransport(handler))).search(
        QUERY, limit=5
    )

    assert seen["summary"] is True


def test_不自己限时间范围():
    """自己限范围会让「范围内无结果」变成常态，而这一路问的正是时效——
    搜不到比搜到旧的更糟。要收紧时间也是问答侧的事。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=PAYLOAD)

    BochaWebSearch(settings(), client=httpx.Client(transport=httpx.MockTransport(handler))).search(
        QUERY, limit=5
    )

    assert seen["freshness"] == "noLimit"


def test_地址可配():
    """换别家就是改这一行加一个适配器。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=PAYLOAD)

    search = BochaWebSearch(
        settings(base_url="https://search.test/v1/web-search"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    search.search(QUERY, limit=5)

    assert seen["url"] == "https://search.test/v1/web-search"


# --- 响应解析 ---


def test_读得出标题地址与摘要():
    assert ok().search(QUERY, limit=5) == [
        WebResult(
            title="1.1 版本更新公告",
            url="https://example.com/patch",
            text="金箍棒的基础伤害下调，新增两件套装。",
            published_at="2026-01-02T10:00:00+08:00",
        )
    ]


def test_没有长摘要时退到短摘要():
    """长摘要是开了 `summary` 才有的，个别页面本来就没有——退到 `snippet` 总比空着强。"""
    payload = json.loads(json.dumps(PAYLOAD))
    del payload["data"]["webPages"]["value"][0]["summary"]

    assert replying(httpx.Response(200, json=payload)).search(QUERY, limit=5) == [
        WebResult(
            title="1.1 版本更新公告",
            url="https://example.com/patch",
            text="短摘要",
            published_at="2026-01-02T10:00:00+08:00",
        )
    ]


def test_发布时间取_datePublished_不取_dateLastCrawled():
    """博查文档写明 `dateLastCrawled` 的 `Z` 结尾实际是 UTC+8（历史命名问题），
    照 ISO 读会差 8 小时——而这一路回答的正是「这是什么时候的事」。"""
    payload = json.loads(json.dumps(PAYLOAD))
    page = payload["data"]["webPages"]["value"][0]
    page["dateLastCrawled"] = "2026-01-02T18:00:00Z"

    assert replying(httpx.Response(200, json=payload)).search(QUERY, limit=5)[0].published_at == (
        "2026-01-02T10:00:00+08:00"
    )


def test_一条结果都没有时是空列表():
    payload = {"code": 200, "data": {"webPages": {"value": []}}}

    assert replying(httpx.Response(200, json=payload)).search(QUERY, limit=5) == []


def test_没有地址的结果丢掉():
    """地址既是引用必须给出来的东西，也是「这是网络来源」的判据
    （`Citation.origin`）——留一条没有地址的进来，它在引用里与语料里查到的切片
    长得一模一样，而那正是「区分标注」要防的。"""
    payload = json.loads(json.dumps(PAYLOAD))
    payload["data"]["webPages"]["value"].append({"name": "没有地址的一条", "summary": "正文"})

    found = replying(httpx.Response(200, json=payload)).search(QUERY, limit=5)

    assert [result.title for result in found] == ["1.1 版本更新公告"]


def test_响应形状不对时返回空并留痕(caplog):
    """搜索服务改了响应结构：这不是「这次搜失败了」，是「搜成功了但一条也没解析出来」，
    要改的是适配器。留一条痕，别让它静默成「确实没搜到」。"""
    with caplog.at_level(logging.WARNING, logger="ragamer.websearch"):
        found = replying(httpx.Response(200, json={"code": 200, "data": {}})).search(QUERY, limit=5)

    assert found == []
    assert any("适配器" in record.getMessage() for record in caplog.records)


# --- 失败分类 ---


@pytest.mark.parametrize("status", [401, 403])
def test_密钥或余额的问题按被拒报出来(status):
    """重试无用，之后每一次都会这样——与「这次没连上」的处置方式相反。"""
    with pytest.raises(WebSearchRejected) as caught:
        replying(httpx.Response(status, text="Invalid API KEY")).search(QUERY, limit=5)

    assert str(status) in str(caught.value)


@pytest.mark.parametrize("status", [429, 500, 502])
def test_限流与服务端出错按这次没成报出来(status):
    with pytest.raises(WebSearchUnavailable):
        replying(httpx.Response(status, text="busy")).search(QUERY, limit=5)


def test_连不上按这次没成报出来():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("连接被拒绝")

    with pytest.raises(WebSearchUnavailable):
        BochaWebSearch(
            settings(), client=httpx.Client(transport=httpx.MockTransport(handler))
        ).search(QUERY, limit=5)


def test_错误信息里带上服务端的原话但压成一行():
    with pytest.raises(WebSearchRejected) as caught:
        replying(httpx.Response(401, text="Invalid\nAPI   KEY")).search(QUERY, limit=5)

    assert "Invalid API KEY" in str(caught.value)


# --- 构造 ---


def test_没配密钥时不构造这一路():
    """组合根据此决定「没有这一路」（`None`）。真构造了说明接线错了，当场报出来。"""
    with pytest.raises(ValueError):
        BochaWebSearch(SearchSettings(api_key=None))


def test_地址可以落日志密钥不行():
    """密钥进日志是事故；地址要能落，排查时才看得见查的是哪儿。"""
    text = repr(ok())

    assert "api.bocha.cn" in text
    assert "test-key" not in text


# --- 假件 ---


def test_假件按脚本回结果并记下每次调用():
    search = FakeWebSearch([WebResult("标题", "https://example.com/p", "正文")])

    found = search.search(QUERY, limit=5)

    assert found == [WebResult("标题", "https://example.com/p", "正文")]
    assert search.calls == [(QUERY, 5)]


def test_假件脚本排空后当场炸():
    """静默返回空列表会让「这一路没被调用」与「调用了但没有结果」分不开。"""
    with pytest.raises(AssertionError):
        FakeWebSearch().search(QUERY, limit=5)
