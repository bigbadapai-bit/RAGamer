"""语言模型适配器：结构化联合输出、流式逐字、三类失败可辨、重试有界、不碰真模型。

真实客户端用 `httpx.MockTransport` 打 —— 走的是完整的请求构造、状态码翻译与
SSE 解析代码路径，只是不经过网络。假件则单独验它自己那套确定性行为。
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from pydantic import BaseModel

from ragamer.config import LlmSettings, Settings
from ragamer.llm import (
    FakeLlm,
    ImagePart,
    LlmClient,
    LlmError,
    LlmInvalidOutput,
    LlmRateLimited,
    LlmRejected,
    LlmRequest,
    LlmTimeout,
    LlmTruncated,
    LlmUnavailable,
    Message,
    OpenAiLlm,
    RetryPolicy,
)


class 联合输出(BaseModel):
    """打标兜底与查询路由共用的那种一次多字段的结构化返回。"""

    subject_name: str
    rewritten_query: str
    routes: list[str]


@pytest.fixture
def llm_config(settings_env) -> LlmSettings:
    """完整配置里的语言模型那一组。"""
    return Settings().llm


def _request(question: str = "二郎神怎么打") -> LlmRequest:
    return LlmRequest(
        messages=[Message("system", "你是游戏攻略助手"), Message("user", question)],
    )


def _llm(handler, config: LlmSettings, *, sleeps: list[float] | None = None) -> OpenAiLlm:
    """接上假传输层的真实客户端。`sleeps` 传列表时记录退避，不真等。"""
    return OpenAiLlm(
        config,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps.append if sleeps is not None else (lambda _: None)),
    )


def _reply(text: str, finish_reason: str = "stop") -> dict:
    return {
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


def _event(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _sse(*pieces: str, finish_reason: str = "stop") -> list[bytes]:
    """一段 SSE 响应体：每片一个事件，末尾带结束原因与 `[DONE]`。"""
    events = [_event({"choices": [{"delta": {"content": piece}}]}) for piece in pieces]
    events.append(_event({"choices": [{"delta": {}, "finish_reason": finish_reason}]}))
    events.append(b"data: [DONE]\n\n")
    return events


def _messages_of(payload: dict) -> str:
    """把一次请求的全部消息拼起来，供断言提示里写了什么。"""
    return "\n".join(message["content"] for message in payload["messages"])


def _streaming(*pieces: str, finish_reason: str = "stop"):
    """假传输层：每次请求都回一段给定内容的 SSE。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(*pieces, finish_reason=finish_reason))

    return handler


def _body(*events: bytes):
    """假传输层：每次请求都把给定的字节按顺序吐出去。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=list(events))

    return handler


def _always_429(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, json={"error": "too many requests"})


def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("读超时", request=request)


def _always_truncated(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_reply("半截答案", finish_reason="length"))


def _unauthorized(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"error": {"message": "invalid api key"}})


# ── 结构化联合输出 ──


def test_一次调用拿回多个字段(llm_config):
    """打标兜底、查询路由都要一次调用吐出多个字段，而不是每个字段调一次。"""
    body = '{"subject_name":"二郎神","rewritten_query":"二郎神 打法","routes":["main","table"]}'
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_reply(body))

    result = _llm(handler, llm_config).complete_structured(_request(), 联合输出)

    assert result.subject_name == "二郎神"
    assert result.rewritten_query == "二郎神 打法"
    assert result.routes == ["main", "table"]
    assert len(calls) == 1


def test_结构化调用把结构与校验要求写进了提示(llm_config):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json=_reply('{"subject_name":"a","rewritten_query":"b","routes":[]}')
        )

    _llm(handler, llm_config).complete_structured(_request(), 联合输出)

    prompt = _messages_of(seen)
    assert seen["response_format"] == {"type": "json_object"}
    for field in 联合输出.model_fields:
        assert field in prompt
    assert "JSON Schema" in prompt
    # 调用方的系统提示还在，结构要求是并进去的，不是顶掉
    assert "你是游戏攻略助手" in prompt


def test_返回不合规时识别并重试(llm_config):
    """返回的不是 JSON 就得认出来，并把毛病带回去让模型改。"""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(200, json=_reply("好的，我看看：二郎神"))
        return httpx.Response(
            200,
            json=_reply('{"subject_name":"二郎神","rewritten_query":"二郎神 打法","routes":[]}'),
        )

    result = _llm(handler, llm_config).complete_structured(_request(), 联合输出)

    assert result.subject_name == "二郎神"
    assert len(payloads) == 2
    # 温度 0 时原样重发只会再错一次，所以第二次要带上上一次哪里不合规
    assert "上一次的返回不符合要求" in _messages_of(payloads[1])
    assert "上一次的返回不符合要求" not in _messages_of(payloads[0])


def test_一直不合规时按次数上限放弃(llm_config):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_reply("{}"))  # 缺字段，过不了 schema

    with pytest.raises(LlmInvalidOutput):
        _llm(handler, llm_config).complete_structured(_request(), 联合输出)

    assert len(calls) == llm_config.max_attempts


# ── 流式 ──


def test_流式逐字产出(llm_config):
    pieces = _llm(_streaming("二", "郎", "神"), llm_config).stream(_request())

    assert list(pieces) == ["二", "郎", "神"]


def test_流式一边收一边吐(llm_config):
    """不是把整段答案收完再给：网络还没读完，第一片就该出来。"""
    received: list[bytes] = []

    def body():
        for event in _sse("二", "郎", "神"):
            received.append(event)
            yield event

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    pieces = _llm(handler, llm_config).stream(_request())
    assert next(pieces) == "二"

    assert len(received) == 1


def test_流式的请求带上了_stream_开关(llm_config):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_sse("好"))

    list(_llm(handler, llm_config).stream(_request()))

    assert seen["stream"] is True


def test_流式被截断时报截断(llm_config):
    """已经吐出来的字不收回，但失败要报出来：调用方自己决定留不留。"""
    pieces = _llm(_streaming("二郎神有", finish_reason="length"), llm_config).stream(_request())

    assert next(pieces) == "二郎神有"
    with pytest.raises(LlmTruncated):
        next(pieces)


def test_坏掉的流不静默给空答案(llm_config):
    """服务端在流里报错、或正文根本不是 SSE：报出来，不能装成「模型没话说」。"""
    error_frame = _body(_event({"error": {"message": "invalid api key"}}), b"data: [DONE]\n\n")
    not_sse = _body(b'{"error": {"message": "bad request"}}')
    not_json = _body(b"data: \xe4\xb8\x8d\xe6\x98\xaf JSON\n\n")
    empty = _body(b"\n\n")

    for broken in (error_frame, not_sse, not_json, empty):
        with pytest.raises(LlmUnavailable):
            list(_llm(broken, llm_config).stream(_request()))

    # 报错信息里带上服务端说的话，好定位
    with pytest.raises(LlmUnavailable) as excinfo:
        list(_llm(error_frame, llm_config).stream(_request()))
    assert "invalid api key" in str(excinfo.value)


def test_流式开头的限流也按上限重试(llm_config):
    """流还没开始，重试不会重复吐字，所以照常重试到上限。"""
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json={})

    with pytest.raises(LlmRateLimited):
        list(_llm(handler, llm_config, sleeps=sleeps).stream(_request()))

    assert len(calls) == llm_config.max_attempts
    assert sleeps == [0.25, 0.5, 1.0, 2.0]


def test_流式中途超时按超时上报(llm_config):
    """超时与连不上在流里也要分得开。"""

    def body():
        yield _sse("二")[0]
        raise httpx.ReadTimeout("读超时")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    pieces = _llm(handler, llm_config).stream(_request())

    assert next(pieces) == "二"
    with pytest.raises(LlmTimeout):
        next(pieces)


# ── 失败可辨 ──


@pytest.mark.parametrize(
    ("handler", "expected"),
    [(_read_timeout, LlmTimeout), (_always_429, LlmRateLimited), (_always_truncated, LlmTruncated)],
    ids=["超时", "限流", "截断"],
)
def test_三种失败各有各的异常类型(llm_config, handler, expected):
    with pytest.raises(expected) as excinfo:
        _llm(handler, llm_config).complete(_request())

    assert type(excinfo.value) is expected
    assert isinstance(excinfo.value, LlmError)


def test_连不上与请求被拒也分得开(llm_config):
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("连接被拒绝", request=request)

    with pytest.raises(LlmUnavailable):
        _llm(refused, llm_config).complete(_request())
    with pytest.raises(LlmRejected):
        _llm(_unauthorized, llm_config).complete(_request())


def test_错误信息里只留主机_不带凭据(llm_config):
    """报错要能定位到打给了谁，但不能把凭据捎出去。"""
    config = llm_config.model_copy(update={"base_url": "https://user:HUNTER2@llm.test:8443/v1"})

    with pytest.raises(LlmRejected) as excinfo:
        _llm(_unauthorized, config).complete(_request())

    message = str(excinfo.value)
    assert message.startswith("https://llm.test:8443 拒绝了请求（HTTP 401）")
    assert "HUNTER2" not in message
    assert config.api_key.get_secret_value() not in message


# ── 重试有界 ──


def test_重试次数到上限就停(llm_config):
    """重试有上限：限流一直不解除时，尝试次数与退避都封在配置里。"""
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json={})

    with pytest.raises(LlmRateLimited):
        _llm(handler, llm_config, sleeps=sleeps).complete(_request())

    assert len(calls) == llm_config.max_attempts == 5
    assert sleeps == [0.25, 0.5, 1.0, 2.0]


def test_限流按服务端给的等待时间退避(llm_config):
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={})
        return httpx.Response(200, json=_reply("好的"))

    assert _llm(handler, llm_config, sleeps=sleeps).complete(_request()) == "好的"
    assert sleeps == [2.0]


def test_服务端给的等待时间超过本地上限时照样听它的(llm_config):
    """按本地上限砍掉只会让下一次照样被限流，白白烧完尝试次数。"""
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "30"}, json={})
        return httpx.Response(200, json=_reply("好的"))

    assert _llm(handler, llm_config, sleeps=sleeps).complete(_request()) == "好的"
    # 配置里的 backoff_max 是 4 秒，服务端说了 30 秒
    assert sleeps == [30.0]


def test_重试成功后拿到的是正常结果(llm_config):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ReadTimeout("读超时", request=request)
        return httpx.Response(200, json=_reply("二郎神有三阶段"))

    assert _llm(handler, llm_config).complete(_request()) == "二郎神有三阶段"
    assert len(calls) == 3


def test_不可重试的失败不浪费尝试(llm_config):
    """截断重发一次还是截断，被拒重发一次还是被拒。"""
    truncated_calls: list[httpx.Request] = []
    rejected_calls: list[httpx.Request] = []

    def truncating(request: httpx.Request) -> httpx.Response:
        truncated_calls.append(request)
        return httpx.Response(200, json=_reply("半截", finish_reason="length"))

    def rejecting(request: httpx.Request) -> httpx.Response:
        rejected_calls.append(request)
        return httpx.Response(401, json={})

    with pytest.raises(LlmTruncated):
        _llm(truncating, llm_config).complete(_request())
    with pytest.raises(LlmRejected):
        _llm(rejecting, llm_config).complete(_request())

    assert len(truncated_calls) == 1
    assert len(rejected_calls) == 1


def test_流式吐过字之后就不再重试(llm_config):
    """中途断线重来会把前半段答案再吐一遍，宁可少答也不能答歪。"""
    calls: list[httpx.Request] = []

    def body():
        yield _sse("二")[0]
        raise httpx.ReadError("连接断了")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=body())

    pieces = _llm(handler, llm_config).stream(_request())

    assert next(pieces) == "二"
    with pytest.raises(LlmUnavailable):
        next(pieces)
    assert len(calls) == 1


def test_流式还没吐字就断了可以重来(llm_config):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("连接被拒绝", request=request)
        return httpx.Response(200, content=_sse("二", "郎", "神"))

    assert list(_llm(handler, llm_config).stream(_request())) == ["二", "郎", "神"]
    assert len(calls) == 2


def test_指数退避逐次翻倍并封顶():
    policy = RetryPolicy(attempts=6, base_delay=0.5, max_delay=2.0)

    assert [policy.delay_for(attempt) for attempt in range(1, 6)] == [0.5, 1.0, 2.0, 2.0, 2.0]


def test_退避与重试次数取配置里的值(llm_config):
    policy = RetryPolicy.from_settings(llm_config)

    assert policy.attempts == llm_config.max_attempts
    assert policy.base_delay == llm_config.backoff_base
    assert policy.max_delay == llm_config.backoff_max


# ── 配置 ──


def test_地址模型名密钥与超时都来自配置(llm_config):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["timeout"] = request.extensions["timeout"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_reply("好的"))

    _llm(handler, llm_config).complete(_request())

    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-llm-api-key"
    assert seen["payload"]["model"] == "test-model"
    assert seen["timeout"]["read"] == 12.5


def test_base_url_结尾的斜杠不影响路径(llm_config):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_reply("好的"))

    config = llm_config.model_copy(update={"base_url": "https://llm.test/v1/"})
    _llm(handler, config).complete(_request())

    assert seen["url"] == "https://llm.test/v1/chat/completions"


def test_没有消息的调用在发出前就被拦下():
    with pytest.raises(ValueError):
        LlmRequest(messages=[])


# ── 带图消息（补图那一票的视觉摘要走它）──


def test_带图的消息发出多模态内容(llm_config):
    """图按 data URI 内联。走 URL 要先有个公网可取的地址，而解析产物里的原图
    在对象存储里，没有这么个地址。"""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_reply("一张攻略截图"))

    client = _llm(handler, llm_config)
    result = client.complete(
        LlmRequest(
            messages=[
                Message("system", "说清图里是什么"),
                Message(
                    "user",
                    "这张图是什么",
                    images=(ImagePart(data=b"\x89PNG-bytes", content_type="image/png"),),
                ),
            ]
        )
    )

    assert result == "一张攻略截图"
    content = payloads[0]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "这张图是什么"}
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"\x89PNG-bytes").decode("ascii")
    )


def test_不带图的消息仍然是字符串形式(llm_config):
    """**不退化成「只含一段文字」的数组**：不是每个 OpenAI 兼容服务端都认后者，
    而纯文本这条路本来一直好好的，不该被补图那一票波及。"""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_reply("嗨"))

    _llm(handler, llm_config).complete(_request())

    assert all(isinstance(message["content"], str) for message in payloads[0]["messages"])


def test_结构化调用也带得上图(llm_config):
    """摘要与联合输出走同一个客户端，两边的带图路径不该只有一条通。"""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        body = '{"subject_name":"二郎神","rewritten_query":"二郎神 打法","routes":["main"]}'
        return httpx.Response(200, json=_reply(body))

    _llm(handler, llm_config).complete_structured(
        LlmRequest(
            messages=[
                Message(
                    "user",
                    "看看这张图",
                    images=(ImagePart(data=b"jpeg-bytes", content_type="image/jpeg"),),
                )
            ]
        ),
        联合输出,
    )

    assert isinstance(payloads[0]["messages"][-1]["content"], list)


def test_首条不是系统提示时结构化调用不丢消息(llm_config):
    """一次调用可以只有一条 user 消息。系统提示要**插在它前面**，不是把它顶掉——
    顶掉之后模型收到的是一段光秃秃的 schema 说明，问题本身没了，而且不报错。"""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        body = '{"subject_name":"二郎神","rewritten_query":"二郎神 打法","routes":[]}'
        return httpx.Response(200, json=_reply(body))

    _llm(handler, llm_config).complete_structured(
        LlmRequest(messages=[Message("user", "二郎神怎么打")]), 联合输出
    )

    prompt = _messages_of(payloads[0])
    assert "二郎神怎么打" in prompt
    assert "JSON Schema" in prompt


# ── 假件 ──


def test_假件与真实客户端是同一个接口(llm_config):
    assert isinstance(OpenAiLlm(llm_config), LlmClient)
    assert isinstance(FakeLlm("嗨"), LlmClient)


def test_同一个接口的两个实现可以互相替换(llm_config):
    """同一组断言在真实客户端与假件上都成立 —— 这才是「可替换」的证据。"""
    答案 = "二郎神有三阶段"
    结构化 = '{"subject_name":"二郎神","rewritten_query":"二郎神 打法","routes":["main"]}'

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = 结构化 if "response_format" in payload else 答案
        if payload.get("stream"):
            return httpx.Response(200, content=_sse(*text))
        return httpx.Response(200, json=_reply(text))

    for client in (_llm(handler, llm_config), FakeLlm(答案, 结构化, 答案)):
        assert isinstance(client, LlmClient)
        assert client.complete(_request()) == 答案
        assert client.complete_structured(_request(), 联合输出).subject_name == "二郎神"
        assert list(client.stream(_request())) == list(答案)


def test_假件按脚本顺序回话并记下收到的问题():
    fake = FakeLlm("一", "二")

    assert fake.complete(_request("第一个问题")) == "一"
    assert fake.complete(_request("第二个问题")) == "二"

    assert [call.messages[-1].content for call in fake.calls] == ["第一个问题", "第二个问题"]


def test_假件流式逐字产出():
    fake = FakeLlm("二郎神怎么打")

    assert list(fake.stream(_request())) == list("二郎神怎么打")


def test_假件可以按更大的粒度吐():
    fake = FakeLlm("二郎神怎么打", chunk_size=3)

    assert list(fake.stream(_request())) == ["二郎神", "怎么打"]


def test_假件能按脚本返回结构化结果():
    fake = FakeLlm({"subject_name": "二郎神", "rewritten_query": "二郎神 打法", "routes": ["main"]})

    result = fake.complete_structured(_request(), 联合输出)

    assert result.subject_name == "二郎神"


def test_假件对不合规的返回给同一个错误类型():
    fake = FakeLlm("二郎神")

    with pytest.raises(LlmInvalidOutput):
        fake.complete_structured(_request(), 联合输出)


def test_假件能按脚本演失败(llm_config):
    fake = FakeLlm(LlmTimeout("超时了"), LlmRateLimited("限流了"))

    with pytest.raises(LlmTimeout):
        fake.complete(_request())
    with pytest.raises(LlmRateLimited):
        fake.complete(_request())


def test_假件脚本排空后立刻报错():
    """少排一条脚本要立刻炸，不能静默返回空串把断言带偏。"""
    fake = FakeLlm("只够一次")

    assert fake.complete(_request()) == "只够一次"
    with pytest.raises(LlmError):
        fake.complete(_request())
    with pytest.raises(LlmError):
        fake.stream(_request())
