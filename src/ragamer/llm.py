"""语言模型适配器：全项目唯一直接和模型服务对话的地方。

打标兜底、查询路由的联合输出、HyDE、多查询改写、生成全部走它。三件事必须一次做对：

- **结构化联合输出**：一次调用同时拿回多个字段（`complete_structured`），
  带 schema 校验；返回不合规按 `LlmInvalidOutput` 识别并重试，
  **重试时把上一次的毛病带回去** —— 温度 0 时原样重发只会再错一次。
- **流式**：`stream` 逐字产出。已经吐出内容之后不再重试，
  否则重试会把前半段答案再吐一遍。
- **失败可辨**：超时、限流、被截断各是各的异常类型（`LlmTimeout` /
  `LlmRateLimited` / `LlmTruncated`），各自知道自己能不能重试。

业务模块只依赖 `LlmClient` 协议：组合根构造 `OpenAiLlm` 注入，测试用 `FakeLlm`，
一次网络请求都不发。
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from ragamer.config import LlmSettings
from ragamer.logging import get_logger

logger = get_logger(__name__)

#: OpenAI 兼容接口的补全路径，接在配置的 base_url 之后。
ENDPOINT_PATH = "/chat/completions"

ModelT = TypeVar("ModelT", bound=BaseModel)
T = TypeVar("T")

Role = Literal["system", "user", "assistant"]


class LlmError(Exception):
    """模型调用失败。基类默认不可重试，能重试的子类把 `retryable` 打开。

    `retry_after` 是服务端给出的等待秒数，重试间隔以它为准。
    """

    retryable = False

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LlmTimeout(LlmError):
    """调用超时。服务端慢或网络抖动，重试有意义。"""

    retryable = True


class LlmRateLimited(LlmError):
    """被限流（HTTP 429）。退避后再来，服务端给了 `Retry-After` 就按它等。"""

    retryable = True


class LlmUnavailable(LlmError):
    """连不上，或服务端 5xx / 返回的结构不认识。"""

    retryable = True


class LlmTruncated(LlmError):
    """返回被 `max_tokens` 截断（`finish_reason == "length"`）。

    不重试：同样的请求会同样地截断，该改的是提示或 `max_tokens`，不是再来一次。
    """


class LlmInvalidOutput(LlmError):
    """返回不合规：不是 JSON，或过不了 schema 校验。重试时会把校验错误带回去。"""

    retryable = True


class LlmRejected(LlmError):
    """请求本身被拒（密钥、模型名、参数有问题）。重试只会再被拒一次。"""


@dataclass(frozen=True)
class RetryPolicy:
    """重试的次数上限与退避。三个取值都来自配置，默认值只在 `LlmSettings` 上有一份。"""

    attempts: int
    base_delay: float
    max_delay: float

    @classmethod
    def from_settings(cls, config: LlmSettings) -> RetryPolicy:
        return cls(
            attempts=config.max_attempts,
            base_delay=config.backoff_base,
            max_delay=config.backoff_max,
        )

    def delay_for(self, failed_attempt: int, retry_after: float | None = None) -> float:
        """第 `failed_attempt` 次失败之后等多久（秒）。

        服务端给了 `Retry-After` 就按它等 —— 它比本地猜的准，按本地上限砍掉只会让
        下一次照样被限流、白白烧完尝试次数。`max_delay` 只管自己算的那条指数退避。
        """
        if retry_after is not None:
            return retry_after
        return min(self.base_delay * 2 ** (failed_attempt - 1), self.max_delay)


@dataclass(frozen=True)
class ImagePart:
    """随消息发出去的一张图。

    `content_type` 是拼 data URI 用的（`data:image/png;base64,…`），模型按它解释字节——
    类型写错在有些服务端上不是报错，而是把图当成坏的直接忽略。认字节的活由调用方做
    （`ragamer.enriching`）：这一层只认 OpenAI 兼容接口的形状，不认识图片格式。
    """

    data: bytes
    content_type: str


@dataclass(frozen=True)
class Message:
    """一条对话消息。带图时 `images` 非空，正文与图一起发出去。"""

    role: Role
    content: str
    #: 跟这条消息一起发出去的图。空元组即纯文本消息——纯文本仍是字符串形式的
    #: `content`，不是只含一段文字的多模态数组：不是每个 OpenAI 兼容服务端都认后者。
    images: tuple[ImagePart, ...] = ()


@dataclass(frozen=True)
class LlmRequest:
    """一次调用要带的东西。多轮对话把历史一并放进 `messages`。"""

    messages: Sequence[Message]
    temperature: float = 0.0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("一次调用至少要有一条消息")


@runtime_checkable
class LlmClient(Protocol):
    """业务模块看到的语言模型。换实现只换这里。"""

    def complete(self, request: LlmRequest) -> str:
        """取一段完整文本。"""

    def stream(self, request: LlmRequest) -> Iterator[str]:
        """逐字取文本。"""

    def complete_structured(self, request: LlmRequest, schema: type[ModelT]) -> ModelT:
        """一次调用同时拿回多个字段，并按 `schema` 校验。

        schema 的字段描述（`Field(description=…)`）会一并写进提示里，所以它是调用方
        给模型下指令的地方。
        """


class OpenAiLlm:
    """OpenAI 兼容接口的实现（DeepSeek、Qwen、vLLM 都适用）。

    地址、密钥、模型名、超时、重试策略全部来自配置，这里不硬编码任何一项。
    """

    def __init__(
        self,
        config: LlmSettings,
        *,
        client: httpx.Client | None = None,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._policy = policy if policy is not None else RetryPolicy.from_settings(config)
        self._client = client if client is not None else httpx.Client()
        self._sleep = sleep
        self._endpoint = config.base_url.rstrip("/") + ENDPOINT_PATH
        self._headers = {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}

    def complete(self, request: LlmRequest) -> str:
        payload = self._payload(request)
        return self._retry(lambda: self._content(self._post(payload)))

    def stream(self, request: LlmRequest) -> Iterator[str]:
        payload = self._payload(request, stream=True)
        attempt = 1
        produced = False
        while True:
            try:
                for piece in self._stream_once(payload):
                    produced = True
                    yield piece
                return
            except LlmError as exc:
                # 已经吐过字就不能重来：重试会把前半段答案再吐一遍
                if produced or not self._can_retry(exc, attempt):
                    raise
                self._wait_before_retry(exc, attempt)
                attempt += 1

    def complete_structured(self, request: LlmRequest, schema: type[ModelT]) -> ModelT:
        repair: str | None = None

        def once() -> ModelT:
            nonlocal repair
            payload = self._payload(request, schema=schema, repair=repair)
            text = self._content(self._post(payload))
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                repair = _repair_note(exc)
                raise LlmInvalidOutput(
                    f"返回不符合 {schema.__name__}：{_one_line(str(exc))}"
                ) from exc

        return self._retry(once)

    def _payload(
        self,
        request: LlmRequest,
        *,
        schema: type[BaseModel] | None = None,
        stream: bool = False,
        repair: str | None = None,
    ) -> dict[str, Any]:
        messages = (
            request.messages if schema is None else _with_schema(request.messages, schema, repair)
        )
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": message.role, "content": _content(message)} for message in messages
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload["stream"] = True
        if schema is not None:
            # 只写 JSON 模式、结构写在提示里：各家对 json_schema 严格模式的支持程度不一
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """发一次请求，把传输层与 HTTP 状态翻译成异常。重试是 `_retry` 的事。"""
        try:
            response = self._client.post(
                self._endpoint,
                json=payload,
                headers=self._headers,
                timeout=self._config.timeout,
            )
        except httpx.HTTPError as exc:
            raise _transport_error(exc, self._endpoint, self._config.timeout) from exc
        self._raise_for_status(response.status_code, response.headers, response.text)
        return response

    def _stream_once(self, payload: dict[str, Any]) -> Iterator[str]:
        """开一次流。未开始的失败可以在外层重试，中途断掉不会重来。"""
        try:
            with self._client.stream(
                "POST",
                self._endpoint,
                json=payload,
                headers=self._headers,
                timeout=self._config.timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response.status_code, response.headers, response.text)
                truncated = False
                for delta in _sse_deltas(response.iter_lines()):
                    if delta.finish_reason == "length":
                        truncated = True
                    if delta.content:
                        yield delta.content
                if truncated:
                    raise LlmTruncated("流式返回被 max_tokens 截断，答案不完整")
        except httpx.HTTPError as exc:
            raise _transport_error(exc, self._endpoint, self._config.timeout) from exc

    def _content(self, response: httpx.Response) -> str:
        """取 `choices[0]` 的正文。截断在进 schema 校验之前就认出来。"""
        try:
            choice = response.json()["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LlmUnavailable(f"模型服务返回的结构不认识：{_excerpt(response.text)}") from exc
        if choice.get("finish_reason") == "length":
            raise LlmTruncated("返回被 max_tokens 截断，答案不完整")
        return (choice.get("message") or {}).get("content") or ""

    def _raise_for_status(self, status: int, headers: httpx.Headers, body: str) -> None:
        if status < 400:
            return
        where = _address(self._endpoint)
        if status == 429:
            raise LlmRateLimited(f"{where} 限流（HTTP 429）", retry_after=_retry_after(headers))
        if status in (408, 504):
            raise LlmTimeout(f"{where} 超时（HTTP {status}）")
        if status >= 500:
            raise LlmUnavailable(f"{where} 服务端出错（HTTP {status}）：{_excerpt(body)}")
        raise LlmRejected(f"{where} 拒绝了请求（HTTP {status}）：{_excerpt(body)}")

    def _retry(self, operation: Callable[[], T]) -> T:
        attempt = 1
        while True:
            try:
                return operation()
            except LlmError as exc:
                if not self._can_retry(exc, attempt):
                    raise
                self._wait_before_retry(exc, attempt)
                attempt += 1

    def _can_retry(self, exc: LlmError, attempt: int) -> bool:
        """还剩尝试次数，且这一类失败值得重试。"""
        return exc.retryable and attempt < self._policy.attempts

    def _wait_before_retry(self, exc: LlmError, attempt: int) -> None:
        delay = self._policy.delay_for(attempt, exc.retry_after)
        logger.warning(
            "模型调用失败（第 %d/%d 次）：%s；%.1f 秒后重试",
            attempt,
            self._policy.attempts,
            exc,
            delay,
        )
        self._sleep(delay)


class FakeLlm:
    """确定性的假模型：按脚本顺序回话，一次网络都不发。

    脚本排空后再被调用会抛错 —— 测试少排了一条时立刻炸，
    而不是静默返回空串把断言带偏。`calls` 记下每一次拿到的请求，供断言。

    不模拟网络故障与重试：重试是客户端与网络之间的策略，要测它请给 `OpenAiLlm`
    换一个 `httpx.MockTransport` 传输层。
    """

    def __init__(self, *replies: str | Mapping[str, Any] | LlmError, chunk_size: int = 1) -> None:
        self.replies: list[str | Mapping[str, Any] | LlmError] = list(replies)
        #: 流式一次吐几个字。默认逐字。
        self.chunk_size = chunk_size
        self.calls: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> str:
        return self._reply(request)

    def stream(self, request: LlmRequest) -> Iterator[str]:
        text = self._reply(request)  # 先取脚本：调用立刻被记下，排空了也立刻报错
        return (
            text[start : start + self.chunk_size] for start in range(0, len(text), self.chunk_size)
        )

    def complete_structured(self, request: LlmRequest, schema: type[ModelT]) -> ModelT:
        text = self._reply(request)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LlmInvalidOutput(f"返回不符合 {schema.__name__}：{_one_line(str(exc))}") from exc

    def _reply(self, request: LlmRequest) -> str:
        self.calls.append(request)
        if not self.replies:
            raise LlmError(
                f"假件没有更多脚本回复了（第 {len(self.calls)} 次调用）—— 测试少排了一条"
            )
        reply = self.replies.pop(0)
        if isinstance(reply, LlmError):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def _content(message: Message) -> str | list[dict[str, Any]]:
    """一条消息的 `content`：不带图就是字符串，带图是多模态数组。

    不带图时不退化成「只含一段文字」的数组——两边的形状本来就不同，
    多绕一层只会让纯文本那边也跟着受服务端实现差异的影响（补图那一票加的）。
    """
    if not message.images:
        return message.content
    return [
        {"type": "text", "text": message.content},
        *(
            {"type": "image_url", "image_url": {"url": _data_uri(image)}}
            for image in message.images
        ),
    ]


def _data_uri(image: ImagePart) -> str:
    """图按 base64 内联进请求体。走 URL 意味着先得有个公网可取的地址，
    而解析产物里的原图在对象存储里，没有这么个地址。"""
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.content_type};base64,{encoded}"


@dataclass(frozen=True)
class _Delta:
    """流里的一片产出，以及它携带的结束原因。"""

    content: str = ""
    finish_reason: str | None = None


#: SSE 的数据行前缀与结束标记。
_SSE_PREFIX = "data:"
_SSE_DONE = "[DONE]"


def _sse_deltas(lines: Iterable[str]) -> Iterator[_Delta]:
    """把 SSE 行流翻译成产出。心跳与 `event:` 之类的字段跳过，坏东西不吞。

    认不出的数据行、服务端在流里报的错、一个数据帧都没返回的流，一律报
    `LlmUnavailable` —— 静默给出空答案比直接报错难查得多。
    """
    seen = False
    for line in lines:
        if not line.startswith(_SSE_PREFIX):
            continue
        data = line[len(_SSE_PREFIX) :].strip()
        seen = True
        if data == _SSE_DONE:
            return
        chunk = _load_frame(data)
        if chunk.get("error"):
            raise LlmUnavailable(f"模型服务在流里报错：{_excerpt(_dump(chunk['error']))}")
        for choice in chunk.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            yield _Delta(
                content=delta.get("content") or "",
                finish_reason=choice.get("finish_reason"),
            )
    if not seen:
        raise LlmUnavailable("模型服务一个数据帧都没返回，这个流是坏的")


def _with_schema(
    messages: Sequence[Message],
    schema: type[BaseModel],
    repair: str | None,
) -> tuple[Message, ...]:
    """把「要输出什么结构」并进系统提示；重试时再带上上一次的毛病。

    首条不是系统提示时**在它前面插一条**，不是把它顶掉——一次调用可以只有一条 user
    消息（补图那一票的带图调用就是），顶掉之后模型收到的是光秃秃的 schema 说明，
    问题本身没了，而且不报错。
    """
    instruction = _schema_instruction(schema)
    if repair is not None:
        instruction = f"{instruction}\n\n{repair}"
    head, rest = messages[0], tuple(messages[1:])
    if head.role == "system":
        return (Message("system", f"{head.content}\n\n{instruction}"), *rest)
    return (Message("system", instruction), head, *rest)


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "只输出一个 JSON 对象，不要解释、前后缀或代码块标记。它必须符合这个 JSON Schema：\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def _repair_note(exc: ValidationError) -> str:
    """把校验错误回给模型 —— 温度 0 时原样重发只会再错一次。"""
    problems = "；".join(
        f"{'.'.join(str(part) for part in error['loc']) or '整体'}：{error['msg']}"
        for error in exc.errors()[:5]
    )
    return f"上一次的返回不符合要求（{problems}）。请重新只输出一个符合上述 JSON Schema 的对象。"


def _retry_after(headers: httpx.Headers) -> float | None:
    """读 `Retry-After`。只认秒数形式，HTTP 日期形式交给退避策略。"""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None


def _load_frame(data: str) -> dict[str, Any]:
    """一个数据帧必须是个 JSON 对象，别的都算流坏了。"""
    try:
        chunk = json.loads(data)
    except ValueError as exc:
        raise LlmUnavailable(f"流里的数据帧不是 JSON：{_excerpt(data)}") from exc
    if not isinstance(chunk, dict):
        raise LlmUnavailable(f"流里的数据帧不是对象：{_excerpt(data)}")
    return chunk


def _dump(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _transport_error(exc: httpx.HTTPError, endpoint: str, timeout: float) -> LlmError:
    """把 httpx 的传输层异常翻成项目自己的类型：超时与连不上要分得开。"""
    if isinstance(exc, httpx.TimeoutException):
        return LlmTimeout(f"{_address(endpoint)} 超时（{timeout} 秒）：{exc!r}")
    return LlmUnavailable(f"连不上模型服务 {_address(endpoint)}：{exc!r}")


def _address(endpoint: str) -> str:
    """只留协议、主机与端口：地址里可能内嵌凭据或查询串，不进错误信息。

    用 `host` 而不是 `netloc` —— 后者连 `user:password@` 一段一起带出来。
    """
    url = httpx.URL(endpoint)
    port = f":{url.port}" if url.port is not None else ""
    return f"{url.scheme}://{url.host}{port}"


def _excerpt(text: str, limit: int = 200) -> str:
    """压平并截断：服务端的错误页可能是几百行 HTML，不该整段进日志。"""
    flat = _one_line(text)
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _one_line(text: str) -> str:
    return " ".join(text.split())
