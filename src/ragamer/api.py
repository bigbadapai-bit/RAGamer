"""HTTP 端点：写入侧与读取侧对外的唯一入口。

六个端点分两组：

**写入侧**两个，对应界面上的两处输入：`POST /api/kb/{game_id}/import` 传文件
（或界面上传的字节），`POST /api/kb/{game_id}/import/urls` 给网址、抓回来再入库。
两条都是批量提交、**逐条独立**：某一条失败时其余照常入库，失败的那条在结果里带
来源与失败阶段；两条走的是同一个导入器、同一条链路，响应形状也逐字相同。

**读取侧**是多轮对话：`POST /api/chat/sessions`、`GET /api/chat/sessions`
（按库列会话）、`GET /api/chat/sessions/{session_id}`、
`GET /api/chat/sessions/{session_id}/ask`。开会话、列会话、把历史读回来、
逐字问一句（SSE）。**澄清反问也在这条流上**：判不准时流里出一个
`clarification` 事件，用户点完带 `pending_id` 再问一次。

**只有会话这一条提问路径**。曾经另有一个不绑会话的 `POST /api/chat`，接上会话之后
它被 `/api/chat/sessions` 影子掉了（`/api/chat/{pending_id}` 那条路由先注册，把
`/api/chat/sessions` 当成了自己的 `pending_id`）——两条入口本来也是同一件事，
收敛掉的那条不再保留。

**写入侧那两条 JSON 是同步的**：调用方拿到的是最终结果，中途看不见进度。页面那条
不同——它提交完就返回，进度靠轮询一个任务快照（见 `ragamer.jobs`）。两条都在
线程池里跑，谁都不占事件循环。

这一层只做 HTTP 这一层的事：会话不存在翻成 404、问题为空翻成 400、一轮问答翻成
SSE 事件、暂停点选错了翻成 422。**「这一轮算不算问完」「要不要写进历史」在
`ragamer.conversations` 里**，这一层不重复判断——判两遍迟早会分岔，而分岔的那一次
表现为「刷新之后历史少了一轮」。

知识库元数据从 MongoDB 读（`knowledge_bases` 集合，id 就是游戏 id）：打标要用的词表
——启用了哪些主体类型、这个游戏的术语映射——就在它里面（docs/ARCHITECTURE.md §2.3），
检索回落哪个版本也从它取（§2.4），**这次走哪几路召回**同样由它覆盖（§3.1 的默认路由表
是出厂值，库里配了 `route_table` 就用配的）。形状与判断在 `ragamer.knowledge`，界面那条
写入路径用的是同一份。知识库列表同时是**游戏候选**：给模型的是显示名，拿回来再换回 id，
理由见 `_games`。
本层**一个适配器都不构造**，全部来自组合根（`ragamer.container`），缝因此立得住。

这是**只有 JSON 端点**的应用；给人看的页面由 `ragamer.web` 挂上去，两者在
`ragamer.app` 里装成同一个应用。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from queue import SimpleQueue
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragamer.answering import Citation
from ragamer.clarifying import Clarification, NotACandidate, UnknownPending
from ragamer.container import Container, build_importer
from ragamer.conversations import (
    Chat,
    Conversation,
    ConversationNotFound,
    Delta,
    Game,
    SessionPage,
    Sources,
    Status,
    build_chat,
    parse_session_cursor,
)
from ragamer.importing import STAGE_LABELS, ImportResult, ProgressEvent
from ragamer.knowledge import (
    KB_COLLECTION,
    KnowledgeBase,
    KnowledgeBaseError,
    readable_knowledge_base,
)
from ragamer.live import CANCELLED, END, TurnRegistry
from ragamer.llm import LlmError
from ragamer.logging import get_logger
from ragamer.routing import RouteTable
from ragamer.sources import SourceDocument
from ragamer.stores.base import UNVERSIONED, StoreError, collection_name
from ragamer.tagging import TagVocabulary
from ragamer.vectors.base import ModelOutputError

logger = get_logger(__name__)

#: SSE 的响应头。`no-cache` 是这条流的标准要求：中间任何一层缓存住它，逐字就变成一次给全。
SSE_HEADERS = {"Cache-Control": "no-cache"}

#: SSE 的起手那一行 `retry:`，把浏览器的重连间隔推得很远。
#:
#: 🔴 `EventSource` 在连接关闭之后会**自动重连同一个 URL**——正常问完关掉连接也一样。
#: 重连就是再问一遍，于是历史里多出一轮一模一样的问答。客户端收到 `done` 就该 `close()`，
#: 但那是页面那一侧的事；这一行是兜底：真忘了关，下一次重连要等一天，而不是默认的三秒。
SSE_RETRY = "retry: 86400000\n\n"


class UrlImport(BaseModel):
    """一次网址导入的请求体。

    网址单独成一个端点而不是塞进上传那一个的多表单字段里：界面上的两处输入本来就分开
    （文件上传 / URL 输入，docs/ARCHITECTURE.md §5），而多部分表单里夹一个网址数组
    两边都不好写。
    """

    urls: list[str] = Field(min_length=1, description="要抓的网页地址，可多条")
    version: str = Field(default=UNVERSIONED, description="这次导入标注的版本，留空即未标注版本")


def create_app(container: Container, chat: Chat | None = None) -> FastAPI:
    """把组合根里那套依赖接成 ASGI 应用。

    `chat` 由 `ragamer.app` 传进来：整站只有一套读取侧，JSON 端点与页面共用同一份
    （缓存与澄清器都在它里面，接两遍就可能两边不一样）。不传就现接一份——
    直接打这个应用的测试走那条。

    写入侧那条链路也不在这里拼：`ragamer.container.build_importer` 一处接完，
    与页面共用同一份。各拼一遍的代价不是重复，而是**缺件不报错**——少接的那几样
    在界面上只是「用不了」，没有任何一处会说自己没接上。
    """
    chat = chat if chat is not None else build_chat(container).chat
    app = FastAPI(title="RAGamer", summary="游戏攻略 RAG 助手")
    importer = build_importer(container)
    #: 这个进程里正在跑的那几轮问答，见 `ragamer.live`。**一个应用一份**——
    #: 它同时是「同一个会话只跑一轮」那把锁，接两份就等于没锁。
    turns = TurnRegistry()

    @app.post("/api/kb/{game_id}/import")
    async def import_sources(
        game_id: str,
        files: Annotated[list[UploadFile], File(description="要导入的资料，可多份")],
        version: Annotated[
            str, Form(description="这次导入标注的版本，留空即未标注版本")
        ] = UNVERSIONED,
    ) -> dict[str, Any]:
        """批量导入。某个文件失败时其余照常入库，失败信息带文件名与失败阶段。

        **这一段同步给出最终结果**（调用方要的就是这个），但它跑在**线程池**里：
        导入是分钟级的一段（MinerU、二次 OCR、出网抓取），压在事件循环里跑会把整个
        应用卡住——别的端点、别的页面全都得排队等它。
        """
        _check_game_id(game_id)
        vocabulary = _vocabulary(container, game_id)
        sources = [
            SourceDocument(filename=file.filename or "", data=await file.read()) for file in files
        ]

        results = await run_in_threadpool(
            importer.batch, sources, game_id=game_id, version=version, vocabulary=vocabulary
        )
        return _response(game_id, version, results)

    @app.post("/api/kb/{game_id}/import/urls")
    async def import_urls(game_id: str, request: UrlImport) -> dict[str, Any]:
        """抓一批网页再入库。某个地址失败时其余照常入库，失败信息带地址与失败阶段。

        抓回来的资料与上传的文件走同一条链路，响应形状也相同——界面上两条输入各是各的
        提交按钮，读结果的地方却可以共用一处。同上，这一段也在线程池里跑。
        """
        _check_game_id(game_id)
        vocabulary = _vocabulary(container, game_id)

        results = await run_in_threadpool(
            importer.batch_urls,
            request.urls,
            game_id=game_id,
            version=request.version,
            vocabulary=vocabulary,
        )
        return _response(game_id, request.version, results)

    @app.post("/api/chat/sessions", status_code=201)
    def create_session(payload: SessionRequest) -> dict[str, Any]:
        """开一次会话，绑一个知识库。

        知识库不存在当场 404——与导入端点同一个口径：那是游戏选错了。
        会话 id 由服务端生成并返回，之后的两次请求都带着它。
        """
        _check_game_id(payload.game_id)
        _knowledge_base(container, payload.game_id)
        conversation = chat.start(game_id=payload.game_id, version=payload.version)
        return _conversation_payload(conversation)

    @app.get("/api/chat/sessions")
    def list_sessions(game_id: str, after: str = "") -> dict[str, Any]:
        """一个知识库下的会话，按最后活跃倒序，**一页一页给**。**左栏那一份列表**。

        只回标题与时间——正文在 `GET /api/chat/sessions/{session_id}` 那一条上取。
        库里一条会话都没有时回空列表，不是 404；**「还没聊过」是正常状态**。
        知识库本身不存在则是 404，与建会话同一个口径：那是游戏选错了。

        `after` 是上一页回的 `next`，**原样带回来即可**；不给就是从最新一页开始。
        响应里的 `next` 空串表示到底了——这是「还有没有更多」唯一的信号。
        """
        _check_game_id(game_id)
        _knowledge_base(container, game_id)
        page = chat.list_for_game(game_id, after=parse_session_cursor(after))
        return _sessions_payload(game_id, page)

    @app.get("/api/chat/sessions/{session_id}")
    def read_session(session_id: str) -> dict[str, Any]:
        """一次会话的全部问答。**刷新页面靠它把历史拿回来**——历史在服务端，不在页面里。

        空会话也是一次正常的返回（`turns` 为空列表），与「没有这个会话」分得很开：
        后者是 404。
        """
        return _conversation_payload(_conversation(chat, session_id))

    @app.get("/api/chat/sessions/{session_id}/ask")
    def ask(
        session_id: str,
        question: str,
        version: str = "",
        pending_id: str = "",
        label: str = "",
    ) -> StreamingResponse:
        """问一句，逐字把答案拿回来（SSE）。

        走 GET 是给浏览器原生的 `EventSource` 留的路——它只会发 GET（架构文档 §5 允许
        对话页那一小块用它）。**客户端收到 `done` 之后必须 `close()`**，理由见 :data:`SSE_RETRY`。

        看着别扭是承认的：**这一条 GET 带写库副作用**（答完要落一轮历史），而「GET 是安全
        方法」正是浏览器敢自动重连的前提。两个约束撞在一起——`EventSource` 只会发 GET，
        而这一轮问答又必须写进去——取的是前者。页面若改用 `fetch` 读流就不必受这条约束，
        :data:`SSE_RETRY` 那行也就可以不看。

        失败分两段，**以「流开没开」为界**：会话不存在（404）与问题为空（400）都在开流
        之前判掉，读知识库、读游戏候选也在这里——这几步不碰模型，快且失败原因明确。
        理解、检索、生成都在流里跑（见 `ragamer.conversations`），它们失败时只剩 `error`
        事件这一条路：响应头那时已经发出去了，状态码改不了。

        这一层**不等正文**：返回的是个还没开始跑的生成器，读正文由 ASGI 那边拉。
        端点本身是同步的，FastAPI 会把它放进线程池——检索与生成都是阻塞调用，
        写在 `async def` 里会把事件循环钉住。

        **这一轮跑在后台线程里，不属于这条请求**（`ragamer.live`）：页面断开、刷新、
        切走都只停转发，那一轮照跑完、照落库——回来刷新就看得到它。真要收手得走
        `POST /api/chat/sessions/{session_id}/cancel`。同一个会话同时只允许一轮，
        第二问当场被挡回来（从流里的 `error` 说出口，理由见上）。
        """
        if not question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        # 先读一次会话：不存在当场 404，顺带拿到它绑的知识库（现行版本要从那里取）。
        # `chat.ask` 自己还会再读一次，那是它的事——会话是上一次请求写下的，
        # 这一层手里这一份只用来决定「去哪个库问」。
        conversation = _conversation(chat, session_id)
        knowledge = _kb_document(container, conversation.game_id)
        # 这几样**在这一侧先取出来**，不进那个后台线程：它们不碰模型，快，而且失败原因
        # 明确（路由表配坏了 422、候选读不出来也是），值得一个正常的状态码。放进线程里
        # 就只剩流里一条 `error`——响应头那时已经发出去了。
        routes = _route_table(knowledge, game_id=conversation.game_id)
        games = _games(container)
        turn = turns.start(
            session_id,
            lambda live: chat.ask(
                session_id,
                question,
                version=version,
                current_version=str(knowledge.get("version", "")),
                games=games,
                routes=routes,
                pending_id=pending_id,
                label=label,
                cancelled=live.cancelled.is_set,
            ),
        )
        if turn is None:
            # 同一会话的第二问。**判在开流之前，却只能从流里说出口**：`EventSource`
            # 在非 2xx 时什么细节都不给（见上面的说明），所以这里回一条只带 `error`
            # 的流，而不是 409。
            return StreamingResponse(
                _refusal("这个会话上已经有一轮在答了，等它答完，或者先把它停掉。"),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )
        return StreamingResponse(
            _events(turn.queue), media_type="text/event-stream", headers=SSE_HEADERS
        )

    @app.post("/api/chat/sessions/{session_id}/cancel")
    def cancel(session_id: str) -> dict[str, Any]:
        """让这个会话上正在跑的那一轮收手。**不可逆**：那一轮不写进会话，当没问过。

        没有在跑的一轮时照样 200（`cancelled` 为 `false`）：用户点「停止」的那一刻
        那一轮可能刚好答完，那不是错误，也不该报成错误。

        **收手不是抢占式的**。跑那一轮的线程要到下一个检查点才看得见这个信号，而检查点
        之间可能隔着一次模型往返、一次向量化或者一次精排（最长二十几秒），见
        `ragamer.live`。会话不存在照样 404——与另外几条端点同一个口径。
        """
        _conversation(chat, session_id)
        return {"session_id": session_id, "cancelled": turns.cancel(session_id)}

    return app


class SessionRequest(BaseModel):
    """新建会话的请求体。"""

    #: 问哪个知识库（游戏 id）。
    game_id: str
    #: 这次会话选定的版本。留空即不选，检索时回落知识库的现行版本。
    version: str = ""


def _events(queue: SimpleQueue[object]) -> Iterator[str]:
    """一轮问答 → SSE 字节流。

    正常那一轮是五种事件，**前四种按发生的先后**：

    | 事件 | 几条 | 什么时候 |
    |---|---|---|
    | `status` | 若干 | 理解、检索、生成三步各自开始之前 |
    | `citations` | 一条 | 检索完、拿到来源（引用 + 图片 + 实际版本） |
    | `delta` | 若干 | 正文一片一片来 |
    | `done` | 一条 | 正常收尾 |
    | `error` | 至多一条 | 中途失败，代替 `done` |

    **判不准的那一轮只有两条**：`status`（正在理解问题）加一条 `clarification`
    （问哪个维度、候选有哪些、回哪个暂停点继续）。它**不发 `done`**——这一轮没走完，
    也不该让页面以为答完了。用户点完候选带 `pending_id` 再问一次，那一轮才从头走一遍。

    `status` 是给「不用干等」用的：提问到第一个字之间隔着两次实打实的等待（一次模型
    往返加一次检索），没有它，界面在那一段里没有任何东西可显示。

    **失败也必须发成一个事件**：响应头在第一个事件之前就出去了，状态码此刻改不了。
    对浏览器原生的 `EventSource` 来说这一条反而是好事——非 2xx 时它不给你任何细节，
    只有流里的 `error` 带得回原因。收不到 `done` 就是这一轮没有正常结束，会话里相应地
    什么都没写（见 `ragamer.conversations`）。

    **这一层不管「断开留不留痕」**：那一轮跑在后台线程里（`ragamer.live`），页面断开
    只是让它不再被转发，答案照常跑完、照常落库。消费方把这里丢掉时收到的是
    `GeneratorExit`——队列还在，跑那一轮的线程照跑不误，这个异常够不着它。

    队列里的哨兵决定怎么收尾：**收到结束哨兵才发 `done`**，而哨兵是那一轮跑完
    （落库之后）才放进去的——所以页面看见 `done` 的时候，刷新一定读得到这一轮。
    被取消的那一轮放的是另一个哨兵，这里直接结束、不发 `done`：它没走完。

    三类失败分得开：**生成挂掉**是模型这一次不行，按 WARNING；**检索或落库挂掉**
    （`StoreError` / `ModelOutputError`）是系统性的，按 ERROR；**暂停点对不上**
    （选的不在候选里、暂停点已经不在了）是这一次请求本身的问题，按 WARNING——
    排查时看的不是同一个地方。失败是后台那一轮把异常放进队列送过来的，分类在
    :func:`_failure` 里做。
    """
    yield SSE_RETRY
    stopped = False
    while True:
        item = queue.get()
        if item is CANCELLED:
            # 被取消的那一轮不发 `done`：它没走完。页面多半已经自己收好了（停止是它点的），
            # 这一条留给「在别处取消」的那种——转发的这一侧只是安静地结束
            return
        if item is END:
            break
        if isinstance(item, BaseException):
            yield _failure(item)
            return
        if isinstance(item, Status):
            yield _event("status", {"text": item.text})
        elif isinstance(item, Clarification):
            # 这一轮没走完：等着用户点。收尾那条 `done` 因此不能发
            stopped = True
            yield _event("clarification", _clarification_payload(item))
        elif isinstance(item, Sources):
            yield _event("citations", _sources_payload(item))
        elif isinstance(item, Delta):
            yield _event("delta", {"text": item.text})
    if not stopped:
        yield _event("done", {})


def _failure(exc: BaseException) -> str:
    """后台那一轮抛出来的失败 → 一条 `error` 事件。三类分开记，口径与原来一致。"""
    if isinstance(exc, LlmError):
        logger.warning("生成中途失败，这一轮不写进会话：%s", exc)
    elif isinstance(exc, (NotACandidate, UnknownPending)):
        logger.warning("暂停点这条路走不通，这一轮不写进会话：%s", exc)
    elif isinstance(exc, (ModelOutputError, StoreError)):
        logger.error("检索或落库失败，这一轮没有答案也不写进会话：%s", exc)
    else:
        # 从前这种异常会把响应头后面的连接直接掐断，原因只留在服务端日志里；
        # 现在它至少还能变成一条说得出口的事件
        logger.error("这一轮意外失败，什么都没写进会话：%s", exc)
    return _event("error", {"message": str(exc)})


def _refusal(message: str) -> Iterator[str]:
    """开流之前就判掉、却只能从流里说出口的那点事（同一会话的第二问）。

    回一整条流而不是一个状态码：浏览器原生的 `EventSource` 在非 2xx 时什么细节都不给
    （见 :func:`ask` 的说明），「这个会话上已经有一轮在答了」这句话只有走 `error`
    才到得了页面。
    """
    yield SSE_RETRY
    yield _event("error", {"message": message})


def _clarification_payload(clarification: Clarification) -> dict[str, Any]:
    """一次反问对外的样子。

    `label` 是摆给人看的字面（游戏是显示名），`value` 是库里的取值（游戏 id）。
    **从暂停点继续认的是 `label`**（`ragamer.clarifying._choice_of` 照它认），
    `value` 一并给出来只是把库里那侧交代清楚——客户端要跳到那个库的页面时不必自己反推。
    """
    return {
        "pending_id": clarification.pending_id,
        "dimension": clarification.dimension,
        "prompt": clarification.prompt,
        "choices": [
            {"label": choice.label, "value": choice.value} for choice in clarification.choices
        ],
    }


def _sources_payload(sources: Sources) -> dict[str, Any]:
    """这一轮的依据与口径。图片与版本跟引用同一批交出去，页面不必再问一次。"""
    return {
        "citations": [_citation_payload(item) for item in sources.citations],
        "images": list(sources.images),
        "version": sources.version,
    }


def _event(name: str, payload: Mapping[str, Any]) -> str:
    """一个 SSE 事件。

    `data` 里放 JSON 而不是裸文本：答案里换行是常事，按 SSE 的多行 `data:` 规则拼要自己
    处理折行与转义，交给 `json.dumps` 就只有一行。`ensure_ascii=False` 是为了日志与
    `curl` 看到的还是中文。
    """
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _response(game_id: str, version: str, results: tuple[ImportResult, ...]) -> dict[str, Any]:
    """两条导入路径共用的响应。形状一样是有意的：界面读结果只有一处。"""
    return {
        "game_id": game_id,
        "version": version,
        "imported": sum(1 for result in results if result.ok),
        "failed": sum(1 for result in results if not result.ok),
        "results": [_result_payload(result) for result in results],
    }


def _check_game_id(game_id: str) -> None:
    """游戏 id 同时是 collection 名，不合法就当场 400。

    在这里拦而不是等入库那一步报错：那会变成一个文件的失败原因，而这明明是整条请求
    的问题——建不出来的 collection 名，换哪个文件都一样。
    """
    try:
        collection_name(game_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _knowledge_base(container: Container, game_id: str) -> KnowledgeBase:
    """这个知识库的元数据。**它是「当前该用哪个版本」的唯一真相来源**（ADR-0004）。

    检索与聚合父块都从这里取现行版本，不各自维护一份。判断在 `ragamer.knowledge` 里
    （界面那条写入路径用的是同一份），这里只把问题翻成它自带的那个状态码——库不存在 404、
    配置读不了 422，两种问题的分法见那边的类文档。
    """
    try:
        return readable_knowledge_base(container.docs, game_id)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


def _vocabulary(container: Container, game_id: str) -> TagVocabulary:
    """这个知识库的打标词表。

    配置里没写的项一律走默认值（全部主体类型、映射为空）——用户自定义库没配映射时
    就是这条降级路径，标签会稀疏但不会漏（docs/ARCHITECTURE.md §2.3）。
    """
    return _knowledge_base(container, game_id).vocabulary


def _kb_document(container: Container, game_id: str) -> dict[str, Any]:
    """这个知识库的元数据，**原始那一份**。

    提问那条路读它而不是读 `_knowledge_base`：路由表要从文档本身建
    （`RouteTable.from_mapping`），而 `KnowledgeBase` 是 `ragamer.knowledge` 归一过的
    形状，不带原样字段。读一次就够——现行版本也从这一份里取。

    库不存在是游戏选错了，当场 404，不静默按默认值把内容查一遍。
    """
    payload = container.docs.get(KB_COLLECTION, game_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"知识库 {game_id} 不存在。先在知识库管理里建一个",
        )
    return payload


def _route_table(knowledge: Mapping[str, Any], *, game_id: str) -> RouteTable:
    """这个知识库的查询路由表。**元数据已经在手里**，所以收的是那份文档而不是再读一次
    （`ask` 那一步为了现行版本本来就要读它，见 `_kb_document`）。

    配置里没写的类型一律走默认值（`ragamer.routing`）——与打标词表同一个姿势：
    库里只写改过的那几行，默认值以后才改得动。而**读不了就是 422**：路由表决定这次
    检索走哪几路，配置写坏时静默按默认值跑，会让人以为「改配置没用」，查无可查。
    """
    try:
        return RouteTable.from_mapping(knowledge)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"知识库 {game_id} 的路由表读不了：{exc}"
        ) from exc


def _games(container: Container) -> tuple[Game, ...]:
    """游戏候选：`(显示名, 知识库 id)`。

    候选值必须是**用户问句里会出现的那种写法**。知识库 id 同时是 Milvus 的 collection 名，
    只能是英文标识符（`collection_name`），用户不会这么问——拿 id 去当候选，模型只会把
    「黑神话」判成不在候选里，这一步于是永远判不出东西来。所以给显示名，拿回来再换回 id
    （`ragamer.conversations._game_id`）。显示名没配就回落 id，与知识库管理页一个口径。
    """
    candidates = []
    for game_id in container.docs.list_ids(KB_COLLECTION):
        knowledge = container.docs.get(KB_COLLECTION, game_id) or {}
        candidates.append((str(knowledge.get("name", "")) or game_id, game_id))
    return tuple(candidates)


def _conversation(chat: Chat, session_id: str) -> Conversation:
    try:
        return chat.open(session_id)
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _sessions_payload(game_id: str, page: SessionPage) -> dict[str, Any]:
    """会话列表对外的样子。

    `ConversationSummary` 天生装不下正文，所以这里不必再挑一遍字段——每一项就三样：
    会话 id、标题、最后活跃时刻。正文在单条会话那一条端点上取。

    `next` 是下一页的游标（空串即到底了）。**它是不透明的**：客户端不必懂它，原样带
    回来即可——排序键与并列时按什么定序都在里面，自己拼是拼不对的。
    """
    return {
        "game_id": game_id,
        "sessions": [asdict(summary) for summary in page.sessions],
        "next": page.next,
    }


def _conversation_payload(conversation: Conversation) -> dict[str, Any]:
    """一次会话对外的样子。

    引用连 `label` 一起给：标题与祖先标题路径怎么拼由 `Citation.label` 定，
    界面照抄就行——两边各拼一遍，迟早会出现「同一个来源在两处叫法不一样」。
    """
    return {
        "session_id": conversation.session_id,
        "game_id": conversation.game_id,
        "version": conversation.version,
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "citations": [_citation_payload(citation) for citation in turn.citations],
            }
            for turn in conversation.turns
        ],
    }


def _citation_payload(citation: Citation) -> dict[str, Any]:
    """引用对外的样子。

    `label` 与 `origin` 在这里补上：前者是显示用的那一行（怎么拼由 `Citation.label`
    定，界面照抄就行），后者是「这条是知识库里查到的还是网上搜来的」。**`origin`
    是个属性、`asdict` 带不出来**，而界面判它不该靠「url 是不是空串」去猜。
    """
    return {**asdict(citation), "label": citation.label, "origin": citation.origin}


def _result_payload(result: ImportResult) -> dict[str, Any]:
    """一个文件的结果。失败时 `stage` 与 `error` 一起给出：界面要能说清卡在哪一步。"""
    return {
        "source": result.source,
        "doc_title": result.doc_title,
        "chunk_count": result.chunk_count,
        "skipped": result.skipped,
        "tags": asdict(result.tags),
        "stage": None if result.stage is None else result.stage.value,
        "error": result.error,
        "progress": [_event_payload(event) for event in result.progress],
    }


def _event_payload(event: ProgressEvent) -> dict[str, Any]:
    return {
        "source": event.source,
        "stage": event.stage.value,
        "stage_label": STAGE_LABELS[event.stage],
        "file_number": event.file_number,
        "file_total": event.file_total,
    }
