"""HTTP 端点：写入侧与读取侧对外的唯一入口。

五个端点分两侧：

- **写入侧** `POST /api/kb/{game_id}/import`。批量提交、**逐文件独立**：某个文件失败时
  其余照常入库，失败的那个在结果里带文件名与失败阶段。
- **读取侧** `POST /api/chat/sessions`、`GET /api/chat/sessions`（按库列会话）、
  `GET /api/chat/sessions/{session_id}`、`GET /api/chat/sessions/{session_id}/ask`。
  开会话、列会话、把历史读回来、逐字问一句（SSE）。对话页那一层在后面的票里接。

对话那几个只做 HTTP 这一层的事：会话不存在翻成 404、问题为空翻成 400、一轮问答翻成
SSE 事件。**「这一轮算不算问完」「要不要写进历史」在 `ragamer.conversations` 里**，
这一层不重复判断——判两遍迟早会分岔，而分岔的那一次表现为「刷新之后历史少了一轮」。

知识库元数据从 MongoDB 读（`knowledge_bases` 集合，id 就是游戏 id）：打标要用的词表
——启用了哪些主体类型、这个游戏的术语映射——就在它里面（docs/ARCHITECTURE.md §2.3），
检索回落哪个版本也从它取（§2.4）。形状与判断在 `ragamer.knowledge`，界面那条写入路径
用的是同一份。知识库列表同时是**游戏候选**：给模型的是显示名，拿回来再换回 id，
理由见 `_games`。
本层**一个适配器都不构造**，全部来自组合根（`ragamer.container`），缝因此立得住。

这是**只有 JSON 端点**的应用；给人看的页面由 `ragamer.web` 挂上去，两者在
`ragamer.app` 里装成同一个应用。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ragamer.answering import Answerer, Citation
from ragamer.container import Container
from ragamer.conversations import (
    Chat,
    Conversation,
    ConversationNotFound,
    ConversationSummary,
    Game,
    Reply,
    Sources,
    Status,
)
from ragamer.importing import STAGE_LABELS, Importer, ImportResult, ProgressEvent
from ragamer.knowledge import (
    KB_COLLECTION,
    KnowledgeBase,
    KnowledgeBaseError,
    readable_knowledge_base,
)
from ragamer.llm import LlmError
from ragamer.logging import get_logger
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


def create_app(container: Container) -> FastAPI:
    """把组合根里那套依赖接成 ASGI 应用。"""
    app = FastAPI(title="RAGamer", summary="游戏攻略 RAG 助手")
    importer = Importer(
        chunks=container.chunks,
        embedder=container.embedder,
        llm=container.llm,
        parser=container.parser,
        objects=container.objects,
        # 导入完成时按游戏前缀清缓存（架构文档 §4）：语料变了，基于旧语料的答案不该再命中
        cache=container.cache,
    )
    chat = Chat(
        docs=container.docs,
        answerer=Answerer(
            chunks=container.chunks,
            embedder=container.embedder,
            reranker=container.reranker,
            llm=container.llm,
        ),
        llm=container.llm,
    )

    @app.post("/api/kb/{game_id}/import")
    async def import_sources(
        game_id: str,
        files: Annotated[list[UploadFile], File(description="要导入的资料，可多份")],
        version: Annotated[
            str, Form(description="这次导入标注的版本，留空即未标注版本")
        ] = UNVERSIONED,
    ) -> dict[str, Any]:
        """批量导入。某个文件失败时其余照常入库，失败信息带文件名与失败阶段。"""
        _check_game_id(game_id)
        vocabulary = _vocabulary(container, game_id)
        sources = [
            SourceDocument(filename=file.filename or "", data=await file.read()) for file in files
        ]

        results = importer.batch(sources, game_id=game_id, version=version, vocabulary=vocabulary)
        return {
            "game_id": game_id,
            "version": version,
            "imported": sum(1 for result in results if result.ok),
            "failed": sum(1 for result in results if not result.ok),
            "results": [_result_payload(result) for result in results],
        }

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
    def list_sessions(game_id: str) -> dict[str, Any]:
        """一个知识库下的会话，按最后活跃倒序。**左栏那一份列表**。

        只回标题与时间——正文在 `GET /api/chat/sessions/{session_id}` 那一条上取。
        库里一条会话都没有时回空列表，不是 404；**「还没聊过」是正常状态**。
        知识库本身不存在则是 404，与建会话同一个口径：那是游戏选错了。
        """
        _check_game_id(game_id)
        _knowledge_base(container, game_id)
        return _sessions_payload(game_id, chat.list_for_game(game_id))

    @app.get("/api/chat/sessions/{session_id}")
    def read_session(session_id: str) -> dict[str, Any]:
        """一次会话的全部问答。**刷新页面靠它把历史拿回来**——历史在服务端，不在页面里。

        空会话也是一次正常的返回（`turns` 为空列表），与「没有这个会话」分得很开：
        后者是 404。
        """
        return _conversation_payload(_conversation(chat, session_id))

    @app.get("/api/chat/sessions/{session_id}/ask")
    def ask(session_id: str, question: str, version: str = "") -> StreamingResponse:
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
        """
        if not question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        # 先读一次会话：不存在当场 404，顺带拿到它绑的知识库（现行版本要从那里取）。
        # `chat.ask` 自己还会再读一次，那是它的事——会话是上一次请求写下的，
        # 这一层手里这一份只用来决定「去哪个库问」。
        knowledge = _knowledge_base(container, _conversation(chat, session_id).game_id)
        replies = chat.ask(
            session_id,
            question,
            version=version,
            current_version=knowledge.version,
            games=_games(container),
        )
        return StreamingResponse(
            _events(replies), media_type="text/event-stream", headers=SSE_HEADERS
        )

    return app


class SessionRequest(BaseModel):
    """新建会话的请求体。"""

    #: 问哪个知识库（游戏 id）。
    game_id: str
    #: 这次会话选定的版本。留空即不选，检索时回落知识库的现行版本。
    version: str = ""


def _events(replies: Iterator[Reply]) -> Iterator[str]:
    """一轮问答 → SSE 字节流。

    五种事件，**前四种按发生的先后**：

    | 事件 | 几条 | 什么时候 |
    |---|---|---|
    | `status` | 若干 | 理解、检索、生成三步各自开始之前 |
    | `citations` | 一条 | 检索完、拿到来源 |
    | `delta` | 若干 | 正文一片一片来 |
    | `done` | 一条 | 正常收尾 |
    | `error` | 至多一条 | 中途失败，代替 `done` |

    `status` 是给「不用干等」用的：提问到第一个字之间隔着两次实打实的等待（一次模型
    往返加一次检索），没有它，界面在那一段里没有任何东西可显示。

    **失败也必须发成一个事件**：响应头在第一个事件之前就出去了，状态码此刻改不了。
    对浏览器原生的 `EventSource` 来说这一条反而是好事——非 2xx 时它不给你任何细节，
    只有流里的 `error` 带得回原因。收不到 `done` 就是这一轮没有正常结束，会话里相应地
    什么都没写（见 `ragamer.conversations`）。

    **兜住「断开不留痕」的不是这里，是落库的时机**：会话只在正文全部收完之后才写，
    所以流在半路停住时它一个字都没写。消费方把生成器丢掉时，这里收到的是
    `GeneratorExit`——它不是 `Exception`，下面那两个 `except` 接不住它，于是它一路把
    上游那个生成器也关掉，模型那边的请求跟着结束。

    两类失败分得开：**生成挂掉**是模型这一次不行，按 WARNING；**检索或落库挂掉**
    （`StoreError` / `ModelOutputError`）是系统性的，按 ERROR——排查时看的不是同一个地方。
    """
    yield SSE_RETRY
    try:
        for reply in replies:
            if isinstance(reply, Status):
                yield _event("status", {"text": reply.text})
            elif isinstance(reply, Sources):
                payload = {"citations": [_citation_payload(item) for item in reply.citations]}
                yield _event("citations", payload)
            else:
                yield _event("delta", {"text": reply.text})
    except LlmError as exc:
        logger.warning("生成中途失败，这一轮不写进会话：%s", exc)
        yield _event("error", {"message": str(exc)})
        return
    except (ModelOutputError, StoreError) as exc:
        logger.error("检索或落库失败，这一轮没有答案也不写进会话：%s", exc)
        yield _event("error", {"message": str(exc)})
        return
    yield _event("done", {})


def _event(name: str, payload: Mapping[str, Any]) -> str:
    """一个 SSE 事件。

    `data` 里放 JSON 而不是裸文本：答案里换行是常事，按 SSE 的多行 `data:` 规则拼要自己
    处理折行与转义，交给 `json.dumps` 就只有一行。`ensure_ascii=False` 是为了日志与
    `curl` 看到的还是中文。
    """
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


def _sessions_payload(game_id: str, summaries: Sequence[ConversationSummary]) -> dict[str, Any]:
    """会话列表对外的样子。

    `ConversationSummary` 天生装不下正文，所以这里不必再挑一遍字段——列表就三样：
    会话 id、标题、最后活跃时刻。正文在单条会话那一条端点上取。
    """
    return {"game_id": game_id, "sessions": [asdict(summary) for summary in summaries]}


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
    return {**asdict(citation), "label": citation.label}


def _result_payload(result: ImportResult) -> dict[str, Any]:
    """一个文件的结果。失败时 `stage` 与 `error` 一起给出：界面要能说清卡在哪一步。"""
    return {
        "filename": result.filename,
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
        "filename": event.filename,
        "stage": event.stage.value,
        "stage_label": STAGE_LABELS[event.stage],
        "file_number": event.file_number,
        "file_total": event.file_total,
    }
