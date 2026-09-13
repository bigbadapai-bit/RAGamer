"""界面：服务端渲染的页面。

模板 + htmx 局部刷新，**没有 JavaScript 构建步骤**（ADR-0005）。每条写入路径都先是一张
普通的 HTML 表单：浏览器禁用 JavaScript 时页面照常能用，只是每次提交整页刷新一次。

htmx 只用在**有一块明显可以就地换掉的结果区**的地方——导入那条路就是：表单同时带
`action`／`method` 与 `hx-post`，服务端按 `HX-Request` 决定回整页还是回片段，两条路走的是
同一段处理逻辑。改配置与删库那几条不挂 htmx：它们的结果是「这个库现在是什么样」，
整页重渲染本来就是对的，硬做成片段反而要把半张页面拆开拼。

这一层不构造任何适配器，数据全来自组合根注入的容器。写入只做三件事——建库、改库、跑导入，
三者都直接用写入侧已有的实现（`ragamer.knowledge` 与 `ragamer.importing`），页面自己不重做
其中的判断：术语映射的增删也是调 `set_term`／`remove_term`，不在页面里拼那份映射。
**删库同理**：页面只负责把「将要清掉什么」摆出来让人确认，清理本身走 `purge_knowledge_base`，
那条路上的三处存储不经过界面。切分预览页是把库里存下来的东西读回来渲染，**没有任何编辑入口**。

改配置与删库这类动作一律走「提交 → 重定向 → 重新渲染」，不直接回 200：刷新一下就把上一次
的删除或改动再提交一遍，是这类页面上最容易踩的一个坑。

对话与评测两个页面在这里只到占位为止，完整形态是后面几张票的事。
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from ragamer.clarifying import Clarification, version_choices
from ragamer.container import Container
from ragamer.conversations import ChatStack, ConversationNotFound, Turn
from ragamer.importing import STAGE_LABELS, Importer, ImportResult
from ragamer.knowledge import (
    KnowledgeBase,
    KnowledgeBaseError,
    PurgeError,
    create_knowledge_base,
    find_knowledge_base,
    knowledge_base_of,
    list_knowledge_bases,
    purge_inventory,
    purge_knowledge_base,
    readable_knowledge_base,
    remove_term,
    set_term,
    update_knowledge_base,
    vocabulary_of,
)
from ragamer.sources import SourceDocument
from ragamer.stores.base import (
    IMAGE_PREFIX,
    UNVERSIONED,
    Chunk,
    StoreError,
    collection_name,
    image_folder,
)
from ragamer.tagging import (
    CONTENT_NATURE_NAMES,
    SUBJECT_TYPE_NAMES,
    SubjectType,
    TagVocabulary,
)

#: 模板目录。跟着包走，装成 wheel 也在。
TEMPLATES = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES))

#: 原图路由的前缀。库里存的图片地址就是对象 key（`images/…`），
#: 页面按 `/images/…` 取——多一层前缀，这个路由就只放行图片那一块。
IMAGE_ROUTE = "/images"

#: 导航。四个位置一次留齐，评测的页面在后面的票里接上。
NAV: tuple[tuple[str, str], ...] = (
    ("对话", "/chat"),
    ("知识库管理", "/kb"),
    ("导入", "/import"),
    ("评测", "/eval"),
)

#: 切片类型在界面上的叫法。`text` 是默认值，列表里不挂徽章。
CHUNK_TYPE_NAMES: Mapping[str, str] = {"text": "正文", "table": "表格", "image": "图片"}

#: 「没有内容」时占位用的短横。留白的单元格容易被看漏。
EMPTY = "—"


def create_router(container: Container, stack: ChatStack) -> APIRouter:
    """把页面挂成一个路由器，由 `ragamer.app` 与 JSON 端点装进同一个应用。

    `stack` 是读取侧接好的那一套（`ragamer.conversations.build_chat`）：对话页与 JSON
    端点打的是同一批会话、同一份缓存，所以**由装配那一处传进来**，页面不再接一遍。
    """
    router = APIRouter()
    # 进度回调用 Importer 的默认实现（落日志）：导入是同步的一整段，
    # 在页面上等的时候只有日志看得见进度。
    importer = Importer(chunks=container.chunks, embedder=container.embedder, llm=container.llm)

    @router.get("/")
    def home() -> RedirectResponse:
        """根路径进知识库管理页——界面上要做的第一件事就是建库。"""
        return RedirectResponse("/kb")

    @router.get("/kb")
    def knowledge_bases(
        request: Request, deleted: str = "", chunks: int = 0, images: int = 0
    ) -> Response:
        """知识库列表与新建表单。每个库点进去配术语映射与版本，或者删掉它。"""
        return _knowledge_bases_page(
            request,
            container,
            message=(
                f"已删除知识库 {deleted}：清掉 {chunks} 条切片、{images} 个原图，"
                "向量库、对象存储与 MongoDB 里的数据都清干净了。"
                if deleted
                else ""
            ),
        )

    @router.post("/kb")
    def create_kb(
        request: Request,
        game_id: Annotated[str, Form()] = "",
        name: Annotated[str, Form()] = "",
        subject_types: Annotated[list[str] | None, Form()] = None,
    ) -> Response:
        """新建一个知识库，建好直接进它的导入页——中间没有别的可做。"""
        game_id = game_id.strip()
        try:
            knowledge_base = KnowledgeBase.new(game_id, name, _subject_types(subject_types))
            create_knowledge_base(container.docs, knowledge_base)
        except ValueError as exc:
            # 表单填错不该是 500：把原因写在表单上方，人当场就能改
            return _knowledge_bases_page(
                request,
                container,
                error=str(exc),
                form={"game_id": game_id, "name": name},
                checked=set(subject_types or ()),
                status_code=400,
            )
        return RedirectResponse(f"/import?{urlencode({'game_id': game_id})}", status_code=303)

    @router.get("/kb/{game_id}")
    def knowledge_base_page(request: Request, game_id: str, saved: bool = False) -> Response:
        """单个库的配置页：名称、启用的类目、术语映射、现行版本，以及删库入口。"""
        return _knowledge_base_page(
            request,
            container,
            game_id,
            message="已保存。下一次导入打标与切分预览读的就是这一份。" if saved else "",
        )

    @router.post("/kb/{game_id}")
    def save_kb(
        request: Request,
        game_id: str,
        name: Annotated[str, Form()] = "",
        version: Annotated[str, Form()] = "",
        subject_types: Annotated[list[str] | None, Form()] = None,
    ) -> Response:
        """保存基本配置。**保存后立即生效**。

        术语映射不在这张表单里：它是一张可增可删的表，混进来就得靠 JavaScript 加行，
        而这里每条写入路径都要是不带脚本也能用的。映射那两条走 `/terms`，同样是存完即生效。
        """
        name = name.strip()
        version = version.strip()
        # 存回表单：出错时人不必把敲过的名字与版本重来一遍
        draft = {
            "name": name or game_id,
            "version": version,
            "checked": set(subject_types or ()),
        }
        try:
            existing = knowledge_base_of(container.docs, game_id)
            update_knowledge_base(
                container.docs,
                replace(
                    existing,
                    name=draft["name"],
                    version=version,
                    vocabulary=TagVocabulary(
                        _subject_types(subject_types), existing.vocabulary.term_mapping
                    ),
                ),
            )
        except (ValueError, KnowledgeBaseError) as exc:
            return _config_error(request, container, game_id, exc, draft=draft)
        return RedirectResponse(f"/kb/{game_id}?saved=1", status_code=303)

    @router.post("/kb/{game_id}/terms")
    def add_term(
        request: Request,
        game_id: str,
        term: Annotated[str, Form()] = "",
        kind: Annotated[str, Form()] = "",
    ) -> Response:
        """给术语映射加一条。叫法与表里已有的完全一样时改掉它的归类，不报错——那是同一条映射。

        只差大小写或首尾空白的那种会被拦下：归一之后它们本来就是同一条，静默顶掉一条不好查。
        """
        term = term.strip()
        if not term:
            return _knowledge_base_page(
                request, container, game_id, error="先填一个这个游戏里的叫法", status_code=400
            )
        try:
            set_term(container.docs, game_id, term, _subject_type(kind))
        except (ValueError, KnowledgeBaseError) as exc:
            return _config_error(request, container, game_id, exc)
        return RedirectResponse(f"/kb/{game_id}", status_code=303)

    @router.post("/kb/{game_id}/terms/delete")
    def delete_term(request: Request, game_id: str, term: Annotated[str, Form()] = "") -> Response:
        """从术语映射里去掉一条。去掉之后语料里的这个叫法就只剩模型兜底那条路了。"""
        try:
            remove_term(container.docs, game_id, term)
        except (ValueError, KnowledgeBaseError) as exc:
            return _config_error(request, container, game_id, exc)
        return RedirectResponse(f"/kb/{game_id}", status_code=303)

    @router.get("/kb/{game_id}/delete")
    def delete_kb_page(request: Request, game_id: str) -> Response:
        """删库前的确认页：先把将要清掉的东西逐条列出来，再让人决定。"""
        return _delete_page(request, container, game_id)

    @router.post("/kb/{game_id}/delete")
    def delete_kb(request: Request, game_id: str, confirm: Annotated[str, Form()] = "") -> Response:
        """真删。三处存储一并清，知识库配置排在最后（见 `purge_knowledge_base`）。"""
        try:
            knowledge_base_of(container.docs, game_id)
        except (ValueError, KnowledgeBaseError) as exc:
            return _knowledge_bases_page(
                request, container, error=str(exc), status_code=_status_of(exc)
            )
        if not confirm:
            # 删除不可逆，确认那一勾不是装饰：没勾就退回去，别把它当成手滑
            return _delete_page(
                request, container, game_id, error="先把那句确认勾上，再点删除", status_code=400
            )
        try:
            inventory = purge_knowledge_base(
                container.chunks, container.docs, container.objects, game_id
            )
        except PurgeError as exc:
            return _delete_page(request, container, game_id, error=str(exc), status_code=exc.status)
        # 回话里带上真清掉的条数：确认页写的是「将要」，这里写的是「已经」，
        # 两个数对得上才说明两处数的真是同一批东西
        query = urlencode(
            {
                "deleted": game_id,
                "chunks": inventory.chunk_count,
                "images": inventory.image_count,
            }
        )
        return RedirectResponse(f"/kb?{query}", status_code=303)

    @router.get("/import")
    def import_page(request: Request, game_id: str = "") -> Response:
        """导入页：选库、传资料、看结果。进度与 URL 导入在后面的票里。"""
        return _import_page(request, container, selected=game_id)

    @router.post("/import")
    async def run_import(
        request: Request,
        game_id: Annotated[str, Form()] = "",
        version: Annotated[str, Form()] = "",
        files: Annotated[list[UploadFile] | None, File()] = None,
    ) -> Response:
        """跑一次导入，把**逐文件**的结果渲染出来。

        htmx 发来的请求只回结果那一块，原生表单提交回整页，两块内容一模一样。
        """

        def reply(
            message: str = "",
            result: Mapping[str, Any] | None = None,
            status: int = 200,
        ) -> Response:
            """结果区该显示什么——**整页与 htmx 片段走同一个出口**。

            片段一律 200：htmx 默认不换入非 2xx 的响应，出错时回 4xx 的话人会对着一个
            空的结果区发呆。整页那条路仍报真实状态码，curl 与别的工具看得见。
            """
            if request.headers.get("HX-Request"):
                return templates.TemplateResponse(
                    request,
                    "partials/import_result.html",
                    {"result": result, "message": message, "empty": EMPTY},
                )
            return _import_page(
                request,
                container,
                selected=game_id,
                version=version,
                message=message,
                result=result,
                status_code=status,
            )

        game_id = game_id.strip()
        version = version.strip()
        sources = [
            SourceDocument(filename=file.filename or "", data=await file.read())
            for file in files or []
            # 空的文件框也会送上来一个 filename 为空的部件，那不是一份资料
            if file.filename
        ]
        if not sources:
            return reply("先选一份资料再提交")
        try:
            collection_name(game_id)
            vocabulary = vocabulary_of(container.docs, game_id)
        except ValueError as exc:
            return reply(str(exc), status=400)
        except KnowledgeBaseError as exc:
            # 状态码跟着异常走：接口与页面两条路翻出来的是同一个（见 ragamer.knowledge）
            return reply(str(exc), status=exc.status)

        results = importer.batch(
            sources,
            game_id=game_id,
            version=version or UNVERSIONED,
            vocabulary=vocabulary,
        )
        return reply(result=_import_result(results, game_id=game_id, version=version))

    @router.get("/kb/{game_id}/preview")
    def preview(
        request: Request, game_id: str, doc_title: str = "", version: str | None = None
    ) -> Response:
        """切分预览（**只读**）：把入库的结果原样渲染出来。

        取切片走的是检索那条路（`fetch_document`），所以显示的就是检索时会看见的：
        该版本的内容与未标注版本的内容一并列出（ADR-0004），后者挂个徽章说明来历。

        不带 `version` 参数时按这个库的**现行版本**取——设了现行版本却看不到它在哪儿生效，
        那个设置就只是一行字。带空串则是明确地按「未标注版本」看。
        """
        if version is None:
            version = _current_version(container, game_id)
        try:
            collection_name(game_id)
        except ValueError as exc:
            return _preview_page(request, game_id, doc_title, version, error=str(exc))
        if not doc_title:
            return _preview_page(request, game_id, doc_title, version, error="没说要预览哪一份文档")
        chunks = container.chunks.fetch_document(game_id, doc_title, version=version)
        return _preview_page(request, game_id, doc_title, version, chunks)

    # --- 对话页（T24）---

    @router.get("/chat")
    def chat_home(request: Request) -> Response:
        """还没选库：左栏把知识库列出来，右边请人选一个。"""
        return _chat_page(request, container, stack)

    @router.get("/chat/{game_id}")
    def chat_game(request: Request, game_id: str) -> Response:
        """选中一个库：左栏列它的近期会话，右边提示开一个或点一个。"""
        return _chat_page(request, container, stack, game_id=game_id)

    @router.get("/chat/{game_id}/{session_id}")
    def chat_session(request: Request, game_id: str, session_id: str) -> Response:
        """一次会话的正文。刷新页面回到这里——历史在服务端存着。"""
        return _chat_page(request, container, stack, game_id=game_id, session_id=session_id)

    @router.get("/chat/{game_id}/{session_id}/turns")
    def chat_turns(request: Request, game_id: str, session_id: str) -> Response:
        """会话正文那一段。**流答完之后 htmx 拿它换掉整块**。

        换掉而不是在页面里拼：引用、图片、版本徽章都由模板渲染，拼第二遍迟早与第一遍
        长得不一样（导入那条路是同一个打法）。片段与整页渲染的是同一份数据，所以两条路
        看到的东西一模一样。
        """
        try:
            conversation = stack.chat.open(session_id)
        except ConversationNotFound as exc:
            return _page(
                request,
                "partials/turns.html",
                "对话",
                "/chat",
                error=str(exc),
                status_code=404,
                turns=[],
                game_id=game_id,
                session_id=session_id,
            )
        return templates.TemplateResponse(
            request,
            "partials/turns.html",
            {"turns": _turn_rows(conversation.turns), "error": ""},
        )

    @router.post("/chat/{game_id}")
    def start_session(request: Request, game_id: str, version: str = Form("")) -> Response:
        """开一次会话，然后转过去。「提交 → 重定向」：刷新一下不会又开一个。"""
        try:
            knowledge = readable_knowledge_base(container.docs, game_id)
        except KnowledgeBaseError as exc:
            return _chat_page(request, container, stack, game_id=game_id, error=str(exc))
        conversation = stack.chat.start(game_id=game_id, version=version or knowledge.version)
        return RedirectResponse(f"/chat/{game_id}/{conversation.session_id}", status_code=303)

    @router.post("/chat/{game_id}/{session_id}/ask")
    def ask_whole(
        request: Request, game_id: str, session_id: str, question: str = Form("")
    ) -> Response:
        """不用 JavaScript 时走这条：整轮跑完再回来。

        页面上的提问框默认由 `EventSource` 接管（逐字流式），这条是它禁用脚本时的退路——
        结果一样，只是答案一次给全。澄清也一样：判不准时这一页把候选按钮渲染出来。
        """
        return _whole_turn(request, container, stack, game_id, session_id, question)

    @router.post("/chat/{game_id}/{session_id}/resolve")
    def resolve_whole(
        request: Request,
        game_id: str,
        session_id: str,
        pending_id: str = Form(...),
        label: str = Form(...),
        question: str = Form(""),
    ) -> Response:
        """不用 JavaScript 时点澄清按钮走这条：从暂停点继续，整轮跑完再回来。"""
        return _whole_turn(
            request,
            container,
            stack,
            game_id,
            session_id,
            question,
            pending_id=pending_id,
            label=label,
        )

    @router.post("/chat/{game_id}/{session_id}/version")
    def change_version(
        request: Request, game_id: str, session_id: str, version: str = Form("")
    ) -> Response:
        """换这次会话选定的版本。**下一轮起按它走**，直到再换一次。"""
        try:
            stack.chat.set_version(session_id, version)
        except ConversationNotFound as exc:
            return _chat_page(
                request, container, stack, game_id=game_id, error=str(exc), status_code=404
            )
        return RedirectResponse(f"/chat/{game_id}/{session_id}", status_code=303)

    @router.get(IMAGE_ROUTE + "/{key:path}")
    def image(key: str) -> Response:
        """原图。库里存的图片地址就是对象 key，答案带回来的也是它，页面照它取。

        路径上那一段 `images/` 与对象 key 的顶层前缀是同一段（`IMAGE_PREFIX`）：
        这样这个路由只放行图片那一块，不会变成一个万能的对象读口。
        """
        address = f"{IMAGE_PREFIX}/{key}"
        try:
            data = container.objects.get(address)
        except StoreError as exc:
            raise HTTPException(status_code=404, detail=f"没有这张图：{exc}") from exc
        return Response(
            content=data,
            media_type=_content_type(address),
            # key 里带着来源内容的摘要（§1.2），同一张图的内容不会变——可以放心让浏览器留着
            headers={"Cache-Control": "private, max-age=86400"},
        )

    @router.get("/eval")
    def evaluation(request: Request) -> Response:
        return _page(
            request,
            "placeholder.html",
            "评测",
            "/eval",
            note="这个页面还没做。评测集定下来之后再开。",
        )

    return router


# --- 页面 ---


def _chat_page(
    request: Request,
    container: Container,
    stack: ChatStack,
    *,
    game_id: str = "",
    session_id: str = "",
    question: str = "",
    error: str = "",
    clarification: Clarification | None = None,
    status_code: int = 200,
) -> Response:
    """对话页。两级导航都在这一页上：左栏是知识库，选中之后下面接着列它的近期会话。

    **左栏是知识库列表，不是「问过哪些游戏」**：建了库一次没聊过也要在里面——它是
    「我能问什么」的入口，不是历史记录。会话列表才是历史，按最后活跃倒序。
    """
    context: dict[str, Any] = {
        "bases": [
            {
                "game_id": base.game_id,
                "name": base.name,
                "version": base.version,
                "current": base.game_id == game_id,
            }
            for base in list_knowledge_bases(container.docs)
        ],
        "game_id": game_id,
        "session_id": session_id,
        "sessions": [],
        "turns": [],
        "versions": [],
        "hot": (),
        "question": question,
        "clarification": _clarification_view(clarification, question) if clarification else None,
    }
    conversation = None
    if session_id:
        try:
            conversation = stack.chat.open(session_id)
        except ConversationNotFound as exc:
            return _page(
                request,
                "chat.html",
                "对话",
                "/chat",
                error=error or str(exc),
                status_code=404,
                **context,
            )
        context["turns"] = _turn_rows(conversation.turns)
    if game_id:
        try:
            knowledge = readable_knowledge_base(container.docs, game_id)
        except (KnowledgeBaseError, ValueError) as exc:
            # id 不合法与库不存在对页面是同一件事：这个库点不进来。两者都归 404——
            # 页面上没有「换个 id 再试」这个动作，那是建库页的事。
            return _page(
                request,
                "chat.html",
                "对话",
                "/chat",
                error=error or str(exc),
                status_code=_status_of(exc) if isinstance(exc, KnowledgeBaseError) else 404,
                **context,
            )
        selected = conversation.version if conversation is not None else knowledge.version
        context["sessions"] = [
            {
                "session_id": summary.session_id,
                "title": summary.title,
                "updated_at": summary.updated_at,
                "current": summary.session_id == session_id,
            }
            for summary in stack.chat.list_for_game(game_id)
        ]
        context["versions"] = _version_options(container, game_id, selected=selected)
        context["hot"] = stack.cache.top_questions(game_id)
    return _page(
        request, "chat.html", "对话", "/chat", error=error, status_code=status_code, **context
    )


def _whole_turn(
    request: Request,
    container: Container,
    stack: ChatStack,
    game_id: str,
    session_id: str,
    question: str,
    *,
    pending_id: str = "",
    label: str = "",
) -> Response:
    """一轮问答跑到底，然后回整页。**这是没开 JavaScript 时的那条路**。

    流式那条路把答案一片一片推给页面；这一条等它全跑完，再把结果渲染回来。两条路走的是
    同一个 `Chat.ask`，所以落库、澄清、版本回落全都一致——差别只在答案怎么出来。

    澄清那一轮不重定向：页面要把候选按钮渲染出来，而那段状态（暂停点、候选）不在会话里，
    只在这一刻手上。所以直接把它渲染进这一页。
    """
    if not question.strip():
        return _chat_page(
            request,
            container,
            stack,
            game_id=game_id,
            session_id=session_id,
            question=question,
            error="问题不能为空",
        )
    try:
        # 先读一次会话拿它绑的库：现行版本要从那里取（ADR-0004）。
        # `chat.ask` 自己还会再读一次，那是它的事——会话不可变，`ask` 落的是新的一份。
        bound = stack.chat.open(session_id).game_id
    except ConversationNotFound as exc:
        return _chat_page(
            request,
            container,
            stack,
            game_id=game_id,
            error=str(exc),
            status_code=404,
        )
    replies = list(
        stack.chat.ask(
            session_id,
            question,
            current_version=_current_version(container, bound),
            pending_id=pending_id,
            label=label,
        )
    )
    outcome = replies[-1] if replies else None
    if isinstance(outcome, Clarification):
        return _chat_page(
            request,
            container,
            stack,
            game_id=game_id,
            session_id=session_id,
            question=question,
            clarification=outcome,
        )
    return RedirectResponse(f"/chat/{game_id}/{session_id}", status_code=303)


def _clarification_view(clarification: Clarification, question: str) -> dict[str, Any]:
    """一次反问渲染成按钮要的那几样。`value` 是回传的取值（游戏是 id），
    `label` 是按钮上显示的（游戏是显示名）——两者不一样的理由见 `Choice`。"""
    return {
        "pending_id": clarification.pending_id,
        "prompt": clarification.prompt,
        "question": question,
        "choices": [
            {"label": choice.label, "value": choice.value} for choice in clarification.choices
        ],
    }


def _version_options(container: Container, game_id: str, *, selected: str) -> list[dict[str, Any]]:
    """版本下拉的选项：这个库里**真实有过**的版本，外加「跟随知识库现行版本」。

    候选只从语料里读（`ragamer.clarifying.version_choices`），不让人手输一个：输一个库里
    没有的版本，检索会静默查空，而界面上看不出区别——与澄清反问那一条是同一个理由。
    """
    base = find_knowledge_base(container.docs, game_id)
    current = base.version if base else ""
    options = [
        {
            "value": "",
            "label": f"跟随现行版本（{current}）" if current else "跟随现行版本",
            "current": selected == "",
        }
    ]
    options += [
        {"value": choice.value, "label": choice.label, "current": selected == choice.value}
        for choice in version_choices(container.chunks, game_id)
    ]
    return options


def _turn_rows(turns: Sequence[Turn]) -> list[dict[str, Any]]:
    """会话里的消息 → 模板要的那几样。用户那一侧只有原话，模型那一侧还带来源与图。"""
    return [
        {
            "role": turn.role,
            "content": turn.content,
            "version": turn.version,
            "citations": [
                {
                    "index": citation.index,
                    "label": citation.label,
                    "doc_title": citation.doc_title,
                    "ancestor_path": citation.ancestor_path,
                }
                for citation in turn.citations
            ],
            # 地址本身就是对象 key（`images/…`），而路由那一段也是 `images/`：
            # 去掉一层再拼，页面上看到的就是 `/images/black_myth/…`，不是 `images` 叠两遍
            "images": [
                {
                    "url": f"{IMAGE_ROUTE}/{address.removeprefix(IMAGE_PREFIX + '/')}",
                    "name": address.rsplit("/", 1)[-1],
                }
                for address in turn.images
            ],
        }
        for turn in turns
    ]


def _content_type(address: str) -> str:
    """按扩展名给一个内容类型。认不出来时按二进制流——浏览器仍会当图显示。"""
    guessed, _ = mimetypes.guess_type(address)
    return guessed or "application/octet-stream"


def _page(
    request: Request,
    template: str,
    title: str,
    section: str,
    *,
    status_code: int = 200,
    **context: Any,
) -> Response:
    """渲染一个整页。`section` 决定导航里哪一项高亮。"""
    return templates.TemplateResponse(
        request,
        template,
        {"title": title, "nav": _nav(section), "empty": EMPTY, **context},
        status_code=status_code,
    )


def _knowledge_bases_page(
    request: Request,
    container: Container,
    *,
    message: str = "",
    error: str = "",
    form: Mapping[str, str] | None = None,
    checked: set[str] | None = None,
    status_code: int = 200,
) -> Response:
    options = _subject_type_options()
    if checked is None:  # 刚打开时七类全勾上
        checked = {option["value"] for option in options}
    return _page(
        request,
        "knowledge_bases.html",
        "知识库管理",
        "/kb",
        bases=_knowledge_base_rows(list_knowledge_bases(container.docs)),
        subject_types=options,
        checked=checked,
        message=message,
        error=error,
        form=form or {"game_id": "", "name": ""},
        status_code=status_code,
    )


def _knowledge_base_page(
    request: Request,
    container: Container,
    game_id: str,
    *,
    message: str = "",
    error: str = "",
    draft: Mapping[str, Any] | None = None,
    status_code: int = 200,
) -> Response:
    """单个库的配置页。

    库不存在、或者 id 不合法时**回列表页**并把原因写在上面，而不是丢一张光秃秃的 404：
    列表页上正好能建一个，那多半就是人本来要做的事。

    `draft` 是刚提交上来的那份值，出错时用它覆盖页面上显示的内容——名字与勾选不必重来。
    传了空集合就是空集合，不能跟「没传」混为一谈：一个类目都没勾正是一种要报出来的错。
    """
    try:
        knowledge_base = knowledge_base_of(container.docs, game_id)
    except (ValueError, KnowledgeBaseError) as exc:
        return _knowledge_bases_page(
            request, container, error=str(exc), status_code=_status_of(exc)
        )
    fields: dict[str, Any] = {
        "name": knowledge_base.name,
        "version": knowledge_base.version,
        "checked": {kind.value for kind in knowledge_base.vocabulary.subject_types},
        **(draft or {}),
    }
    return _page(
        request,
        "knowledge_base.html",
        knowledge_base.name,
        "/kb",
        game_id=game_id,
        problem=knowledge_base.problem,
        subject_types=_subject_type_options(),
        # 加映射时只列这个库启用的类目：归到一个没启用的类目上，那条映射永远落不进标签
        mapping_kinds=[
            {"value": kind.value, "label": SUBJECT_TYPE_NAMES[kind]}
            for kind in knowledge_base.vocabulary.subject_types
        ],
        rows=_mapping_rows(knowledge_base),
        message=message,
        error=error,
        status_code=status_code,
        **fields,
    )


def _config_error(
    request: Request,
    container: Container,
    game_id: str,
    exc: Exception,
    *,
    draft: Mapping[str, Any] | None = None,
) -> Response:
    """改配置出错时**回这个库的配置页**，把原因写在上面。

    库不存在或 id 不合法时，`_knowledge_base_page` 自己会退到列表页——那儿正好能建一个。
    状态码跟着异常走，与 JSON 端点那边翻出来的是同一个。
    """
    return _knowledge_base_page(
        request, container, game_id, error=str(exc), draft=draft, status_code=_status_of(exc)
    )


def _current_version(container: Container, game_id: str) -> str:
    """这个库的现行版本。

    读不出来（库不在、配置坏了、id 不合法）就按未标注版本走：这一处只影响预览默认取哪个
    版本的切片，为它把整个预览页拦下来不值得。
    """
    try:
        return knowledge_base_of(container.docs, game_id).version
    except (ValueError, KnowledgeBaseError):
        return UNVERSIONED


def _delete_page(
    request: Request,
    container: Container,
    game_id: str,
    *,
    error: str = "",
    status_code: int = 200,
) -> Response:
    """删库确认页。

    条数取自 `purge_inventory`，与实际清理**同一份数法**：页面上写「12 张原图」而真删掉
    15 张，那份确认就成了摆设。数不出来时干脆不让人确认——闭着眼睛删不是确认。
    """
    try:
        knowledge_base = knowledge_base_of(container.docs, game_id)
    except (ValueError, KnowledgeBaseError) as exc:
        return _knowledge_bases_page(
            request, container, error=str(exc), status_code=_status_of(exc)
        )
    try:
        inventory = purge_inventory(container.chunks, container.objects, game_id)
    except Exception as exc:
        # 条数报不出来就不让人确认：闭着眼睛删不是确认。兜住全部异常的理由与
        # `purge_knowledge_base` 那边一样——适配器只把「连不上」包成 StoreError
        return _knowledge_bases_page(
            request, container, error=f"清点不出这个库占用的数据：{exc}", status_code=500
        )
    return _page(
        request,
        "knowledge_base_delete.html",
        f"删除知识库 {knowledge_base.name}",
        "/kb",
        game_id=game_id,
        name=knowledge_base.name,
        inventory=inventory,
        # 带尾随斜杠，与实际清理用的是同一个前缀（见 `image_folder`）——页面上写着
        # `images/black_myth`，删的时候按它去匹配，就会连 `black_myth_2` 的图一起收走
        prefix=image_folder(game_id),
        error=error,
        status_code=status_code,
    )


def _import_page(
    request: Request,
    container: Container,
    *,
    selected: str = "",
    version: str = "",
    message: str = "",
    result: Mapping[str, Any] | None = None,
    status_code: int = 200,
) -> Response:
    """导入页整页。`message` 与 `result` 落在结果区里，与 htmx 拿到的片段是同一份内容。

    **不经 base.html 的那条 `error` 通道**：那条画在表单上方，片段换入时看不见；
    要显示的东西得在结果区里，两条路才一致。
    """
    bases = _knowledge_base_rows(list_knowledge_bases(container.docs))
    known = {base["game_id"] for base in bases}
    return _page(
        request,
        "import.html",
        "导入",
        "/import",
        bases=bases,
        # 直接打开这个页面时默认选中第一个库，省得每回都挑一次
        selected=selected if selected in known else next(iter(sorted(known)), ""),
        version=version,
        message=message,
        result=result,
        status_code=status_code,
    )


def _preview_page(
    request: Request,
    game_id: str,
    doc_title: str,
    version: str,
    chunks: Sequence[Chunk] = (),
    error: str = "",
) -> Response:
    if not chunks and not error:
        error = "这个文档在这个版本下没有切片。是不是版本不对，或者它其实没导进来？"
    return _page(
        request,
        "preview.html",
        "切分预览",
        "/kb",
        game_id=game_id,
        doc_title=doc_title,
        version=version,
        version_label=version or "未标注版本",
        subject_name=chunks[0].subject_name if chunks else "",
        chunks=_chunk_rows(chunks, version),
        error=error,
        status_code=200 if chunks else 404,
    )


# --- 页面要用的数据 ---


def _nav(current: str) -> list[dict[str, Any]]:
    return [{"label": label, "url": url, "active": url == current} for label, url in NAV]


def _knowledge_base_rows(bases: Sequence[KnowledgeBase]) -> list[dict[str, Any]]:
    return [
        {
            "game_id": base.game_id,
            "name": base.name,
            "version": base.version,
            "subject_types": _names(
                SUBJECT_TYPE_NAMES, (kind.value for kind in base.vocabulary.subject_types)
            ),
            "problem": base.problem,
        }
        for base in bases
    ]


def _mapping_rows(knowledge_base: KnowledgeBase) -> list[dict[str, Any]]:
    """术语映射表，按叫法排列。

    `enabled` 为假的那几条落不进标签——映射归到的类目这个库没启用，`_classify` 会丢掉它。
    表上照旧列出来（那是库里真有的数据），但标一句，免得人对着一条永远不生效的映射发呆。
    """
    enabled = set(knowledge_base.vocabulary.subject_types)
    return [
        {
            "term": term,
            "kind": SUBJECT_TYPE_NAMES[kind],
            "enabled": kind in enabled,
        }
        for term, kind in sorted(knowledge_base.vocabulary.term_mapping.items())
    ]


def _subject_type_options() -> list[dict[str, str]]:
    return [{"value": kind.value, "label": SUBJECT_TYPE_NAMES[kind]} for kind in SubjectType]


def _status_of(exc: Exception) -> int:
    """这个异常该报哪个状态码。

    `KnowledgeBaseError` 自己带着状态码（见 `ragamer.knowledge`），两个 HTTP 面从这里取，
    翻法才一致；`ValueError` 那几个是表单填错，400。
    """
    return int(getattr(exc, "status", 400))


def _import_result(
    results: Sequence[ImportResult], *, game_id: str, version: str
) -> dict[str, Any]:
    """一次导入的逐文件结果。**失败也是结果**，和成功的一起列出来。"""
    rows = [
        {
            "filename": result.filename,
            "doc_title": result.doc_title,
            "ok": result.ok,
            "chunk_count": result.chunk_count,
            "skipped": result.skipped,
            "stage_label": EMPTY if result.stage is None else STAGE_LABELS[result.stage],
            "error": result.error or "",
            "subject_name": result.tags.subject_name,
            "subject_types": _names(SUBJECT_TYPE_NAMES, result.tags.subject_type),
            "content_natures": _names(CONTENT_NATURE_NAMES, result.tags.content_nature),
            "preview_url": _preview_url(game_id, result.doc_title, version),
        }
        for result in results
    ]
    return {
        "rows": rows,
        "imported": sum(1 for row in rows if row["ok"]),
        "failed": sum(1 for row in rows if not row["ok"]),
    }


def _chunk_rows(chunks: Sequence[Chunk], version: str) -> list[dict[str, Any]]:
    return [
        {
            "index": chunk.chunk_index,
            "content": chunk.content,
            "content_meta": chunk.content_meta,
            "ancestor_path": chunk.ancestor_path,
            "subject_types": _names(SUBJECT_TYPE_NAMES, chunk.subject_type),
            "content_natures": _names(CONTENT_NATURE_NAMES, chunk.content_nature),
            "game_terms": list(chunk.game_terms),
            "chunk_type": CHUNK_TYPE_NAMES.get(chunk.chunk_type, chunk.chunk_type),
            "is_plain_text": chunk.chunk_type == "text",
            # 未标注版本的内容在任何版本下都会被一并取回（ADR-0004）。混进来时标一下，
            # 免得看的人把它当成这次导入的那一份
            "other_version": chunk.version != version,
        }
        for chunk in chunks
    ]


def _preview_url(game_id: str, doc_title: str, version: str) -> str:
    return f"/kb/{game_id}/preview?{urlencode({'doc_title': doc_title, 'version': version})}"


def _subject_type(value: str) -> SubjectType:
    """表单里的一个类目取值 → 枚举。认不出的当场报错，不静默丢掉。

    静默丢掉的后果是标签少一片，而页面上看不出任何异样——这个词表是用户自己填的，
    填错了要当场告诉他。
    """
    try:
        return SubjectType(value)
    except ValueError as exc:
        known = "、".join(kind.value for kind in SubjectType)
        raise ValueError(f"不认识的主体类型 {value!r}。可用的有：{known}") from exc


def _subject_types(values: Sequence[str] | None) -> tuple[SubjectType, ...]:
    """表单里勾选的类目 → 枚举。

    一个都没勾时返回空元组，由 `TagVocabulary` 报「至少要启用一个主体类型」——
    这里不替它兜底成「全开」。
    """
    return tuple(_subject_type(value) for value in values or ())


def _names(labels: Mapping[Any, str], values: Iterable[str]) -> list[str]:
    """枚举取值 → 界面上的中文叫法。认不出的取值原样显示，不吞掉。"""
    return [labels.get(value, value) for value in values]
