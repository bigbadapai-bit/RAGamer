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
那条路上的四处存储不经过界面。切分预览页是把库里存下来的东西读回来渲染，**没有任何编辑入口**。

改配置与删库这类动作一律走「提交 → 重定向 → 重新渲染」，不直接回 200：刷新一下就把上一次
的删除或改动再提交一遍，是这类页面上最容易踩的一个坑。

对话页（T24）也在这里：两级导航、会话正文、澄清按钮、版本切换与热门问题。评测那个
页面只到占位为止，完整形态是后面几张票的事。
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

from ragamer.clarifying import Clarification, NotACandidate, UnknownPending, version_choices
from ragamer.container import Container, build_importer
from ragamer.conversations import (
    CONVERSATIONS,
    ChatStack,
    ConversationNotFound,
    Turn,
    parse_session_cursor,
)
from ragamer.importing import STAGE_LABELS, ImportResult, SourceKind
from ragamer.jobs import ImportJobs, JobItem, JobSnapshot
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
from ragamer.llm import LlmError
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
from ragamer.web.markdown import to_html

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

#: 任务号认不出来时说的话。服务重启过、或者那条早被丢掉了（只留最近几条），两种都算。
_GONE = "这个导入任务不在了：服务重启过，或者它已经跑完很久、被新任务挤掉了。再提交一次吧。"


def create_router(container: Container, stack: ChatStack) -> APIRouter:
    """把页面挂成一个路由器，由 `ragamer.app` 与 JSON 端点装进同一个应用。

    `stack` 是读取侧接好的那一套（`ragamer.conversations.build_chat`）：对话页与 JSON
    端点打的是同一批会话、同一份缓存，所以**由装配那一处传进来**，页面不再接一遍。
    """
    router = APIRouter()
    # 与 JSON 端点共用组合根里那一份接线（进度回调也是 Importer 的默认实现：落日志）。
    # 各自拼一遍的话，页面这条会静默少接几样——PDF、图片与网址在界面上就永远用不了。
    importer = build_importer(container)
    # 导入在它自己的线程里跑，页面按任务号来问进度（见 `ragamer.jobs`）
    jobs = ImportJobs(importer)

    @router.get("/")
    def home() -> RedirectResponse:
        """根路径进知识库管理页——界面上要做的第一件事就是建库。"""
        return RedirectResponse("/kb")

    @router.get("/kb")
    def knowledge_bases(
        request: Request,
        deleted: str = "",
        chunks: int = 0,
        images: int = 0,
        sessions: int = 0,
    ) -> Response:
        """知识库列表与新建表单。每个库点进去配术语映射与版本，或者删掉它。"""
        return _knowledge_bases_page(
            request,
            container,
            message=(
                f"已删除知识库 {deleted}：清掉 {chunks} 条切片、{images} 个原图、"
                f"{sessions} 条会话，缓存也一并清空了。"
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
        """真删。**四处一并清**（向量库、对象存储、会话、缓存），配置排在最后。"""
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
                chunks=container.chunks,
                docs=container.docs,
                objects=container.objects,
                cache=container.cache,
                game_id=game_id,
                sessions=CONVERSATIONS,
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
                "sessions": inventory.session_count,
            }
        )
        return RedirectResponse(f"/kb?{query}", status_code=303)

    @router.get("/import")
    def import_page(request: Request, game_id: str = "", job: str = "") -> Response:
        """导入页：选库、传资料或填网址、看这一次跑到哪了。

        `job` 是任务号。它让**刷新之后还看得见这一批**——不带脚本那条路提交完就是
        跳到这里来的（见 `run_import`）。
        """
        return _import_page(request, container, jobs, selected=game_id, job=job)

    @router.get("/import/result")
    def import_result(request: Request, job: str = "") -> Response:
        """只回结果那一块。跑着的时候 htmx 每秒来问一次，跑到哪就换到哪。"""
        snapshot = jobs.snapshot(job)
        return templates.TemplateResponse(
            request,
            "partials/import_result.html",
            {
                "result": None
                if snapshot is None
                else _job_view(snapshot, accept=_accept(container)),
                "notice": "" if snapshot is not None else _GONE,
                "empty": EMPTY,
            },
        )

    @router.post("/import")
    async def run_import(
        request: Request,
        game_id: Annotated[str, Form()] = "",
        version: Annotated[str, Form()] = "",
        urls: Annotated[str, Form()] = "",
        files: Annotated[list[UploadFile] | None, File()] = None,
    ) -> Response:
        """收下一次提交，**立刻返回**——导入在后台跑，结果区自己轮询。

        文件与网址是同一次提交的两半：一次请求里两条都收，交给同一个任务——编号连着排，
        文档标识也共用一份认领表，同一次提交里一个文件与一个网址撞上才不会互相覆盖。

        导入是分钟级的一段（MinerU、二次 OCR、出网抓取），压在请求里跑的话页面什么都
        看不见，而且会把整个应用卡住（见 `ragamer.jobs`）。

        htmx 发来的请求只回结果那一块并换掉地址栏；原生表单提交回一个重定向，
        两条路都落在同一个页面上。
        """

        def reply(
            message: str = "",
            status: int = 200,
        ) -> Response:
            """出错时说一句人话——**整页与 htmx 片段走同一个出口**。

            片段一律 200：htmx 默认不换入非 2xx 的响应，出错时回 4xx 的话人会对着一个
            空的结果区发呆。整页那条路仍报真实状态码，curl 与别的工具看得见。
            """
            if request.headers.get("HX-Request"):
                return templates.TemplateResponse(
                    request,
                    "partials/import_result.html",
                    {"result": None, "notice": message, "empty": EMPTY},
                )
            return _import_page(
                request,
                container,
                jobs,
                selected=game_id,
                version=version,
                message=message,
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
        links = _url_lines(urls)
        if not sources and not links:
            return reply("先选一份资料或填一个网址再提交")
        try:
            collection_name(game_id)
            vocabulary = vocabulary_of(container.docs, game_id)
        except ValueError as exc:
            return reply(str(exc), status=400)
        except KnowledgeBaseError as exc:
            # 状态码跟着异常走：接口与页面两条路翻出来的是同一个（见 ragamer.knowledge）
            return reply(str(exc), status=exc.status)

        job_id = jobs.submit(
            sources,
            urls=links,
            game_id=game_id,
            version=version or UNVERSIONED,
            vocabulary=vocabulary,
        )
        where = f"/import?{urlencode({'game_id': game_id, 'job': job_id})}"
        if request.headers.get("HX-Request"):
            response = templates.TemplateResponse(
                request,
                "partials/import_result.html",
                {
                    "result": _job_view(jobs.snapshot(job_id), accept=_accept(container)),
                    "notice": "",
                    "empty": EMPTY,
                },
            )
            # 地址栏跟着变：刷新之后还看得见这一批，而不是回到一张空表单
            response.headers["HX-Push-Url"] = where
            return response
        # 不带脚本那条路：提交 → 重定向 → 重新渲染（与本仓别的写入路径同一条规矩）
        return RedirectResponse(where, status_code=303)

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
    def chat_game(request: Request, game_id: str, after: str = "", selected: str = "") -> Response:
        """选中一个库：左栏列它的近期会话，右边提示开一个或点一个。

        `after` 是上一页最后一条的位置——**左栏滚到底时 htmx 拿它取下一页**。
        片段与整页走同一份数据、同一段渲染（导入那条路是同一个打法），所以两条路
        看到的东西一模一样，只是长过了多少条不同。
        """
        if after and request.headers.get("HX-Request"):
            return templates.TemplateResponse(
                request,
                "partials/session_list.html",
                _session_list(stack, game_id, after=after, selected=selected),
            )
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
    def start_session(
        request: Request,
        game_id: str,
        version: str = Form(""),
        question: str = Form(""),
    ) -> Response:
        """开一次会话，然后转过去。「提交 → 重定向」：刷新一下不会又开一个。

        带 `question` 进来时顺手把那一轮也问了——热门问题在**还没开会话**的那一页上就是
        这么用的：按钮上挂着问题，点一下从这里开一个会话再问。
        """
        try:
            knowledge = readable_knowledge_base(container.docs, game_id)
        except KnowledgeBaseError as exc:
            return _chat_page(request, container, stack, game_id=game_id, error=str(exc))
        conversation = stack.chat.start(game_id=game_id, version=version or knowledge.version)
        if not question.strip():
            return RedirectResponse(f"/chat/{game_id}/{conversation.session_id}", status_code=303)
        return _whole_turn(request, container, stack, game_id, conversation.session_id, question)

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
        """换这次会话选定的版本。**下一轮起按它走**，直到再换一次。

        空串是「跟随知识库的现行版本」，其余必须是这个库里**真实有过**的版本：
        换成一个库里没有的版本，检索会静默查空，而界面上看不出区别——
        与澄清反问的候选是同一条理由。下拉里只摆真有的，这里再拦一道是为了
        绕过页面的请求（改过的表单、直接打接口）也拦得住。
        """
        if version and version not in {
            choice.value for choice in version_choices(container.chunks, game_id)
        }:
            return _chat_page(
                request,
                container,
                stack,
                game_id=game_id,
                session_id=session_id,
                error=f"知识库 {game_id} 里没有版本 {version}。下拉里列的是这个库里真有的版本",
            )
        try:
            stack.chat.set_version(session_id, version)
        except ConversationNotFound as exc:
            return _chat_page(
                request,
                container,
                stack,
                game_id=game_id,
                session_id=session_id,
                error=str(exc),
                status_code=404,
            )
        return RedirectResponse(f"/chat/{game_id}/{session_id}", status_code=303)

    @router.post("/chat/{game_id}/{session_id}/delete")
    def delete_session(request: Request, game_id: str, session_id: str) -> Response:
        """删掉一次会话，回到这个库的会话列表。

        **不要二次确认**（与删库那一套不同）：一次会话就是一段问答，删错了重问一遍就是，
        而删库是不可逆地清掉语料。按钮上写着「删除」，不再多一步。
        """
        try:
            stack.chat.delete(session_id)
        except ConversationNotFound as exc:
            # 那一页已经不是最新的了（另一个标签页删过、或者会话早被删掉）：照实说
            return _chat_page(
                request,
                container,
                stack,
                game_id=game_id,
                session_id=session_id,
                error=str(exc),
                status_code=404,
            )
        return RedirectResponse(f"/chat/{game_id}", status_code=303)

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
            {**row, "current": row["game_id"] == game_id}
            for row in _knowledge_base_rows(list_knowledge_bases(container.docs))
        ],
        "game_id": game_id,
        "session_id": session_id,
        "sessions": [],
        "selected": session_id,
        "next": "",
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
        context.update(_session_list(stack, game_id, selected=session_id))
        context["versions"] = _version_options(
            container, game_id, selected=conversation.version if conversation else knowledge.version
        )
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

    澄清那一轮不重定向：页面要把候选按钮渲染出来，而那段状态（暂停点、候选）就在这一刻
    手上，不在会话里。所以直接把它渲染进这一页。

    说不通的几种各回一页：库读不了、会话没了、点的是个过期的候选、模型挂了。
    **这条路上没有「流已经开了」这个约束**，所以状态码该怎么给就怎么给
    （接口那条发 `error` 事件，是因为响应头早发出去了）。
    """
    if not question.strip():
        return _wrong(
            request, container, stack, game_id, session_id, "问题不能为空", question=question
        )
    try:
        # 先读一次会话拿它绑的库：现行版本要从那里取（ADR-0004）。
        # `chat.ask` 自己还会再读一次，那是它的事——会话不可变，`ask` 落的是新的一份。
        bound = stack.chat.open(session_id).game_id
        # **要用读得出来的那个库**：配置坏掉的库不能被拿来干活（`ragamer.knowledge`），
        # 照它作答会按「七类全开、映射为空」落标签，看起来一切正常，错的全在标签里。
        knowledge = readable_knowledge_base(container.docs, bound)
    except ConversationNotFound as exc:
        return _wrong(request, container, stack, game_id, session_id, str(exc), status=404)
    except KnowledgeBaseError as exc:
        return _wrong(
            request, container, stack, game_id, session_id, str(exc), status=_status_of(exc)
        )
    try:
        replies = list(
            stack.chat.ask(
                session_id,
                question,
                current_version=knowledge.version,
                pending_id=pending_id,
                label=label,
            )
        )
    except (NotACandidate, UnknownPending) as exc:
        # 用户点的是个过期的候选：页面停在原处，把原因写出来让他重来一次
        return _wrong(request, container, stack, game_id, session_id, str(exc), status=422)
    except LlmError as exc:
        return _wrong(request, container, stack, game_id, session_id, str(exc), status=502)
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


def _wrong(
    request: Request,
    container: Container,
    stack: ChatStack,
    game_id: str,
    session_id: str,
    error: str,
    *,
    question: str = "",
    status: int = 400,
) -> Response:
    """这一轮没走成，把原因渲染回这一页。**重定向会把它丢掉**——闪一下就没了。"""
    return _chat_page(
        request,
        container,
        stack,
        game_id=game_id,
        session_id=session_id,
        question=question,
        error=error,
        status_code=status,
    )


def _clarification_view(clarification: Clarification, question: str) -> dict[str, Any]:
    """一次反问渲染成按钮要的那几样。按钮上显示 `label`、回传的也是它
    （`ragamer.clarifying` 照它认候选）；`question` 一并带上，用户点完那次请求
    还要靠它记下这一轮的用户原话。"""
    return {
        "pending_id": clarification.pending_id,
        "prompt": clarification.prompt,
        "question": question,
        "choices": [{"label": choice.label} for choice in clarification.choices],
    }


def _session_list(
    stack: ChatStack, game_id: str, *, after: str = "", selected: str = ""
) -> dict[str, Any]:
    """会话列表那一段的上下文。**整页与「滚到底取下一页」两条路共用它**。

    共用的不只是数据，还有渲染它的那一个模板（`partials/session_list.html`）：
    两条路各拼一遍的话，第一页与后面几页迟早长得不一样。
    """
    page = stack.chat.list_for_game(game_id, after=parse_session_cursor(after))
    return {
        "game_id": game_id,
        "selected": selected,
        # 空串即到底了：那个哨兵元素因此不再出现
        "next": page.next,
        "sessions": [
            {
                "session_id": summary.session_id,
                "title": summary.title,
                "updated_at": summary.updated_at,
            }
            for summary in page.sessions
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
    if selected and all(option["value"] != selected for option in options):
        # 会话上那个版本在这个库里已经没有了（语料重导过，或者会话是走 JSON 端点建的，那边
        # 不校验）。不摆出来的话，下拉会静默落到「跟随现行版本」——而这一轮实际按另一个
        # 版本检索，页面上看不出区别。
        options.append(
            {
                "value": selected,
                "label": f"{selected}（这个库里已经没有它了）",
                "current": True,
            }
        )
    return options


def _turn_rows(turns: Sequence[Turn]) -> list[dict[str, Any]]:
    """会话里的消息 → 模板要的那几样。用户那一侧只有原话，模型那一侧还带来源与图。

    `body` 是**排过版的那一份**（`ragamer.web.markdown` 认的那几种记号）：模型写的是
    Markdown，而模板原先直接输出纯文本，星号与列表符会原样露在页面上。排版只做一次、
    只在这一处：流式那一轮的正文由页面自己拼，答完被这整块换掉（`turns.html`）。

    `content` 保留原样（存进会话、也走 API 的就是它），`body` 只给页面用。
    """
    return [
        {
            "role": turn.role,
            "content": turn.content,
            "body": to_html(turn.content),
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
    """渲染一个整页。

    `section` 决定导航里哪一项高亮；上下文里的 `game_id`／`session_id` 决定每一项指向
    哪——同一段里的不同位置落点不同（见 `_nav`）。两者都从 `context` 里取，因为它们
    本来就在那儿，不必每个调用点再传一遍。
    """
    return templates.TemplateResponse(
        request,
        template,
        {
            "title": title,
            "nav": _nav(
                section,
                game_id=str(context.get("game_id") or ""),
                session_id=str(context.get("session_id") or ""),
            ),
            "empty": EMPTY,
            **context,
        },
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
        inventory = purge_inventory(
            chunks=container.chunks,
            docs=container.docs,
            objects=container.objects,
            game_id=game_id,
            sessions=CONVERSATIONS,
        )
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
        # 集合名从常量来，不在模板里写死：`CONVERSATIONS` 一改名，页面就开始撒谎
        sessions=CONVERSATIONS,
        error=error,
        status_code=status_code,
    )


def _accept(container: Container) -> str:
    """能传什么格式由解析适配器说了算，不在模板里再抄一份——抄的那份迟早对不上。"""
    return ",".join(container.parser.SUFFIXES)


def _import_page(
    request: Request,
    container: Container,
    jobs: ImportJobs,
    *,
    selected: str = "",
    version: str = "",
    job: str = "",
    message: str = "",
    status_code: int = 200,
) -> Response:
    """导入页整页。`message` 与结果区里的东西，与 htmx 拿到的片段是同一份内容。

    **不经 base.html 的那条 `error` 通道**：那条画在表单上方，片段换入时看不见；
    要显示的东西得在结果区里，两条路才一致。

    `job` 没带上时**自己认领一条**（`ImportJobs.latest`）：导入是分钟级的一段，切到别的
    页再切回来时地址栏里那个任务号多半已经没了，而那一批还在后台跑着。认领到的那批决定
    这一页的主题，库与版本都跟着它走——否则会出现「显示着 A 库的结果、表单却停在 B 库」。
    """
    bases = _knowledge_base_rows(list_knowledge_bases(container.docs))
    known = {base["game_id"] for base in bases}
    if job:
        snapshot = jobs.snapshot(job)
        if snapshot is None:
            # 服务重启过、或者这条早被丢掉了：说清楚，别让人对着一个空结果区猜
            message = message or _GONE
    else:
        snapshot = jobs.latest(selected)
    if snapshot is not None:
        selected = snapshot.game_id
        # 提交失败时回填的那份版本优先：那一次提交的值比上一批的更能说明人想干什么
        version = version or snapshot.version
    # 直接打开这个页面时默认选中第一个库，省得每回都挑一次；认领到库里已经没有了的那一批
    # （它的库刚被删掉）也退到同一个落点
    if selected not in known:
        selected = next(iter(sorted(known)), "")
    return _page(
        request,
        "import.html",
        "导入",
        "/import",
        bases=bases,
        # 给外壳与导航用：这一页此刻是在哪个库里
        game_id=selected,
        selected=selected,
        version=version,
        accept=_accept(container),
        message=message,
        result=None if snapshot is None else _job_view(snapshot, accept=_accept(container)),
        # 跑着的时候整页自己刷新（没脚本那条路的进度）；有 htmx 时不需要它
        running=bool(snapshot is not None and snapshot.running),
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


def _nav(current: str, *, game_id: str = "", session_id: str = "") -> list[dict[str, Any]]:
    """导航四项。`active` 说这一页属于哪一段，`url` 说**从这儿点它去哪儿**。

    两件事分开是因为四项写成四个裸地址时，「切到别的页再切回来」会把上下文丢光：在
    `/chat/黑神话/s1` 上点「对话」回到 `/chat`、在 `/import?job=…` 上点「导入」回到一张
    空表单。而 URL 本来就带得住这些，所以每一项按当前位置补上它。
    """
    return [
        {
            "label": label,
            "url": _nav_url(url, game_id=game_id, session_id=session_id),
            "active": url == current,
        }
        for label, url in NAV
    ]


def _nav_url(url: str, *, game_id: str, session_id: str) -> str:
    """一个导航项在当前位置下该指向哪。认不出来（评测那项）的原样返回。"""
    if not game_id:
        return url
    if url == "/kb":
        return f"/kb/{game_id}"
    if url == "/import":
        return f"/import?{urlencode({'game_id': game_id})}"
    if url == "/chat":
        # 在某个会话里就回那个会话；只在库那一层就回那个库
        return f"/chat/{game_id}/{session_id}" if session_id else f"/chat/{game_id}"
    return url


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


def _job_view(snapshot: JobSnapshot, *, accept: str) -> dict[str, Any]:
    """一个导入任务在界面上的样子。跑着的、跑完的、整批没跑起来的，都用同一份形状。

    跑着的每一秒被重新渲染一次（htmx 轮询这个片段），所以这里**不许有副作用**，
    也不要在模板里做判断——数字与句子都在这里算好。任务号认不出来时**不叫它**：
    由调用方统一说「这个任务不在了」那一句。
    """
    results = list(snapshot.results)
    ok = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    return {
        "job_id": snapshot.job_id,
        "game_id": snapshot.game_id,
        "accept": accept,
        "running": snapshot.running,
        "error": snapshot.error,
        "headline": _headline(snapshot, ok=len(ok), failed=len(failed)),
        # 这一批实际落下的东西：进了多少切片、覆盖到哪些标签（验收要的那两句）。
        # 跑着的时候不报——那是半截账，读了会当成总数。
        "summary": _labels_text(ok) if snapshot.finished else "",
        "chunks": sum(result.chunk_count for result in ok),
        "rows": [_job_row(item, snapshot) for item in snapshot.items],
        "retry": _retry_payload(failed, version=snapshot.version),
    }


def _headline(snapshot: JobSnapshot, *, ok: int, failed: int) -> str:
    """结果区顶上那一行。跑着的时候说的是「已经跑完几条」，不是最终账单。"""
    if snapshot.error:
        counts = f"共 {snapshot.total} 条：这一批没能跑起来"
    elif snapshot.running:
        counts = f"共 {snapshot.total} 条，已跑完 {len(snapshot.results)} 条 · 还在跑"
    else:
        counts = f"共 {snapshot.total} 条，成功 {ok} 条，失败 {failed} 条"
    # 空串就是未标注版本。界面上直接显示空串的话，那一行读起来像没渲染出来
    return f"{counts}。标注版本：{snapshot.version or '未标注版本'}。"


def _job_row(item: JobItem, snapshot: JobSnapshot) -> dict[str, Any]:
    """一条资料此刻的样子。**它走到哪一步就是它在界面上的进度**。"""
    result = item.result
    if result is None:
        # 整批没跑起来时，每一条都不该还挂着「排队中」——那一批永远不会轮到它
        if snapshot.error:
            status, label = "unstarted", "没能开始"
        elif item.stage is not None:
            status, label = "running", f"正在{STAGE_LABELS[item.stage]}"
        else:
            status, label = "queued", "排队中"
        return _empty_row(item, status=status, status_label=label)
    return _empty_row(
        item,
        status="ok" if result.ok else "failed",
        status_label="成功" if result.ok else "失败",
        # 只列走过的阶段：卡在归一化的那条不该显示它走过切分
        steps=[STAGE_LABELS[event.stage] for event in result.progress],
        doc_title=result.doc_title,
        chunk_count=result.chunk_count,
        skipped=result.skipped,
        subject_name=result.tags.subject_name,
        subject_types=_names(SUBJECT_TYPE_NAMES, result.tags.subject_type),
        content_natures=_names(CONTENT_NATURE_NAMES, result.tags.content_nature),
        stage_label=EMPTY if result.stage is None else STAGE_LABELS[result.stage],
        error=result.error or "",
        # 撞了标题的那条重试不得：单独重试它，写下去就是把它撞的那一份删掉
        collides_with=result.collides_with,
        preview_url=_preview_url(snapshot.game_id, result.doc_title, snapshot.version),
    )


def _empty_row(item: JobItem, *, status: str, status_label: str, **filled: Any) -> dict[str, Any]:
    """一条行的底子：还没有结果的那些字段都留空。"""
    row: dict[str, Any] = {
        "source": item.source,
        "status": status,
        "status_label": status_label,
        "steps": [STAGE_LABELS[stage] for stage in item.stages],
        "doc_title": "",
        "chunk_count": 0,
        "skipped": 0,
        "subject_name": "",
        "subject_types": [],
        "content_natures": [],
        "stage_label": "",
        "error": "",
        "collides_with": "",
        "preview_url": "",
    }
    row.update(filled)
    return row


def _labels_text(results: Sequence[ImportResult]) -> str:
    """这一批成功的那几条合起来覆盖到哪些标签（验收要的那一句）。

    一份资料一个切片都可能是空的（比如整篇都没读出结构），所以是并集而不是「第一条的」。
    一条标签都没有时明说，不留一句「覆盖到的标签：」在那儿吊着。
    """
    groups = (
        ("主体类型", _union(SUBJECT_TYPE_NAMES, [r.tags.subject_type for r in results])),
        ("内容性质", _union(CONTENT_NATURE_NAMES, [r.tags.content_nature for r in results])),
        ("游戏术语", sorted({term for result in results for term in result.tags.game_terms})),
    )
    covered = [f"{name} {'、'.join(values)}" for name, values in groups if values]
    if not covered:
        return "这一批一条标签都没打上——语料本身没有可读的结构，模型那条路也没给出结论。"
    return "覆盖到的标签：" + "；".join(covered) + "。"


def _union(labels: Mapping[Any, str], groups: Iterable[Sequence[str]]) -> list[str]:
    """并集，按词表里的顺序——界面上两个字段的排列才稳定。"""
    values = {value for group in groups for value in group}
    ordered = [name for value, name in labels.items() if value in values]
    # 词表里没有的取值原样补在后面，不吞掉
    return ordered + sorted(_names(labels, values - set(labels)))


def _retry_payload(failed: Sequence[ImportResult], *, version: str) -> dict[str, Any]:
    """「只重试失败的那些」要带上的东西。

    网址能原样带上（它本身就是那条资料的凭据）；**文件带不了**——字节在浏览器那边，
    提交完就不在页面上了，只能请人重新选中。

    **撞了标题的那条不进重试**：它失败是因为同一次提交里另有两条落成了同一个文档标题，
    单独重试它，它就成了那一批里唯一的一条，写下去会把它撞赢的那份整份替掉——
    重试按钮不该是把人送进这个坑的那只手。
    """
    retryable = [result for result in failed if not result.collides_with]
    return {
        "urls": [result.source for result in retryable if result.kind is SourceKind.URL],
        "files": [result.source for result in retryable if result.kind is SourceKind.FILE],
        "blocked": [
            {"source": result.source, "other": result.collides_with}
            for result in failed
            if result.collides_with
        ],
        "version": version,
    }


def _url_lines(raw: str) -> list[str]:
    """网址输入框里的地址：一行一条，空行与前后空白丢掉。

    重复的地址不去重——两条一样的地址就是同一份资料，导入侧按幂等处理（主键稳定，
    写下去等于没写），这里替它做主反而会让人以为自己只填了一条。
    """
    return [line.strip() for line in raw.splitlines() if line.strip()]


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
