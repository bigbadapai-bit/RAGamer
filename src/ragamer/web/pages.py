"""界面：服务端渲染的页面。

模板 + htmx 局部刷新，**没有 JavaScript 构建步骤**（ADR-0005）。每条写入路径都先是一张
普通的 HTML 表单：浏览器禁用 JavaScript 时页面照常能用，只是每次提交整页刷新一次；
htmx 在的时候把结果那一块换掉，人不用跳走。表单同时带 `action`／`method` 与 `hx-post`，
服务端按 `HX-Request` 决定回整页还是回片段——两条路走的是同一段处理逻辑。

这一层不构造任何适配器，数据全来自组合根注入的容器。写入只做两件事——建知识库、跑导入，
两者都直接用写入侧已有的实现（`ragamer.knowledge` 与 `ragamer.importing`），页面自己不
重做其中的判断。切分预览页是把库里存下来的东西读回来渲染，**没有任何编辑入口**。

几个页面在这里都只到骨架为止：知识库管理、导入、对话的完整形态是后面几张票的事。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from ragamer.container import Container
from ragamer.importing import STAGE_LABELS, Importer, ImportResult
from ragamer.knowledge import (
    KnowledgeBase,
    KnowledgeBaseError,
    create_knowledge_base,
    list_knowledge_bases,
    vocabulary_of,
)
from ragamer.sources import SourceDocument
from ragamer.stores.base import UNVERSIONED, Chunk, collection_name
from ragamer.tagging import CONTENT_NATURE_NAMES, SUBJECT_TYPE_NAMES, SubjectType

#: 模板目录。跟着包走，装成 wheel 也在。
TEMPLATES = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES))

#: 导航。四个位置一次留齐，对话与评测的页面在后面的票里接上。
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


def create_router(container: Container) -> APIRouter:
    """把页面挂成一个路由器，由 `ragamer.app` 与 JSON 端点装进同一个应用。"""
    router = APIRouter()
    # 进度回调用 Importer 的默认实现（落日志）：导入是同步的一整段，
    # 在页面上等的时候只有日志看得见进度。
    importer = Importer(chunks=container.chunks, embedder=container.embedder, llm=container.llm)

    @router.get("/")
    def home() -> RedirectResponse:
        """根路径进知识库管理页——界面上要做的第一件事就是建库。"""
        return RedirectResponse("/kb")

    @router.get("/kb")
    def knowledge_bases(request: Request) -> Response:
        """知识库列表与新建表单。术语映射与删库在后面的票里。"""
        return _knowledge_bases_page(request, container)

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
        request: Request, game_id: str, doc_title: str = "", version: str = UNVERSIONED
    ) -> Response:
        """切分预览（**只读**）：把入库的结果原样渲染出来。

        取切片走的是检索那条路（`fetch_document`），所以显示的就是检索时会看见的：
        该版本的内容与未标注版本的内容一并列出（ADR-0004），后者挂个徽章说明来历。
        """
        try:
            collection_name(game_id)
        except ValueError as exc:
            return _preview_page(request, game_id, doc_title, version, error=str(exc))
        if not doc_title:
            return _preview_page(request, game_id, doc_title, version, error="没说要预览哪一份文档")
        chunks = container.chunks.fetch_document(game_id, doc_title, version=version)
        return _preview_page(request, game_id, doc_title, version, chunks)

    # 两个还没做的页面。留位置是这一票的要求，所以点进来要有一句人话，而不是一个 404。
    @router.get("/chat")
    def chat(request: Request) -> Response:
        return _page(
            request,
            "placeholder.html",
            "对话",
            "/chat",
            note="这个页面还没做。等检索链路与澄清反问接上之后，问答开在这里。",
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
    error: str = "",
    form: Mapping[str, str] | None = None,
    checked: set[str] | None = None,
    status_code: int = 200,
) -> Response:
    options = [{"value": kind.value, "label": SUBJECT_TYPE_NAMES[kind]} for kind in SubjectType]
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
        error=error,
        form=form or {"game_id": "", "name": ""},
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
            "subject_types": _names(
                SUBJECT_TYPE_NAMES, (kind.value for kind in base.vocabulary.subject_types)
            ),
            "problem": base.problem,
        }
        for base in bases
    ]


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


def _subject_types(values: Sequence[str] | None) -> tuple[SubjectType, ...]:
    """表单里勾选的类目 → 枚举。认不出的取值当场报错，不静默丢掉。

    一个都没勾时返回空元组，由 `TagVocabulary` 报「至少要启用一个主体类型」——
    这里不替它兜底成「全开」。
    """
    kinds: list[SubjectType] = []
    for value in values or ():
        try:
            kinds.append(SubjectType(value))
        except ValueError as exc:
            known = "、".join(kind.value for kind in SubjectType)
            raise ValueError(f"不认识的主体类型 {value!r}。可用的有：{known}") from exc
    return tuple(kinds)


def _names(labels: Mapping[Any, str], values: Iterable[str]) -> list[str]:
    """枚举取值 → 界面上的中文叫法。认不出的取值原样显示，不吞掉。"""
    return [labels.get(value, value) for value in values]
