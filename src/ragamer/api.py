"""HTTP 端点：写入侧对外的唯一入口。

两个端点，对应界面上的两处输入：

- `POST /api/kb/{game_id}/import` —— 上传文件（或界面上传的字节）。
- `POST /api/kb/{game_id}/import/urls` —— 给网址，抓回来再入库。

两条都是批量提交、**逐条独立**：某一条失败时其余照常入库，失败的那条在结果里带来源与
失败阶段。两条走的是同一个导入器、同一条链路，响应形状也逐字相同。读取侧（提问）与
几个页面在后面的票里接。

知识库元数据从 MongoDB 读（`knowledge_bases` 集合，id 就是游戏 id）：打标要用的词表
——启用了哪些主体类型、这个游戏的术语映射——就在它里面（docs/ARCHITECTURE.md §2.3），
形状与判断在 `ragamer.knowledge`，界面那条写入路径用的是同一份。
本层**一个适配器都不构造**，全部来自组合根（`ragamer.container`），缝因此立得住。

这是**只有 JSON 端点**的应用；给人看的页面由 `ragamer.web` 挂上去，两者在
`ragamer.app` 里装成同一个应用。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ragamer.container import Container
from ragamer.enriching import ImageEnricher
from ragamer.importing import STAGE_LABELS, Importer, ImportResult, ProgressEvent
from ragamer.knowledge import KnowledgeBaseError, vocabulary_of
from ragamer.sources import SourceDocument
from ragamer.stores.base import UNVERSIONED, collection_name
from ragamer.tagging import TagVocabulary


class UrlImport(BaseModel):
    """一次网址导入的请求体。

    网址单独成一个端点而不是塞进上传那一个的多表单字段里：界面上的两处输入本来就分开
    （文件上传 / URL 输入，docs/ARCHITECTURE.md §5），而多部分表单里夹一个网址数组
    两边都不好写。
    """

    urls: list[str] = Field(min_length=1, description="要抓的网页地址，可多条")
    version: str = Field(default=UNVERSIONED, description="这次导入标注的版本，留空即未标注版本")


def create_app(container: Container) -> FastAPI:
    """把组合根里那套依赖接成 ASGI 应用。"""
    app = FastAPI(title="RAGamer", summary="游戏攻略 RAG 助手")
    importer = Importer(
        chunks=container.chunks,
        embedder=container.embedder,
        llm=container.llm,
        parser=container.parser,
        objects=container.objects,
        # 补图在这一层装好再交进去：它要的三样（对象存储、OCR 引擎、视觉模型）
        # 都是组合根造的，编排器只该看见一个 `ImageEnricher`
        enricher=ImageEnricher(
            objects=container.objects,
            ocr=container.ocr,
            vision=container.vision,
        ),
        crawler=container.crawler,
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
        return _response(game_id, version, results)

    @app.post("/api/kb/{game_id}/import/urls")
    async def import_urls(game_id: str, request: UrlImport) -> dict[str, Any]:
        """抓一批网页再入库。某个地址失败时其余照常入库，失败信息带地址与失败阶段。

        抓回来的资料与上传的文件走同一条链路，响应形状也相同——界面上两条输入各是各的
        提交按钮，读结果的地方却可以共用一处。
        """
        _check_game_id(game_id)
        vocabulary = _vocabulary(container, game_id)

        results = importer.batch_urls(
            request.urls, game_id=game_id, version=request.version, vocabulary=vocabulary
        )
        return _response(game_id, request.version, results)

    return app


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


def _vocabulary(container: Container, game_id: str) -> TagVocabulary:
    """这个知识库的打标词表。

    判断在 `ragamer.knowledge` 里（界面那条写入路径用的是同一份），这里只把问题翻成
    它自带的那个状态码——库不存在 404、配置读不了 422，两种问题的分法见那边的类文档。
    """
    try:
        return vocabulary_of(container.docs, game_id)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


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
