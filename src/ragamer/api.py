"""HTTP 端点：写入侧对外的唯一入口。

一个端点——`POST /api/kb/{game_id}/import`。批量提交、**逐文件独立**：某个文件失败时
其余照常入库，失败的那个在结果里带文件名与失败阶段。读取侧（提问）在后面的票里接。

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

from ragamer.container import Container
from ragamer.importing import STAGE_LABELS, Importer, ImportResult, ProgressEvent
from ragamer.knowledge import KnowledgeBaseError, vocabulary_of
from ragamer.sources import SourceDocument
from ragamer.stores.base import UNVERSIONED, collection_name
from ragamer.tagging import TagVocabulary


def create_app(container: Container) -> FastAPI:
    """把组合根里那套依赖接成 ASGI 应用。"""
    app = FastAPI(title="RAGamer", summary="游戏攻略 RAG 助手")
    importer = Importer(
        chunks=container.chunks,
        embedder=container.embedder,
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

    return app


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
