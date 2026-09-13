"""HTTP 端点：写入侧对外的唯一入口。

一个端点——`POST /api/kb/{game_id}/import`。批量提交、**逐文件独立**：某个文件失败时
其余照常入库，失败的那个在结果里带文件名与失败阶段。读取侧（提问）与几个页面在后面的
票里接。

知识库元数据从 MongoDB 读（`knowledge_bases` 集合，id 就是游戏 id）：打标要用的词表
——启用了哪些主体类型、这个游戏的术语映射——就在它里面（docs/ARCHITECTURE.md §2.3）。
本层**一个适配器都不构造**，全部来自组合根（`ragamer.container`），缝因此立得住。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from ragamer.container import Container
from ragamer.importing import STAGE_LABELS, Importer, ImportResult, ProgressEvent
from ragamer.logging import get_logger
from ragamer.sources import SourceDocument
from ragamer.stores.base import UNVERSIONED, collection_name
from ragamer.tagging import TagVocabulary

logger = get_logger(__name__)

#: 知识库元数据所在的集合，文档 id 就是游戏 id。
KB_COLLECTION = "knowledge_bases"


def create_app(container: Container) -> FastAPI:
    """把组合根里那套依赖接成 ASGI 应用。"""
    app = FastAPI(title="RAGamer", summary="游戏攻略 RAG 助手")
    importer = Importer(
        chunks=container.chunks,
        embedder=container.embedder,
        llm=container.llm,
        parser=container.parser,
        objects=container.objects,
        on_progress=_log_progress,
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


def _log_progress(event: ProgressEvent) -> None:
    """导入是同步的一整段，日志是它在跑的时候唯一看得见的进度窗口。

    响应里的 `progress` 是跑完之后才拿得到的账单；一批几十份资料时，
    那之前能看到的只有这几行。
    """
    logger.info(
        "导入 %s：[%d/%d] %s",
        event.filename,
        event.file_number,
        event.file_total,
        STAGE_LABELS[event.stage],
    )


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

    配置里没写的项一律走默认值（全部主体类型、映射为空）——用户自定义库没配映射时
    就是这条降级路径，标签会稀疏但不会漏（docs/ARCHITECTURE.md §2.3）。
    知识库本身不存在是另一回事：那是游戏选错了，当场 404，不静默按默认词表建内容。
    """
    payload = container.docs.get(KB_COLLECTION, game_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"知识库 {game_id} 不存在。先在知识库管理里建一个，再导入资料",
        )
    try:
        return TagVocabulary.from_mapping(payload)
    except ValueError as exc:
        # 库里配了个不认识的主体类型：是知识库自己的数据坏了，不是这份资料的错，
        # 也不该长成一个 500——那样界面上只会看见「服务器错误」，查无可查
        raise HTTPException(
            status_code=422, detail=f"知识库 {game_id} 的配置读不了：{exc}"
        ) from exc


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
