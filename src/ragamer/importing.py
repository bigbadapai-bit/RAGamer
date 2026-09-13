"""导入编排器：写入侧的唯一入口。

**两个入口，一条链路**：`import_one` 收本地文件（或界面上传的字节），`import_url` 收网址，
两条在归一化那一步合流（`import_url` 只是换了个取正文的方式），此后切分、打标、
向量化、入库完全不区分来源——验收要求的「抓取结果与本地文件走完全相同的后续链路」，
与其靠两处代码长得一样来保证，不如让它们本来就是同一处。

一份资料从归一化一路送到库里，中间串起补图、切分、打标、向量化。链路上每一段都自己
有模块，这里不重做其中任何一件，只负责把线接对——而接错的代价恰恰最大，
所以编排层自己扛三件事：

- **进度可上报**：每进入一个阶段回调一次，界面上的「还要等多久、卡在哪一步」来自它。
- **一批里某个文件失败不牵连其余**：异常在逐个文件那一层兜住，记下文件名与失败阶段。
  兜住不等于吞掉——每一次失败都落日志（带调用栈），也落进那个文件的结果里。
- **同一份资料重复导入不产生重复切片**，靠两个机制：
  一是**切片主键由导入侧分配**，取「游戏 + 文档标题 + 版本 + 切片序号」的哈希，
  重导算出的 id 与上次完全一致，覆盖写入天然幂等（服务端自增做不到这一点）；
  二是**入库时按文档整体替换**，先删掉这份文档在这个版本下的旧切片再写。
  少了第二条，新一次切出来的片数变少时，只靠覆盖会留下一截旧切片——
  查得出来、还会进聚合父块，而且不报错。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from ragamer.chunking import ChunkRules, chunk_document
from ragamer.llm import LlmClient
from ragamer.logging import get_logger
from ragamer.sources import (
    ImageEnricher,
    MarkdownParser,
    NormalizedDoc,
    PageCrawler,
    SourceDocument,
    SourceParser,
)
from ragamer.stores.base import UNVERSIONED, Chunk, ChunkStore
from ragamer.tagging import TaggedChunk, TagVocabulary, tag_document
from ragamer.vectors.base import Embedder, ModelOutputError

logger = get_logger(__name__)

#: 文档大标题。MediaWiki 页面的条目名就在这里。
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class ImportStage(StrEnum):
    """导入的六个阶段，声明顺序即执行顺序。失败信息里点名的就是其中一个。"""

    NORMALIZE = "normalize"
    ENRICH = "enrich"
    CHUNK = "chunk"
    TAG = "tag"
    EMBED = "embed"
    STORE = "store"


#: 阶段的中文叫法。日志与界面上的「失败在哪一步」都用它。
STAGE_LABELS: dict[ImportStage, str] = {
    ImportStage.NORMALIZE: "归一化",
    ImportStage.ENRICH: "补图",
    ImportStage.CHUNK: "切分",
    ImportStage.TAG: "打标",
    ImportStage.EMBED: "向量化",
    ImportStage.STORE: "入库",
}


@dataclass(frozen=True)
class ProgressEvent:
    """一次阶段推进。每进入一个阶段发一条，阶段本身是「将要开始做」而不是「已经做完」。"""

    #: 这一条是从哪来的：文件名，或网址。
    source: str
    stage: ImportStage
    #: 这一批里的第几个文件，从 1 起。
    file_number: int
    #: 这一批一共几个文件。
    file_total: int


#: 进度回调。每进入一个阶段调一次。
ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class CoveredTags:
    """这次导入实际落下的标签。界面上的「覆盖到了哪些标签」来自这里。

    取的是并集：主体名是文档级的、只有一个，主体类型与游戏术语同样是文档级的，
    内容性质是切片级的、同一份文档里可以并存几种。
    """

    subject_name: str = ""
    subject_type: tuple[str, ...] = ()
    content_nature: tuple[str, ...] = ()
    game_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportResult:
    """一个文件的导入结果。**失败也是结果**——一批里它失败了，其余照跑。"""

    #: 这一条是从哪来的：文件名，或网址。
    source: str
    #: 存进库里叫什么。同时是「重导时替换掉哪一批切片」的依据。
    doc_title: str
    #: 入库的切片数。
    chunk_count: int
    #: 没有可向量化正文、因而没有入库的切片数。今天的切分器不会产出正文为空的切片
    #: （`_split` 已经滤掉空段），真实来源是补图那一层——OCR 与摘要都失败的图片切片。
    skipped: int
    tags: CoveredTags
    #: 失败发生在哪一步；成功时是 `None`。
    stage: ImportStage | None = None
    #: 失败原因；成功时是 `None`。
    error: str | None = None
    #: 这个文件走过的阶段，按先后。走到哪就报到哪。
    progress: tuple[ProgressEvent, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None


def document_title(markdown: str, filename: str) -> str:
    """这份资料存进库里叫什么。

    先取正文的一级标题（词条页的条目名就在这里），取不到才回落到文件名。
    它对同一份资料必须稳定：文档标题同时是「重导时替换掉哪一批切片」的依据，
    换个文件名重传会变成两份文档——这是回落带来的已知代价。

    按行扫，**不认围栏代码块**：正文若以一段代码开头，代码里的 `#` 会被当成大标题。
    这与打标读结构（`ragamer.tagging` 的已知限）同源，是同一处妥协；
    但这里的影响面更大——标题是替换键，改错了旧的那批切片会留在库里。
    """
    title = _TITLE.search(markdown)
    return title.group(1).strip() if title is not None else Path(filename).stem


def chunk_id(*, game_id: str, doc_title: str, version: str, chunk_index: int) -> int:
    """切片的主键。**由导入侧分配，不用服务端自增**（docs/ARCHITECTURE.md §2.2）。

    取「游戏 + 文档标题 + 版本 + 切片序号」的哈希：同一份资料每次重导算出的 id
    完全一致，覆盖写入于是天然幂等。服务端自增的 id 每次都不一样，覆盖无从谈起。

    四个字段用空字符分隔：`("ab", "c")` 与 `("a", "bc")` 拼出来必须不是同一个键，
    否则两款游戏的同名文档会互相覆盖，而且不报错。右移一位避开 INT64 的符号位。
    """
    key = "\x00".join((game_id, doc_title, version, str(chunk_index)))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def content_hash(content: str) -> str:
    """正文的摘要。为将来的增量导入预留——v1 走全量幂等重导，还不比较它。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Importer:
    """一次导入的接线。外部依赖由组合根注入（见 `ragamer.container`）。

    做成不可变对象而不是一组自由函数：六个阶段之间的东西（词表、规则、回调）
    每次都一样，只有资料本身在变，那就不该每调一次重传一遍。
    """

    chunks: ChunkStore
    embedder: Embedder
    #: 打标兜底用的模型。不传时结构读不出来的标签留空，不阻断入库。
    llm: LlmClient | None = None
    #: 解析适配器。默认只认 md／txt，另外两条来源在后面的票里接上。
    parser: SourceParser = field(default_factory=MarkdownParser)
    #: 网页来源。组合根总会接上；留成可空只是为了让不碰网址的测试不必造一个假件。
    crawler: PageCrawler | None = None
    #: 补图。没接上时这一步不做、也不上报——报了就是假进度。
    enricher: ImageEnricher | None = None
    #: 切分参数。不传用 `ChunkRules` 的默认值。
    rules: ChunkRules | None = None
    on_progress: ProgressCallback | None = None

    def batch(
        self,
        sources: Sequence[SourceDocument],
        *,
        game_id: str,
        version: str = UNVERSIONED,
        vocabulary: TagVocabulary | None = None,
    ) -> tuple[ImportResult, ...]:
        """一批资料，顺序即提交顺序。**逐个独立**：某个文件失败时其余照常入库。"""
        total = len(sources)
        results = tuple(
            self.import_one(
                source,
                game_id=game_id,
                version=version,
                vocabulary=vocabulary,
                file_number=number,
                file_total=total,
            )
            for number, source in enumerate(sources, start=1)
        )
        _warn_on_repeated_documents(results, version=version)
        return results

    def batch_urls(
        self,
        urls: Sequence[str],
        *,
        game_id: str,
        version: str = UNVERSIONED,
        vocabulary: TagVocabulary | None = None,
    ) -> tuple[ImportResult, ...]:
        """一批网址，顺序即提交顺序。**逐个独立**：某个地址失败时其余照常入库。

        与 :meth:`batch` 分成两个方法而不是合成一个收「文件或网址」的入口：
        两者的差别只在归一化那一步，那就不该让类型判断扩散到这一层。
        """
        total = len(urls)
        results = tuple(
            self.import_url(
                url,
                game_id=game_id,
                version=version,
                vocabulary=vocabulary,
                file_number=number,
                file_total=total,
            )
            for number, url in enumerate(urls, start=1)
        )
        _warn_on_repeated_documents(results, version=version)
        return results

    def import_one(
        self,
        source: SourceDocument,
        *,
        game_id: str,
        version: str = UNVERSIONED,
        vocabulary: TagVocabulary | None = None,
        file_number: int = 1,
        file_total: int = 1,
    ) -> ImportResult:
        """导入一份资料。失败不抛异常，落进结果的 `stage` 与 `error`。

        :param game_id: 进哪个游戏知识库。
        :param version: 这次导入标注的版本；空串即「未标注版本」。
        :param vocabulary: 该知识库的词表。不传则全部主体类型、映射为空。
        """
        return self._run(
            source.filename,
            lambda: self.parser.parse(source),
            game_id=game_id,
            version=version,
            vocabulary=vocabulary,
            file_number=file_number,
            file_total=file_total,
        )

    def import_url(
        self,
        url: str,
        *,
        game_id: str,
        version: str = UNVERSIONED,
        vocabulary: TagVocabulary | None = None,
        file_number: int = 1,
        file_total: int = 1,
    ) -> ImportResult:
        """抓一个网址再导入。**抓完之后走的是同一条链路**：切分与打标不知道这份资料从哪来。

        结果与进度事件里的 `source` 报的是这个地址——网页没有文件名，
        而「这一批里是哪一条失败了」总得有个能指认的东西。
        """
        crawler = self.crawler
        if crawler is None:
            raise ValueError("这个导入器没有接上抓取器，导入不了网址（组合根里接）")
        return self._run(
            url,
            lambda: crawler.crawl(url),
            game_id=game_id,
            version=version,
            vocabulary=vocabulary,
            file_number=file_number,
            file_total=file_total,
        )

    def _run(
        self,
        source: str,
        normalize: Callable[[], NormalizedDoc],
        *,
        game_id: str,
        version: str,
        vocabulary: TagVocabulary | None,
        file_number: int,
        file_total: int,
    ) -> ImportResult:
        """六个阶段走一遍。`normalize` 是这里唯一的变量：本地资料读字节、网址去抓。

        两条入口合流得这么早是有意的——「抓回来的与本地文件走完全相同的后续链路」
        这条验收要求，与其靠两处代码长得一样来保证，不如让它们本来就是同一处。
        """
        reported: list[ProgressEvent] = []
        current = ImportStage.NORMALIZE

        def enter(stage: ImportStage) -> None:
            nonlocal current
            current = stage
            event = ProgressEvent(source, stage, file_number, file_total)
            reported.append(event)
            if self.on_progress is not None:
                self.on_progress(event)

        doc_title = ""
        try:
            enter(ImportStage.NORMALIZE)
            doc = normalize()
            doc_title = document_title(doc.markdown, source)
            if self.enricher is not None:
                enter(ImportStage.ENRICH)
                doc = self.enricher.enrich(doc)
            enter(ImportStage.CHUNK)
            chunks = chunk_document(doc.markdown, self.rules)
            enter(ImportStage.TAG)
            tagged = tag_document(doc.markdown, chunks, vocabulary=vocabulary, llm=self.llm)
            enter(ImportStage.EMBED)
            rows, skipped = self._vectorize(
                tagged,
                game_id=game_id,
                version=version,
                doc_title=doc_title,
                source_url=doc.source_url,
            )
            enter(ImportStage.STORE)
            self._store(game_id, doc_title, version, rows)
        # 兜住全部异常是「一个文件失败不牵连其余」要求的：读文件、抓网页、调模型、写库
        # 都可能以各自的异常类型挂掉，而这一批的其余文件不该跟着陪葬。兜住不等于吞掉
        # ——错误原文进结果、带调用栈进日志，两处都留痕。
        except Exception as exc:
            logger.error(
                "导入失败：%s · %s：%s",
                source,
                STAGE_LABELS[current],
                exc,
                exc_info=True,
            )
            return ImportResult(
                source=source,
                doc_title=doc_title,
                chunk_count=0,
                skipped=0,
                tags=CoveredTags(),
                stage=current,
                error=str(exc),
                progress=tuple(reported),
            )
        return ImportResult(
            source=source,
            doc_title=doc_title,
            chunk_count=len(rows),
            skipped=skipped,
            tags=_covered(tagged),
            progress=tuple(reported),
        )

    def _vectorize(
        self,
        tagged: Sequence[TaggedChunk],
        *,
        game_id: str,
        version: str,
        doc_title: str,
        source_url: str,
    ) -> tuple[list[Chunk], int]:
        """打标后的切片 → 可入库的切片，并报出被丢掉的条数。

        没有可向量化正文的切片在这里丢掉，计入 `skipped`：表格的长文本列整列降级进
        `content_meta` 之后，那一片的正文可以是空的。空正文算出来的向量没有意义，
        写进去只会让检索冒出一条什么都没有的命中。

        `chunk_index` 保留原值不重编：它的含义是「在源文档中的顺序」，
        不是「入库之后的第几条」，中间少几条不该让后面全体挪号。
        """
        rows: list[Chunk] = []
        skipped = 0
        for item in tagged:
            content = item.chunk.content.strip()
            if not content:
                skipped += 1
                continue
            rows.append(
                Chunk(
                    chunk_id=chunk_id(
                        game_id=game_id,
                        doc_title=doc_title,
                        version=version,
                        chunk_index=item.chunk.chunk_index,
                    ),
                    content=content,
                    content_meta=item.chunk.content_meta,
                    ancestor_path=item.chunk.ancestor_path,
                    chunk_index=item.chunk.chunk_index,
                    subject_name=item.subject_name,
                    subject_type=tuple(kind.value for kind in item.subject_type),
                    content_nature=tuple(nature.value for nature in item.content_nature),
                    game_terms=item.game_terms,
                    game_id=game_id,
                    version=version,
                    doc_title=doc_title,
                    chunk_type=item.chunk.chunk_type,
                    content_hash=content_hash(content),
                    source_url=source_url,
                )
            )
        if not rows:
            return [], skipped
        embedding = self.embedder.embed([row.content for row in rows])
        if len(embedding) != len(rows):
            # 按短的一边截齐会写进一批没有向量的切片，缺向量的那些查不出来也不报错
            raise ModelOutputError(f"向量化返回了 {len(embedding)} 条，喂进去的是 {len(rows)} 条")
        return [
            replace(row, dense_vector=dense, sparse_vector=sparse)
            for row, dense, sparse in zip(rows, embedding.dense, embedding.sparse, strict=True)
        ], skipped

    def _store(self, game_id: str, doc_title: str, version: str, rows: Sequence[Chunk]) -> None:
        """按文档整体替换：先删这份文档在这个版本下的旧切片，再写新的。

        删除排在最后而不是最早：归一化、切分、打标、向量化都可能失败，那些时候旧的一批
        应该原样留着，而不是先被删光、留下一个空文档。空的一批也要走删除——
        文档被改成一个切片都不剩时，旧的那批正该被清掉。
        """
        self.chunks.ensure_collection(game_id)
        self.chunks.delete_document(game_id, doc_title, version=version)
        self.chunks.upsert(game_id, rows)


def _covered(tagged: Sequence[TaggedChunk]) -> CoveredTags:
    """这次实际落下的标签，按字段各取并集。"""
    return CoveredTags(
        subject_name=next((item.subject_name for item in tagged if item.subject_name), ""),
        subject_type=_unique(kind.value for item in tagged for kind in item.subject_type),
        content_nature=_unique(nature.value for item in tagged for nature in item.content_nature),
        game_terms=_unique(term for item in tagged for term in item.game_terms),
    )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """去重并保持原次序。"""
    return tuple(dict.fromkeys(values))


def _warn_on_repeated_documents(results: Sequence[ImportResult], *, version: str) -> None:
    """同一批里两份文件切成同一个文档标识时提醒一声。

    文档标识是「文档标题 + 版本」，也是重导替换的范围：两份一级标题相同的文件会互相
    覆盖——后写的那份先把前一份删掉，两条结果却都报成功，库里最终只剩一份。
    这不是错（同名即同一份文档），但界面上「导入了 2 份」的读法要打折扣，所以留一条痕。
    """
    claimed: dict[str, str] = {}
    for result in results:
        if not result.ok:
            continue
        first = claimed.setdefault(result.doc_title, result.source)
        if first != result.source:
            logger.warning(
                "同一批里 %s 与 %s 切出了同一个文档标题 %r（版本 %r）：后写的覆盖了前一份",
                first,
                result.source,
                result.doc_title,
                version,
            )
