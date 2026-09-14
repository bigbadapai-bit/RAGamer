"""导入编排器：写入侧的唯一入口。

**两类来源，一条链路**：`batch` 收本地资料（或界面上传的字节）与网址，两条在归一化那一步
合流（网址只是换了个取正文的方式），此后补图、切分、打标、向量化、入库完全不区分来源——
验收要求的「抓取结果与本地文件走完全相同的后续链路」，与其靠两处代码长得一样来保证，
不如让它们本来就是同一处。单条也有入口（`import_one`／`import_url`），那是给不带批次的
调用方用的，走的是同一段 `_run`。

一份资料从归一化一路送到库里，中间串起补图、切分、打标、向量化。链路上每一段都自己
有模块，这里不重做其中任何一件，只负责把线接对——而接错的代价恰恰最大，
所以编排层自己扛四件事：

- **进度可上报**：每进入一个阶段回调一次，界面上的「还要等多久、卡在哪一步」来自它。
- **一批里某一条失败不牵连其余**：异常在逐条那一层兜住，记下它从哪来、失败在哪一步。
  兜住不等于吞掉——每一次失败都落日志（带调用栈），也落进那一条的结果里。
- **同一份资料重复导入不产生重复切片**，靠两个机制：
  一是**切片主键由导入侧分配**，取「游戏 + 文档标题 + 版本 + 切片序号」的哈希，
  重导算出的 id 与上次完全一致，覆盖写入天然幂等（服务端自增做不到这一点）；
  二是**入库时按文档整体替换**，先删掉这份文档在这个版本下的旧切片再写。
  少了第二条，新一次切出来的片数变少时，只靠覆盖会留下一截旧切片——
  查得出来、还会进聚合父块，而且不报错。
- **一次提交里两条不会互相覆盖**：文档标识（标题 + 版本）是替换的范围，两条落成同一个
  标识时后一条带着 :class:`DocumentCollision` 失败。跨提交的重导不在此列——那是「同名即
  同一份文档」，是有意的。
- **语料变了要让缓存失效**（`docs/ARCHITECTURE.md` §4）。缓存里存的是基于旧语料写出来的
  答案，新资料进来之后它们可能已经不对了——而「不对」是看不出来的：同样的答案、
  同样漂亮的引用。按这个游戏的缓存前缀批量删（`{前缀}:cache:{游戏}:*`），
  是那儿唯一要动缓存的地方。
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from pathlib import Path

from ragamer.caching.base import AnswerCache, CacheError
from ragamer.chunking import ChunkRules, chunk_document
from ragamer.llm import LlmClient
from ragamer.logging import get_logger
from ragamer.sources import (
    Enricher,
    MarkdownParser,
    NormalizedDoc,
    PageCrawler,
    SourceDocument,
    SourceError,
    SourceParser,
    fetch_images,
    image_refs,
    publish_assets,
)
from ragamer.stores.base import UNVERSIONED, Chunk, ChunkStore, ObjectStore
from ragamer.tagging import TaggedChunk, TagVocabulary, tag_document
from ragamer.vectors.base import Embedder, ModelOutputError

logger = get_logger(__name__)

#: 一批里最多几条同时跑。见 `_in_parallel`：并行的是「等」，本地重活仍是串行的。
#: 上限还受外部额度约束（MinerU 单账号并发、视觉模型限流），开大了会被挡回来。
BATCH_WORKERS = 4

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


class SourceKind(StrEnum):
    """一条资料是从哪来的。差异只到取正文那一步（`_Entry.normalize`），但界面要留一份——

    「重试失败的那些」得知道失败的那条该回填到网址框还是文件框里，
    靠 `source` 长得像不像地址来猜，迟早会把一个叫得怪的文件名派去抓。
    """

    FILE = "file"
    URL = "url"


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
class _Entry:
    """编排器内部的一条待导入：显示名、怎么拿到归一化文档、以及它在「是不是同一份」上的身份。"""

    #: 这一条是从哪来的：文件名，或网址。进度与结果里指认它用的就是它。
    source: str
    kind: SourceKind
    #: 取正文那一步。本地资料读字节再解析，网址去抓——差异只到这一层为止。
    normalize: Callable[[], NormalizedDoc]
    #: 同一份资料的凭据。同一次提交里两条身份相同即同一份，不是撞车。
    identity: str


class DocumentCollision(ValueError):
    """这一条与同一次提交里的另一条落成同一个文档标识。

    写下去会把那一份整份替掉，所以当场失败。**带上撞的是谁**：界面要据此把这一条
    挡在「重试」之外——单独重试它就等于把它撞的那一份删掉。
    """

    def __init__(self, other: str, doc_title: str) -> None:
        super().__init__(
            f"这一批里的 {other} 也叫「{doc_title}」。文档标题同时是重导替换的范围："
            "两条落成同一个标题只会留下一份。改掉其中一份的标题再导——"
            "分两次导入也留不下两份，后一次会替掉前一次。"
        )
        self.other = other


def log_progress(event: ProgressEvent) -> None:
    """进度回调的默认实现：每进入一个阶段落一行日志。

    导入是同步的一整段，日志是它在跑的时候唯一看得见的进度窗口。响应里的 `progress`
    是跑完之后才拿得到的账单；一批几十份资料时，那之前能看到的只有这几行。
    界面与 JSON 端点两条写入路径共用这一份——各写一遍的话，换个措辞就是两套进度。
    """
    logger.info(
        "导入 %s：[%d/%d] %s",
        event.source,
        event.file_number,
        event.file_total,
        STAGE_LABELS[event.stage],
    )


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
    """一条资料的导入结果。**失败也是结果**——一批里它失败了，其余照跑。"""

    #: 这一条是从哪来的：文件名，或网址。
    source: str
    #: 存进库里叫什么。同时是「重导时替换掉哪一批切片」的依据。
    doc_title: str
    #: 入库的切片数。
    chunk_count: int
    #: 没有可向量化正文、因而没有入库的切片数。切分器会产出正文为空的切片，但只在一种
    #: 情况下：一整段正文里只有图片，替代文本又都空着——图片地址在切分时就被摘走了
    #: （`ragamer.chunking`），剩下的正文一个字都没有。这种段本来就没有可检索的正文，
    #: 写进去只会让检索冒出一条什么都没有的命中。**备了图片却没有替代文本的图**
    #: 因此会连它的地址一起没有落点，靠的是补图那一层给写摘要（`ragamer.enriching`）。
    skipped: int
    tags: CoveredTags
    #: 本地文件还是网址。界面据此决定重试时回填到哪个框里。
    kind: SourceKind = SourceKind.FILE
    #: 失败发生在哪一步；成功时是 `None`。
    stage: ImportStage | None = None
    #: 失败原因；成功时是 `None`。
    error: str | None = None
    #: 撞车时撞的是谁（同一次提交里的另一条）。**这条失败不能靠重试解决**：
    #: 单独重试它，它就成了那一批里唯一的一条，写下去会把它撞赢的那份整份替掉。
    collides_with: str = ""
    #: 这条资料走过的阶段，按先后。走到哪就报到哪。
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


def _source_digest(data: bytes) -> str:
    """来源文件的摘要，进原图的对象 key（见 `ragamer.stores.base.image_key`）。

    取前 16 位十六进制（64 bit）：同一份文件重导算出同一个 key，图片原地覆盖，
    与切片主键由导入侧分配是同一套幂等思路；不同文件即使同名也各有各的一层。
    """
    return hashlib.sha256(data).hexdigest()[:16]


def _url_digest(url: str) -> str:
    """网址的来源摘要，与 :func:`_source_digest` 同一个用处。

    文件那条路摘要取自文件的字节；网址没有字节可摘，取地址本身——同一个网址重导
    落在同一批 key 上，图片原地覆盖。
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


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
    #: 解析适配器。默认只认 md／txt；PDF 与图片由组合根接上 MinerU（`ragamer.mineru`）。
    parser: SourceParser = field(default_factory=MarkdownParser)
    #: 对象存储。解析产物里的原图存进它，正文里的引用改指对象 key。
    #: 不传时 md／txt 照跑；真来了附件还没有它，那一步会当场报错而不是把图丢掉。
    objects: ObjectStore | None = None
    #: 网页来源。组合根总会接上；留成可空只是为了让不碰网址的测试不必造一个假件。
    crawler: PageCrawler | None = None
    #: 补图。没接上时这一步不做、也不上报——报了就是假进度。
    enricher: Enricher | None = None
    #: 切分参数。不传用 `ChunkRules` 的默认值。
    rules: ChunkRules | None = None
    #: 进度回调。默认落日志；显式传 `None` 表示这一段完全不出声。
    on_progress: ProgressCallback | None = log_progress
    #: 答案缓存。不传时这一步不做——没接缓存就没有要失效的东西，报出来反而是假动作。
    cache: AnswerCache | None = None

    def batch(
        self,
        sources: Sequence[SourceDocument] = (),
        *,
        urls: Sequence[str] = (),
        game_id: str,
        version: str = UNVERSIONED,
        vocabulary: TagVocabulary | None = None,
    ) -> tuple[ImportResult, ...]:
        """一次提交：若干份本地资料、若干个网址，按提交顺序一条条来。

        **逐个独立**：某一条失败时其余照常入库。两类来源合成一个入口是因为它们本来
        就是「同一次提交」的两半——编号连着往后排，文档标识也共用一份认领表，
        同一次提交里一个文件与一个网址撞上才不会互相覆盖。

        **最多 :data:`BATCH_WORKERS` 条同时跑**（见 `_in_parallel`）：一条接一条时，
        等 MinerU 解析、等抓取、等视觉摘要、等打标的那几段全是空的，而这些正是这一批
        的绝大部分时间。
        """
        total = len(sources) + len(urls)
        claimed = _Claims()
        entries = [self._file_entry(source, game_id) for source in sources]
        entries += [self._url_entry(url, game_id) for url in urls]

        def run(numbered: tuple[int, _Entry]) -> ImportResult:
            number, entry = numbered
            return self._run(
                entry,
                game_id=game_id,
                version=version,
                vocabulary=vocabulary,
                file_number=number,
                file_total=total,
                claimed=claimed,
            )

        results = _in_parallel(run, entries)
        self._invalidate(game_id, results)
        return results

    def batch_urls(
        self,
        urls: Sequence[str],
        *,
        game_id: str,
        version: str = UNVERSIONED,
        vocabulary: TagVocabulary | None = None,
    ) -> tuple[ImportResult, ...]:
        """一批网址。与 :meth:`batch` 是同一个入口，只是这一批里没有本地资料。"""
        return self.batch((), urls=urls, game_id=game_id, version=version, vocabulary=vocabulary)

    def _invalidate(self, game_id: str, results: Sequence[ImportResult]) -> None:
        """这一批之后，把这个游戏的缓存按前缀批量删（架构文档 §4）。

        **一个文件都没成时不删**：库里什么都没变，删了只是让一批热问题白重算一遍。
        删失败也不影响这一批的结果——导入本身已经成功了，缓存的账是另一本；
        缓存不通时照常作答，只是下次仍要重算（见 `ragamer.caching.answer`）。
        """
        if self.cache is None or not any(result.ok for result in results):
            return
        try:
            deleted = self.cache.invalidate(game_id)
        except CacheError as exc:
            logger.warning(
                "导入完成，但 %s 的缓存没清掉，下次提问仍可能命中旧答案：%s", game_id, exc
            )
            return
        logger.info("导入完成，按前缀清掉 %s 的 %d 条缓存", game_id, deleted)

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
            self._file_entry(source, game_id),
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
        return self._run(
            self._url_entry(url, game_id),
            game_id=game_id,
            version=version,
            vocabulary=vocabulary,
            file_number=file_number,
            file_total=file_total,
        )

    def _file_entry(self, source: SourceDocument, game_id: str) -> _Entry:
        """一份本地资料 → 这一次要跑的那条。原图在这里进对象存储。"""
        return _Entry(
            source=source.filename,
            kind=SourceKind.FILE,
            normalize=lambda: self._collect(
                self.parser.parse(source),
                game_id=game_id,
                # 来源文件的字节摘要进对象 key（见 `image_key`）：同一份文件重导
                # 算出的 key 完全一致，图片原地覆盖
                digest=_source_digest(source.data),
                label=source.filename,
            ),
            # 同一批里再提交一次同一份文件是幂等的，靠它认出来
            identity=_source_digest(source.data),
        )

    def _url_entry(self, url: str, game_id: str) -> _Entry:
        """一个网址 → 这一次要跑的那条。抓不了的当场报错，不兜成一条失败的结果。"""
        crawler = self.crawler
        if crawler is None:
            raise ValueError("这个导入器没有接上抓取器，导入不了网址（组合根里接）")
        return _Entry(
            source=url,
            kind=SourceKind.URL,
            normalize=lambda: self._collect(
                crawler.crawl(url), game_id=game_id, digest=_url_digest(url), label=url
            ),
            identity=url,
        )

    def _run(
        self,
        entry: _Entry,
        *,
        game_id: str,
        version: str,
        vocabulary: TagVocabulary | None,
        file_number: int,
        file_total: int,
        claimed: _Claims | None = None,
    ) -> ImportResult:
        """六个阶段走一遍。`entry.normalize` 是这里唯一的变量：本地资料读字节、网址去抓。

        两条入口合流得这么早是有意的——「抓回来的与本地文件走完全相同的后续链路」
        这条验收要求，与其靠两处代码长得一样来保证，不如让它们本来就是同一处。

        `claimed` 是这一批已经占下的文档标识。单条导入没有第二个来源，不必传。
        """
        source = entry.source
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
            doc = entry.normalize()
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
            # 查标识、写库、占下它三步连在一起（见 `_Claims.store`）：几条并行时中间插进
            # 另一条同名资料，两条会都写下去，后写的那条把前一条整批删掉
            write = partial(self._store, game_id, doc_title, version, rows)
            if claimed is None:
                write()
            else:
                claimed.store(doc_title, version, entry, write)
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
                kind=entry.kind,
                stage=current,
                error=str(exc),
                # 撞车单独标出来：这条失败重试不得，界面要拦住
                collides_with=exc.other if isinstance(exc, DocumentCollision) else "",
                progress=tuple(reported),
            )
        return ImportResult(
            source=source,
            doc_title=doc_title,
            chunk_count=len(rows),
            skipped=skipped,
            tags=_covered(tagged),
            kind=entry.kind,
            progress=tuple(reported),
        )

    def _collect(
        self, doc: NormalizedDoc, *, game_id: str, digest: str, label: str
    ) -> NormalizedDoc:
        """把这份资料里的图片收进对象存储：外链先拉下来，再连同自带的附件一起发出去。

        两条来源都走它——网页与 md 的图之前一直是外链，补图那一层按对象 key 取原图，
        那些图因此一张都补不上（见 `ragamer.sources.fetch_images`）。

        没接抓取器时外链取不回来，那就留着它们在正文里当外链。**留一条 warning**：
        那多半是接线漏了（网页那条来源干脆是当场报错，见 `_url_entry`），
        而静默跳过的样子与「这份资料本来就没有外链图」一模一样。
        """
        if self.crawler is None:
            # 数的是**正文里**的引用，不是 `doc.images`：后者由解析适配器填，
            # 而这一层要问的是「这份资料手上到底有几张取不到的图」
            external = [
                ref for ref in image_refs(doc.markdown) if ref.startswith(("http://", "https://"))
            ]
            if external:
                logger.warning(
                    "%s：有 %d 张外链图，但这个导入器没接抓取器，取不回来——"
                    "它们不会进对象存储，也补不上摘要",
                    label,
                    len(external),
                )
        else:
            doc = fetch_images(doc, crawler=self.crawler)
        return self._publish(doc, game_id=game_id, digest=digest, label=label)

    def _publish(
        self, doc: NormalizedDoc, *, game_id: str, digest: str, label: str
    ) -> NormalizedDoc:
        """附件进对象存储，正文与条目级结构的引用改指对象 key（`sources`）。

        没有附件就直接过。**有附件却没接对象存储是接线错了**：图片会连着正文里的引用
        一起悬空，而且整份资料照样报成功——宁可当场炸，让那一条带着原因失败。
        """
        if not doc.assets:
            return doc
        if self.objects is None:
            raise SourceError(
                f"{label}：资料里有 {len(doc.assets)} 张原图，但没有接对象存储，"
                "它们会连同正文里的引用一起悬空"
            )
        return publish_assets(doc, self.objects, game_id=game_id, digest=digest)

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
                    image_urls=item.chunk.image_urls,
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


def _in_parallel(
    run: Callable[[tuple[int, _Entry]], ImportResult], entries: Sequence[_Entry]
) -> tuple[ImportResult, ...]:
    """一批按提交顺序编号、**最多 :data:`BATCH_WORKERS` 条同时跑**，结果按提交顺序返回。

    并行是为了把**等**重叠起来：MinerU 解析、抓取、视觉摘要、打标都是等网络，而这些
    占了这一批的绝大部分时间。本地重活（向量化、二次 OCR）在各自的适配器里串行
    （`BgeM3Embedder`、`RapidOcrEngine`），所以这个数不是「同时几个重活」。

    **一条时不惊动线程池**：绝大多数提交就是一条，为它起一批线程不值当。

    `Executor.map` 保序，所以结果与提交顺序一一对应——进度事件里的编号（`file_number`）
    也是按这个顺序，界面上「第 3 条」在两处指的是同一条。
    """
    numbered = list(enumerate(entries, start=1))
    if len(numbered) <= 1:
        return tuple(run(item) for item in numbered)
    with ThreadPoolExecutor(
        max_workers=min(BATCH_WORKERS, len(numbered)), thread_name_prefix="ragamer-import"
    ) as pool:
        return tuple(pool.map(run, numbered))


class _Claims:
    """这一批里已经占下的文档标识。**带锁**：几条并行跑时，查与占之间有窗口。

    文档标识（标题 + 版本）同时是重导替换的范围：两条落成同一个标识时，后写的那条会把
    前一条整批删掉，而两条结果都报成功——库里只剩一条，界面上却看着两份都进来了。
    宁可少一份、并且说清为什么。同一份资料重复提交是例外：标识一样、内容也一样，
    写下去等于没写。

    **查标识 → 写库 → 占下它三步连在一起**（`store`）：分开放的话，两条同名资料会同时
    查到「没人占」，于是都写下去——正是上面那个后果。锁按标识分而不是整批一把：
    不同标题之间没有互相等的理由，写库那一步不必跟着串行。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._taken: dict[tuple[str, str], _Entry] = {}
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}

    def store(self, doc_title: str, version: str, entry: _Entry, write: Callable[[], None]) -> None:
        """`write` 是真正入库那一步。查、写、占都在这一把（按标识的）锁里。

        **写进库了才算占下**：写失败的那条不占，后面同名的那条该去写它自己的。
        """
        key = (doc_title, version)
        with self._key_lock(key):
            with self._guard:
                first = self._taken.get(key)
                if first is not None and first.identity != entry.identity:
                    raise DocumentCollision(first.source, doc_title)
            write()
            with self._guard:
                self._taken.setdefault(key, entry)

    def _key_lock(self, key: tuple[str, str]) -> threading.Lock:
        """这个标识的那把锁。**取锁本身也在锁里**：两个线程同时建会各拿一把，等于没锁。"""
        with self._guard:
            return self._key_locks.setdefault(key, threading.Lock())
