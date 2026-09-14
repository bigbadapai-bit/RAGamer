"""导入编排器：进度、逐文件独立、幂等重导。

外部依赖全换内存假件——这一层验的是**接线**（字段有没有填全、失败在哪一步被记下、
重导是不是覆盖同一批主键），不是切分与打标的内部逻辑，那两个各有自己的测试。
"""

from __future__ import annotations

import logging

import pytest

from ragamer import importing
from ragamer.answering import Citation
from ragamer.caching import CachedAnswer, CacheUnavailableError, InMemoryAnswerCache, cache_key
from ragamer.chunking import Chunk, ChunkRules
from ragamer.importing import (
    Importer,
    ImportStage,
    SourceKind,
    chunk_id,
    content_hash,
    document_title,
)
from ragamer.sources import NormalizedDoc, SourceAsset, SourceDocument
from ragamer.stores.base import UNVERSIONED, StoreUnavailableError, image_prefix
from ragamer.stores.memory import InMemoryChunkStore, InMemoryObjectStore
from ragamer.tagging import ContentNature, SubjectType, TagVocabulary
from ragamer.vectors.base import ModelUnavailableError
from ragamer.vectors.fake import FakeEmbedder

from .conftest import FakeCrawler

RULES = ChunkRules(max_chars=200, min_chars=40, heading_density=0.02)

GAME = "black_myth"

#: 一份截图的字节。独立上传的攻略截图走的正是带附件那条路。
SCREENSHOT = "截图".encode()

#: 一份词条页：结构齐全，打标全部走结构那条路，一次模型都不调。
WIKI_ARTICLE = """\
# 二郎神

{{信息框
| 名称 = 二郎神
| 类型 = BOSS
}}

二郎神是隐藏 BOSS，需要三阶段打完。

## 获取方式

在第三章的隐藏区域遇到。

## 属性

血量 12000，抗性偏高。

## 打法

先定身，再贴身输出。

[[Category:角色]]
[[Category:妖王]]
"""

#: 另一份词条页：一级标题与上面那份不同，一批里各成一篇。
SECOND_ARTICLE = """\
# 白骨精

{{信息框
| 名称 = 白骨精
| 类型 = 妖王
}}

白骨精在第二章，总共三条命。

## 打法

先破盾，再贴身。
"""

#: 同一份资料的另一版：只剩开头，用来验「重导之后旧的那一截被清掉」。
SHORT_ARTICLE = """\
# 二郎神

{{信息框
| 名称 = 二郎神
| 类型 = BOSS
}}

二郎神是隐藏 BOSS，需要三阶段打完。
"""

BLACK_MYTH = TagVocabulary(
    subject_types=tuple(SubjectType),
    term_mapping={"妖王": SubjectType.CHARACTER},
)


class FailingEmbedder:
    """一调就炸的向量化。用来验证「入库之前失败时，旧的那一批原样留着」。"""

    def embed(self, texts):
        raise ModelUnavailableError("权重取不下来")


class RecordingEnricher:
    """记下补过图的文档，然后把原文原样还回去。真实实现要等二次 OCR 那一票。"""

    def __init__(self) -> None:
        self.docs: list[NormalizedDoc] = []

    def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
        self.docs.append(doc)
        return doc


class StubParser:
    """直接给一份归一化文档的假解析器。

    真去造 MinerU 的字节没有意义——这里验的是编排器拿到带附件的文档之后怎么接
    （`tests/test_mineru.py` 验的是那份文档怎么来的）。
    """

    SUFFIXES = (".png",)

    def __init__(self, doc: NormalizedDoc) -> None:
        self.doc = doc

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        return self.doc


def scanned_doc() -> NormalizedDoc:
    """一份带附件的归一化文档：独立上传的攻略截图就是这样。"""
    return NormalizedDoc(
        markdown="# 二郎神\n\n![立绘](images/a.png)\n\n打法：先定身。\n",
        images=("images/a.png",),
        assets=(SourceAsset("images/a.png", b"PNG", "image/png"),),
    )


def make_importer(chunks=None, *, embedder=None, **kwargs) -> Importer:
    return Importer(
        chunks=chunks if chunks is not None else InMemoryChunkStore(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
        rules=RULES,
        **kwargs,
    )


def markdown(name: str = "二郎神.md", text: str = WIKI_ARTICLE) -> SourceDocument:
    return SourceDocument(filename=name, data=text.encode("utf-8"))


def stored(chunks: InMemoryChunkStore, version: str = UNVERSIONED) -> list:
    """库里这份文档的全部切片，按顺序。"""
    return chunks.fetch_document(GAME, "二郎神", version=version)


# --- 一路走到入库 ---


def test_一份资料从归一化一路走到入库():
    chunks = InMemoryChunkStore()

    result = make_importer(chunks).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    assert result.ok
    assert result.doc_title == "二郎神"
    assert result.chunk_count == len(stored(chunks))
    assert result.chunk_count > 1  # 确实切出了多片，这条才验得到东西


def test_入库的切片每个字段都填上了():
    """写入的字段与建表时的显式声明一一对应，不依赖动态字段。

    除了两个向量，其余字段缺了都不会报错：内容摘要缺了增量导入静默失效、版本缺了
    过滤静默落空、标签缺了检索静默漏召回。所以在这里逐个钉住。
    """
    chunks = InMemoryChunkStore()
    make_importer(chunks).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    for chunk in stored(chunks):
        assert chunk.content
        assert chunk.ancestor_path  # 祖先标题路径：脱离页面也要认得出属于谁
        assert chunk.doc_title == "二郎神"
        assert chunk.game_id == GAME
        assert chunk.version == UNVERSIONED
        # 本地文件没有来源地址。网页那一条落的是抓取时的最终地址，见下面网址那一节
        assert chunk.source_url == ""
        assert chunk.chunk_type in ("text", "table", "image")
        assert chunk.content_hash
        assert chunk.dense_vector is not None
        assert chunk.sparse_vector is not None


def test_两层标签与切片类型一起落库():
    chunks = InMemoryChunkStore()
    result = make_importer(chunks).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    assert result.tags.subject_name == "二郎神"
    assert result.tags.subject_type == (SubjectType.CHARACTER.value,)
    assert set(result.tags.content_nature) >= {ContentNature.WHERE, ContentNature.STATS}
    # 主体名与主体类型是文档级，回写到每一个切片
    assert {chunk.subject_name for chunk in stored(chunks)} == {"二郎神"}
    assert all(chunk.subject_type == ("character",) for chunk in stored(chunks))
    # Infobox 整块一片，与表格同标 `table`
    assert {chunk.chunk_type for chunk in stored(chunks)} == {"text", "table"}


def test_内容性质逐切片判定_同一文档里两种性质并存():
    chunks = InMemoryChunkStore()
    make_importer(chunks).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    natures = [chunk.content_nature for chunk in stored(chunks)]

    assert (ContentNature.WHERE.value,) in natures
    assert (ContentNature.STATS.value,) in natures
    assert (ContentNature.GUIDE.value,) in natures


# --- 幂等重导 ---


def test_同一份资料导入两次_库里的切片数量不变():
    chunks = InMemoryChunkStore()
    importer = make_importer(chunks)

    first = importer.import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    before = [chunk.chunk_id for chunk in stored(chunks)]
    second = importer.import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    assert second.chunk_count == first.chunk_count
    assert [chunk.chunk_id for chunk in stored(chunks)] == before


def test_重导时片数变少_旧的那一截被清掉():
    """只靠覆盖写入会留下一截旧切片——查得出来、还进聚合父块，而且不报错。"""
    chunks = InMemoryChunkStore()
    importer = make_importer(chunks)

    long = importer.import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    short = importer.import_one(markdown(text=SHORT_ARTICLE), game_id=GAME, vocabulary=BLACK_MYTH)

    assert short.chunk_count < long.chunk_count  # 确实变少了，这条才验得到东西
    assert len(stored(chunks)) == short.chunk_count
    assert [chunk.chunk_index for chunk in stored(chunks)] == list(range(short.chunk_count))


def test_不同版本的一份资料并存():
    """新版本作为新文档导入并与旧版本并存，不替换（ADR-0004）。"""
    chunks = InMemoryChunkStore()
    importer = make_importer(chunks)

    importer.import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    importer.import_one(markdown(), game_id=GAME, version="2.0", vocabulary=BLACK_MYTH)

    assert stored(chunks, version=UNVERSIONED)
    assert stored(chunks, version="2.0")
    # 未标注版本的那一批没有被 2.0 的重导删掉
    assert {chunk.version for chunk in stored(chunks, version=UNVERSIONED)} == {UNVERSIONED}


def test_同一批里两份同标题的文件不互相覆盖():
    """文档标题是重导替换的范围：两份一级标题相同的文件，后一份会把前一份删掉。

    一批里出现这种撞车时后一份失败，**宁可少一份也不静默丢一份**——两条都报成功而
    库里只剩一条，是这一层最难查的一种结果。
    """
    chunks = InMemoryChunkStore()
    reference = InMemoryChunkStore()
    make_importer(reference).import_one(markdown("甲.md"), game_id=GAME, vocabulary=BLACK_MYTH)

    results = make_importer(chunks).batch(
        [markdown("甲.md"), markdown("乙.md", text=SHORT_ARTICLE)],
        game_id=GAME,
        vocabulary=BLACK_MYTH,
    )

    assert [result.ok for result in results] == [True, False]
    assert results[1].stage is ImportStage.STORE
    assert "甲.md" in results[1].error and "二郎神" in results[1].error
    # 「分两次导入」不是出路：后一次会把前一次整份替掉，两份留不下。文案不能把人往那儿引
    assert "分两次导入也留不下两份" in results[1].error
    assert results[1].chunk_count == 0
    # 撞车结构化地带出来：界面据此把这一条挡在「重试」之外（单独重试会删掉甲）
    assert results[1].collides_with == "甲.md"
    assert results[0].collides_with == ""
    # 乙是同一篇的短版本：真写下去的话甲那一批切片会被它替掉
    assert [chunk.content for chunk in stored(chunks)] == [
        chunk.content for chunk in stored(reference)
    ]


def test_同一批里同一份资料提交两次不算撞车():
    """同一份内容再导一次是幂等的：文档标识一样，切片主键也一样，写下去等于没写。

    撞车的判断因此不只看标题：**同一批里同一份资料重复提交要放行**，
    否则「全选文件夹」把同一个文件带进来两次就成了一个假失败。
    """
    chunks = InMemoryChunkStore()

    results = make_importer(chunks).batch(
        [markdown("二郎神.md"), markdown("二郎神.md")], game_id=GAME, vocabulary=BLACK_MYTH
    )

    assert [result.ok for result in results] == [True, True]
    assert len(stored(chunks)) == results[0].chunk_count


def test_写失败的那条不占文档标识():
    """认领发生在真的写进库之后。

    前一条写挂了、后一条同名，后一条该报自己的写错误，而不是被误报成「跟前面那条撞了」
    ——后者会把人打发去改标题，白改一场。
    """

    class FailingStore(InMemoryChunkStore):
        def upsert(self, game_id: str, chunks) -> None:
            raise StoreUnavailableError("Milvus", "milvus.test:19530", 2.5, "连接被拒绝")

    results = make_importer(FailingStore()).batch(
        [markdown("甲.md"), markdown("乙.md", text=SHORT_ARTICLE)],
        game_id=GAME,
        vocabulary=BLACK_MYTH,
    )

    assert [result.ok for result in results] == [False, False]
    assert [result.collides_with for result in results] == ["", ""]
    assert all(result.stage is ImportStage.STORE for result in results)


def test_结果里带着来源的种类():
    """界面靠它决定失败的那条回填到网址框还是文件框，不靠字符串长得像不像地址。"""
    importer = make_importer(InMemoryChunkStore(), crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE}))

    results = importer.batch([markdown()], urls=[MISSING_URL], game_id=GAME, vocabulary=BLACK_MYTH)

    assert [result.kind for result in results] == [SourceKind.FILE, SourceKind.URL]
    assert not results[1].ok  # 抓取失败那条也带得出种类，重试才知道往哪儿回填


def test_两个不同网址切成同一个标题时后一条失败():
    """撞车不只发生在文件之间：两条来源只要落成同一个文档标识就一样会互相覆盖。"""
    other_url = "https://wiki.test/wiki/二郎神/打法"
    chunks = InMemoryChunkStore()
    importer = make_importer(
        chunks, crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE, other_url: WIKI_ARTICLE})
    )

    results = importer.batch_urls([PAGE_URL, other_url], game_id=GAME, vocabulary=BLACK_MYTH)

    assert [result.ok for result in results] == [True, False]
    assert PAGE_URL in results[1].error
    assert {chunk.source_url for chunk in stored(chunks)} == {PAGE_URL}


def test_同一标题的两个版本互不相干():
    """文档标识是「标题 + 版本」：两个版本本来就要并存（ADR-0004），不是撞车。"""
    chunks = InMemoryChunkStore()
    importer = make_importer(chunks)

    importer.batch([markdown("甲.md")], game_id=GAME, version="1.0", vocabulary=BLACK_MYTH)
    results = importer.batch(
        [markdown("乙.md")], game_id=GAME, version="2.0", vocabulary=BLACK_MYTH
    )

    assert results[0].ok
    assert stored(chunks, version="1.0") and stored(chunks, version="2.0")


def test_重导时换了版本不动别的版本():
    chunks = InMemoryChunkStore()
    importer = make_importer(chunks)
    importer.import_one(markdown(), game_id=GAME, version="1.0", vocabulary=BLACK_MYTH)
    before = len(stored(chunks, version="1.0"))

    importer.import_one(
        markdown(text=SHORT_ARTICLE), game_id=GAME, version="2.0", vocabulary=BLACK_MYTH
    )

    assert len(stored(chunks, version="1.0")) == before


# --- 一批里失败不牵连其余 ---


def test_某个文件失败时其余照常入库():
    chunks = InMemoryChunkStore()
    bad = SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7")

    results = make_importer(chunks).batch(
        [markdown("甲.md"), bad, markdown("乙.md")], game_id=GAME, vocabulary=BLACK_MYTH
    )

    assert [result.ok for result in results] == [True, False, True]
    assert chunks.fetch_document(GAME, "二郎神", version=UNVERSIONED)


def test_失败信息带文件名与失败阶段():
    results = make_importer().batch(
        [SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7")],
        game_id=GAME,
        vocabulary=BLACK_MYTH,
    )

    assert results[0].source == "攻略.pdf"
    assert results[0].stage is ImportStage.NORMALIZE
    assert "攻略.pdf" in results[0].error


def test_向量化那一步失败时阶段报得出来():
    results = make_importer(embedder=FailingEmbedder()).batch(
        [markdown()], game_id=GAME, vocabulary=BLACK_MYTH
    )

    assert results[0].stage is ImportStage.EMBED
    assert "权重取不下来" in results[0].error


def test_入库之前失败时旧的那一批原样留着():
    """删除排在最后而不是最早：中途失败不该留下一个空文档。"""
    chunks = InMemoryChunkStore()
    make_importer(chunks).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    before = [chunk.chunk_id for chunk in stored(chunks)]

    result = make_importer(chunks, embedder=FailingEmbedder()).import_one(
        markdown(text=SHORT_ARTICLE), game_id=GAME, vocabulary=BLACK_MYTH
    )

    assert not result.ok
    assert [chunk.chunk_id for chunk in stored(chunks)] == before


# --- 进度上报 ---


def test_每进入一个阶段上报一次():
    seen: list[tuple[str, ImportStage, int, int]] = []
    make_importer(
        on_progress=lambda event: seen.append(
            (event.source, event.stage, event.file_number, event.file_total)
        )
    ).batch([markdown("甲.md"), markdown("乙.md")], game_id=GAME, vocabulary=BLACK_MYTH)

    assert [stage for _, stage, _, _ in seen[:5]] == [
        ImportStage.NORMALIZE,
        ImportStage.CHUNK,
        ImportStage.TAG,
        ImportStage.EMBED,
        ImportStage.STORE,
    ]
    assert [(number, total) for _, _, number, total in seen] == [(1, 2)] * 5 + [(2, 2)] * 5


def test_补图没接上时不报这一步():
    """报了就是假进度：界面会显示一个根本没在跑的阶段。"""
    seen: list[ImportStage] = []
    importer = make_importer(on_progress=lambda event: seen.append(event.stage))

    importer.import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    assert ImportStage.ENRICH not in seen

    enricher = RecordingEnricher()
    Importer(
        chunks=InMemoryChunkStore(),
        embedder=FakeEmbedder(),
        rules=RULES,
        enricher=enricher,
    ).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    assert len(enricher.docs) == 1


def test_走到哪报到哪_失败时后面的阶段不出现():
    seen: list[ImportStage] = []
    results = make_importer(on_progress=lambda event: seen.append(event.stage)).batch(
        [SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7")],
        game_id=GAME,
        vocabulary=BLACK_MYTH,
    )

    assert seen == [ImportStage.NORMALIZE]
    assert [event.stage for event in results[0].progress] == [ImportStage.NORMALIZE]


def test_结果里带着这个文件走过的阶段():
    result = make_importer().import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    assert [event.stage for event in result.progress][-1] is ImportStage.STORE
    first = result.progress[0]
    assert (first.source, first.file_number, first.file_total) == ("二郎神.md", 1, 1)


# --- 跳过的条数 ---


def test_没有正文的切片不入库并计入跳过(monkeypatch):
    """正文为空的切片没什么可向量化的，写进去只会让检索冒出一条空命中。

    今天的切分器不会产出这种片（`_split` 已经滤掉空段），这里换掉切分器把这一手
    单独验出来——补图那一层接上之后，OCR 与摘要都失败的图片切片正是这种。
    """
    blank = Chunk("   ", 0, "")
    monkeypatch.setattr(importing, "chunk_document", lambda markdown, rules: [blank])
    chunks = InMemoryChunkStore()

    result = make_importer(chunks).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)

    assert result.ok
    assert (result.chunk_count, result.skipped) == (0, 1)
    assert stored(chunks) == []


# --- 解析产物里的附件 ---


def test_附件存进对象存储_切片带着对象_key():
    """独立上传的截图走的正是这条路：图先落对象存储，切片带着它的 key。

    key 落在切片的 `image_urls` 上而**不在正文里**：正文里只有替代文本，地址是由
    这一路单独带着去回显的（`ragamer.chunking`）。
    """
    chunks = InMemoryChunkStore()
    objects = InMemoryObjectStore()
    importer = make_importer(chunks, parser=StubParser(scanned_doc()), objects=objects)

    result = importer.import_one(SourceDocument("攻略.png", SCREENSHOT), game_id=GAME)

    assert result.ok
    keys = objects.list_keys(image_prefix(GAME))
    assert len(keys) == 1
    assert objects.get(keys[0]) == b"PNG"
    assert all(keys[0] in chunk.image_urls for chunk in stored(chunks))
    assert all(keys[0] not in chunk.content for chunk in stored(chunks))


def test_同一份资料重导_图片原样覆盖同一个对象():
    """对象 key 取自来源文件的字节，与切片主键是同一套幂等思路。"""
    objects = InMemoryObjectStore()
    importer = make_importer(parser=StubParser(scanned_doc()), objects=objects)
    source = SourceDocument("攻略.png", SCREENSHOT)

    importer.import_one(source, game_id=GAME)
    importer.import_one(source, game_id=GAME)

    assert len(objects.list_keys(image_prefix(GAME))) == 1


def test_有附件却没接对象存储时那个文件失败并说清原因():
    """静默把图丢掉是最坏的结果：库里一堆指向不存在对象的引用，而且什么都不报。"""
    result = make_importer(parser=StubParser(scanned_doc())).import_one(
        SourceDocument("攻略.png", SCREENSHOT), game_id=GAME
    )

    assert not result.ok
    assert result.stage is ImportStage.NORMALIZE
    assert "对象存储" in (result.error or "")


# --- 主键与摘要 ---


def test_同一份资料的切片主键稳定_换个游戏或版本就不一样():
    key = {"game_id": GAME, "doc_title": "二郎神", "version": UNVERSIONED, "chunk_index": 3}

    assert chunk_id(**key) == chunk_id(**key)
    assert chunk_id(**key) != chunk_id(**{**key, "game_id": "wukong"})
    assert chunk_id(**key) != chunk_id(**{**key, "version": "2.0"})
    assert chunk_id(**key) != chunk_id(**{**key, "chunk_index": 4})
    assert chunk_id(**key) >= 0  # 右移一位，避开 INT64 的符号位


def test_游戏与标题的拼接不会串味():
    """`("ab", "c")` 与 `("a", "bc")` 拼出来必须是两个键，否则两份文档互相覆盖。"""
    shared = {"version": UNVERSIONED, "chunk_index": 0}

    assert chunk_id(game_id="ab", doc_title="c", **shared) != chunk_id(
        game_id="a", doc_title="bc", **shared
    )


def test_内容摘要只随正文变():
    assert content_hash("正文") == content_hash("正文")
    assert content_hash("正文") != content_hash("正文 ")
    assert len(content_hash("正文")) <= 64  # schema 给 content_hash 的长度上限


# --- 文档标题 ---


def test_文档标题取一级标题():
    assert document_title("# 二郎神\n\n正文", "x.md") == "二郎神"


def test_没有一级标题时回落到文件名():
    assert document_title("没有标题的正文", "两郎神.md") == "两郎神"
    assert document_title("", "路径/两郎神.md") == "两郎神"


@pytest.mark.parametrize("version", [UNVERSIONED, "2.0"])
def test_版本原样落库(version):
    chunks = InMemoryChunkStore()
    make_importer(chunks).import_one(markdown(), game_id=GAME, version=version)

    assert {chunk.version for chunk in stored(chunks, version=version)} == {version}


# --- 缓存失效 ---


class RecordingCache(InMemoryAnswerCache):
    """记下每一次按前缀删，其余行为与内存假件一致。"""

    def __init__(self) -> None:
        super().__init__()
        self.invalidated: list[str] = []

    def invalidate(self, game_id: str) -> int:
        self.invalidated.append(game_id)
        return super().invalidate(game_id)


class UnreachableCache(RecordingCache):
    """一个连不上的缓存：按前缀删会抛「不可达」。"""

    def invalidate(self, game_id: str) -> int:
        self.invalidated.append(game_id)
        raise CacheUnavailableError("Redis", "redis.test:6379", 5.0, "连接被拒绝")


def test_导入完成后按游戏前缀清缓存():
    """语料变了，基于旧语料的答案不该再命中（架构文档 §4）。"""
    cache = RecordingCache()
    cached_answer = CachedAnswer("先定身再贴身输出[1]。", (Citation(1, "二郎神", ""),))
    cache.set(cache_key(GAME, UNVERSIONED, "二郎神怎么打"), cached_answer)

    make_importer(cache=cache).batch([markdown()], game_id=GAME)

    assert cache.invalidated == [GAME]
    assert cache.get(cache_key(GAME, UNVERSIONED, "二郎神怎么打")) is None


def test_一个文件都没成时不清缓存():
    """库里什么都没变，删了只是让一批热问题白重算一遍。"""
    cache = RecordingCache()
    importer = make_importer(cache=cache)

    results = importer.batch([SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7")], game_id=GAME)

    assert not any(result.ok for result in results)
    assert cache.invalidated == []


def test_一批里有一个成功就清缓存():
    cache = RecordingCache()

    make_importer(cache=cache).batch(
        [SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7"), markdown()], game_id=GAME
    )

    assert cache.invalidated == [GAME]


def test_缓存清不掉不影响这一批的结果(caplog):
    """导入本身已经成功了，缓存的账是另一本——缓存不通时照常作答，只是下次仍要重算。"""
    chunks = InMemoryChunkStore()

    with caplog.at_level(logging.WARNING):
        results = make_importer(chunks, cache=UnreachableCache()).batch([markdown()], game_id=GAME)

    assert results[0].ok
    assert stored(chunks)
    assert "没清掉" in caplog.text


def test_没接缓存时照常导入():
    """没接缓存就没有要失效的东西，这一步整个不做。"""
    chunks = InMemoryChunkStore()

    results = make_importer(chunks).batch([markdown()], game_id=GAME)

    assert results[0].ok
    assert stored(chunks)


# --- 网址那条入口 ---

PAGE_URL = "https://wiki.test/wiki/二郎神"
MISSING_URL = "https://wiki.test/wiki/没有这页"


def test_网址导入走的是同一条链路():
    """「抓取结果与本地文件走完全相同的后续链路」——同一份正文，两条入口切出来的片
    除来源地址之外逐字相同：切分与打标确实不感知来源。"""
    from_file = InMemoryChunkStore()
    from_url = InMemoryChunkStore()
    make_importer(from_file).import_one(markdown(), game_id=GAME, vocabulary=BLACK_MYTH)
    make_importer(from_url, crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE})).import_url(
        PAGE_URL, game_id=GAME, vocabulary=BLACK_MYTH
    )

    fields = (
        "chunk_id",
        "content",
        "ancestor_path",
        "chunk_index",
        "chunk_type",
        "doc_title",
        "subject_name",
        "subject_type",
        "content_nature",
        "game_terms",
        "content_hash",
    )
    reference = stored(from_file)
    fetched = stored(from_url)
    assert len(reference) == len(fetched) > 1  # 确实切出了多片，这条才验得到东西
    assert [tuple(getattr(chunk, name) for name in fields) for chunk in reference] == [
        tuple(getattr(chunk, name) for name in fields) for chunk in fetched
    ]


def test_网址导入把来源地址落进每一片():
    """答案的引用里要显示的就是它。"""
    chunks = InMemoryChunkStore()

    result = make_importer(chunks, crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE})).import_url(
        PAGE_URL, game_id=GAME, vocabulary=BLACK_MYTH
    )

    assert result.ok
    assert {chunk.source_url for chunk in stored(chunks)} == {PAGE_URL}


def test_抓取失败落在归一化那一步():
    chunks = InMemoryChunkStore()

    result = make_importer(chunks, crawler=FakeCrawler()).import_url(
        PAGE_URL, game_id=GAME, vocabulary=BLACK_MYTH
    )

    assert not result.ok
    assert result.stage is ImportStage.NORMALIZE
    assert PAGE_URL in result.source
    assert stored(chunks) == []


def test_网址导入的进度事件里报的是地址():
    """网页没有文件名，进度与结果里能指认这一条的就是地址。"""
    events = []
    make_importer(
        InMemoryChunkStore(),
        crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE}),
        on_progress=events.append,
    ).import_url(PAGE_URL, game_id=GAME)

    assert events
    assert {event.source for event in events} == {PAGE_URL}


def test_一批网址逐个独立():
    chunks = InMemoryChunkStore()
    importer = make_importer(chunks, crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE}))

    results = importer.batch_urls([PAGE_URL, MISSING_URL], game_id=GAME, vocabulary=BLACK_MYTH)

    assert [result.ok for result in results] == [True, False]
    assert stored(chunks)  # 失败的那一条没有牵连成功的那一条


def test_一次提交里文件与网址混在一起():
    """界面上一次提交可以同时带文件与网址，那它们就是**一批**：

    编号连着往后排（进度里说得出「第几条 / 共几条」），两条来源也共用同一份文档标识
    的认领表——否则同一次提交里一个文件与一个网址撞上，谁也拦不住。
    """
    chunks = InMemoryChunkStore()
    seen: list[importing.ProgressEvent] = []
    page = "# 金角大王\n\n银角大王的哥哥，拿着紫金红葫芦与羊脂玉净瓶，在平顶山占山为王。\n"
    importer = make_importer(
        chunks, crawler=FakeCrawler(**{PAGE_URL: page}), on_progress=seen.append
    )

    results = importer.batch(
        [markdown("甲.md"), markdown("乙.md", text=SECOND_ARTICLE)],
        urls=[PAGE_URL, MISSING_URL],
        game_id=GAME,
        vocabulary=BLACK_MYTH,
    )

    assert [result.source for result in results] == ["甲.md", "乙.md", PAGE_URL, MISSING_URL]
    assert [result.ok for result in results] == [True, True, True, False]
    assert {event.file_total for event in seen} == {4}
    assert [event.file_number for event in seen if event.stage is ImportStage.NORMALIZE] == [
        1,
        2,
        3,
        4,
    ]


def test_没接抓取器时导入网址当场报错():
    """接线漏了是程序错，不是这份资料错——不该被兜成一个「失败的结果」：
    那样它就跟在一批正常的业务失败里，谁也不觉得要去修。"""
    with pytest.raises(ValueError):
        make_importer(InMemoryChunkStore()).import_url(PAGE_URL, game_id=GAME)


# --- 图片收进对象存储 ---


#: 网页来源的图：正文在 wiki.test、图在 cdn.test（bwiki 就是这个形状）。
#: 130px 那份是立绘、18px 那份是行内图标——判据在 `ragamer.sources.is_icon`。
CDN_IMAGE = "https://cdn.test/thumb/abc.png/130px-立绘.png"
CDN_ICON = "https://cdn.test/thumb/abc.png/18px-图标.png"
#: 上面那张缩略图的**原图**：wiki 是按显示宽度另存一份缩略图，原图在 `thumb` 的上一层
#: （`ragamer.sources.original_ref`）。收的是它——130px 那份放大就是糊的。
CDN_ORIGINAL = "https://cdn.test/abc.png"


def test_网址导入时外链图落进对象存储():
    """网页来源的图以前是外链，而补图那一层按对象 key 取原图，那些图一张都补不上。

    归一化那一层现在把它们收进来，「出了这一层只剩对象 key 一种形态」这句话
    对网页来源也才成立。
    """
    chunks = InMemoryChunkStore()
    objects = InMemoryObjectStore()
    crawler = FakeCrawler(
        images={CDN_ORIGINAL: b"PNG"},
        **{PAGE_URL: f"# 二郎神\n\n![立绘]({CDN_IMAGE})\n\n正文。\n"},
    )
    importer = make_importer(chunks, crawler=crawler, objects=objects)

    result = importer.import_url(PAGE_URL, game_id=GAME)

    assert result.ok, result.error
    keys = objects.list_keys(image_prefix(GAME))
    assert len(keys) == 1
    assert objects.get(keys[0]) == b"PNG"
    # 取的是原图：缩略图那份只有 130px 宽，回显出来放大就糊
    assert crawler.images_requested == [CDN_ORIGINAL]
    # 正文里的引用改指对象 key，切片带着它去回显（正文里只有替代文本）
    assert all(keys[0] in chunk.image_urls for chunk in stored(chunks))
    assert all(CDN_IMAGE not in chunk.content for chunk in stored(chunks))


def test_本地_md_里的外链图同样收进来():
    """md 与网页一样带着外链——攻略站另存成 md 是常事，那份资料的图也该收进来。"""
    chunks = InMemoryChunkStore()
    objects = InMemoryObjectStore()
    importer = make_importer(
        chunks, crawler=FakeCrawler(images={CDN_ORIGINAL: b"PNG"}), objects=objects
    )

    result = importer.import_one(
        markdown(text=f"# 二郎神\n\n![立绘]({CDN_IMAGE})\n\n正文。\n"), game_id=GAME
    )

    assert result.ok, result.error
    assert len(objects.list_keys(image_prefix(GAME))) == 1


def test_行内图标不收进对象存储():
    """18px 的图标不值得占一份存储，也不值得为它调一次视觉模型。

    它也**不进切片的图片字段**：回显是拿地址去对象存储取的，没存下来的地址带进答案
    只会是一条取不到的死图（页面上报「没有这张图」）。图标是版面装饰，本来也不该显示。
    """
    chunks = InMemoryChunkStore()
    objects = InMemoryObjectStore()
    crawler = FakeCrawler(**{PAGE_URL: f"# 二郎神\n\n![图标]({CDN_ICON})\n\n正文。\n"})
    importer = make_importer(chunks, crawler=crawler, objects=objects)

    result = importer.import_url(PAGE_URL, game_id=GAME)

    assert result.ok, result.error
    assert objects.list_keys(image_prefix(GAME)) == []
    assert crawler.images_requested == []
    assert all(chunk.image_urls == () for chunk in stored(chunks))
    assert all(CDN_ICON not in chunk.content for chunk in stored(chunks))


def test_图取不到时那份资料照常入库(caplog):
    """与「一张图补不上不让整份资料失败」同一条规矩。"""
    chunks = InMemoryChunkStore()
    crawler = FakeCrawler(**{PAGE_URL: f"# 二郎神\n\n![立绘]({CDN_IMAGE})\n\n正文。\n"})
    importer = make_importer(chunks, crawler=crawler, objects=InMemoryObjectStore())

    with caplog.at_level("WARNING"):
        result = importer.import_url(PAGE_URL, game_id=GAME)

    assert result.ok, result.error
    assert stored(chunks)
    assert "取不到" in caplog.text


def test_没接抓取器时外链图留痕(caplog):
    """接线漏了与「这份资料本来就没有外链图」长得一模一样，所以要留一条痕。"""
    importer = make_importer(InMemoryChunkStore())  # 没接抓取器

    with caplog.at_level(logging.WARNING, logger="ragamer.importing"):
        result = importer.import_one(
            markdown(text=f"# 二郎神\n\n![立绘]({CDN_IMAGE})\n\n正文。\n"), game_id=GAME
        )

    assert result.ok, result.error
    assert "外链图" in caplog.text


def test_同一个网址重导时图片原地覆盖():
    """来源摘要取自地址本身，与文件那条路取自字节是同一套幂等思路。"""
    objects = InMemoryObjectStore()
    crawler = FakeCrawler(
        images={CDN_ORIGINAL: b"PNG"},
        **{PAGE_URL: f"# 二郎神\n\n![立绘]({CDN_IMAGE})\n\n正文。\n"},
    )
    importer = make_importer(crawler=crawler, objects=objects)

    importer.import_url(PAGE_URL, game_id=GAME)
    keys = objects.list_keys(image_prefix(GAME))
    importer.import_url(PAGE_URL, game_id=GAME)

    assert objects.list_keys(image_prefix(GAME)) == keys
