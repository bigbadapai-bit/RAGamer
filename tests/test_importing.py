"""导入编排器：进度、逐文件独立、幂等重导。

外部依赖全换内存假件——这一层验的是**接线**（字段有没有填全、失败在哪一步被记下、
重导是不是覆盖同一批主键），不是切分与打标的内部逻辑，那两个各有自己的测试。
"""

from __future__ import annotations

import logging

import pytest

from ragamer import importing
from ragamer.chunking import Chunk, ChunkRules
from ragamer.importing import (
    Importer,
    ImportStage,
    chunk_id,
    content_hash,
    document_title,
)
from ragamer.sources import NormalizedDoc, SourceAsset, SourceDocument
from ragamer.stores.base import UNVERSIONED, image_prefix
from ragamer.stores.memory import InMemoryChunkStore, InMemoryObjectStore
from ragamer.tagging import ContentNature, SubjectType, TagVocabulary
from ragamer.vectors.base import ModelUnavailableError
from ragamer.vectors.fake import FakeEmbedder

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


def test_同一批里两份同标题的文件留一条覆盖的痕(caplog):
    """同名即同一份文档（文档标题是重导替换的范围），但两条结果都会报成功。

    这不是错，是界面上「导入了 2 份」的读法要打折扣——所以留一条日志，不假装没发生。
    """
    chunks = InMemoryChunkStore()

    with caplog.at_level(logging.WARNING, logger="ragamer.importing"):
        results = make_importer(chunks).batch(
            [markdown("甲.md"), markdown("乙.md")], game_id=GAME, vocabulary=BLACK_MYTH
        )

    assert [result.ok for result in results] == [True, True]
    assert "甲.md" in caplog.text and "乙.md" in caplog.text
    assert len(stored(chunks)) == results[-1].chunk_count  # 库里只剩后写的那一份


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

    assert results[0].filename == "攻略.pdf"
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
            (event.filename, event.stage, event.file_number, event.file_total)
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
    assert (first.filename, first.file_number, first.file_total) == ("二郎神.md", 1, 1)


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


def test_附件存进对象存储_正文里的引用改指对象_key():
    """独立上传的截图走的正是这条路：图先落对象存储，切片正文里留的是它的 key。"""
    chunks = InMemoryChunkStore()
    objects = InMemoryObjectStore()
    importer = make_importer(chunks, parser=StubParser(scanned_doc()), objects=objects)

    result = importer.import_one(SourceDocument("攻略.png", SCREENSHOT), game_id=GAME)

    assert result.ok
    keys = objects.list_keys(image_prefix(GAME))
    assert len(keys) == 1
    assert objects.get(keys[0]) == b"PNG"
    assert all(keys[0] in chunk.content for chunk in stored(chunks))


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
