"""集成测试：真的连云端。默认不跑，`uv run pytest -m integration` 才跑。

假件覆盖不到的东西都在这里——建表参数服务端收不收、过滤表达式的语法对不对、
写进去的切片检索得到不。跑之前需要一份填好的 `.env`。

**它会在配置的 database 里建一个探针 collection，跑完删掉**，只碰探针自己的数据。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ragamer.config import ConfigError, Settings, load_settings
from ragamer.container import build_container
from ragamer.logging import setup_logging
from ragamer.stores.base import Chunk, ChunkFilter, DocumentSummary
from ragamer.stores.chunks import DENSE_DIM

from .conftest import fake_vector, make_chunk

pytestmark = pytest.mark.integration

#: 探针用的游戏 id。带 it_ 前缀，一眼看出是集成测试留下的。
PROBE_GAME = "ragamer_it_probe"
PROBE_PREFIX = "ragamer_it_probe/"
PROBE_COLLECTION = "ragamer_it_probe"


@pytest.fixture(scope="module")
def settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        pytest.skip(f"没有可用的配置，跳过集成测试：{exc}")


@pytest.fixture(scope="module", autouse=True)
def logging_setup(settings: Settings) -> None:
    setup_logging(settings.log_level)


@pytest.fixture
def probe_chunks() -> Iterator[list[Chunk]]:
    """两个切片，稠密向量按维度补齐——服务端会校维度。"""
    yield [
        make_chunk(
            1,
            game_id=PROBE_GAME,
            doc_title="探针文档",
            chunk_index=0,
            content="二郎神第一阶段的打法",
            subject_type=("character", "skill"),
            content_nature=("guide",),
            # 数组字段的容量、元素长度与服务端的收法只有真库验得了
            image_urls=(
                "https://patchwiki.biligame.com/images/wukong/thumb/b/b1/x.png/18px-%E5%9B%BE%E6%A0%87.png",
                "images/black_myth/0123456789abcdef/phase2.jpg",
            ),
            dense_vector=fake_vector(1, DENSE_DIM),
            sparse_vector={1: 0.5, 7: 0.25},
        ),
        make_chunk(
            2,
            game_id=PROBE_GAME,
            doc_title="探针文档",
            chunk_index=1,
            content="二郎神掉落的物品",
            subject_type=("character",),
            content_nature=("where",),
            dense_vector=fake_vector(2, DENSE_DIM),
            sparse_vector={2: 0.5},
        ),
    ]


def test_建表_写入_检索_取回_删表(settings: Settings, probe_chunks: list[Chunk]):
    """一条走完的链路。任何一步不对，都说明接到云端这层出了问题。"""
    container = build_container(settings)
    # 各服务自检自己那一个：某个服务没配好，不该把别的服务的用例一起拖死
    container.chunks.check()
    try:
        container.chunks.ensure_collection(PROBE_GAME)
        # 建表是幂等的：再跑一次不该炸
        container.chunks.ensure_collection(PROBE_GAME)

        container.chunks.upsert(PROBE_GAME, probe_chunks)

        hits = container.chunks.search(
            PROBE_GAME,
            dense=probe_chunks[0].dense_vector or (),
            sparse=probe_chunks[0].sparse_vector,
            where=ChunkFilter(version="1.0", subject_types=("skill",)),
            limit=5,
        )
        assert [hit.chunk.chunk_id for hit in hits] == [1]
        assert hits[0].chunk.subject_type == ("character", "skill")
        # 图片地址是数组字段，长度与顺序都要原样回来
        assert hits[0].chunk.image_urls == probe_chunks[0].image_urls

        # 聚合父块走的那条路：按 doc_title 回查同文档的兄弟切片
        siblings = container.chunks.fetch_document(PROBE_GAME, "探针文档", version="1.0")
        assert [chunk.chunk_id for chunk in siblings] == [1, 2]
        assert [chunk.chunk_index for chunk in siblings] == [0, 1]
    finally:
        container.chunks.drop(PROBE_GAME)


def test_未标注版本的切片在版本过滤下也取得到(settings: Settings, probe_chunks: list[Chunk]):
    """漏掉"未标注版本"这一支是静默失效，只能靠真检索抓。"""
    container = build_container(settings)
    container.chunks.check()
    unversioned = make_chunk(
        3,
        game_id=PROBE_GAME,
        doc_title="探针文档",
        chunk_index=2,
        content="世界观设定，不随版本变化",
        version="",
        dense_vector=fake_vector(3, DENSE_DIM),
        sparse_vector={3: 0.5},
    )
    try:
        container.chunks.ensure_collection(PROBE_GAME)
        container.chunks.upsert(PROBE_GAME, [*probe_chunks, unversioned])

        siblings = container.chunks.fetch_document(PROBE_GAME, "探针文档", version="1.0")

        assert [chunk.chunk_id for chunk in siblings] == [1, 2, 3]
    finally:
        container.chunks.drop(PROBE_GAME)


def test_列版本扫得完整且不漏未标注版本(settings: Settings, probe_chunks: list[Chunk]):
    """列版本扫的是整个 collection，服务端那条 `version != ""` 的表达式只有真库验得了。

    两个版本 + 一批未标注版本的切片：列出来的是两个版本，未标注版本不在其中
    （它在库里的取值是空串，与「没判出来」共用同一个字面）。
    """
    container = build_container(settings)
    container.chunks.check()
    other_version = make_chunk(
        3,
        game_id=PROBE_GAME,
        doc_title="探针文档",
        chunk_index=2,
        version="2.0",
        dense_vector=fake_vector(3, DENSE_DIM),
        sparse_vector={3: 0.5},
    )
    unversioned = make_chunk(
        4,
        game_id=PROBE_GAME,
        doc_title="探针文档",
        chunk_index=3,
        version="",
        dense_vector=fake_vector(4, DENSE_DIM),
        sparse_vector={4: 0.5},
    )
    try:
        container.chunks.ensure_collection(PROBE_GAME)
        container.chunks.upsert(PROBE_GAME, [*probe_chunks, other_version, unversioned])

        assert container.chunks.versions(PROBE_GAME) == ("1.0", "2.0")
    finally:
        container.chunks.drop(PROBE_GAME)


def test_数资料要按标题与版本分组(settings: Settings, probe_chunks: list[Chunk]):
    """知识库管理页靠它列「我导进了什么」。

    分组、片数、以及未标注版本（空串）要原样回来——这三样都只有真库验得了：
    换成分页迭代扫之后，同一份文档的切片会落在不同页上，数漏了页面上就少一片。
    """
    container = build_container(settings)
    container.chunks.check()
    other_doc = make_chunk(
        3,
        game_id=PROBE_GAME,
        doc_title="另一份资料",
        chunk_index=0,
        dense_vector=fake_vector(3, DENSE_DIM),
        sparse_vector={3: 0.5},
    )
    other_version = make_chunk(
        4,
        game_id=PROBE_GAME,
        doc_title="探针文档",
        chunk_index=2,
        version="2.0",
        dense_vector=fake_vector(4, DENSE_DIM),
        sparse_vector={4: 0.5},
    )
    try:
        container.chunks.ensure_collection(PROBE_GAME)
        container.chunks.upsert(PROBE_GAME, [*probe_chunks, other_doc, other_version])

        # 按 (标题, 版本) 字面升序——码点，不是拼音（「另」排在「探」前面）
        assert container.chunks.documents(PROBE_GAME) == (
            DocumentSummary("另一份资料", "1.0", 1),
            DocumentSummary("探针文档", "1.0", 2),
            DocumentSummary("探针文档", "2.0", 1),
        )
    finally:
        container.chunks.drop(PROBE_GAME)


def test_列出库里引用到的图片_key(settings: Settings, probe_chunks: list[Chunk]):
    """`image_urls` 是数组字段，扫全库只取它这一列要真的回得来（见 `ChunkStore.image_keys`）。

    探针那一批里本来就摆了一条外链与一条对象 key：前者不该混进来。
    """
    container = build_container(settings)
    container.chunks.check()
    try:
        container.chunks.ensure_collection(PROBE_GAME)
        container.chunks.upsert(PROBE_GAME, list(probe_chunks))

        assert container.chunks.image_keys(PROBE_GAME) == (
            "images/black_myth/0123456789abcdef/phase2.jpg",
        )
        # 探针那一批全属于同一份：跳过它就什么都不剩
        assert container.chunks.image_keys(PROBE_GAME, excluding=("探针文档", "1.0")) == ()
    finally:
        container.chunks.drop(PROBE_GAME)


def test_列一个还没有资料的库得到空(settings: Settings):
    """建了库、一份资料都没导：页面上那句「还没有导入任何资料」靠它，扫不存在的表会炸。"""
    container = build_container(settings)
    container.chunks.check()

    assert container.chunks.documents(PROBE_GAME) == ()


def test_列一个还没有切片的库的版本得到空(settings: Settings):
    """建了库、一份资料都没导。这里要的是不报错——扫一个不存在的 collection 会炸。"""
    container = build_container(settings)
    container.chunks.check()

    assert container.chunks.versions(PROBE_GAME) == ()


def test_对象存储能存能取能清(settings: Settings):
    container = build_container(settings)
    container.objects.check()
    try:
        container.objects.put(PROBE_PREFIX + "探针.txt", b"probe", content_type="text/plain")

        assert container.objects.get(PROBE_PREFIX + "探针.txt") == b"probe"
        assert container.objects.delete_prefix(PROBE_PREFIX) == 1
        assert container.objects.list_keys(PROBE_PREFIX) == []
    finally:
        container.objects.delete_prefix(PROBE_PREFIX)


def test_文档存储能存能取能删(settings: Settings):
    container = build_container(settings)
    container.docs.check()
    document = {"probe": PROBE_GAME}
    try:
        container.docs.put(PROBE_COLLECTION, "probe", document)

        assert container.docs.get(PROBE_COLLECTION, "probe") == document
        assert container.docs.list_ids(PROBE_COLLECTION) == ["probe"]
    finally:
        container.docs.delete(PROBE_COLLECTION, "probe")
