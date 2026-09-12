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
from ragamer.stores.base import Chunk, ChunkFilter
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
    container.check()
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

        # 聚合父块走的那条路：按 doc_title 回查同文档的兄弟切片
        siblings = container.chunks.fetch_document(PROBE_GAME, "探针文档", version="1.0")
        assert [chunk.chunk_id for chunk in siblings] == [1, 2]
        assert [chunk.chunk_index for chunk in siblings] == [0, 1]
    finally:
        container.chunks.drop(PROBE_GAME)


def test_未标注版本的切片在版本过滤下也取得到(settings: Settings, probe_chunks: list[Chunk]):
    """漏掉"未标注版本"这一支是静默失效，只能靠真检索抓。"""
    container = build_container(settings)
    container.check()
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


def test_对象存储能存能取能清(settings: Settings):
    container = build_container(settings)
    container.check()
    try:
        container.objects.put(PROBE_PREFIX + "探针.txt", b"probe", content_type="text/plain")

        assert container.objects.get(PROBE_PREFIX + "探针.txt") == b"probe"
        assert container.objects.delete_prefix(PROBE_PREFIX) == 1
        assert container.objects.list_keys(PROBE_PREFIX) == []
    finally:
        container.objects.delete_prefix(PROBE_PREFIX)


def test_文档存储能存能取能删(settings: Settings):
    container = build_container(settings)
    container.check()
    document = {"probe": PROBE_GAME}
    try:
        container.docs.put(PROBE_COLLECTION, "probe", document)

        assert container.docs.get(PROBE_COLLECTION, "probe") == document
        assert container.docs.list_ids(PROBE_COLLECTION) == ["probe"]
    finally:
        container.docs.delete(PROBE_COLLECTION, "probe")
