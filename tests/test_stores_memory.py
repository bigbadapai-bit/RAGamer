"""内存假件：与真实客户端实现同一组协议，且行为一致。

这组断言就是「可替换」的判据——同一组期望，换成真实客户端也该成立（那部分留给集成测试）。
过滤语义与 collection 名走同一份规则，所以在内存上跑过的行为接到云端仍然成立。
"""

from __future__ import annotations

import pytest

from ragamer.config import load_settings
from ragamer.container import build_container
from ragamer.stores import (
    ChunkFilter,
    ChunkStore,
    DocStore,
    InMemoryChunkStore,
    InMemoryDocStore,
    InMemoryObjectStore,
    MilvusChunkStore,
    MinioObjectStore,
    MongoDocStore,
    ObjectStore,
    Store,
    StoreError,
)
from ragamer.stores.base import UNVERSIONED

from .conftest import fake_vector, make_chunk


def test_三个真实客户端都满足协议(settings_env):
    settings = load_settings(env_file=None)
    stores: list[Store] = [
        MilvusChunkStore(settings.milvus, timeout=settings.store_timeout_seconds),
        MongoDocStore(settings.mongo, timeout=settings.store_timeout_seconds),
        MinioObjectStore(settings.minio, timeout=settings.store_timeout_seconds),
    ]

    for store in stores:
        assert isinstance(store, Store)
        assert store.name and store.address


@pytest.mark.parametrize(
    ("factory", "protocol"),
    [
        (InMemoryChunkStore, ChunkStore),
        (InMemoryDocStore, DocStore),
        (InMemoryObjectStore, ObjectStore),
    ],
)
def test_内存假件实现自己那一组协议(factory, protocol):
    assert isinstance(factory(), protocol)


def test_协议之间不互相将就(memory_container):
    """切片存储不是文档存储——三组协议各自可判别，组合根才能互相替换。"""
    assert isinstance(memory_container.chunks, ChunkStore)
    assert isinstance(memory_container.docs, DocStore)
    assert isinstance(memory_container.objects, ObjectStore)
    assert not isinstance(memory_container.chunks, DocStore)


def test_组合根造出的是真实客户端且每次都是新的一套(settings_env):
    container = build_container(load_settings(env_file=None))
    other = build_container(load_settings(env_file=None))

    assert isinstance(container.chunks, ChunkStore)
    assert isinstance(container.docs, DocStore)
    assert isinstance(container.objects, ObjectStore)
    # 每次构造都是新的一套，不存在跨调用的共享状态
    assert container.chunks is not other.chunks


def test_切片写入后能按主体类型过滤检索(memory_container):
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, subject_name="二郎神", subject_type=("character", "skill")),
            make_chunk(2, subject_name="寒江雪", subject_type=("character",)),
        ],
    )

    hits = store.search(
        "black_myth", dense=fake_vector(1), where=ChunkFilter(subject_types=("skill",))
    )

    assert [hit.chunk.subject_name for hit in hits] == ["二郎神"]


def test_检索按分数排序并遵守条数上限(memory_container):
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert("black_myth", [make_chunk(index) for index in (1, 2, 3)])

    hits = store.search("black_myth", dense=fake_vector(2), limit=2)

    assert len(hits) == 2
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_版本过滤一并纳入未标注版本(memory_container):
    """漏掉未标注版本，用户切到历史版本后世界观类问题会全部答不出。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, version="1.0"),
            make_chunk(2, version="2.0"),
            make_chunk(3, version=UNVERSIONED),
        ],
    )

    hits = store.search("black_myth", dense=fake_vector(1), where=ChunkFilter(version="1.0"))

    assert sorted(hit.chunk.chunk_id for hit in hits) == [1, 3]


def test_列出一个库里真实存在过的版本(memory_container):
    """澄清反问的版本候选只能来自这里（§3.4）：模型自己编的版本号，用户选了也检索不到。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, version="2.0"),
            make_chunk(2, version="1.0"),
            make_chunk(3, version="2.0"),
            make_chunk(4, version=UNVERSIONED),
        ],
    )

    assert store.versions("black_myth") == ("1.0", "2.0")


def test_还没有切片的游戏列不出任何版本(memory_container):
    """建了库、一份资料都没导：这是正常状态，不是错误——空列表就是「没有可选的历史版本」。"""
    assert memory_container.chunks.versions("black_myth") == ()


def test_版本只按这个游戏列(memory_container):
    """一个游戏一个 collection（ADR-0002）：别的游戏的版本不该出现在候选里。"""
    store = memory_container.chunks
    store.upsert("black_myth", [make_chunk(1, version="1.0")])
    store.upsert("yanyun", [make_chunk(2, version="3.0", game_id="yanyun")])

    assert store.versions("black_myth") == ("1.0",)


def test_取一份文档的全部切片按顺序且不混版本(memory_container):
    """聚合父块靠它：命中并截断之后回查同文档的兄弟切片。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, doc_title="二郎神", chunk_index=0, version="1.0"),
            make_chunk(2, doc_title="二郎神", chunk_index=1, version="1.0"),
            make_chunk(3, doc_title="二郎神", chunk_index=0, version="2.0"),
            make_chunk(4, doc_title="寒江雪", chunk_index=0, version="1.0"),
        ],
    )

    chunks = store.fetch_document("black_myth", "二郎神", version="1.0")

    assert [chunk.chunk_id for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_回查不传版本时两个版本都取回(memory_container):
    """问题与知识库都给不出版本时 `version_filter` 有意收窄成不过滤，聚合跟随同一口径；
    回查因此也要能表达「不按版本筛」，否则那一种情形下父块会凭空少掉一半切片。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, doc_title="二郎神", chunk_index=0, version="1.0"),
            make_chunk(2, doc_title="二郎神", chunk_index=1, version="2.0"),
            make_chunk(3, doc_title="二郎神", chunk_index=2, version=UNVERSIONED),
        ],
    )

    chunks = store.fetch_document("black_myth", "二郎神", version=None)

    assert [chunk.chunk_id for chunk in chunks] == [1, 2, 3]


def test_按文档删只删这个版本的切片(memory_container):
    """重导 1.0 版不该连带删掉未标注版本——那是「新版本与旧版本并存」要留的（ADR-0004）。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, doc_title="二郎神", chunk_index=0, version="1.0"),
            make_chunk(2, doc_title="二郎神", chunk_index=1, version="1.0"),
            make_chunk(3, doc_title="二郎神", chunk_index=0, version=UNVERSIONED),
            make_chunk(4, doc_title="寒江雪", chunk_index=0, version="1.0"),
        ],
    )

    store.delete_document("black_myth", "二郎神", version="1.0")

    assert [
        chunk.chunk_id for chunk in store.fetch_document("black_myth", "二郎神", version="1.0")
    ] == [3]
    assert [
        chunk.chunk_id for chunk in store.fetch_document("black_myth", "寒江雪", version="1.0")
    ] == [4]


def test_同一份文档的另一个版本不受影响(memory_container):
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert(
        "black_myth",
        [
            make_chunk(1, doc_title="二郎神", chunk_index=0, version="2.0"),
            make_chunk(2, doc_title="二郎神", chunk_index=0, version="1.0"),
        ],
    )

    store.delete_document("black_myth", "二郎神", version="1.0")

    assert [
        chunk.chunk_id for chunk in store.fetch_document("black_myth", "二郎神", version="2.0")
    ] == [1]


def test_删不存在的文档不报错(memory_container):
    memory_container.chunks.delete_document("black_myth", "查无此页", version="1.0")


def test_同一个切片_id_重复写入是覆盖不是追加(memory_container):
    """重导一份文档就是覆盖同一批 id，不产生重复切片。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert("black_myth", [make_chunk(1, content="旧正文")])
    store.upsert("black_myth", [make_chunk(1, content="新正文")])

    hits = store.search("black_myth", dense=fake_vector(1))

    assert [hit.chunk.content for hit in hits] == ["新正文"]


def test_缺向量的切片入库时报错(memory_container):
    """静默写一条检索不到的切片，是查不出也不报错的失败形态。"""
    store = memory_container.chunks
    store.ensure_collection("black_myth")

    with pytest.raises(ValueError, match="还没有向量"):
        store.upsert("black_myth", [make_chunk(1, dense_vector=None)])


def test_游戏_id_不合法时当场报错(memory_container):
    """真实客户端上头会炸得更晚，规则放在共享层，两边一致。"""
    with pytest.raises(ValueError, match="不合法"):
        memory_container.chunks.ensure_collection("黑神话·悟空")


def test_删库后检索不到该游戏的切片(memory_container):
    store = memory_container.chunks
    store.ensure_collection("black_myth")
    store.upsert("black_myth", [make_chunk(1)])

    store.drop("black_myth")

    assert store.search("black_myth", dense=fake_vector(1)) == []


def test_数出这个游戏有多少切片(memory_container):
    """没建过表的游戏算 0 条，与真实适配器同一条口径。"""
    store = memory_container.chunks
    assert store.count("black_myth") == 0

    store.upsert("black_myth", [make_chunk(1), make_chunk(2)])

    assert store.count("black_myth") == 2
    assert store.count("zelda") == 0


def test_文档存储的增改删查(memory_container):
    docs = memory_container.docs
    assert docs.get("knowledge_bases", "black_myth") is None

    docs.put("knowledge_bases", "black_myth", {"current_version": "1.0"})
    docs.put("knowledge_bases", "black_myth", {"current_version": "2.0"})

    assert docs.get("knowledge_bases", "black_myth") == {"current_version": "2.0"}
    assert docs.list_ids("knowledge_bases") == ["black_myth"]

    docs.delete("knowledge_bases", "black_myth")

    assert docs.get("knowledge_bases", "black_myth") is None
    assert docs.list_ids("knowledge_bases") == []


def test_文档的_id_由参数给而不是载荷(memory_container):
    """载荷里混进 `_id`，真实客户端那边会被判为不可改字段而整条写不进。"""
    docs = memory_container.docs

    docs.put("knowledge_bases", "black_myth", {"_id": "别的", "current_version": "2.0"})

    assert docs.get("knowledge_bases", "black_myth") == {"current_version": "2.0"}
    assert docs.list_ids("knowledge_bases") == ["black_myth"]


def test_按字段取一批文档(memory_container):
    """等值匹配、投影、排序、截断，与真实那边同一套语义——列表那类查询走的就是它。

    一次把四个旋钮都用上：结果里**没有 `turns`**（投影挡住了正文），
    顺序是倒序，条数是上限，别个库的没进来。
    """
    docs = memory_container.docs
    docs.put(
        "conversations",
        "s1",
        {"game_id": "black_myth", "title": "先问的", "updated_at": "2026-09-13T01:00:00+00:00"},
    )
    docs.put(
        "conversations",
        "s2",
        {"game_id": "black_myth", "title": "后问的", "updated_at": "2026-09-13T02:00:00+00:00"},
    )
    docs.put(
        "conversations",
        "s3",
        {"game_id": "another_game", "title": "别个库的", "updated_at": "2026-09-13T03:00:00+00:00"},
    )

    found = docs.find(
        "conversations",
        {"game_id": "black_myth"},
        fields=("title", "updated_at"),
        order_by="updated_at",
        descending=True,
        limit=1,
    )

    assert found == [{"_id": "s2", "title": "后问的", "updated_at": "2026-09-13T02:00:00+00:00"}]


def test_按字段取不到时返回空列表(memory_container):
    """「一条都没有」是列表的正常状态，不是错误。"""
    assert memory_container.docs.find("conversations", {"game_id": "还没有这个库"}) == []
    assert memory_container.docs.find("还没有这个集合") == []


def test_不投影时整份返回(memory_container):
    docs = memory_container.docs
    docs.put("conversations", "s1", {"game_id": "black_myth", "turns": [{"role": "user"}]})

    assert docs.find("conversations") == [
        {"_id": "s1", "game_id": "black_myth", "turns": [{"role": "user"}]}
    ]


def test_对象存储按前缀列出与清理(memory_container):
    objects = memory_container.objects
    objects.ensure_bucket()
    objects.put("images/二郎神.png", b"a")
    objects.put("images/寒江雪.png", b"b")
    objects.put("docs/readme.md", b"c")

    assert objects.list_keys("images/") == ["images/二郎神.png", "images/寒江雪.png"]
    assert objects.get("images/二郎神.png") == b"a"

    assert objects.delete_prefix("images/") == 2
    assert objects.list_keys("images/") == []
    # 前缀之外的没被牵连
    assert objects.list_keys() == ["docs/readme.md"]


def test_前导斜杠不影响前缀匹配(memory_container):
    """原项目的 list 去前导 `/` 而 put 不去，清旧图时一个也匹配不上、静默失效。"""
    objects = memory_container.objects
    objects.put("/images/二郎神.png", b"a")

    assert objects.list_keys("images/") == ["images/二郎神.png"]
    assert objects.delete_prefix("/images/") == 1


def test_取不存在的对象报存储错误(memory_container):
    with pytest.raises(StoreError):
        memory_container.objects.get("images/没有这张.png")


def test_自检在内存版上通过(memory_container):
    """内存版必须能跑通自检，否则测试里就没法把整条链路拉起来。"""
    memory_container.check()

    assert [store.name for store in memory_container.stores()] == ["内存假件"] * 3
