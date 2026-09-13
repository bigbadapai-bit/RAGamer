"""切片存储（Milvus）。

建表参数与过滤表达式是纯函数，直接断言；发出去的请求形状用假客户端断言。
真的建出表、真的检索属于云端行为，留给集成测试。
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from pymilvus import DataType
from pymilvus import MilvusClient as RealMilvusClient

from ragamer.config import MilvusSettings, load_settings
from ragamer.stores import chunks
from ragamer.stores.base import (
    UNVERSIONED,
    ChunkFilter,
    StoreError,
    StoreUnavailableError,
)
from ragamer.stores.chunks import (
    DENSE_DIM,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    MilvusChunkStore,
    chunk_index_params,
    chunk_schema,
    collection_name,
    filter_expression,
)

from .conftest import fake_vector, make_chunk

#: 与 schema 一一对应的字段，顺序即声明顺序。
FIELDS = (
    "chunk_id",
    "content",
    "content_meta",
    "ancestor_path",
    "chunk_index",
    "subject_name",
    "game_id",
    "version",
    "doc_title",
    "chunk_type",
    "content_hash",
    "subject_type",
    "content_nature",
    "game_terms",
    "dense_vector",
    "sparse_vector",
)


class FakeMilvusClient:
    """替代 `pymilvus.MilvusClient`：记录调用、返回预置结果，不连服务。"""

    instances: ClassVar[list[FakeMilvusClient]] = []
    databases: ClassVar[list[str]] = ["default"]
    #: `query` 的返回
    rows: ClassVar[list[dict[str, Any]]] = []
    #: `search` / `hybrid_search` 的返回（外层那圈是"每个查询一路"）
    hits: ClassVar[list[dict[str, Any]]] = []
    #: 开连之前就预置成「已经存在的表」。默认空——多数用例要看着表被建出来，
    #: 查询与删除那几条要的是「表已经在」，用 `existing` fixture 预置。
    existing: ClassVar[set[str]] = set()

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.collections: set[str] = set(self.existing)
        #: 每次调用时的连接上下文（真客户端是把 db_name 塞进每条 RPC 的）
        self.context: list[tuple[str, str]] = []
        self.db_name = ""
        FakeMilvusClient.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.databases = ["default"]
        cls.rows = []
        cls.hits = []
        cls.existing = set()

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        self.context.append((name, self.db_name))

    # schema 与索引参数用真实现造：这两个对象本身就是要断言的东西
    @classmethod
    def create_schema(cls, **kwargs: Any) -> Any:
        return RealMilvusClient.create_schema(**kwargs)

    @classmethod
    def prepare_index_params(cls) -> Any:
        return RealMilvusClient.prepare_index_params()

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]

    def list_databases(self, timeout: float | None = None) -> list[str]:
        self._record("list_databases")
        return list(self.databases)

    def create_database(self, db_name: str, **kwargs: Any) -> None:
        self._record("create_database", db_name=db_name)
        self.databases.append(db_name)

    def use_database(self, db_name: str, **kwargs: Any) -> None:
        self._record("use_database", db_name=db_name)
        self.db_name = db_name

    def has_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self._record("has_collection", collection_name=collection_name)
        return collection_name in self.collections

    def create_collection(self, **kwargs: Any) -> None:
        self._record("create_collection", **kwargs)
        self.collections.add(kwargs["collection_name"])

    def upsert(self, **kwargs: Any) -> None:
        self._record("upsert", **kwargs)

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._record("search", **kwargs)
        return [list(self.hits)]

    def hybrid_search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._record("hybrid_search", **kwargs)
        return [list(self.hits)]

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("query", **kwargs)
        return list(self.rows)

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self._record("delete", **kwargs)
        return {"delete_count": len(kwargs.get("ids") or ())}

    def drop_collection(self, collection_name: str, **kwargs: Any) -> None:
        self._record("drop_collection", collection_name=collection_name)
        self.collections.discard(collection_name)


@pytest.fixture
def milvus(monkeypatch: pytest.MonkeyPatch) -> type[FakeMilvusClient]:
    """把 `chunks` 模块里的 `MilvusClient` 换成假件。"""
    FakeMilvusClient.reset()
    monkeypatch.setattr(chunks, "MilvusClient", FakeMilvusClient)
    return FakeMilvusClient


@pytest.fixture
def existing(milvus: type[FakeMilvusClient]) -> type[FakeMilvusClient]:
    """预置「black_myth 这张表已经在」。

    `fetch_document` 会先问一句有没有这张表（没有就如实返回空，见适配器里的说明），
    所以查询与删除那几条用例得先让假件知道表在。
    """
    milvus.existing.add("black_myth")
    return milvus


@pytest.fixture
def store(milvus: type[FakeMilvusClient], settings_env) -> MilvusChunkStore:
    return MilvusChunkStore(load_settings(env_file=None).milvus, timeout=2.5)


def _client(milvus: type[FakeMilvusClient]) -> FakeMilvusClient:
    assert milvus.instances, "还没有连过"
    return milvus.instances[0]


def test_建表显式声明全部字段并关闭动态字段():
    """坑 #1：开了动态字段，Milvus v3.0.0 重启后段加载失败。"""
    schema = chunk_schema()

    assert schema.enable_dynamic_field is False
    assert {field.name for field in schema.fields} == set(FIELDS)


def test_两个标签字段是数组():
    """主体类型不互斥（"二郎神的技能"同时属于角色与技能），过滤用包含判断。"""
    fields = {field.name: field for field in chunk_schema().fields}

    for name in ("subject_type", "content_nature", "game_terms"):
        assert fields[name].dtype == DataType.ARRAY
        assert fields[name].element_type == DataType.VARCHAR
    # 七类主体类型 + 一些余量
    assert fields["subject_type"].max_capacity >= 7


def test_稠密向量维度是_BGE_M3_的_1024():
    fields = {field.name: field for field in chunk_schema().fields}

    assert fields["dense_vector"].dtype == DataType.FLOAT_VECTOR
    assert fields["dense_vector"].params["dim"] == DENSE_DIM
    assert DENSE_DIM == 1024
    assert fields["sparse_vector"].dtype == DataType.SPARSE_FLOAT_VECTOR


def test_稀疏索引显式指定_DAAT_MAXSCORE_与_IP_度量():
    """坑 #2 / #3：不指定，Milvus 3.0 对 IP 走 SINDI，默认配置下稀疏检索慢。"""
    indexes = {param.field_name: param.to_dict() for param in chunk_index_params()}

    assert indexes["dense_vector"]["metric_type"] == "IP"
    assert indexes["sparse_vector"]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert indexes["sparse_vector"]["metric_type"] == "IP"
    assert indexes["sparse_vector"]["inverted_index_algo"] == "DAAT_MAXSCORE"


def test_倒排索引建在回查兄弟切片的两个字段上():
    """聚合父块靠 doc_title 与 version 回查同文档的切片。"""
    indexes = {param.field_name: param.to_dict() for param in chunk_index_params()}

    assert indexes["doc_title"]["index_type"] == "INVERTED"
    assert indexes["version"]["index_type"] == "INVERTED"


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        (None, ""),
        (ChunkFilter(), ""),
        (ChunkFilter(version="1.2"), 'version in ["1.2", ""]'),
        (ChunkFilter(doc_title="二郎神"), 'doc_title == "二郎神"'),
        (ChunkFilter(subject_name="二郎神"), 'subject_name == "二郎神"'),
        (ChunkFilter(chunk_type="table"), 'chunk_type == "table"'),
        (
            ChunkFilter(subject_types=("character", "skill")),
            'ARRAY_CONTAINS_ANY(subject_type, ["character", "skill"])',
        ),
        (
            ChunkFilter(version="1.2", doc_title="二郎神", content_natures=("guide",)),
            'version in ["1.2", ""] and doc_title == "二郎神"'
            ' and ARRAY_CONTAINS_ANY(content_nature, ["guide"])',
        ),
    ],
)
def test_过滤条件翻成表达式(where: ChunkFilter | None, expected: str):
    assert filter_expression(where) == expected


def test_数组过滤用_ANY_而不是_CONTAINS():
    """`ARRAY_CONTAINS(field, [列表])` 在 Milvus 3.0 上解析失败：它只收单个字面量。

    服务端原话：`cannot cast value to VarChar, value: array_val:{...}`。
    语义上这里要的也正是"任一命中"——主体类型不互斥。
    """
    expression = filter_expression(ChunkFilter(subject_types=("character", "skill")))

    assert expression == 'ARRAY_CONTAINS_ANY(subject_type, ["character", "skill"])'
    assert "ARRAY_CONTAINS(" not in expression


def test_版本过滤带上未标注版本():
    """漏掉空串这一支，用户切到历史版本后世界观类问题会全部答不出。"""
    assert filter_expression(ChunkFilter(version="1.2")) == 'version in ["1.2", ""]'


def test_表达式里的取值被转义():
    """原项目把主体名插值进表达式，写库侧转义了而查询侧没有，构成注入。"""
    assert (
        filter_expression(ChunkFilter(doc_title='二郎神" or chunk_id > 0 or "'))
        == 'doc_title == "二郎神\\" or chunk_id > 0 or \\""'
    )
    # 反斜杠要先转，否则服务端会把 `\"` 拆成「转义的反斜杠 + 结束引号」
    assert filter_expression(ChunkFilter(subject_name="C:\\神")) == 'subject_name == "C:\\\\神"'


@pytest.mark.parametrize("game_id", ["black_myth", "BlackMyth2", "_x"])
def test_合法的游戏_id_直接当_collection_名(game_id: str):
    assert collection_name(game_id) == game_id


@pytest.mark.parametrize("game_id", ["黑神话·悟空", "black-myth", "2black", "", "x" * 256])
def test_不合法的游戏_id_当场报错(game_id: str):
    """不做字符替换式规整：`a-b` 与 `a_b` 会映到同一个 collection，两款游戏就混一起了。"""
    with pytest.raises(ValueError, match="不合法"):
        collection_name(game_id)


def test_连接时带上配置里的地址与鉴权_token(store, milvus):
    """云端向量库必须带 token：不带就是匿名访问。"""
    store.check()

    assert _client(milvus).init_kwargs["uri"] == "http://milvus.test:19530"
    assert _client(milvus).init_kwargs["token"] == "test-milvus-token"
    assert _client(milvus).init_kwargs["timeout"] == 2.5


def test_自检时建出配置指定的_database(store, milvus):
    """与原项目共用实例，靠 database 隔离——撞进 default 就会捞到对方的数据。"""
    store.check()

    assert _client(milvus).called("create_database") == [{"db_name": "ragamer-test"}]
    assert _client(milvus).called("use_database") == [{"db_name": "ragamer-test"}]


def test_database_已存在时不重复创建(store, milvus):
    milvus.databases.append("ragamer-test")

    store.check()

    assert _client(milvus).called("create_database") == []
    # 仍然要切过去：不切的话后面的请求会打到 default
    assert _client(milvus).called("use_database") == [{"db_name": "ragamer-test"}]


#: 会读写数据的调用。建库/切库那几步本来就在切之前，不在此列。
DATA_CALLS = {"create_collection", "has_collection", "upsert", "search", "hybrid_search", "query"}


def test_每条数据路径都落在配置的_database_上(store, existing):
    """切库只做在 ensure_collection 里的话，没先建表的那条路径会静默落进 default。

    这是静默失效：数据写进了与原项目共用的库，本地看着一切正常。
    """
    store.upsert("black_myth", [make_chunk(1)])
    store.search("black_myth", dense=fake_vector(1))
    store.fetch_document("black_myth", "二郎神", version="1.0")
    store.ensure_collection("black_myth")
    store.drop("black_myth")

    contexts = [db for call, db in _client(existing).context if call in DATA_CALLS]

    assert contexts, "一条数据调用都没记到，测试本身失效了"
    assert set(contexts) == {"ragamer-test"}


def test_连过之后不再重复切库(store, milvus):
    store.check()
    _client(milvus).calls.clear()

    store.upsert("black_myth", [make_chunk(1)])

    assert _client(milvus).called("use_database") == []


def test_建_collection_时带上完整_schema_与索引参数(store, milvus):
    store.ensure_collection("black_myth")

    created = _client(milvus).called("create_collection")[0]
    assert created["collection_name"] == "black_myth"
    assert created["schema"].enable_dynamic_field is False
    assert {field.name for field in created["schema"].fields} == set(FIELDS)
    assert {param.field_name for param in created["index_params"]} >= {
        "dense_vector",
        "sparse_vector",
    }


def test_建_collection_时用_Strong_一致性(store, milvus):
    """Bounded（pymilvus 的默认）下刚写入的数据还在增长段，混合检索会直接报错。"""
    store.ensure_collection("black_myth")

    assert _client(milvus).called("create_collection")[0]["consistency_level"] == "Strong"


def test_重复确保同一个_collection_时只建一次(store, milvus):
    store.ensure_collection("black_myth")
    _client(milvus).calls.clear()

    store.ensure_collection("black_myth")

    assert _client(milvus).called("create_collection") == []
    assert _client(milvus).called("create_database") == []


def test_入库的行带上全部字段与两个向量(store, milvus):
    store.upsert("black_myth", [make_chunk(1)])

    rows = _client(milvus).called("upsert")[0]["data"]
    assert _client(milvus).called("upsert")[0]["collection_name"] == "black_myth"
    assert set(rows[0]) == set(FIELDS)
    assert rows[0]["dense_vector"] == list(fake_vector(1))
    assert rows[0]["sparse_vector"] == {1: 1.0}
    assert rows[0]["subject_type"] == []


def test_空的一批切片不落库也不连服务(store, milvus):
    store.upsert("black_myth", [])

    assert milvus.instances == []


def test_混合检索把两路向量与过滤条件一起发出去(store, milvus):
    store.search(
        "black_myth",
        dense=fake_vector(1),
        sparse={1: 0.5},
        where=ChunkFilter(version="1.0"),
        limit=5,
    )

    call = _client(milvus).called("hybrid_search")[0]
    dense_request, sparse_request = call["reqs"]
    assert dense_request.anns_field == "dense_vector"
    assert sparse_request.anns_field == "sparse_vector"
    assert dense_request.expr == 'version in ["1.0", ""]'
    # 两路权重是原项目标定过的值，且要开归一化后再融合
    assert call["ranker"].dict() == {
        "strategy": "weighted",
        "params": {"weights": [DENSE_WEIGHT, SPARSE_WEIGHT], "norm_score": True},
    }
    assert call["limit"] == 5
    # 不取向量：聚合父块只用得到正文与元数据
    assert "dense_vector" not in call["output_fields"]


def test_不给稀疏向量时退化成单路检索(store, milvus):
    """元数据过滤路这类场景只需要稠密一路。"""
    store.search("black_myth", dense=fake_vector(1), where=ChunkFilter(subject_name="二郎神"))

    assert _client(milvus).called("hybrid_search") == []
    call = _client(milvus).called("search")[0]
    assert call["anns_field"] == "dense_vector"
    assert call["filter"] == 'subject_name == "二郎神"'


def test_检索结果还原成切片与分数(store, milvus):
    milvus.hits = [
        {
            "chunk_id": 7,
            "distance": 0.75,
            "entity": {
                **_row_of(make_chunk(7)),
                "dense_vector": None,
            },
        }
    ]

    hits = store.search("black_myth", dense=fake_vector(7))

    assert [hit.chunk.chunk_id for hit in hits] == [7]
    assert hits[0].score == pytest.approx(0.75)
    assert hits[0].chunk.content == "正文7"


def test_主键挂在顶层时也认得出来(store, milvus):
    """搜索结果的行是 `{"id": …, "distance": …, "entity": {...}}`。"""
    milvus.hits = [{"id": 7, "distance": 0.5, "entity": {**_row_of(make_chunk(7))}}]
    milvus.hits[0]["entity"].pop("chunk_id")

    hits = store.search("black_myth", dense=fake_vector(7))

    assert hits[0].chunk.chunk_id == 7


def test_行里没有主键时明确报错(store, milvus):
    """静默造一条 id 不明的切片，要到拼聚合父块时才发现就晚了。"""
    milvus.hits = [{"distance": 0.5, "entity": {}}]

    with pytest.raises(StoreError, match="没有主键"):
        store.search("black_myth", dense=fake_vector(7))


def test_空数组回_None_也当空处理(store, existing):
    row = _row_of(make_chunk(1))
    row["subject_type"] = None
    existing.rows = [row]

    assert store.fetch_document("black_myth", "二郎神", version="1.0")[0].subject_type == ()


def test_取一份文档的切片按顺序返回(store, existing):
    existing.rows = [_row_of(make_chunk(index)) for index in (2, 0, 1)]

    chunks_ = store.fetch_document("black_myth", "二郎神", version="1.0")

    assert [chunk.chunk_index for chunk in chunks_] == [0, 1, 2]
    call = _client(existing).called("query")[0]
    assert call["filter"] == 'version in ["1.0", ""] and doc_title == "二郎神"'


def test_表还没建起来时取文档返回空_而不是把供应商的异常漏出去(store, milvus):
    """切分预览页可以直接翻一个空库：没有表就是没有切片，如实说，不报错。"""
    chunks_ = store.fetch_document("black_myth", "二郎神", version="1.0")

    assert chunks_ == []
    assert _client(milvus).called("query") == []


def test_按文档删只删这个版本的切片(store, existing):
    """重导 1.0 版不该连带删掉未标注版本——那是「新版本与旧版本并存」要留的（ADR-0004）。"""
    existing.rows = [
        _row_of(make_chunk(1, chunk_index=0, version="1.0")),
        _row_of(make_chunk(2, chunk_index=1, version="1.0")),
        _row_of(make_chunk(3, chunk_index=0, version=UNVERSIONED)),
        _row_of(make_chunk(4, chunk_index=0, version="2.0")),
    ]

    store.delete_document("black_myth", "二郎神", version="1.0")

    assert _client(existing).called("delete") == [
        {"collection_name": "black_myth", "ids": [1, 2], "timeout": 2.5}
    ]


def test_按文档删时查回来的一批不带版本过滤之外的口径(store, existing):
    """查询那一趟仍走既有的表达式生成，删除不另开一个转义点。"""
    existing.rows = []

    store.delete_document("black_myth", "二郎神", version="1.0")

    call = _client(existing).called("query")[0]
    assert call["filter"] == 'version in ["1.0", ""] and doc_title == "二郎神"'
    assert _client(existing).called("delete") == []


def test_删库是幂等的(store, milvus):
    store.drop("black_myth")
    assert _client(milvus).called("drop_collection") == []

    _client(milvus).collections.add("black_myth")
    store.drop("black_myth")

    assert _client(milvus).called("drop_collection") == [{"collection_name": "black_myth"}]


def test_数出这个游戏有多少切片(store, existing):
    """删库前的确认页要报出「将清掉多少条」，`count(*)` 是 Milvus 的算法。"""
    existing.rows = [{"count(*)": 128}]

    assert store.count("black_myth") == 128

    call = _client(existing).called("query")[0]
    assert call["collection_name"] == "black_myth"
    assert call["output_fields"] == ["count(*)"]
    assert call["filter"] == ""


def test_数切片时表还没建起来算零条(store, milvus):
    """空库也数得出「0 条」——为它报一个供应商的异常，删库那一步就白断了。"""
    assert store.count("black_myth") == 0
    assert _client(milvus).called("query") == []


def test_远端不可达时在超时内失败并点名服务与地址():
    """验收标准里的快速失败：不能挂在启动上，且要说得清是哪个服务、哪个地址。"""
    settings = MilvusSettings(uri="http://127.0.0.1:1", token="tok", db="ragamer_test")
    store = MilvusChunkStore(settings, timeout=1.0)

    start = time.monotonic()
    with pytest.raises(StoreUnavailableError) as excinfo:
        store.check()
    elapsed = time.monotonic() - start

    assert elapsed < 5, f"没有在设定的超时内失败：{elapsed:.1f} 秒"
    assert "Milvus" in str(excinfo.value)
    assert "127.0.0.1:1" in str(excinfo.value)


def test_不可达时的报错里没有凭据():
    settings = MilvusSettings(
        uri="http://root:MILVUS-PW@127.0.0.1:1", token="tok", db="ragamer_test"
    )
    store = MilvusChunkStore(settings, timeout=1.0)

    with pytest.raises(StoreUnavailableError) as excinfo:
        store.check()

    assert "MILVUS-PW" not in str(excinfo.value)
    assert "127.0.0.1:1" in str(excinfo.value)


def _row_of(chunk) -> dict[str, Any]:
    """Milvus 的一行：字段平铺，向量按不取处理。

    直接用适配器自己的行构造减掉向量，字段列表就不会与实现各写一份、慢慢漂开。
    """
    return {key: value for key, value in chunks._row(chunk).items() if not key.endswith("_vector")}
