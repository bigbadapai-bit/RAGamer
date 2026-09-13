"""切片存储的 Milvus 实现。

三处「必须继承的坑」落在这里，且都做成了纯函数，测试直接断言、不需要连云端：

- 建表显式声明全部字段并关闭动态字段（坑 #1：开了动态字段，Milvus v3.0.0 的
  JSON Shredding 有路径重复拼接缺陷，重启后段加载失败）
- 稠密向量用 IP 度量（坑 #2：向量化时归一化过，IP 等价余弦但省掉余弦的计算开销）
- 稀疏向量显式指定 `DAAT_MAXSCORE`（坑 #3：不指定时 Milvus 3.0 对 IP 走 SINDI，
  那个算法要服务端开会话开关才生效，不开就是默认配置下的慢）

过滤条件只从 :class:`~ragamer.stores.base.ChunkFilter` 的结构化字段生成，
取值的转义在 :func:`filter_expression` 一处完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    MilvusClient,
    WeightedRanker,
)
from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client import IndexParams

from ragamer.config import MilvusSettings
from ragamer.logging import get_logger
from ragamer.redaction import redact_address
from ragamer.stores.base import (
    MILVUS,
    UNVERSIONED,
    Chunk,
    ChunkFilter,
    ChunkHit,
    StoreError,
    collection_name,
    require_vectors,
    unavailable,
)

# 稠密向量的维度由产出向量的那一侧定义：schema 里的 FLOAT_VECTOR 必须与向量化模型
# 对得上，写死在两处迟早会漂。这里取来用，本模块对外仍叫 DENSE_DIM。
from ragamer.vectors.base import DENSE_DIM

logger = get_logger(__name__)

#: 稠密与稀疏两路在混合检索里的权重。原项目标定过的值，本项目还没有评测集，
#: 先照搬；调参要等评测（见 docs/ARCHITECTURE.md §11）。
DENSE_WEIGHT = 0.8
SPARSE_WEIGHT = 0.2

#: 归一化过的稠密向量配 IP 度量（坑 #2）。
DENSE_METRIC = "IP"
SPARSE_METRIC = "IP"

#: 稀疏索引构建时丢弃最小的这一比例取值。
SPARSE_DROP_RATIO_BUILD = 0.2
SPARSE_INDEX_ALGO = "DAAT_MAXSCORE"

#: 建索引的标量字段。聚合父块要靠它们回查同文档的兄弟切片。
INDEXED_SCALARS = ("doc_title", "version")

#: 一致性级别。不能用默认的 Bounded：写入的数据先落在增长段上，此时做混合检索
#: 服务端会直接报 `service internal error: unsupported ID type`（Milvus 3.0 实测），
#: 而"导入完立刻提问"正是本项目的主流程。Strong 让检索等到最新时间戳，
#: 本项目这个数据量下代价可接受。
CONSISTENCY_LEVEL = "Strong"

#: 各 VARCHAR 字段的长度上限。Milvus 要求显式给，超长会在写入时报错——
#: 报错好过静默截断，但上限要留得够宽，别把正常内容卡在门外。
_MAX_LENGTH = {
    "content": 65535,
    "content_meta": 65535,
    "ancestor_path": 1024,
    "subject_name": 512,
    "game_id": 128,
    "version": 64,
    "doc_title": 512,
    "source_url": 2048,
    "chunk_type": 16,
    "content_hash": 64,
}
_ARRAY_MAX_CAPACITY = {"subject_type": 8, "content_nature": 8, "game_terms": 64}
_ARRAY_ELEMENT_LENGTH = {"subject_type": 32, "content_nature": 32, "game_terms": 64}

#: 检索结果里要取回的标量字段。不取向量的原因很直接：聚合父块只用得到正文与元数据，
#: 取回来是白搬一遍数据。
SCALAR_FIELDS = (
    "content",
    "content_meta",
    "ancestor_path",
    "chunk_index",
    "subject_name",
    "subject_type",
    "content_nature",
    "game_terms",
    "game_id",
    "version",
    "doc_title",
    "source_url",
    "chunk_type",
    "content_hash",
)


def chunk_schema() -> CollectionSchema:
    """`chunks` collection 的 schema：全部字段显式声明，动态字段关闭（坑 #1）。

    `auto_id=False` —— 主键由导入侧分配，不由服务端生成。重导一份文档要能覆盖同一批
    `chunk_id`（全量幂等重导），服务端自增的 id 每次都不一样，覆盖就无从谈起。
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    for name in ("content", "content_meta", "ancestor_path"):
        schema.add_field(name, DataType.VARCHAR, max_length=_MAX_LENGTH[name])
    schema.add_field("chunk_index", DataType.INT64)
    for name in (
        "subject_name",
        "game_id",
        "version",
        "doc_title",
        "source_url",
        "chunk_type",
        "content_hash",
    ):
        schema.add_field(name, DataType.VARCHAR, max_length=_MAX_LENGTH[name])
    # 两个标签字段都是数组：主体类型不互斥（"二郎神的技能"同时属于角色与技能）
    for name in ("subject_type", "content_nature", "game_terms"):
        schema.add_field(
            name,
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=_ARRAY_MAX_CAPACITY[name],
            max_length=_ARRAY_ELEMENT_LENGTH[name],
        )
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    return schema


def chunk_index_params() -> IndexParams:
    """两个向量字段与两个标量字段的索引。"""
    params = MilvusClient.prepare_index_params()
    # 具体索引类型交给服务端挑：本项目还没有评测集，先不把猜测固化进建表参数
    params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type=DENSE_METRIC)
    params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=SPARSE_METRIC,
        params={
            "inverted_index_algo": SPARSE_INDEX_ALGO,
            "drop_ratio_build": SPARSE_DROP_RATIO_BUILD,
        },
    )
    for name in INDEXED_SCALARS:
        params.add_index(field_name=name, index_type="INVERTED")
    return params


def filter_expression(where: ChunkFilter | None) -> str:
    """结构化过滤条件 → Milvus 表达式。

    🔴 取值一律经 :func:`_quote` 转义，且接口上没有别的地方能塞进字符串表达式。
    """
    if where is None:
        return ""
    clauses: list[str] = []
    if where.version is not None:
        # 「未标注版本」必须一并纳入，否则切到历史版本后世界观类问题会全部答不出
        clauses.append(f"version in [{_quote(where.version)}, {_quote(UNVERSIONED)}]")
    for name, value in (
        ("doc_title", where.doc_title),
        ("subject_name", where.subject_name),
        ("chunk_type", where.chunk_type),
    ):
        if value is not None:
            clauses.append(f"{name} == {_quote(value)}")
    for name, values in (
        ("subject_type", where.subject_types),
        ("content_nature", where.content_natures),
        ("game_terms", where.game_terms),
    ):
        if values:
            # ANY 而不是 ARRAY_CONTAINS：后者只收单个字面量，给列表会在服务端解析失败
            # （"cannot cast value to VarChar, value: array_val"）。语义上这里要的正是
            # 「任一命中」——主体类型不互斥，"二郎神的技能"同时属于角色与技能。
            literals = ", ".join(_quote(value) for value in values)
            clauses.append(f"ARRAY_CONTAINS_ANY({name}, [{literals}])")
    return " and ".join(clauses)


def _quote(value: str) -> str:
    """表达式里的字符串字面量。

    反斜杠必须先转：否则 `\\"` 会被服务端拆成「转义的反斜杠 + 结束引号」，
    后面剩下的内容就成了表达式的一部分。
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _row(chunk: Chunk) -> dict[str, Any]:
    """切片 → 一行。"""
    require_vectors(chunk)
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "content_meta": chunk.content_meta,
        "ancestor_path": chunk.ancestor_path,
        "chunk_index": chunk.chunk_index,
        "subject_name": chunk.subject_name,
        "subject_type": list(chunk.subject_type),
        "content_nature": list(chunk.content_nature),
        "game_terms": list(chunk.game_terms),
        "game_id": chunk.game_id,
        "version": chunk.version,
        "doc_title": chunk.doc_title,
        "source_url": chunk.source_url,
        "chunk_type": chunk.chunk_type,
        "content_hash": chunk.content_hash,
        "dense_vector": list(chunk.dense_vector),
        "sparse_vector": dict(chunk.sparse_vector),
    }


def _chunk(fields: Mapping[str, Any]) -> Chunk:
    """一行 → 切片。向量按不取处理（见 `SCALAR_FIELDS`）。

    主键两种形态都认：`query` 的行走服务端拼装，字段名就是 `chunk_id`；
    `search` 那边主键挂在顶层。取不到就报错——静默造出一条 id 不明的切片，
    后面拼聚合父块时才发现，就晚了。
    """
    chunk_id = fields.get("chunk_id", fields.get("id"))
    if chunk_id is None:
        raise StoreError(f"Milvus 返回的行里没有主键，拿不到切片序号：{sorted(fields)}")
    return Chunk(
        chunk_id=int(chunk_id),
        content=fields["content"],
        content_meta=fields["content_meta"],
        ancestor_path=fields["ancestor_path"],
        chunk_index=int(fields["chunk_index"]),
        subject_name=fields["subject_name"],
        # 空数组有的版本回 None，有的回 []
        subject_type=tuple(fields.get("subject_type") or ()),
        content_nature=tuple(fields.get("content_nature") or ()),
        game_terms=tuple(fields.get("game_terms") or ()),
        game_id=fields["game_id"],
        version=fields["version"],
        doc_title=fields["doc_title"],
        source_url=fields["source_url"],
        chunk_type=fields["chunk_type"],
        content_hash=fields["content_hash"],
    )


def _hit(row: Mapping[str, Any]) -> ChunkHit:
    """搜索结果 → 命中。

    行是 `{"<主键名>": …, "distance": …, "entity": {字段: 取值}}`，标量字段在 `entity` 里。
    """
    fields = {**(row.get("entity") or {}), "chunk_id": row.get("chunk_id", row.get("id"))}
    return ChunkHit(chunk=_chunk(fields), score=float(row.get("distance", 0.0)))


class MilvusChunkStore:
    """Milvus 上的切片存储。

    一个游戏一个 collection，全部落在配置指定的 database 里——与原项目共用实例但库不同，
    两边可以同时跑。构造不碰网络，第一次用到才连：连不上是运行期的事，
    应该在自检里报出来，而不是在导入模块时把进程带走。
    """

    def __init__(self, settings: MilvusSettings, *, timeout: float) -> None:
        self.name = MILVUS
        self.address = redact_address(settings.uri)
        self._uri = settings.uri
        self._token = settings.token.get_secret_value()
        self._db = settings.db
        self._timeout = timeout
        self._milvus: MilvusClient | None = None

    def check(self) -> None:
        """连通性自检：地址通不通、token 认不认，顺带确保本项目的 database 存在。

        库的确保放在这里而不是等到第一次入库：命名空间必须在任何一条数据路径之前定下来。
        否则哪条路径忘了切库，数据就静默落进 `default`——那是与原项目共用的库。
        """
        try:
            self._client().list_databases(timeout=self._timeout)
        except (MilvusException, OSError, ValueError) as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def ensure_collection(self, game_id: str) -> None:
        """确保该游戏的 collection 存在，索引也一并建好。"""
        client = self._client()
        name = collection_name(game_id)
        if not client.has_collection(name, timeout=self._timeout):
            client.create_collection(
                collection_name=name,
                schema=chunk_schema(),
                index_params=chunk_index_params(),
                consistency_level=CONSISTENCY_LEVEL,
                timeout=self._timeout,
            )
            logger.info("新建 Milvus collection %s", name)

    def upsert(self, game_id: str, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        self._client().upsert(
            collection_name=collection_name(game_id),
            data=[_row(chunk) for chunk in chunks],
            timeout=self._timeout,
        )

    def search(
        self,
        game_id: str,
        *,
        dense: Sequence[float],
        sparse: Mapping[int, float] | None = None,
        where: ChunkFilter | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        client = self._client()
        name = collection_name(game_id)
        expression = filter_expression(where)
        dense_request = AnnSearchRequest(
            data=[list(dense)],
            anns_field="dense_vector",
            param={"metric_type": DENSE_METRIC},
            limit=limit,
            expr=expression or None,
        )
        if sparse is None:
            rows = client.search(
                collection_name=name,
                data=[list(dense)],
                filter=expression,
                limit=limit,
                output_fields=list(SCALAR_FIELDS),
                anns_field="dense_vector",
                search_params={"metric_type": DENSE_METRIC},
                timeout=self._timeout,
            )[0]
            return [_hit(row) for row in rows]
        rows = client.hybrid_search(
            collection_name=name,
            reqs=[
                dense_request,
                AnnSearchRequest(
                    data=[dict(sparse)],
                    anns_field="sparse_vector",
                    param={"metric_type": SPARSE_METRIC},
                    limit=limit,
                    expr=expression or None,
                ),
            ],
            ranker=WeightedRanker(DENSE_WEIGHT, SPARSE_WEIGHT),
            limit=limit,
            output_fields=list(SCALAR_FIELDS),
            timeout=self._timeout,
        )[0]
        return [_hit(row) for row in rows]

    def fetch_document(self, game_id: str, doc_title: str, *, version: str) -> list[Chunk]:
        client = self._client()
        name = collection_name(game_id)
        if not client.has_collection(name, timeout=self._timeout):
            # 这个库还什么都没导进来过。没有表就是没有切片，如实返回空——直接查会把
            # 供应商的异常漏给调用方，而内存假件在这条路径上返回的是空列表。
            # 切分预览页可以直接翻一个空库，靠的就是这一条。
            return []
        rows = client.query(
            collection_name=name,
            filter=filter_expression(ChunkFilter(doc_title=doc_title, version=version)),
            output_fields=list(SCALAR_FIELDS),
            timeout=self._timeout,
        )
        return sorted((_chunk(row) for row in rows), key=lambda chunk: chunk.chunk_index)

    def delete_document(self, game_id: str, doc_title: str, *, version: str) -> None:
        """先按文档查回主键，再只删版本精确对上的那些（见协议里的说明）。

        `fetch_document` 的版本过滤是「该版本或未标注版本」，比删除要宽一档，
        所以查回来的行还要自己再筛一遍。按主键删而不是按表达式删：取值的转义
        因此只有 `filter_expression` 一处，删除这条路不会长出第二个转义点。
        """
        stale = [
            chunk.chunk_id
            for chunk in self.fetch_document(game_id, doc_title, version=version)
            if chunk.version == version
        ]
        if not stale:
            return
        self._client().delete(
            collection_name=collection_name(game_id), ids=stale, timeout=self._timeout
        )
        logger.info("删除 %s 在版本 %r 下的 %d 个旧切片", doc_title, version, len(stale))

    def count(self, game_id: str) -> int:
        client = self._client()
        name = collection_name(game_id)
        if not client.has_collection(name, timeout=self._timeout):
            return 0  # 没有表就是没有切片，与 `fetch_document` 同一条口径
        rows = client.query(
            collection_name=name,
            # `count(*)` 要求过滤表达式为空：带上条件就成了「符合条件的行数」，
            # 而这里问的是整张表有多少行
            filter="",
            output_fields=["count(*)"],
            timeout=self._timeout,
        )
        return int(rows[0]["count(*)"])

    def drop(self, game_id: str) -> None:
        client = self._client()
        name = collection_name(game_id)
        if client.has_collection(name, timeout=self._timeout):
            client.drop_collection(name, timeout=self._timeout)
            logger.info("删除 Milvus collection %s", name)

    def _client(self) -> MilvusClient:
        """拿到一个绑定在本项目 database 上的客户端。

        连线与切库都只做一次，且**任何一条数据路径都从这里走**——切库只发生在
        `ensure_collection` 里的话，没先建表就入库的那条路径会静默落到 `default`。

        `MilvusClient` 的构造函数本身就会连服务端，所以连不上也在这里翻译。
        """
        if self._milvus is None:
            try:
                client = MilvusClient(
                    uri=self._uri,
                    # 云端向量库必须带 token：不带就是匿名访问，服务端开了鉴权就连不上
                    token=self._token,
                    timeout=self._timeout,
                )
                self._bind_database(client)
            except (MilvusException, OSError, ValueError) as exc:
                raise unavailable(self.name, self.address, self._timeout, exc) from exc
            self._milvus = client
        return self._milvus

    def _bind_database(self, client: MilvusClient) -> None:
        """把连接上下文切到本项目的 database，没有就建一个。"""
        if self._db not in client.list_databases(timeout=self._timeout):
            client.create_database(self._db, timeout=self._timeout)
            logger.info("新建 Milvus database %s", self._db)
        # 只切一次连接上下文，之后每条 RPC 都带着它
        client.use_database(self._db)
