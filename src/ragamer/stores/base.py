"""存储适配器的共同约定：协议、共享类型与错误。

业务层只认这里的协议。真实客户端（`chunks` / `documents` / `objects`）与内存假件
（`memory`）实现的是同一组协议，所以在组合根把三者换掉，整条链路照跑——
后面两条测试缝能立起来，靠的就是这一层。

构造一律在组合根（`ragamer.container`）发生一次：模块级单例会让测试换不掉实现。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

#: 三个服务在错误信息里的名字。
MILVUS = "Milvus"
MONGO = "MongoDB"
MINIO = "MinIO"

#: 未标注版本的切片的 `version` 取值（空串）。检索任何版本时都要一并纳入它。
UNVERSIONED = ""

ChunkType = Literal["text", "table", "image"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """一个切片的落库形态。

    字段与 `chunks` collection 的 schema 一一对应（见 docs/ARCHITECTURE.md §2.2）。
    两个向量在切分与打标阶段还是空的，由向量化环节填上——入库时缺向量直接报错，
    不静默写一条检索不到的切片。
    """

    chunk_id: int
    content: str
    ancestor_path: str
    chunk_index: int
    subject_name: str
    game_id: str
    version: str
    doc_title: str
    chunk_type: ChunkType = "text"
    content_meta: str = ""
    subject_type: tuple[str, ...] = ()
    content_nature: tuple[str, ...] = ()
    game_terms: tuple[str, ...] = ()
    content_hash: str = ""
    dense_vector: tuple[float, ...] | None = None
    sparse_vector: Mapping[int, float] | None = None


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """一次检索命中的切片与它的分数。"""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class ChunkFilter:
    """结构化过滤条件。

    🔴 接口上不接受字符串形式的过滤表达式。原项目把主体名直接插值进 Milvus 表达式，
    写库侧转义了而查询侧没有，构成注入。表达式只由适配器从这些字段生成，
    取值的转义在 `ragamer.stores.chunks` 里一处完成——这类缺陷在结构上就无处可入。

    `version` 传值时的语义是「该版本 **或** 未标注版本」：漏掉后半句，
    用户切到历史版本后世界观类问题会全部答不出，而且是静默失效。
    """

    version: str | None = None
    doc_title: str | None = None
    subject_name: str | None = None
    chunk_type: ChunkType | None = None
    #: 数组字段：任一命中即可（主体类型不互斥，"二郎神的技能"同时属于角色与技能）
    subject_types: tuple[str, ...] = ()
    content_natures: tuple[str, ...] = ()
    game_terms: tuple[str, ...] = ()


def matches(chunk: Chunk, where: ChunkFilter | None) -> bool:
    """结构化过滤的语义。

    内存假件照它实现；Milvus 适配器把同一组语义翻成表达式。两处的判断必须一致，
    否则缝里跑过的行为与云端跑的行为对不上。
    """
    if where is None:
        return True
    if where.version is not None and chunk.version not in (where.version, UNVERSIONED):
        return False
    for wanted, actual in (
        (where.doc_title, chunk.doc_title),
        (where.subject_name, chunk.subject_name),
        (where.chunk_type, chunk.chunk_type),
    ):
        if wanted is not None and actual != wanted:
            return False
    return all(
        not wanted or bool(set(wanted) & set(actual))
        for wanted, actual in (
            (where.subject_types, chunk.subject_type),
            (where.content_natures, chunk.content_nature),
            (where.game_terms, chunk.game_terms),
        )
    )


def matches_where(document: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """`DocStore.find` 的等值匹配语义。

    内存假件照它实现；Mongo 适配器把同一组语义翻成 filter 交给服务端。两处的判断必须
    一致，否则缝里跑过的行为与云端跑的行为对不上——与 :func:`matches` 同一个理由，
    只是这里只有等值这一条，所以短得多。
    """
    return all(document.get(key) == value for key, value in (where or {}).items())


def require_vectors(chunk: Chunk) -> None:
    """入库前必须已经向量化。

    向量缺失时静默写进去，结果是一条检索永远命不中的切片——查不出、也不报错。
    宁可当场炸。
    """
    if chunk.dense_vector is None or chunk.sparse_vector is None:
        raise ValueError(f"切片 {chunk.chunk_id} 还没有向量，向量化之后才能入库")


#: collection 名的规则：字母或下划线开头，只含字母、数字、下划线，不超过 255 字符。
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_NAME_LENGTH = 255


def collection_name(game_id: str) -> str:
    """游戏 id → collection 名。一个游戏一个 collection（ADR-0002）。

    不做「把非法字符换成下划线」式的规整：那会把 `a-b` 与 `a_b` 映到同一个名字，
    两款游戏的切片就混进同一个 collection 了——而互相隔离正是命名空间要保证的事。
    游戏 id 因此限定成合法标识符，不合法在这里当场报出来。真实适配器与内存假件
    共用这一条规则，换掉后端不会换掉命名空间。
    """
    if len(game_id) > MAX_NAME_LENGTH or not _NAME_PATTERN.match(game_id):
        raise ValueError(
            f"游戏 id 不合法：{game_id!r}。"
            f"collection 名只能是字母或下划线开头的字母、数字、下划线，"
            f"且不超过 {MAX_NAME_LENGTH} 字符"
        )
    return game_id


class StoreError(Exception):
    """存储相关的失败。启动自检与业务调用都按这个基类兜。"""


class StoreUnavailableError(StoreError):
    """某个存储连不上：超时、地址不通、凭据被拒。

    信息里点名是哪个服务、哪个地址——地址已抹掉凭据，可以安全落日志。
    """

    def __init__(self, service: str, address: str, timeout: float, reason: str) -> None:
        self.service = service
        self.address = address
        self.timeout = timeout
        self.reason = reason
        super().__init__(f"{service} 不可达（地址 {address}，超时 {timeout:g} 秒）：{reason}")


def unavailable(
    service: str, address: str, timeout: float, exc: Exception
) -> StoreUnavailableError:
    """把一个连接失败翻成 :class:`StoreUnavailableError`。

    三个适配器的失败都从这里出：措辞一致，不会这个说"连接被拒绝"、那个说"不可达"。
    """
    return StoreUnavailableError(service, address, timeout, str(exc) or type(exc).__name__)


class StoreCheckError(StoreError):
    """启动自检有失败项。

    三个服务全查一遍再一起报：启动时一次看清全部问题，而不是修一个重启一次。
    """

    def __init__(self, failures: Sequence[StoreUnavailableError]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "存储连通性自检失败：\n" + "\n".join(f"  - {failure}" for failure in self.failures)
        )


@runtime_checkable
class Store(Protocol):
    """外部存储客户端的共同部分。"""

    #: 服务名与地址，用于出错信息（地址不含凭据）
    name: str
    address: str

    def check(self) -> None:
        """连通性自检，顺带确保这个服务上属于本项目的命名空间存在。

        连不上时抛 :class:`StoreUnavailableError`。命名空间的确保放在自检里，
        是因为它必须在任何一条数据路径之前落定：漏了哪一步，数据就静默落到
        与原项目共用的那个命名空间里。
        """
        ...


@runtime_checkable
class ChunkStore(Store, Protocol):
    """切片存储（Milvus）。接口刻意做小，且只收结构化过滤条件。"""

    def ensure_collection(self, game_id: str) -> None:
        """确保该游戏的 collection 存在。一个游戏一个 collection。"""
        ...

    def upsert(self, game_id: str, chunks: Sequence[Chunk]) -> None:
        """按 `chunk_id` 覆盖写入。重导一份文档就是覆盖同一批 id，不产生重复切片。"""
        ...

    def search(
        self,
        game_id: str,
        *,
        dense: Sequence[float],
        sparse: Mapping[int, float] | None = None,
        where: ChunkFilter | None = None,
        limit: int = 10,
    ) -> list[ChunkHit]:
        """混合检索：稠密 + 稀疏两路在同一批数据上融合（原项目权重 (0.8, 0.2)）。

        只给稠密不给稀疏时退化成单路检索，供元数据过滤路这类场景使用。
        """
        ...

    def versions(self, game_id: str) -> tuple[str, ...]:
        """这个库里**真实存在过**的版本，去重后按字面升序。

        澄清反问的版本候选只能来自这里（§3.4）：版本不是一份独立的数据，它只是切片上的
        一个字段，没有别的地方问得出「这个库有哪些版本」。让模型自己编版本号，用户选了
        也检索不到——而「选了却没东西」与「选了查得到」在界面上长得一样。

        **未标注版本不在其中**（`:data:`UNVERSIONED``）：它在库里的取值是空串，
        与「没判出来」共用同一个字面，摆到按钮上用户点了也说不清自己选了什么；
        它本来也不需要选——检索恒把它一并纳入（ADR-0004）。

        库还不存在（建了库但一份资料都没导）时返回空元组，不报错。
        """
        ...

    def fetch_document(self, game_id: str, doc_title: str, *, version: str | None) -> list[Chunk]:
        """取一份文档的全部切片，按 `chunk_index` 升序。

        聚合父块靠它：命中并截断之后按 `doc_title` 回查兄弟切片，父块因此永远与子块同源。
        `version` 是必给的，但允许 `None`——两个取值对应检索侧的两套口径：

        - 传值：过滤条件恒为「该版本**或**未标注版本」。少了后半句，同一个父块里会拼进
          不同版本的切片，而且不会报错（ADR-0004）。
        - 传 `None`：不过滤。跟随 `ragamer.query.version_filter` 在问题与知识库都给不出
          版本时的收窄——聚合与检索必须是同一套口径，聚合另立一套就等于把版本判错两次。
        """
        ...

    def delete_document(self, game_id: str, doc_title: str, *, version: str) -> None:
        """删掉一份文档在**这个版本**下的全部切片。

        重导一份资料时先清后写（见 `ragamer.importing`）：新一次切出来的片数变少时，
        只靠覆盖写入会留下一截旧切片——查得出来、还会进聚合父块，而且不报错。

        `version` 是**精确匹配**，不含「或未标注版本」那半句：那是检索的规则，不是删除的
        规则。照检索的口径删，重导 1.0 版会连带删掉同一份文档的未标注版本——而那正是
        「新版本与旧版本并存」（ADR-0004）要求留下来的东西。
        """
        ...

    def count(self, game_id: str) -> int:
        """该游戏 collection 里的切片数。**还没建表就是 0**，不报错。

        删库前的确认页用它说清「将清掉多少条」——报一个异常的话，一个空库会把
        整条确认路径断在那里，而「0 条」本来就是一个说得通的答案。
        """
        ...

    def drop(self, game_id: str) -> None:
        """删掉该游戏的 collection。删库要清四处，这是其中一处。"""
        ...


@runtime_checkable
class DocStore(Store, Protocol):
    """文档存储（MongoDB）：会话、术语映射、知识库元数据。

    知识库元数据在这里是「当前该用哪个版本」的唯一真相来源，检索与聚合父块都从这取，
    不各自维护一份。各类文档的形状由使用它的模块定义，这一层只负责存取与命名空间。
    """

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        """按 id 取一份文档；没有则返回 `None`。"""
        ...

    def put(self, collection: str, doc_id: str, document: Mapping[str, Any]) -> None:
        """整体覆盖写入一份文档。"""
        ...

    def delete(self, collection: str, doc_id: str) -> None:
        """删一份文档；不存在也算成功。"""
        ...

    def list_ids(self, collection: str) -> list[str]:
        """列出该集合的全部文档 id，按字典序。"""
        ...

    def find(
        self,
        collection: str,
        where: Mapping[str, Any] | None = None,
        *,
        fields: Sequence[str] = (),
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
        after: tuple[Any, str] | None = None,
    ) -> list[dict[str, Any]]:
        """按字段取一批文档。**只读**。

        有 `get` 还要有它的理由只有一个：**列表**。界面上的会话列表、知识库列表要的是
        「某个游戏下的最近若干条，且不要正文」——先 `list_ids` 再逐条 `get` 会把每份文档
        都读出来，几十条会话就是几十次往返，而其中有用的只有标题和时间那两个字段。

        - `where`：**等值**匹配，`None` 即不过滤，语义见 :func:`matches_where`。刻意只做到
          等值——比较、数组包含那些是 `ChunkFilter` 的事，那边有 `matches()` 兜住两套实现
          的口径；这一层再长出一套过滤语义，两个后端就会有对不上的地方。
        - `fields`：空即整份返回；非空只返回这几项。**列表这类场景必须给**，否则把一堆
          用不上的正文拖回来，正是这个方法要避免的事。
        - `order_by` / `descending`：按哪个字段排、正序还是倒序。不给 `order_by` 时
          顺序由存储自己定（Mongo 不保证），**不要依赖它**。给的字段要在每份文档里都存在：
          **缺这个字段的文档怎么排，两个后端不保证一致**（Mongo 把缺的当 null，内存假件
          当空串），别拿一个可能缺的字段来排。
        - `limit`：条数上限。
        - `after`：**翻页游标**——只要排在「`(order_by` 的值`, 文档 id)` 这一条之后」的。
          与 `order_by` 必须一起给：没有排序键就无从谈「之后」。取值是上一页最后一条的
          那两个字段（`_id` 由返回的每一条带上）。

          🔴 **游标落在两个字段上，不能只落排序键**：排序键在这里是可变的（会话每落一次库
          就刷新 `updated_at`），单靠它在并列值上会漏条或重条。排序同理——给了 `order_by`
          就一定带上 `_id` 作次键，两个后端才算出同一个次序，游标也才接得上。

        **返回的每一条都带 `_id`**。它是文档 id，批量取的时候调用方就是靠它认人的。
        `get` 那边把 `_id` 摘掉是因为 id 本来就是调用方给的，这里正好反过来。

        一条都没命中时返回空列表，不报错——**「一条都没有」是列表的正常状态**，
        与「这个集合不存在」也不作区分，两者对调用方是同一件事。

        :raises ValueError: 给了 `after` 却没给 `order_by`。
        """
        ...

    def ensure_indexes(self, collection: str, fields: Sequence[tuple[str, int]]) -> None:
        """确保这个集合上有这个复合索引。**幂等**：已经有了就什么都不做。

        列表查询与它的翻页靠它。没有索引时 `find` 是「全表扫 + 内存排序」，一页一次；
        翻页把这份成本乘以页数——Mongo 的阻塞排序超了内存还是**直接报错**，不是变慢。

        与 `ObjectStore.ensure_bucket` 同一个打法：存储层保证自己那一侧的形态，
        调用方不必知道那边具体是索引还是桶。**但这一条不走 `check()`**——适配器不知道
        业务层查哪些集合，而自检那条路是刻意只读的。调用点在 `ragamer.app` 的启动那一段。
        """
        ...


@runtime_checkable
class ObjectStore(Store, Protocol):
    """对象存储（MinIO）：原图等二进制内容。"""

    def ensure_bucket(self) -> None:
        """确保桶存在。不存在则创建，已存在不动它。"""
        ...

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        """写入一个对象，按 key 覆盖。"""
        ...

    def get(self, key: str) -> bytes:
        """读一个对象；不存在时抛 :class:`StoreError`。"""
        ...

    def delete(self, key: str) -> None:
        """删一个对象；不存在也算成功。"""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """列出前缀下的全部 key，按字典序。"""
        ...

    def delete_prefix(self, prefix: str) -> int:
        """按前缀批量删，返回删掉的个数。删库时清原图走它。"""
        ...


def normalize_prefix(prefix: str) -> str:
    """统一去掉前导 `/`。

    原项目的 list 去前导 `/` 而 put 不去，于是"清旧图"按前缀删时一个也匹配不上，
    静默失效。列表与按前缀删共用这一个函数，前缀对不上的可能就不存在了。
    """
    return prefix.lstrip("/")


#: 原图在对象存储里的顶层前缀。删库清原图按它下面那一级走。
IMAGE_PREFIX = "images"


def image_prefix(game_id: str, digest: str = "") -> str:
    """一个游戏的原图前缀；给了 `digest` 就再收窄到这一份来源文件。

    对象名分两级：游戏一级、来源文件一级。**写入与清理共用这一个函数**——
    原项目那处坑是 list 与 put 各拼一遍前缀、两处对不上，于是清旧图静默失效；
    这里只要两处都调它，前缀就没有对不上的余地。

    清点与清理按 :func:`image_folder` 走，不必知道当初导过哪些文件。
    """
    return "/".join(part for part in (IMAGE_PREFIX, game_id, digest) if part)


def image_folder(game_id: str) -> str:
    """这个游戏的原图那一层，**带尾随斜杠**。按前缀清点与清理走它。

    尾随的斜杠不是装饰：`delete_prefix` / `list_keys` 比的是**字符串前缀**，不是目录。
    拿 `image_prefix(game_id)`（`images/black_myth`）去删，id 为 `black_myth_2` 的那个库
    的原图会被一并收走——而且不报错，人只会在很久以后发现另一个库的图没了。
    一游戏一 collection 要保证的正是互相隔离（ADR-0002），边界就在这里补上，
    不指望每个调用点都记得自己加。
    """
    return f"{image_prefix(game_id)}/"


def image_key(game_id: str, digest: str, name: str) -> str:
    """一个附件的对象 key。

    `digest` 取自来源文件的字节：同一份文件重导算出的 key 完全一致，图片原地覆盖，
    与切片主键由导入侧分配（`ragamer.importing.chunk_id`）是同一套幂等思路；
    不同文件即使同名也各有各的一层，不会互相覆盖。

    **代价**：同一份资料改了内容再导，算出的 digest 变了，上一版的图片会留在旧的
    那一层——它按游戏一级清理时一并收走（`image_folder(game_id)`）。比按文件名分层强：
    那样两份同名不同内容的截图会互相覆盖，答案是配错图，而且不报错。
    """
    return f"{image_prefix(game_id, digest)}/{name}"
