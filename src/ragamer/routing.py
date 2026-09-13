"""查询路由：按查询类型决定这次走哪几路召回，不做无差别全跑。

**判定本身不新增任何模型调用**：路由标签是联合输出节点（`ragamer.query`）多吐出来的
一个字段，那一步本来就要问模型一次（`docs/ARCHITECTURE.md` §3.1）。

三件事在这里定死：

- **判不出就按事实型那一行走**。模型给了词表以外的标签、那一步整个挂掉，都只影响
  走哪几路，不影响答不答得出来——这一层不阻断作答（与 `ragamer.query` 对理解失败的
  处理同一条口径）。**不另立一份「默认组合」**：另立的那一份与表里的一行迟早会分岔。
- **组合是数据**。默认表照 §3.1 摆，知识库里配了 `route_table` 就整行覆盖。姿势与
  打标词表（`ragamer.tagging.TagVocabulary`）一致：每个库一份的配置，不是进程级开关，
  也不是环境变量。**界面改的是「走哪几路」，不是分类学本身**——六类与六路是代码里的
  词表，配置只能在这两者之间连线。
- **没接上的路跳过，但不静默**。六条路现在接了两条（主检索、元数据过滤），其余四条在
  后面的票里补。选中了没接上的那条要留一条 warning 再跳过：静默跳过会让「还没做」与
  「组合配错了」在日志里长得一模一样。

六类问题与六路召回的对应关系见 :data:`DEFAULT_ROUTES`；配置的读写格式见
:meth:`RouteTable.from_mapping`。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ragamer.tagging import ContentNature, SubjectType


class QueryType(StrEnum):
    """问题属于哪一类。六类，由联合输出节点顺带判定。

    划的是**该走哪几路召回**，不是语义分类本身：事实型与表格型问的都是数值，分成两类
    只因为一个去捞正文、一个去捞表格切片。
    """

    FACTUAL = "factual"
    GUIDE = "guide"
    COMPARISON = "comparison"
    VAGUE = "vague"
    RECENCY = "recency"
    TABULAR = "tabular"


#: 每一类在提示词与日志里的叫法。同时进提示词——模型吐回来的就是这几个词。
QUERY_TYPE_LABELS: Mapping[QueryType, str] = {
    QueryType.FACTUAL: "事实型",
    QueryType.GUIDE: "攻略型",
    QueryType.COMPARISON: "对比型",
    QueryType.VAGUE: "模糊型",
    QueryType.RECENCY: "时效型",
    QueryType.TABULAR: "表格型",
}

#: 每一类长什么样。**只给类型名不够**：模型判「表格型」与「事实型」的界线时会靠猜，
#: 给一个典型问法才判得稳。也进提示词。
QUERY_TYPE_HINTS: Mapping[QueryType, str] = {
    QueryType.FACTUAL: "问某个具体数值，如「二郎神血量多少」",
    QueryType.GUIDE: "问怎么打、怎么过，如「二郎神怎么打」",
    QueryType.COMPARISON: "在两样东西之间比，如「二郎神和大圣哪个强」",
    QueryType.VAGUE: "口语、指代不清，如「那个很难的 BOSS 怎么过」",
    QueryType.RECENCY: "问某个版本改了什么，如「这版本还掉不掉金箍棒」",
    QueryType.TABULAR: "问一项属性数值，如「寒江雪的属性」",
}


class RecallPath(StrEnum):
    """一条**彼此独立**的召回策略。

    与「混合检索」不是一回事：稠密 + 稀疏是同一批数据上的一路内部机制，这里是多路之间
    的关系（§3.2）。名字只列出来源与手段，各自的实现在 `ragamer.retrieval`。
    """

    #: 主混合检索路：稠密 + 稀疏向量在同一批数据上融合
    MAIN = "main"
    #: 元数据过滤路：按主体类型、内容性质、版本直接取候选，单路稠密检索
    METADATA = "metadata"
    #: 多查询改写路：一个问题扩成若干等价问法分别检索
    MULTI_QUERY = "multi_query"
    #: HyDE 路：先生成假想答案，再拿它去检索
    HYDE = "hyde"
    #: 结构化表格路：数值类问题定向取表格切片
    TABLE = "table"
    #: 联网兜底路：时效性问题走外部检索
    WEB = "web"


#: 已经接上的路。其余四条在后面的票里补——选中了没接上的那条会跳过并留痕。
WIRED_PATHS: frozenset[RecallPath] = frozenset({RecallPath.MAIN, RecallPath.METADATA})

#: 默认路由表（§3.1 的那张表）：查询类型 → 走哪几路。
#:
#: 三处与原文的出入，都是有意的：
#:
#: - 「+ HyDE」「+ 联网」那两行是简写，展开成「事实型那一组再加一条」——只有这一个读法
#:   讲得通，原表里它们没有另给基础组合。
#: - 表格型除了表格路还留着主检索。两者不互斥：属性写在正文里而不是表格里是常事，
#:   只留表格路会让这一类问题一条候选都取不到，那正是「不阻断作答」的反面。
#: - 判不出类型时用的也是这张表里事实型那一行，不另立一份默认组合。
DEFAULT_ROUTES: Mapping[QueryType, tuple[RecallPath, ...]] = {
    QueryType.FACTUAL: (RecallPath.MAIN, RecallPath.METADATA),
    QueryType.GUIDE: (RecallPath.MAIN, RecallPath.MULTI_QUERY, RecallPath.METADATA),
    QueryType.COMPARISON: (RecallPath.MULTI_QUERY, RecallPath.MAIN),
    QueryType.VAGUE: (RecallPath.MAIN, RecallPath.METADATA, RecallPath.HYDE),
    QueryType.RECENCY: (RecallPath.MAIN, RecallPath.METADATA, RecallPath.WEB),
    QueryType.TABULAR: (RecallPath.TABLE, RecallPath.MAIN),
}

#: 查询类型 → 元数据过滤路收窄到哪一类内容性质。
#:
#: 内容性质说的是「一个切片回答的是哪一类问题」，与「问题属于哪一类」是同一根轴的两面，
#: 所以这是一张固定推导，**不进可覆盖的那张表**——界面改的是走哪几路。
#:
#: 模糊型与时效型不收窄：这两类问的恰恰是「说不清问的是哪一类」，按性质过滤会把本来
#: 该捞上来的东西筛掉。
NATURES_FOR_TYPE: Mapping[QueryType, tuple[ContentNature, ...]] = {
    QueryType.FACTUAL: (ContentNature.STATS,),
    QueryType.GUIDE: (ContentNature.GUIDE,),
    QueryType.COMPARISON: (ContentNature.REVIEW,),
    QueryType.TABULAR: (ContentNature.STATS,),
    QueryType.VAGUE: (),
    QueryType.RECENCY: (),
}


@dataclass(frozen=True)
class Route:
    """这次走哪几路，以及元数据过滤路该收窄到哪些标签。

    三个字段绑成一个，是为了让「走哪几路」与「过滤条件」出自同一处决策：分开放，
    调用方就有机会把元数据路的过滤条件配上另一条路的组合，而不报错。
    """

    #: 按顺序走的路。顺序不影响结果（候选取回后交给 RRF 融合），只影响日志。
    paths: tuple[RecallPath, ...]
    #: 元数据过滤路按这些内容性质取候选。空 = 不按性质收窄。由查询类型推出来。
    content_natures: tuple[ContentNature, ...] = ()
    #: 元数据过滤路按这些主体类型取候选。空 = 不限。
    #:
    #: 默认表里一律留空：问题问的是哪个主体类型，判它要多一次模型调用或者一次术语扫描，
    #: 而这一票的头一条就是「不新增任何模型调用」。库里配了就用配的那个——这一维因此
    #: 是能用的，只是默认不用。
    subject_types: tuple[SubjectType, ...] = ()


@dataclass(frozen=True)
class RouteTable:
    """查询类型 → 这一类的召回组合。默认值见 :data:`DEFAULT_ROUTES`。

    表里**六类齐全**：缺一类的表在 :meth:`route_for` 上会以 KeyError 炸在检索那一步，
    那时已经离配置的出处很远了。配置读进来的表是补齐过的，构造时就拦下。

    不可变，与 `ragamer.tagging.TagVocabulary` 同一个理由：一份配置被一次接线反复使用，
    可变对象上改一处会静默影响所有用到它的地方。
    """

    routes: Mapping[QueryType, Route]

    def __post_init__(self) -> None:
        missing = [kind for kind in QueryType if kind not in self.routes]
        if missing:
            names = "、".join(QUERY_TYPE_LABELS[kind] for kind in missing)
            raise ValueError(f"路由表缺了这几类问题：{names}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RouteTable:
        """从知识库的配置里读路由表。**配置里没写的类型一律走默认值。**

        配置长这样（`:data:`DEFAULT_ROUTES` 的机器形式，键是 `QueryType` 的取值）::

            {"route_table": {"factual": ["main", "metadata"],
                             "guide": {"paths": ["main", "multi_query"],
                                       "subject_types": ["character"]}}}

        一行可以只给路（数组），也可以给全（对象，多两个可选的过滤字段）；给了对象时
        没写的过滤字段仍走默认。**只覆盖写了的那几行**，其余照旧——整表覆盖会让一份
        只改了对比型的配置把另外五类一起打回出厂值。

        :raises ValueError: 类型名或路名不在词表里、某一行一条路都没给。
        """
        raw = payload.get("route_table") or {}
        if not isinstance(raw, Mapping):
            raise ValueError("route_table 应当是一个对象：查询类型 → 这一类走哪几路")
        routes = dict(DEFAULT_TABLE.routes)
        for key, row in raw.items():
            kind = _query_type(key)
            routes[kind] = _route(kind, row)
        return cls(routes)

    def to_mapping(self) -> dict[str, Any]:
        """路由表落成配置里的样子。**只写与默认值不同的那几行。**

        全写回去会让默认值以后改不动——库里存着的那一份会把新默认整个盖住，
        而它看上去只是一份正常的配置。
        """
        return {
            kind.value: {
                "paths": [path.value for path in route.paths],
                "subject_types": [kind.value for kind in route.subject_types],
                "content_natures": [nature.value for nature in route.content_natures],
            }
            for kind, route in self.routes.items()
            if route != DEFAULT_TABLE.routes[kind]
        }

    def route_for(self, query_type: QueryType | None) -> Route:
        """这一类问题走哪几路。**判不出（`None`）时按事实型那一行走。**

        判不出就退回默认组合，不阻断作答；而默认组合就是表里的一行，所以界面把那一行
        改掉时，判不出时的行为跟着一起改——这是有意的，两份默认值没有分岔的机会。
        """
        return self.routes[query_type or QueryType.FACTUAL]


#: 默认路由表的成品。判不出类型时用的那一行也在里面，见 :meth:`RouteTable.route_for`。
DEFAULT_TABLE = RouteTable(
    {kind: Route(paths, NATURES_FOR_TYPE[kind]) for kind, paths in DEFAULT_ROUTES.items()}
)


def parse_query_type(value: str) -> QueryType | None:
    """把模型吐回来的标签对到词表里的那一个，对不上返回 `None`（判不出）。

    中文叫法与英文取值都认：提示词里给的是中文叫法，而模型偶尔会把 schema 里的取值
    原样吐回来。两边都认，比让这一条在两种写法之间随机失效要好。

    **不抛异常**：判不出只是退回默认组合，不该让整个提问失败。
    """
    wanted = " ".join(value.split())
    if not wanted:
        return None
    for kind in QueryType:
        if wanted in (QUERY_TYPE_LABELS[kind], kind.value):
            return kind
    return None


def _query_type(key: Any) -> QueryType:
    """配置里的键 → 查询类型。不认识的当场报出来，别留到检索那一步才炸。"""
    kind = parse_query_type(str(key))
    if kind is None:
        known = "、".join(sorted(item.value for item in QueryType))
        raise ValueError(f"路由表里有个不认识的查询类型 {key!r}，只认这六个：{known}")
    return kind


def _route(kind: QueryType, row: Any) -> Route:
    """配置里的一行 → :class:`Route`。数组只给路，对象可以连过滤条件一起给。

    两种写法先归一成一种：数组包成只写了 `paths` 的对象，往下就只有一条代码路径。
    三个取值各自「没写就取默认」的规则因此只写在各自的取值函数里，不在这里分叉。
    """
    fields = row if isinstance(row, Mapping) else {"paths": row}
    paths = _row_paths(fields.get("paths"), kind)
    if not paths:
        # 一条路都不走 = 这一类问题一条候选都取不到，而它不会报错，只会永远答「没找到」
        raise ValueError(f"{QUERY_TYPE_LABELS[kind]}这一行一条路都没给")
    return Route(
        paths,
        _natures(fields.get("content_natures"), kind),
        _subjects(fields.get("subject_types")),
    )


def _row_paths(raw: Any, kind: QueryType) -> tuple[RecallPath, ...]:
    return _enums(raw, RecallPath, what=f"{QUERY_TYPE_LABELS[kind]}这一行走的路")


def _natures(raw: Any, kind: QueryType) -> tuple[ContentNature, ...]:
    if raw is None:
        # 没写就是按查询类型那一栏推出来的默认，不是「不限」
        return NATURES_FOR_TYPE[kind]
    return _enums(raw, ContentNature, what=f"{QUERY_TYPE_LABELS[kind]}这一行的内容性质")


def _subjects(raw: Any) -> tuple[SubjectType, ...]:
    if raw is None:
        # 没写是「不限」：默认表里这一维一律留空，理由见 :class:`Route`
        return ()
    return _enums(raw, SubjectType, what="路由表里的主体类型")


def _enums[EnumT: StrEnum](raw: Any, vocabulary: type[EnumT], *, what: str) -> tuple[EnumT, ...]:
    """配置里的一串标签 → 枚举成员。

    认不出就报出来：**留空是「不限」，与写错不是一回事**（默认表里主体类型就一律留空），
    静默当成不限会让「配置没生效」看起来和「配置本来就没配」一样。

    `what` 是出错信息里那个主语，由调用方给——三处取值的叫法不同，报错时要说得出
    是哪一行哪一栏。
    """
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{what}应当是一个数组")
    known = {item.value: item for item in vocabulary}
    unknown = [item for item in raw if str(item) not in known]
    if unknown:
        names = "、".join(sorted(known))
        raise ValueError(f"{what}里有不认识的取值 {unknown}，只认这些：{names}")
    return tuple(known[str(item)] for item in raw)
