"""查询路由：六类问题各自走哪几路，以及这张表怎么被知识库覆盖。

三组用例：**默认表照 §3.1 摆**（逐类断言——它是这一层的行为本身）、**覆盖是逐行的**
（只改一类不影响另外五类）、**判不出与写错分得开**（前者退回默认组合继续作答，
后者当场报出来，别留到检索那一步）。

选路的失效是**静默**的：走错了路照样出答案，只是答案的依据少了或偏了。所以这里断的是
「哪一类走哪几路」这个映射本身，而不是「有没有报错」。
"""

from __future__ import annotations

import pytest

from ragamer.routing import (
    DEFAULT_ROUTES,
    DEFAULT_TABLE,
    NATURES_FOR_TYPE,
    QUERY_TYPE_LABELS,
    WIRED_PATHS,
    QueryType,
    RecallPath,
    Route,
    RouteTable,
    parse_query_type,
)
from ragamer.tagging import ContentNature, SubjectType


def paths_of(query_type: QueryType | None) -> tuple[RecallPath, ...]:
    return DEFAULT_TABLE.route_for(query_type).paths


# --- 默认路由表 ---


def test_六类问题都有一条组合():
    """少一类就是那一类问题一条候选都取不到，而它报出来的只是「知识库里没有找到资料」。"""
    assert set(DEFAULT_ROUTES) == set(QueryType)
    assert all(paths_of(kind) for kind in QueryType)


def test_默认表照_3_1_摆():
    """逐类照抄 `docs/ARCHITECTURE.md` §3.1 那张表。改这张表等于改行为。

    两处展开，都是把原文的简写读通：
    「+ HyDE」「+ 联网」没有另给基础组合，只能读成「事实型那一组再加一条」；
    表格型补上主检索，理由见 :data:`ragamer.routing.DEFAULT_ROUTES` 的说明。
    """
    assert paths_of(QueryType.FACTUAL) == (RecallPath.MAIN, RecallPath.METADATA)
    assert paths_of(QueryType.GUIDE) == (
        RecallPath.MAIN,
        RecallPath.MULTI_QUERY,
        RecallPath.METADATA,
    )
    assert paths_of(QueryType.COMPARISON) == (RecallPath.MULTI_QUERY, RecallPath.MAIN)
    assert paths_of(QueryType.VAGUE) == (
        RecallPath.MAIN,
        RecallPath.METADATA,
        RecallPath.HYDE,
    )
    assert paths_of(QueryType.RECENCY) == (RecallPath.MAIN, RecallPath.METADATA, RecallPath.WEB)
    assert RecallPath.TABLE in paths_of(QueryType.TABULAR)


def test_多出来的那几条路是逐类加在事实型那一组上的():
    """模糊型与时效型在原文里只写了「+ 某一路」——那就是事实型那一组再加一条。

    这样读的另一个好处：拿掉加的那一条，剩下的正好是事实型那一组。
    """
    for kind, extra in ((QueryType.VAGUE, RecallPath.HYDE), (QueryType.RECENCY, RecallPath.WEB)):
        assert set(paths_of(kind)) - {extra} == set(paths_of(QueryType.FACTUAL))


def test_判不出类型时按事实型那一行走():
    """判不出不是错误，是这一层的一条降级路径：按默认组合继续，不阻断作答。"""
    assert DEFAULT_TABLE.route_for(None) == DEFAULT_TABLE.route_for(QueryType.FACTUAL)


def test_判不出时的组合跟着事实型那一行改():
    """**不另立一份默认组合**：另立的那一份与表里的一行迟早会分岔，而分岔之后
    「界面改了配置」与「判不出时走的还是老一套」在日志里看不出来。"""
    table = RouteTable.from_mapping({"route_table": {"factual": ["main"]}})

    assert table.route_for(None).paths == (RecallPath.MAIN,)


def test_元数据过滤路按内容性质收窄():
    """内容性质是切片那一侧的轴，「问题属于哪一类」是问句这一侧的轴，两者对得上。

    模糊型与时效型不收窄：这两类问的恰恰是「说不清问的是哪一类」，按性质过滤会把
    本该捞上来的东西筛掉。
    """
    assert DEFAULT_TABLE.route_for(QueryType.FACTUAL).content_natures == (ContentNature.STATS,)
    assert DEFAULT_TABLE.route_for(QueryType.GUIDE).content_natures == (ContentNature.GUIDE,)
    assert DEFAULT_TABLE.route_for(QueryType.COMPARISON).content_natures == (ContentNature.REVIEW,)
    assert DEFAULT_TABLE.route_for(QueryType.VAGUE).content_natures == ()
    assert DEFAULT_TABLE.route_for(QueryType.RECENCY).content_natures == ()


def test_默认表里不按主体类型收窄():
    """问题问的是哪个主体类型，判它要多一次模型调用或一次术语扫描，而这一票的头一条
    就是「不新增任何模型调用」。字段留着——库里配了就用配的那个。"""
    assert all(DEFAULT_TABLE.route_for(kind).subject_types == () for kind in QueryType)
    assert DEFAULT_TABLE.route_for(QueryType.FACTUAL).content_natures  # 另一维是默认就有的


def test_接上的路逐条列出来():
    """这张清单与 `ragamer.retrieval` 里真正实现的那些对齐，对不上的表现是
    「配了那条路却一条候选都没多出来」。**新加一条路时它必须跟着改**——用例在这里
    兜住，免得有人顺手把它改成 `frozenset(RecallPath)` 图省事。"""
    assert WIRED_PATHS == {
        RecallPath.MAIN,
        RecallPath.METADATA,
        RecallPath.MULTI_QUERY,
        RecallPath.HYDE,
        RecallPath.TABLE,
        RecallPath.WEB,
    }


# --- 覆盖 ---


def test_只覆盖写了的那几行():
    """一份只改了对比型的配置不该把另外五类一起打回出厂值。"""
    table = RouteTable.from_mapping({"route_table": {"comparison": ["main"]}})

    assert table.route_for(QueryType.COMPARISON).paths == (RecallPath.MAIN,)
    assert table.route_for(QueryType.GUIDE) == DEFAULT_TABLE.route_for(QueryType.GUIDE)


def test_一行可以连过滤条件一起给():
    """数组只给路，对象可以连主体类型与内容性质一起给——界面要能改这两维。"""
    table = RouteTable.from_mapping(
        {
            "route_table": {
                "factual": {
                    "paths": ["main", "metadata"],
                    "subject_types": ["character"],
                    "content_natures": ["stats"],
                }
            }
        }
    )

    route = table.route_for(QueryType.FACTUAL)
    assert route.paths == (RecallPath.MAIN, RecallPath.METADATA)
    assert route.subject_types == (SubjectType.CHARACTER,)
    assert route.content_natures == (ContentNature.STATS,)


def test_对象形式里没写的过滤字段仍走默认():
    """只写了路的那一行不该把内容性质一起清空——清空是「不限」，与「没配」不是一回事。"""
    table = RouteTable.from_mapping({"route_table": {"guide": {"paths": ["main"]}}})

    route = table.route_for(QueryType.GUIDE)
    assert route.content_natures == NATURES_FOR_TYPE[QueryType.GUIDE]


def test_配置里不认识的查询类型当场报错():
    """留到检索那一步才炸的话，离配置的出处已经很远了，而那时报出来的是一条
    「知识库里没有找到资料」。"""
    with pytest.raises(ValueError, match="不认识的查询类型"):
        RouteTable.from_mapping({"route_table": {"玄学": ["main"]}})


def test_配置里不认识的路当场报错():
    with pytest.raises(ValueError, match="不认识的取值"):
        RouteTable.from_mapping({"route_table": {"factual": ["main", "算命"]}})


def test_配置里一行不给路也当场报错():
    """一条路都不走 = 这一类问题一条候选都取不到，而它不报错，只是永远答「没找到」。"""
    with pytest.raises(ValueError, match="一条路都没给"):
        RouteTable.from_mapping({"route_table": {"factual": []}})


def test_配置里的过滤取值不合法也当场报错():
    """认不出与留空要分得开：留空是「不限」，拼错是配置事故。"""
    with pytest.raises(ValueError, match="主体类型"):
        RouteTable.from_mapping(
            {"route_table": {"factual": {"paths": ["main"], "subject_types": ["妖怪"]}}}
        )


def test_缺一类的表构造不出来():
    """`route_for` 上会以 KeyError 炸在检索那一步，那时已经离配置的出处很远了。"""
    with pytest.raises(ValueError, match="缺了这几类问题"):
        RouteTable({QueryType.FACTUAL: Route((RecallPath.MAIN,))})


def test_没配路由表时整张表都是默认值():
    """知识库里没写 `route_table` 是最常见的一种——它不是错误，是走默认。"""
    assert RouteTable.from_mapping({}).routes == DEFAULT_TABLE.routes


def test_落成配置时只写与默认不同的那几行():
    """全写回去会让默认值以后改不动：库里存着的那一份会把新默认整个盖住，
    而它看上去只是一份正常的配置。"""
    table = RouteTable.from_mapping({"route_table": {"comparison": ["main"]}})

    # 改过的那一行写全三个字段（写出去的是**生效值**，不是当初配置里写的那半截）
    assert table.to_mapping() == {
        "comparison": {"paths": ["main"], "subject_types": [], "content_natures": ["review"]}
    }
    # 其余五类与默认一模一样，一个键都不出现
    assert DEFAULT_TABLE.to_mapping() == {}


def test_落成配置再读回来是同一张表():
    """写出去的那份要能被自己读回来——界面存下来的配置走的就是这一条路。"""
    table = RouteTable.from_mapping(
        {
            "route_table": {
                "comparison": ["main"],
                "factual": {"paths": ["main", "metadata"], "subject_types": ["character"]},
            }
        }
    )

    assert RouteTable.from_mapping({"route_table": table.to_mapping()}) == table


# --- 模型吐回来的标签 ---


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("事实型", QueryType.FACTUAL),
        ("tabular", QueryType.TABULAR),
        ("  攻略型 ", QueryType.GUIDE),
    ],
)
def test_词表里的标签认得出来(given, expected):
    assert parse_query_type(given) == expected


@pytest.mark.parametrize("given", ["", "   ", "玄学型", "factual 型"])
def test_词表以外的标签一律判不出(given):
    """**不抛异常**：判不出只是退回默认组合，不该让整个提问失败。"""
    assert parse_query_type(given) is None


def test_六类的叫法不重不漏():
    """叫法同时进提示词与日志，两个地方都得能一眼看出是哪一类。"""
    assert {QUERY_TYPE_LABELS[kind] for kind in QueryType} == {
        "事实型",
        "攻略型",
        "对比型",
        "模糊型",
        "时效型",
        "表格型",
    }
