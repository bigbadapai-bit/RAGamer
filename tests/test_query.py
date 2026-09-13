"""提问理解：一次调用判定游戏、版本与规范问法，以及检索用的版本过滤。

改用例名一律用 `CONTEXT.md` 的词汇（版本、未标注版本、切片），不用实现术语。
模型那一段接 `FakeLlm`——它一次网络都不发；版本过滤那一段接内存假件，
过滤语义与真实适配器同一份规则。版本过滤的失效是**静默**的：
漏掉「未标注版本」这一支，用户切到历史版本后世界观类问法会全部答不出，
而漏掉整个过滤则是近重复内容互相稀释——两种都不报错，只能靠断言抓。
"""

from __future__ import annotations

import logging

from ragamer.llm import FakeLlm, LlmRejected, LlmTimeout, Message
from ragamer.query import (
    TEMPERATURE,
    Understanding,
    normalize_query,
    understand,
    version_filter,
)
from ragamer.routing import QUERY_TYPE_LABELS, QueryType
from ragamer.stores.base import UNVERSIONED, ChunkFilter, matches
from ragamer.stores.memory import InMemoryChunkStore

from .conftest import fake_vector, make_chunk

GAME = "black_myth"
GAMES = ["黑神话·悟空", "燕云十六声"]
VERSIONS = ["1.0", "2.0"]

#: 模型一次要吐的那四个字段。
JOINT_OUTPUT = {
    "game": "黑神话·悟空",
    "version": "2.0",
    "rewritten_query": "二郎神怎么打",
    "route": "攻略型",
}


def _reply(**overrides: object) -> dict[str, object]:
    return {**JOINT_OUTPUT, **overrides}


def test_一次调用同时拿回游戏版本规范问法与路由标签():
    """四个结果出自同一次调用。脚本只排了一条——真调第二次会当场炸（FakeLlm）。"""
    llm = FakeLlm(_reply())

    result = understand(
        "那它怎么打",
        llm=llm,
        games=GAMES,
        versions=VERSIONS,
        history=[Message("user", "二郎神是谁")],
    )

    assert result == Understanding("黑神话·悟空", "2.0", "二郎神怎么打", QueryType.GUIDE)
    assert len(llm.calls) == 1


def test_路由标签也认模型吐回来的英文取值():
    """提示词给的是中文叫法，而模型偶尔会把 schema 里的取值原样吐回来。

    两种写法都认，比让这一条在两者之间随机失效要好——失效时是静默的：路由退回默认
    组合，答案照样出得来，只是走的不是本该走的那几路。
    """
    llm = FakeLlm(_reply(route="tabular"))

    assert understand("寒江雪的属性", llm=llm, games=GAMES).query_type is QueryType.TABULAR


def test_路由标签不在词表里时判不出并留痕(caplog):
    """判不出只是退回默认组合，**不牵连**另外三个字段，也不让整个提问失败。"""
    llm = FakeLlm(_reply(route="玄学型"))

    with caplog.at_level(logging.WARNING, logger="ragamer.query"):
        result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.query_type is None
    assert result.game == "黑神话·悟空"
    assert result.rewritten_query == "二郎神怎么打"
    assert any("查询类型不在词表里" in record.getMessage() for record in caplog.records)


def test_按提示留空串是判不出而不是认出个没见过的取值(caplog):
    """提示词请模型判不出时留空串。那是一个明确答案，不该报成「不在词表里」——
    真正的跑偏与「本来就没判出来」在排查时看的是不同的地方。"""
    llm = FakeLlm(_reply(route=""))

    with caplog.at_level(logging.WARNING, logger="ragamer.query"):
        result = understand("那个很难的 BOSS 怎么过", llm=llm, games=GAMES)

    assert result.query_type is None
    assert not [record for record in caplog.records if "查询类型" in record.getMessage()]


def test_模型整个漏掉查询类型时另外三个字段照用(caplog):
    """字段**缺席**与「按提示留了空串」要分得开，而两者的走法一样（都回落默认组合）。

    缺席只赔上它自己：`route` 有默认值，另外三个字段照常带走。把它们一起赔进去，
    等于为一个本来就定义了回落的字段牺牲三个没得回落的——重试一次还好，重试用尽
    就整套降级了。
    """
    llm = FakeLlm({key: value for key, value in JOINT_OUTPUT.items() if key != "route"})

    with caplog.at_level(logging.WARNING, logger="ragamer.query"):
        result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result == Understanding("黑神话·悟空", "2.0", "二郎神怎么打")
    assert any("整个缺席" in record.getMessage() for record in caplog.records)


def test_提示词里写全了六类问题():
    """只给类型名不够：模型判「表格型」与「事实型」的界线时会猜，而这两类问的都是数值。
    叫法与典型问法都要进提示词（与游戏、版本候选同一个姿势）。"""
    llm = FakeLlm(_reply())

    understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    system = llm.calls[0].messages[0].content
    assert all(QUERY_TYPE_LABELS[kind] in system for kind in QueryType)


def test_指代被改写成带主体的规范问法():
    """**接线**的断言：模型吐回来的规范问法要原样带走，指代要补成主体名得靠上一轮的话。

    改写本身做不做得对是模型的事，假件证不了——那一条在
    `tests/test_query_integration.py` 里拿真模型问，默认不跑。这里只钉住
    「历史带上了、改写结果没被丢掉」这两件接线上的事。
    """
    llm = FakeLlm(_reply(rewritten_query="二郎神怎么打"))

    result = understand(
        "那它怎么打",
        llm=llm,
        games=GAMES,
        versions=VERSIONS,
        history=[Message("user", "二郎神是谁"), Message("assistant", "二郎神是隐藏 BOSS")],
    )

    assert result.rewritten_query == "二郎神怎么打"
    sent = [(message.role, message.content) for message in llm.calls[0].messages]
    assert ("user", "二郎神是谁") in sent
    assert sent[-1] == ("user", "那它怎么打")


def test_问法与改写结果都稳定可复现():
    """改写结果要能当缓存 key 用：同一个问法每次归一成形同一个样子。

    真正被断言的是归一化只压平空白、不做同义合并（后两行）——那是纯函数，
    钉得住。温度那一行钉的是本模块的取值，**挡不住把参数删掉**：
    `LlmRequest.temperature` 的默认值也是 0，删了照样过；它挡的是将来有人
    把 `TEMPERATURE` 调高。同一个问法两次问出同一个改写要真模型才算数，
    在 `tests/test_query_integration.py` 里。
    """
    llm = FakeLlm(_reply(rewritten_query="  二郎神   怎么打  "))

    result = understand("那它怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.rewritten_query == "二郎神 怎么打"
    assert llm.calls[0].temperature == TEMPERATURE == 0.0
    assert normalize_query(" 二郎神 怎么打") == normalize_query("二郎神   怎么打  ")


def test_模型给出了候选里没有的游戏时丢掉():
    """模型自己编的游戏名在库里不存在，照它检索只会查空——留空交回给调用方。"""
    llm = FakeLlm(_reply(game="塞尔达传说"))

    result = understand("那个 BOSS 怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.game == ""
    assert result.version == "2.0"  # 同一字段的毛病不牵连另一个


def test_没有任何候选时两个字段都留空():
    """调用方拿不出真实候选时不许模型自己编——候选为空即无解。"""
    llm = FakeLlm(_reply())

    result = understand("二郎神怎么打", llm=llm)

    assert result.game == ""
    assert result.version == ""
    assert result.rewritten_query == "二郎神怎么打"


def test_改写为空时退回原问法():
    """模型只吐了游戏与版本、没吐改写：拿原问法继续，别把空串传下去。"""
    llm = FakeLlm(_reply(rewritten_query=""))

    result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.rewritten_query == "二郎神怎么打"
    assert result.game == "黑神话·悟空"


def test_模型失败时按原问法降级():
    """这一步失败不该让整个提问失败：原问法照样能检索。"""
    llm = FakeLlm(LlmTimeout("模型服务超时"))

    result = understand(" 二郎神  怎么打 ", llm=llm, games=GAMES, versions=VERSIONS)

    assert result == Understanding("", "", "二郎神 怎么打")


def test_模型被拒时按_ERROR_留痕(caplog):
    """密钥或模型名配错，之后每条提问都会这样降级——只留 WARNING 会看不出根因。"""
    llm = FakeLlm(LlmRejected("模型服务拒绝了请求（HTTP 401）"))

    with caplog.at_level(logging.ERROR, logger="ragamer.query"):
        understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_版本过滤恒包含未标注版本():
    """ADR-0004：过滤条件必须是「所选版本**或**未标注版本」。

    这一层能做的只是保证不把条件构造歪——「或未标注版本」那半句是
    `stores/base.py` 的 `matches` 与 Milvus 适配器的表达式共有的语义，
    它们各自有自己的测试。这里钉的是：经过 `version_filter` 之后那半句还在。
    """
    where = version_filter("2.0", current_version="1.0")

    assert matches(make_chunk(1, version="2.0"), where)
    assert matches(make_chunk(2, version=UNVERSIONED), where)
    assert not matches(make_chunk(3, version="1.0"), where)


def test_问题没点名版本时用知识库的现行版本():
    """问题里没提版本，就用知识库标着的那一个——它是「当前该用哪个版本」的唯一真相来源。"""
    where = version_filter("", current_version="1.0")

    assert matches(make_chunk(1, version="1.0"), where)
    assert matches(make_chunk(2, version=UNVERSIONED), where)
    assert not matches(make_chunk(3, version="2.0"), where)


def test_两处都判不出时不按版本过滤并留痕(caplog):
    """没有版本可依据时不做过滤，但要留痕（ADR-0004 的「默认开启」在这里有意收窄）。

    过滤成「只留未标注版本」会把标了版本的资料整批漏掉，那是静默的；
    不过滤只是近重复互相稀释，看得见。知识库没标现行版本是该被修掉的配置问题，
    所以这件事不能悄悄发生。
    """
    with caplog.at_level(logging.WARNING, logger="ragamer.query"):
        where = version_filter("", current_version="")

    assert where == ChunkFilter(version=None)
    assert matches(make_chunk(1, version="2.0"), where)
    assert "不按版本过滤" in caplog.text


def test_标注版本的资料与未标注版本的资料都能被检索到():
    """同一份资料的两个版本副本并存时，两个版本各自的检索都要带上未标注的那一支。"""
    store = InMemoryChunkStore()
    store.upsert(
        GAME,
        [
            make_chunk(1, version="2.0", content="2.0 版打法"),
            make_chunk(2, version=UNVERSIONED, content="世界观设定"),
        ],
    )

    hits = store.search(
        GAME,
        dense=fake_vector(1),
        where=version_filter("2.0", current_version="1.0"),
        limit=10,
    )

    assert {hit.chunk.content for hit in hits} == {"2.0 版打法", "世界观设定"}
