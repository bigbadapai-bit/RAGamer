"""提问理解：一次调用判定游戏、版本与规范问法，以及检索用的版本过滤。

改用例名一律用 `CONTEXT.md` 的词汇（版本、未标注版本、切片），不用实现术语。
模型那一段接 `FakeLlm`——它一次网络都不发；版本过滤那一段接内存假件，
过滤语义与真实适配器同一份规则。版本过滤的失效是**静默**的：
漏掉「未标注版本」这一支，用户切到历史版本后世界观类问法会全部答不出，
而漏掉整个过滤则是近重复内容互相稀释——两种都不报错，只能靠断言抓。
"""

from __future__ import annotations

import logging

import pytest

from ragamer.llm import FakeLlm, LlmRejected, LlmTimeout, Message
from ragamer.query import (
    TEMPERATURE,
    Understanding,
    UnderstandingMemo,
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

#: 模型一次要吐的那几个字段。确定度与取值成对：取值是候选里的哪一个，确定度是它有多稳。
JOINT_OUTPUT = {
    "game": "黑神话·悟空",
    "game_confidence": 0.9,
    "version": "2.0",
    "version_confidence": 0.8,
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

    assert result == Understanding("黑神话·悟空", "2.0", "二郎神怎么打", 0.9, 0.8, QueryType.GUIDE)
    assert len(llm.calls) == 1


def test_确定度随取值一起带回来():
    """两档分开处理（§3.4 坑 #13）全靠这个数：**取值的确定度**是澄清反问唯一可判的依据。

    取值是对的、确定度是低的，这正是「接近但不肯定」——丢掉确定度就只剩非黑即白，
    要么一律不问、要么一律再问一遍。
    """
    llm = FakeLlm(_reply(game_confidence=0.55, version_confidence=0.7))

    result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert (result.game, result.game_confidence) == ("黑神话·悟空", 0.55)
    assert (result.version, result.version_confidence) == ("2.0", 0.7)


def test_取值被丢掉时确定度一并归零():
    """模型判出的游戏不在候选里：取值丢掉，确定度也不能留着。

    留着的后果是「一个确定度高到不用反问的取值，却是空的」——调用方照着确定度分流，
    会走进「确定」那一支，然后拿着空游戏去检索。
    """
    llm = FakeLlm(_reply(game="塞尔达传说", game_confidence=0.99))

    result = understand("那个 BOSS 怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.game == ""
    assert result.game_confidence == 0.0


def test_确定度超出_0_到_1_时按模型返回不合规处理():
    """确定度的量纲是 0~1，两个阈值（0.65 / 0.50）按它标定。

    给个 1.5 或 -0.2 就当成不合规退回去重问：**夹到边界上会静默改变那次分级**——
    1.5 夹成 1 看着无害，-0.2 夹成 0 却是把一次「模型其实很确定」判成了「判不出来」。
    """
    llm = FakeLlm(_reply(game_confidence=1.5))

    result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result == Understanding("", "", "二郎神怎么打")


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

    assert result == Understanding("黑神话·悟空", "2.0", "二郎神怎么打", 0.9, 0.8)
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


# ── 理解结果的备忘 ───────────────────────────────────────────────────────────
#
# 它要解决的是**改写会抖**：温度已经钉死在 0，模型在 0 下仍会给出不同的改写（实测
# 同一个问题问 6 次得到 3 种）。而答案缓存的键建在改写之上（`ragamer.caching`），
# 于是同一句话问两次有可能算成两个键、双双未命中，各花掉一次完整检索。
# 记下之后，改写对「同一句话 + 同一段历史」就是确定的。


def test_同一句问话配同一段历史只问一次模型():
    llm = FakeLlm(_reply())
    memo = UnderstandingMemo()

    first = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)
    second = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)

    # 脚本只排了一条：第二次若真去问，FakeLlm 会当场炸
    assert len(llm.calls) == 1
    assert second == first


def test_归一之后相同的问法共用一条():
    """多打几个空格是同一句话——与缓存键、与 `_pick` 比候选用的是同一个归一口径。"""
    llm = FakeLlm(_reply())
    memo = UnderstandingMemo()

    understand("二郎神  怎么打", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)
    understand(" 二郎神 怎么打 ", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)

    assert len(llm.calls) == 1


def test_历史不同就是不同的问题():
    """「那它怎么打」配上不同上文本来就是不同的问题。

    只按问法记的话，后一个会话会拿到前一个会话的指代补全结果——而那一轮检索的是
    另一个主体，界面上完全看不出来。
    """
    llm = FakeLlm(_reply(rewritten_query="二郎神怎么打"), _reply(rewritten_query="大圣怎么打"))
    memo = UnderstandingMemo()

    first = understand(
        "那它怎么打",
        llm=llm,
        games=GAMES,
        versions=VERSIONS,
        history=[Message("user", "二郎神是谁")],
        memo=memo,
    )
    second = understand(
        "那它怎么打",
        llm=llm,
        games=GAMES,
        versions=VERSIONS,
        history=[Message("user", "大圣是谁")],
        memo=memo,
    )

    assert len(llm.calls) == 2
    assert first.rewritten_query != second.rewritten_query


def test_候选变了就不再复用():
    """冻住的取值必须仍是当前候选里的一个。

    `_pick` 只接受候选里有的取值，所以算出来的游戏**一定曾是候选之一**；冻住之后
    知识库被改名或删掉，那个标签就成了候选里根本没有的——而 `ragamer.clarifying._game`
    认不出它是**当场抛 `NotACandidate`**，一条本来问得通的提问直接变成报错。
    版本那侧不校验，轻一些但同样是错的：会照一个已经不在库里的版本去检索。
    """
    llm = FakeLlm(_reply(), _reply())
    memo = UnderstandingMemo()

    understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)
    understand("二郎神怎么打", llm=llm, games=["燕云十六声"], versions=[], memo=memo)

    assert len(llm.calls) == 2


def test_降级的结果不入备忘():
    """一次模型抖动不该被钉成整个进程生命周期里的固定行为。

    降级交回的是「按原问法继续、游戏与版本留空」，它与「模型判出来就是空的」长得
    一模一样——记进来的话，之后所有相同的提问都拿不到真判定，而且看不出为什么。
    """
    llm = FakeLlm(LlmTimeout("模型服务超时"), _reply())
    memo = UnderstandingMemo()

    degraded = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)
    recovered = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)

    assert degraded == Understanding("", "", "二郎神怎么打")
    assert len(llm.calls) == 2  # 第二次照问，没有被那条降级顶掉
    assert recovered == Understanding(
        "黑神话·悟空", "2.0", "二郎神怎么打", 0.9, 0.8, QueryType.GUIDE
    )


def test_不给备忘就每次都问模型():
    """`None` 就是「不记」：行为与没有这一层时完全一样，所以它是安全的缺省。"""
    llm = FakeLlm(_reply(), _reply())

    understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)
    understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert len(llm.calls) == 2


def test_备忘满了丢最久没用过的那条():
    """键里带着整段历史，不封顶就会随着「问过多少种问法」一直长下去。"""
    llm = FakeLlm(_reply(), _reply(), _reply())
    memo = UnderstandingMemo(size=2)

    understand("一", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)
    understand("二", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)
    understand("一", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)  # 一 → 最新
    understand("三", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)  # 挤掉最久没用的二
    understand("一", llm=llm, games=GAMES, versions=VERSIONS, memo=memo)  # 还在，不再问

    assert len(llm.calls) == 3


def test_容量至少是一条():
    """容量给 0 的话 `set` 会当场把刚记下的那条丢掉，而那是不报错的。"""
    with pytest.raises(ValueError):
        UnderstandingMemo(size=0)
