"""提问理解：一次调用判定游戏、版本与规范问法，以及检索用的版本过滤。

改用例名一律用 `CONTEXT.md` 的词汇（版本、未标注版本、切片），不用实现术语。
模型那一段接 `FakeLlm`——它一次网络都不发；版本过滤那一段接内存假件，
过滤语义与真实适配器同一份规则。版本过滤的失效是**静默**的：
漏掉「未标注版本」这一支，用户切到历史版本后世界观类问法会全部答不出，
而漏掉整个过滤则是近重复内容互相稀释——两种都不报错，只能靠断言抓。
"""

from __future__ import annotations

from ragamer.llm import FakeLlm, LlmTimeout, Message
from ragamer.query import Understanding, normalize_query, understand, version_filter
from ragamer.stores.base import UNVERSIONED, ChunkFilter, matches
from ragamer.stores.memory import InMemoryChunkStore

from .conftest import fake_vector, make_chunk

GAME = "black_myth"
GAMES = ["黑神话·悟空", "燕云十六声"]
VERSIONS = ["1.0", "2.0"]

#: 模型一次要吐的那三个字段。
JOINT_OUTPUT = {
    "game": "黑神话·悟空",
    "version": "2.0",
    "rewritten_query": "二郎神怎么打",
}


def reply(**overrides: object) -> dict[str, object]:
    return {**JOINT_OUTPUT, **overrides}


def test_一次调用同时拿回游戏版本与规范问法():
    """三个结果出自同一次调用。脚本只排了一条——真调第二次会当场炸（FakeLlm）。"""
    llm = FakeLlm(reply())

    result = understand(
        "那它怎么打",
        llm=llm,
        games=GAMES,
        versions=VERSIONS,
        history=[Message("user", "二郎神是谁")],
    )

    assert result == Understanding("黑神话·悟空", "2.0", "二郎神怎么打")
    assert len(llm.calls) == 1


def test_指代被改写成带主体的规范问法():
    """「那它怎么打」这类指代要补成带主体的问法，上一轮的话得一并喂进去。"""
    llm = FakeLlm(reply(rewritten_query="二郎神怎么打"))

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

    温度钉死 0 是这件事的一半——温度没钉住，同一个问题每次问出来的改写都不一样，
    缓存就永远命不中。另一半是归一化只压平空白，不做同义合并。
    """
    llm = FakeLlm(reply(rewritten_query="  二郎神   怎么打  "))

    result = understand("那它怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.rewritten_query == "二郎神 怎么打"
    assert llm.calls[0].temperature == 0.0
    assert normalize_query(" 二郎神 怎么打") == normalize_query("二郎神   怎么打  ")


def test_模型给出了候选里没有的游戏时丢掉():
    """模型自己编的游戏名在库里不存在，照它检索只会查空——留空交回给调用方。"""
    llm = FakeLlm(reply(game="塞尔达传说"))

    result = understand("那个 BOSS 怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.game == ""
    assert result.version == "2.0"  # 同一字段的毛病不牵连另一个


def test_没有任何候选时两个字段都留空():
    """调用方拿不出真实候选时不许模型自己编——候选为空即无解。"""
    llm = FakeLlm(reply())

    result = understand("二郎神怎么打", llm=llm)

    assert result.game == ""
    assert result.version == ""
    assert result.rewritten_query == "二郎神怎么打"


def test_改写为空时退回原问法():
    """模型只吐了游戏与版本、没吐改写：拿原问法继续，别把空串传下去。"""
    llm = FakeLlm(reply(rewritten_query=""))

    result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.rewritten_query == "二郎神怎么打"
    assert result.game == "黑神话·悟空"


def test_模型失败时按原问法降级():
    """这一步失败不该让整个提问失败：原问法照样能检索。"""
    llm = FakeLlm(LlmTimeout("模型服务超时"))

    result = understand(" 二郎神  怎么打 ", llm=llm, games=GAMES, versions=VERSIONS)

    assert result == Understanding("", "", "二郎神 怎么打")


def test_版本过滤恒包含未标注版本():
    """ADR-0004：过滤条件必须是「所选版本**或**未标注版本」。"""
    where = version_filter("2.0", current_version="1.0")

    assert matches(make_chunk(1, version="2.0"), where)
    assert matches(make_chunk(2, version=UNVERSIONED), where)
    assert not matches(make_chunk(3, version="1.0"), where)


def test_问题没点名版本时用知识库的当前生效版本():
    """问题里没提版本，就用知识库标着的那一个——它是「当前该用哪个版本」的唯一真相来源。"""
    where = version_filter(UNVERSIONED, current_version="1.0")

    assert matches(make_chunk(1, version="1.0"), where)
    assert matches(make_chunk(2, version=UNVERSIONED), where)
    assert not matches(make_chunk(3, version="2.0"), where)


def test_两处都判不出时不按版本过滤():
    """没有版本可依据时不做过滤：宁可近重复互相稀释，也不能静默漏掉整批内容。"""
    where = version_filter(UNVERSIONED, current_version=UNVERSIONED)

    assert where == ChunkFilter(version=None)
    assert matches(make_chunk(1, version="2.0"), where)


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
