"""澄清反问：拿不准时暂停问，用户点完从暂停点继续。

四组用例：**分级**（确定的直接用、接近但不肯定的反问——两档分开）、
**候选的来源**（全部来自语料，模型编的取值进不来）、
**恢复**（选完的取值回到检索里、重复恢复结果一致且不多写记录）、
**说不通的几种**。

这一层**只管判**：判完的落脚点（`Resolved`）交给谁去生成是调用方的事
（`ragamer.conversations.Chat` 拿着它流式生成并落库）。所以要看「检索按哪个版本过滤」
这类事，用例得自己把生成那一段接上（`answered`）——判定与生成在实现里分开，是因为
对话那一侧要边吐边落库，而「判完就作答」那条一次性的入口在接上会话之后没有调用方了。

候选那一条是本模块存在的理由：让模型自由生成澄清选项，它会给出库里根本没有的选项，
用户选了也检索不到——而「选了没东西」与「选了查得到」在界面上长得一样。
"""

from __future__ import annotations

import pytest

from ragamer.answering import Answer, Answerer
from ragamer.clarifying import (
    CONFIDENT,
    GAME,
    PENDING_COLLECTION,
    UNSURE,
    VERSION,
    Clarification,
    Clarifier,
    NoKnowledgeBase,
    NotACandidate,
    Resolved,
    UnknownPending,
    game_choices,
    version_choices,
)
from ragamer.knowledge import KB_COLLECTION, find_knowledge_base
from ragamer.llm import FakeLlm
from ragamer.retrieval import MAX_CHUNKS
from ragamer.stores.base import UNVERSIONED
from ragamer.stores.memory import InMemoryChunkStore, InMemoryDocStore
from ragamer.vectors.fake import FakeEmbedder, FakeReranker

from .conftest import joint_reply, make_chunk

GAME_ID = "black_myth"
OTHER_ID = "yanyun"
QUESTION = "二郎神怎么打"

#: 两个库。显示名与 id 不同——这是常态（库里的 id 要是合法标识符，界面上要认得出是哪款游戏）。
BLACK_MYTH = {"name": "黑神话·悟空", "version": "1.0"}
YANYUN = {"name": "燕云十六声", "version": "3.0"}

#: 一段答案。编号指的就是提示词里那批切片的编号。
REPLY = "先定身再贴身输出[1]。"


def store(*chunks) -> InMemoryChunkStore:
    chunks_store = InMemoryChunkStore()
    chunks_store.upsert(GAME_ID, [chunk for chunk in chunks if chunk.game_id == GAME_ID])
    chunks_store.upsert(OTHER_ID, [chunk for chunk in chunks if chunk.game_id == OTHER_ID])
    return chunks_store


def docs(**payloads: dict) -> InMemoryDocStore:
    """按「游戏 id = 元数据」建几个库。键名就是 id——它同时是 collection 名。"""
    store_ = InMemoryDocStore()
    for game_id, payload in payloads.items():
        store_.put(KB_COLLECTION, game_id, payload)
    return store_


def both_bases() -> InMemoryDocStore:
    return docs(**{GAME_ID: BLACK_MYTH, OTHER_ID: YANYUN})


def clarifier(chunks=None, doc_store=None, *replies, llm=None) -> Clarifier:
    """判定这一层的内存版：假向量、假精排、假模型，一行云端代码都不碰。

    `llm` 显式给时用它。要数调用条数与看提示词的用例得自己攥着那个假件。
    """
    chunks = chunks if chunks is not None else store(make_chunk(1, content=QUESTION, version="1.0"))
    doc_store = doc_store if doc_store is not None else both_bases()
    llm = llm if llm is not None else FakeLlm(*replies)
    return Clarifier(chunks=chunks, docs=doc_store, llm=llm)


def answered(resolved: Resolved, chunks, doc_store, llm) -> Answer:
    """把判定的落脚点交给生成那一段，返回答案。

    判定与生成在实现里是分开的（`Clarifier` 只判，`ragamer.answering` 才生成），
    要看「选完继续之后检索按哪个版本过滤」，用例里就得自己把这一段接上。
    现行版本照调用方的口径现读一次——它是「当前该用哪个版本」的唯一真相来源。
    """
    base = find_knowledge_base(doc_store, resolved.game_id)
    return Answerer(
        chunks=chunks, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm
    ).answer(
        resolved.rewritten_query,
        game_id=resolved.game_id,
        version=resolved.version,
        current_version=base.version if base else "",
    )


def prompt_of(answer: Answer, llm: FakeLlm) -> str:
    """刚刚那次生成交给模型的提示词。检索按哪批切片走，只有它看得见。

    答案本身看不出这一点：短文档整篇交给生成时，引用上的祖先标题路径是空串。
    """
    assert answer.text == REPLY  # 走完了生成那一段，最后一条调用才是它
    return llm.calls[-1].messages[0].content


# --- 分级：确定与接近但不肯定分开处理 ---


def test_确定度足够时直接用不再反问():
    """两档里的第一档：模型判得出游戏与版本、确定度过线，就别再拿问题去烦用户。"""
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")), both_bases(), joint_reply()
    )

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Resolved)
    assert (outcome.game_id, outcome.version) == (GAME_ID, "1.0")
    assert asker.docs.list_ids(PENDING_COLLECTION) == []


def test_接近但不肯定时暂停并给出候选():
    """第二档：判得出取值但确定度落在两档之间——**猜错的成本远高于反问**（§3.4 坑 #13）。"""
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        both_bases(),
        joint_reply(game_confidence=UNSURE),
    )

    outcome = asker.decide(QUESTION)

    assert isinstance(outcome, Clarification)
    assert outcome.dimension == GAME
    assert [choice.label for choice in outcome.choices] == ["黑神话·悟空", "燕云十六声"]
    assert [choice.value for choice in outcome.choices] == [GAME_ID, OTHER_ID]


def test_反问时不调生成():
    """暂停就是暂停：这一层只调一次模型（理解），生成那一段根本不进来。"""
    asker = clarifier(store(), both_bases(), joint_reply(game_confidence=UNSURE), REPLY)

    asker.decide(QUESTION)

    assert len(asker.llm.calls) == 1


def test_两档的边界落在阈值上():
    """`>= CONFIDENT` 才算确定：0.65 与 0.64 是两档，不是同一个数四舍五入。"""

    def asks(confidence: float) -> bool:
        return isinstance(
            clarifier(store(), both_bases(), joint_reply(game_confidence=confidence)).decide(
                QUESTION
            ),
            Clarification,
        )

    assert not asks(CONFIDENT)
    assert asks(CONFIDENT - 0.01)


def test_压根没判出游戏时也反问而不用调用方之外的库():
    """游戏留空、调用方也没指定：没有任何依据，只能问——这时候选是全部库。"""
    asker = clarifier(store(), both_bases(), joint_reply(game=""))

    outcome = asker.decide(QUESTION)

    assert isinstance(outcome, Clarification)
    assert outcome.dimension == GAME
    assert len(outcome.choices) == 2


def test_调用方已经定下游戏时不再问游戏():
    """页面/会话里已经选定了一个库：没判出别的游戏就按它走，不必多问一遍。"""
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        both_bases(),
        joint_reply(game="", version=""),
    )

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Resolved)
    assert outcome.game_id == GAME_ID


def test_只有一个库时没有可问的直接用它():
    """候选只有一个时答案已经被它定死了，问一句只是多一次点击。

    用户问的要是另一款游戏（库里没有），得到的是 `ragamer.answering` 的「没找到」回复
    ——那比一个只有一个按钮的反问更说得清。
    """
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        docs(black_myth=BLACK_MYTH),
        joint_reply(game="", version=""),
    )

    outcome = asker.decide(QUESTION)

    assert isinstance(outcome, Resolved)
    assert outcome.game_id == GAME_ID


def test_模型判的游戏与调用方给的一致时不再问():
    """两档之间也不见得该问：模型说的正是会话里那一个，问了也只有同一个答案。

    真正拿不准的是**对不上**——那种才把候选摆出来（下一条）。
    """
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        both_bases(),
        joint_reply(game_confidence=UNSURE),
    )

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Resolved)
    assert outcome.game_id == GAME_ID


def test_模型判的游戏与调用方给的对不上时问一问():
    """会话里选的是这个库，模型读出的是另一个库、又不太肯定：这才值得问一句。"""
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        both_bases(),
        joint_reply(game="燕云十六声", game_confidence=UNSURE),
    )

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Clarification)
    assert outcome.dimension == GAME


def test_一个库都没有时当场报错且不白调一次模型():
    """没有候选可问，也没有已定的游戏可依：这一问无解，不能等走到检索那一步才炸。

    而且**拦在理解之前**：候选为空时模型判不出任何游戏，这一次理解注定白调。
    """
    asker = clarifier(InMemoryChunkStore(), InMemoryDocStore(), joint_reply(game=""))

    with pytest.raises(NoKnowledgeBase):
        asker.decide(QUESTION)

    assert asker.llm.calls == []


# --- 候选的来源：语料里真有的取值 ---


def test_模型编出来的游戏进不了候选():
    """库里的游戏只有那两个，模型说「塞尔达传说」——取值丢掉，候选照旧只有库里那些。"""
    asker = clarifier(store(), both_bases(), joint_reply(game="塞尔达传说", game_confidence=0.99))

    outcome = asker.decide(QUESTION)

    assert isinstance(outcome, Clarification)
    assert "塞尔达传说" not in [choice.label for choice in outcome.choices]


def test_版本候选来自语料且不含未标注版本():
    """版本候选是这个库里真实出现过的版本。

    **未标注版本不在候选里**：它在库里的取值是空串，与「没判出来」共用同一个字面，
    摆到按钮上用户点了也说不清选了什么；它本来也不必选——检索恒把它一并纳入（ADR-0004）。
    """
    chunks = store(
        make_chunk(1, content=QUESTION, version="1.0"),
        make_chunk(2, content="世界观设定", version=UNVERSIONED),
        make_chunk(3, content="旧版打法", version="0.9"),
    )
    asker = clarifier(chunks, both_bases(), joint_reply(version_confidence=UNSURE))

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Clarification)
    assert outcome.dimension == VERSION
    assert [choice.label for choice in outcome.choices] == ["0.9", "1.0"]


def test_库里没有别的版本时不反问版本():
    """只有一个版本可选时问「你问的是哪个版本」是明知故问——没有候选就不成立一次反问。"""
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        both_bases(),
        joint_reply(version_confidence=UNSURE),
    )

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Resolved)
    assert outcome.version == ""


def test_模型判成另一个库时它那条版本判断作废():
    """版本候选是照调用方那个库给的：拿 A 库的版本列表去认 B 库的版本，认出来也不作数。

    不作废的话，用户会拿到一个「属于另一个库」的版本去检索——查空且不报错。
    """
    llm = FakeLlm(joint_reply(game="燕云十六声", version="3.0"), REPLY)
    chunks = store(
        make_chunk(1, content="黑神话的正文", version="1.0"),
        make_chunk(2, content="燕云的正文", version="3.0", game_id=OTHER_ID),
    )
    asker = clarifier(chunks, both_bases(), llm=llm)

    outcome = asker.decide(QUESTION, game_id=GAME_ID)

    assert isinstance(outcome, Resolved)
    # 换了库，它那条版本判断作废；落到检索时按新库的现行版本走（3.0）
    assert (outcome.game_id, outcome.version) == (OTHER_ID, "")
    prompt = prompt_of(answered(outcome, chunks, both_bases(), llm), llm)
    assert "燕云的正文" in prompt
    assert "黑神话的正文" not in prompt


def test_候选来源函数直接给出语料里的取值():
    """两条候选读取的口径单列出来钉一下：调用方换了也还是这两条。"""
    chunks = store(make_chunk(1, version="1.0"), make_chunk(2, version="2.0"))

    assert [choice.value for choice in game_choices(both_bases())] == [GAME_ID, OTHER_ID]
    assert [choice.value for choice in version_choices(chunks, GAME_ID)] == ["1.0", "2.0"]


def test_两个库重名时候选标签带上_id_以免认错():
    """标签是模型与用户认的依据。两个库同名，标签分不开就会认到另一个库上去。"""
    same_name = docs(black_myth={"name": "同一个名字"}, yanyun={"name": "同一个名字"})

    labels = [choice.label for choice in game_choices(same_name)]

    assert labels == [f"同一个名字（{GAME_ID}）", f"同一个名字（{OTHER_ID}）"]


# --- 恢复：从暂停点继续 ---


def test_用户选完从暂停点继续落到那个库上():
    """选完继续：落脚点指向用户选的那一项，而且**不再问一遍模型**。"""
    chunks = store(make_chunk(1, content=QUESTION, version="1.0", doc_title="二郎神"))
    llm = FakeLlm(joint_reply(game_confidence=UNSURE), REPLY)
    asker = clarifier(chunks, both_bases(), llm=llm)

    pending = asker.decide(QUESTION)
    resolved = asker.resolve(pending.pending_id, "黑神话·悟空")

    assert resolved.game_id == GAME_ID
    assert len(llm.calls) == 1  # 只有理解那一次：恢复不重新理解
    answer = answered(resolved, chunks, both_bases(), llm)
    assert answer.text == REPLY
    assert [citation.doc_title for citation in answer.citations] == ["二郎神"]


def test_版本还原样回到检索里():
    """选了哪个版本，检索就按哪个版本过滤——「选完继续」的意思正在这里。"""
    llm = FakeLlm(joint_reply(version_confidence=UNSURE), REPLY)
    chunks = store(
        make_chunk(1, content="旧版的正文", version="1.0"),
        make_chunk(2, content="新版的正文", version="2.0"),
    )
    asker = clarifier(chunks, both_bases(), llm=llm)

    pending = asker.decide(QUESTION, game_id=GAME_ID)
    resolved = asker.resolve(pending.pending_id, "2.0")

    prompt = prompt_of(answered(resolved, chunks, both_bases(), llm), llm)
    assert "新版的正文" in prompt
    assert "旧版的正文" not in prompt


def test_选回同一个库时已经确定的版本照旧算数():
    """模型判出的版本是照那个库的版本列表判的，用户点的还是它，就没有理由作废。

    不作数的话，一次**确定**的判断会在点完游戏之后凭空消失，答案悄悄换了个版本过滤。
    """
    llm = FakeLlm(joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    chunks = store(
        make_chunk(1, content="旧版的正文", version="1.0"),
        make_chunk(2, content="新版的正文", version="2.0"),
        make_chunk(3, content="燕云的正文", version="3.0", game_id=OTHER_ID),
    )
    asker = clarifier(chunks, both_bases(), llm=llm)

    pending = asker.decide(QUESTION, game_id=GAME_ID)
    resolved = asker.resolve(pending.pending_id, "黑神话·悟空")

    assert resolved.version == "1.0"
    prompt = prompt_of(answered(resolved, chunks, both_bases(), llm), llm)
    assert "旧版的正文" in prompt
    assert "新版的正文" not in prompt


def test_换成另一个库时那条版本判断作废():
    """同一个暂停点里换成另一个库：拿 A 库判出的版本去检索 B 库只会查空。"""
    llm = FakeLlm(joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    chunks = store(
        make_chunk(1, content="旧版的正文", version="1.0"),
        make_chunk(2, content="新版的正文", version="2.0"),
        make_chunk(3, content="燕云的正文", version="3.0", game_id=OTHER_ID),
    )
    asker = clarifier(chunks, both_bases(), llm=llm)

    pending = asker.decide(QUESTION, game_id=GAME_ID)
    resolved = asker.resolve(pending.pending_id, "燕云十六声")

    # 燕云那个库的现行版本是 3.0，判作废之后就按它走
    assert (resolved.game_id, resolved.version) == (OTHER_ID, "")
    prompt = prompt_of(answered(resolved, chunks, both_bases(), llm), llm)
    assert "燕云的正文" in prompt
    assert "旧版的正文" not in prompt


def test_恢复多次结果一致且不多写记录():
    """§3.4 的坑：恢复会把那一步从头重跑，副作用必须外提或幂等。

    这里的做法是**恢复这条路上一个写操作都没有**：待澄清记录只在暂停的那一刻写一条，
    恢复多少次都是拿它算一遍。这条用例把「结果一致」与「记录没多出来」一起钉住。
    """
    asker = clarifier(
        store(make_chunk(1, content=QUESTION, version="1.0")),
        both_bases(),
        joint_reply(game_confidence=UNSURE),
    )

    pending = asker.decide(QUESTION)
    first = asker.resolve(pending.pending_id, "黑神话·悟空")
    second = asker.resolve(pending.pending_id, "黑神话·悟空")

    assert first == second
    assert asker.docs.list_ids(PENDING_COLLECTION) == [pending.pending_id]


def test_选了候选之外的东西当场报错():
    """按钮之外的值说明这次请求不是这份暂停点发出来的，不能拿它去检索。"""
    asker = clarifier(store(), both_bases(), joint_reply(game_confidence=UNSURE))

    pending = asker.decide(QUESTION)

    with pytest.raises(NotACandidate):
        asker.resolve(pending.pending_id, "塞尔达传说")


def test_恢复一个不存在的暂停点当场报错():
    asker = clarifier(store(), both_bases())

    with pytest.raises(UnknownPending):
        asker.resolve("没有这个暂停点", "黑神话·悟空")


# --- 说不通的几种 ---


def test_空问题不反问也不调模型():
    """空问题连理解都不必做：照它问下去会得到一次「你问的是哪款游戏」的反问。"""
    asker = clarifier(store(), both_bases())

    with pytest.raises(ValueError):
        asker.decide("   ")

    assert asker.llm.calls == []


def test_暂停点里存着恢复所需的那几样():
    """恢复不该重新问一遍模型：改写后的问法、已经定下来的取值、候选都存下来。"""
    asker = clarifier(
        store(), both_bases(), joint_reply(game_confidence=UNSURE, rewritten_query="二郎神 怎么打")
    )

    pending = asker.decide(QUESTION)
    saved = asker.docs.get(PENDING_COLLECTION, pending.pending_id)

    assert saved == {
        "rewritten_query": "二郎神 怎么打",
        "dimension": GAME,
        "game_id": "",
        "version": "",
        "choices": [
            {"label": "黑神话·悟空", "value": GAME_ID},
            {"label": "燕云十六声", "value": OTHER_ID},
        ],
        # 选路要它：恢复那一轮不再问一次模型，查询类型只能一并存下来
        "query_type": "",
    }
    assert len(asker.llm.calls) == 1


def test_候选条数少时也不截断候选():
    """反问的候选是**按钮**，不是交给生成的资料：`MAX_CHUNKS` 那道截断管不到这里。

    两个上界挨着写在一处很容易被当成同一个，所以钉一条：候选给多少就是多少。
    """
    many = docs(
        **{f"game_{index:02d}": {"name": f"游戏{index}"} for index in range(MAX_CHUNKS + 3)}
    )

    assert len(game_choices(many)) == MAX_CHUNKS + 3
