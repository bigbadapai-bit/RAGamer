"""生成：把检索到的切片交给模型，产出带引用的答案。

五组用例：**缝合**（预置一批候选，断言引用的顺序与条数——这是本票的验收口径）、
**聚合父块**（命中细粒度切片，交给生成的是整页）、**检索不到时**（明确回复，不硬编）、
**版本过滤**（所选版本与未标注版本一并纳入）、**边界与失败**（空问题、生成失败、
越界的编号）。模型接 `FakeLlm`（一次网络都不发），检索那一段接内存假件与确定性假件。

引用是**答案可核对的前提**：答案里写了什么，得能顺着编号找回原文。所以这里断言的
不只是「有几条引用」，还有顺序（编号与一个个父块一一对应）与标签（标题 + 祖先标题路径）。
"""

from __future__ import annotations

import logging

import pytest

from ragamer.answering import MAX_IMAGES, NOT_FOUND, TEMPERATURE, Answer, Answerer, Citation
from ragamer.llm import FakeLlm, LlmTimeout, Message
from ragamer.retrieval import MAX_PARENT_CHARS
from ragamer.routing import RecallPath, Route
from ragamer.stores.base import UNVERSIONED
from ragamer.stores.memory import InMemoryChunkStore
from ragamer.vectors.fake import FakeEmbedder, FakeReranker
from ragamer.websearch import FakeWebSearch, WebResult

from .conftest import RecordingChunkStore, ScriptedReranker, chunk_store, make_chunk

GAME = "black_myth"
#: 问题里的字全落在这几条正文里，所以它们与问题的词重合度一样、分数并列。
#: 并列时按切片序号定序（见 `ragamer.retrieval`），引用顺序因此是确定的。
QUESTION = "二郎神怎么打"

#: 一段答案。编号指的就是提示词里那批切片的编号。
REPLY = "先定身再贴身输出[1]，二阶段躲开红光[2]。"


#: 四份切片同属一篇文档：前三份与问题高度重合，第四份只有「二郎神」三个字重合。
#: 第四份与第三份之间差 0.5 分，是断崖——它自己不进命中，但它所属的文档进了，
#: 于是它仍然随父块一起交给生成（这正是父子块要的效果）。
BOSS_CHUNKS = (
    make_chunk(
        1, content="二郎神怎么打：先定身", doc_title="二郎神", ancestor_path="二郎神 › 打法"
    ),
    make_chunk(
        2,
        content="二郎神怎么打的第二阶段",
        doc_title="二郎神",
        ancestor_path="二郎神 › 打法 › 第二阶段",
    ),
    make_chunk(
        3,
        content="二郎神怎么打的逃课打法",
        doc_title="二郎神",
        ancestor_path="二郎神 › 打法 › 逃课",
    ),
    make_chunk(
        4, content="二郎神的获取方式", doc_title="二郎神", ancestor_path="二郎神 › 获取方式"
    ),
)


def answerer(store: InMemoryChunkStore, llm, reranker=None, search=None) -> Answerer:
    """整条读取链的内存版：假向量、假精排、假模型，一行云端代码都不碰。

    默认的精排按词重合度打分；要摆出**指定的分数落差**（断崖）时换个 `ScriptedReranker`。
    `search` 不给就是没配联网那一路，与「整组配置没填」同一个状态。
    """
    return Answerer(
        chunks=store,
        embedder=FakeEmbedder(),
        reranker=reranker or FakeReranker(),
        llm=llm,
        search=search,
    )


# --- 缝合 ---


def test_预置一批候选时引用的顺序与条数():
    """引用按命中顺序编号，**一篇文档一条**——同一文档命中多条不会重复占号。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="二郎神怎么打：先定身", doc_title="二郎神"),
        make_chunk(2, content="二郎神怎么打的逃课打法", doc_title="二郎神"),
        make_chunk(3, content="二郎神怎么打的外传", doc_title="二郎神外传"),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert [citation.index for citation in answer.citations] == [1, 2]
    assert [citation.label for citation in answer.citations] == ["二郎神", "二郎神外传"]
    assert answer.text == REPLY


def test_引用的编号与提示词里的父块一一对应():
    """清单上第 2 条要能顺着编号找回第二个父块——编号是给人核对来源列表用的。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="二郎神怎么打：先定身", doc_title="二郎神"),
        make_chunk(2, content="二郎神怎么打的外传", doc_title="二郎神外传"),
    )
    llm = FakeLlm(REPLY)

    answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    system = llm.calls[0].messages[0]
    assert system.role == "system"
    for citation in (Citation(1, "二郎神", ""), Citation(2, "二郎神外传", "")):
        assert f"[{citation.index}] {citation.label}" in system.content
    assert system.content.index("[1]") < system.content.index("[2]")
    assert llm.calls[0].messages[1] == Message("user", QUESTION)


def test_提示词不让把编号写进正文():
    """编号是给清单对号、给人核对来源列表用的，**不进正文**：挂在句末既指不出是
    清单上哪一条，又铺满整段——一段答案全出自同一个父块时，每一行都带 `[1]`。
    来源由界面单独列出来（`turns.html` 那份清单），正文里不需要再挂一遍。
    """
    llm = FakeLlm(REPLY)

    answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert "不要写引用编号" in llm.calls[0].messages[0].content


def test_附加文本也进提示词():
    """表格里整列降级进 `content_meta` 的长文本**必须带给模型**：它不参与向量化，
    再不给模型就等于整列丢掉（架构文档 §2.2 / §2.5）。"""
    store = chunk_store(
        GAME, make_chunk(1, content="二郎神怎么打", content_meta="| 说明 | 先定身 |")
    )
    llm = FakeLlm(REPLY)

    answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert "先定身" in llm.calls[0].messages[0].content


def test_生成温度钉零():
    """答案要照着给定的内容写，不追求多样性；温度 0 才能让同一个问题两次问出同一个答案。"""
    llm = FakeLlm(REPLY)

    answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert llm.calls[0].temperature == TEMPERATURE == 0.0


# --- 网络来源 ---

#: 联网那条路搜回来的一条。时效型问题本地语料答不上来，靠的就是它。
WEB_RESULT = WebResult(
    "1.1 版本更新公告",
    "https://example.com/patch",
    "金箍棒的基础伤害下调，新增两件套装。",
    "2026-01-02",
)

#: 只有联网那一路时的一次提问：本地库是空的。
WEB_ONLY = Route((RecallPath.WEB,))


def web(llm, *results: WebResult) -> Answerer:
    """只接上联网那一路的读取链。"""
    return answerer(InMemoryChunkStore(), llm, search=FakeWebSearch(list(results) or [WEB_RESULT]))


def test_网络来源在提示词里带前缀():
    """网络内容与语料混在一份答案里而不标出来，等于把「这是我们语料里写的」与
    「这是网上说的」说成同一件事。"""
    llm = FakeLlm("金箍棒的基础伤害下调了[1]。")

    web(llm).answer(QUESTION, game_id=GAME, route=WEB_ONLY)

    system = llm.calls[0].messages[0].content
    assert "【网络】" in system
    assert WEB_RESULT.url in system
    assert WEB_RESULT.text in system


def test_提示词交代了网络内容要说明出处():
    """模型不交代出处，读的人就分不清哪句是查到的、哪句是搜来的。"""
    llm = FakeLlm("金箍棒改了[1]。")

    web(llm).answer(QUESTION, game_id=GAME, route=WEB_ONLY)

    assert "说明这是网络上的说法" in llm.calls[0].messages[0].content


def test_引用带上地址与来路():
    """`url` 非空即网络来源——界面据此标出来，读的人据此判断可信度。"""
    llm = FakeLlm("金箍棒的基础伤害下调了[1]。")

    answer = web(llm).answer(QUESTION, game_id=GAME, route=WEB_ONLY)

    citation = answer.citations[0]
    assert (citation.index, citation.origin, citation.url) == (1, "web", WEB_RESULT.url)
    assert citation.label == f"{WEB_RESULT.title}（{WEB_RESULT.url}，{WEB_RESULT.published_at}）"


def test_只有网络来源时也作答():
    """本地一条都没有、网上有：这一路的意义就在这里，别按「没找到」处理。"""
    reply = "金箍棒的基础伤害下调了[1]。"
    llm = FakeLlm(reply)

    answer = web(llm).answer(QUESTION, game_id=GAME, route=WEB_ONLY)

    assert answer.text == reply
    assert answer.text != NOT_FOUND
    assert [citation.origin for citation in answer.citations] == ["web"]


def test_本地没配联网时按没找到处理():
    """没配那一组配置时这一路直接跳过——不是「搜了但没有结果」，是根本没有这一路。"""
    llm = FakeLlm()

    answer = answerer(InMemoryChunkStore(), llm).answer(QUESTION, game_id=GAME, route=WEB_ONLY)

    assert answer == Answer(NOT_FOUND, ())
    assert llm.calls == []


def test_语料在前网络在后且编号连续():
    """模型看到的是一份资料清单：断号会让它以为中间还有没给它的东西。"""
    store = chunk_store(GAME, make_chunk(1, content="二郎神怎么打：先定身", doc_title="二郎神"))
    llm = FakeLlm("先定身[1]，另外公告说改了[2]。")

    answer = answerer(store, llm, search=FakeWebSearch([WEB_RESULT])).answer(
        QUESTION, game_id=GAME, route=Route((RecallPath.MAIN, RecallPath.WEB))
    )

    assert [(citation.index, citation.origin) for citation in answer.citations] == [
        (1, "local"),
        (2, "web"),
    ]


def test_语料里的引用没有地址():
    """语料是导进来的资料，没有「原文地址」可给——编一个出来比留空更糟。"""
    store = chunk_store(GAME, make_chunk(1, content="二郎神怎么打：先定身", doc_title="二郎神"))
    llm = FakeLlm("先定身[1]。")

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME)

    citation = answer.citations[0]
    assert (citation.origin, citation.url) == ("local", "")
    assert citation.label == "二郎神"


# --- 聚合父块 ---


def test_交给生成的是命中切片所属的整页():
    """问「二郎神怎么打」，模型拿到的却是整页——包括被断崖切掉的「获取方式」。
    追问「怎么获得」不必重新检索，这就是父子块的意义（§2.5）。"""
    llm = FakeLlm(REPLY)

    answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(QUESTION, game_id=GAME, version="1.0")

    system = llm.calls[0].messages[0].content
    assert "二郎神的获取方式" in system
    # 顺序与源文档一致：块内按 chunk_index 升序，与命中的先后无关
    assert system.index("二郎神怎么打：先定身") < system.index("二郎神怎么打的第二阶段")


def test_整页只有一条引用():
    """四条切片同属一篇文档，引用就只有一条——引用指向文档，不是指向某一句话。"""
    llm = FakeLlm(REPLY)

    answer = answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(
        QUESTION, game_id=GAME, version="1.0"
    )

    assert [citation.index for citation in answer.citations] == [1]
    assert [citation.label for citation in answer.citations] == ["二郎神"]


def test_被断崖截断的另一个文档不回查():
    """**聚合发生在截断之后**：断崖切掉的切片所属的文档不该被回查，
    否则会为即将被丢弃的切片白拼一遍父块。"""
    store = RecordingChunkStore()
    store.upsert(
        GAME,
        [
            make_chunk(1, content=QUESTION, doc_title="二郎神"),
            make_chunk(2, content="无关的一条", doc_title="世界观"),
        ],
    )
    llm = FakeLlm(REPLY)

    answerer(store, llm, reranker=ScriptedReranker({QUESTION: 0.9, "无关的一条": 0.2})).answer(
        QUESTION, game_id=GAME, version="1.0"
    )

    assert store.fetched == ["二郎神"]


def test_超长文档只交命中所在的小节():
    """整页超过上限就收敛到命中的那一节，不把整页喂进去（§2.5 第三条约束）。"""
    filler = "长" * MAX_PARENT_CHARS
    store = chunk_store(
        GAME,
        make_chunk(1, content=f"{QUESTION}{filler}", ancestor_path="二郎神 › 背景"),
        make_chunk(2, content=f"{QUESTION}{filler}", ancestor_path="二郎神 › 打法"),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert [citation.label for citation in answer.citations] == [
        "二郎神 › 背景",
        "二郎神 › 打法",
    ]


# --- 检索不到时 ---


def test_库里没有相关内容时给明确回复():
    """**不硬编一个答案**：没有内容可依据时不调模型——让它自由发挥只会得到一段
    编造的游戏攻略。回复是这里的常量，不是模型写的。"""
    llm = FakeLlm()  # 一条脚本都没排，真被调用会当场炸

    answer = answerer(InMemoryChunkStore(), llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.text == NOT_FOUND
    assert answer.citations == ()
    assert llm.calls == []


def test_版本过滤把候选滤空时也给明确回复():
    store = chunk_store(GAME, make_chunk(1, content=QUESTION, version="2.0"))
    llm = FakeLlm()

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.text == NOT_FOUND
    assert llm.calls == []


def test_命中却聚合不出父块时按检索不到处理():
    """索引与数据对不上：检索命中了，按 `doc_title` 回查却一条都拿不到。
    这时手里没有任何内容可依据，同样不调模型——让它自由发挥只会得到一段编造的游戏攻略。"""

    class Vanishing(InMemoryChunkStore):
        def fetch_document(self, game_id: str, doc_title: str, *, version: str | None):
            return []

    store = Vanishing()
    store.upsert(GAME, [make_chunk(1, content=QUESTION)])
    llm = FakeLlm()  # 一条脚本都没排，真被调用会当场炸

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.text == NOT_FOUND
    assert answer.citations == ()
    assert llm.calls == []


# --- 版本过滤 ---


def test_所选版本与未标注版本一起检索到且不混版本():
    """漏掉「未标注版本」那一支，用户切到历史版本后世界观类问题会全部答不出（ADR-0004）；
    而回查兄弟切片若不带同一个版本过滤，父块里会静默混进另一个版本的正文。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content=QUESTION, ancestor_path="二郎神 › 打法", version="1.0"),
        make_chunk(2, content="2.0 才有的打法", ancestor_path="二郎神 › 打法（旧）", version="2.0"),
        make_chunk(
            3,
            content=QUESTION,
            doc_title="世界观",
            ancestor_path="世界观 › 二郎神",
            version=UNVERSIONED,
        ),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert [citation.label for citation in answer.citations] == ["二郎神", "世界观"]
    assert "2.0 才有的打法" not in llm.calls[0].messages[0].content


def test_问题没点名版本时用知识库标的现行版本():
    """`knowledge_bases` 是「当前该用哪个版本」的唯一真相来源，检索不自己维护一份。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="初版的打法", ancestor_path="二郎神 › 打法 · 初版", version="1.0"),
        make_chunk(2, content=QUESTION, ancestor_path="二郎神 › 打法 · 新版", version="2.0"),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, current_version="2.0")

    assert [citation.label for citation in answer.citations] == ["二郎神"]
    assert "初版的打法" not in llm.calls[0].messages[0].content


# --- 图片 ---


def test_答案带出切片自己的图片地址():
    """图片地址取自切片自己的字段，随答案一起交回，用户不必跳出去找原图。

    **不是从正文里扫出来的**：地址在切分时就摘走了（`ragamer.chunking`），
    正文里只有替代文本。落在 `content_meta` 里的图（表格的长文本列整列降级在那里）
    同样进这个字段，所以这里只认字段、不看它来自哪一段正文。"""
    store = chunk_store(
        GAME,
        make_chunk(
            1,
            content="二郎神怎么打\n打法",
            content_meta="| 图 | 第二形态 |",
            image_urls=("images/black_myth/ab12cd/boss.jpg", "images/black_myth/ab12cd/phase2.jpg"),
        ),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.images == (
        "images/black_myth/ab12cd/boss.jpg",
        "images/black_myth/ab12cd/phase2.jpg",
    )


def test_同一张图出现两次只交回一次():
    store = chunk_store(
        GAME,
        make_chunk(
            1,
            content="二郎神怎么打",
            chunk_index=1,
            image_urls=("images/black_myth/ab12cd/boss.jpg",),
        ),
        make_chunk(
            2,
            content="第二阶段",
            chunk_index=2,
            image_urls=("images/black_myth/ab12cd/boss.jpg",),
        ),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.images == ("images/black_myth/ab12cd/boss.jpg",)


def test_检索不到时没有图片与引用():
    llm = FakeLlm()

    answer = answerer(InMemoryChunkStore(), llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.images == ()


def test_至多带出_MAX_IMAGES_张图():
    """聚合的是父块而不是命中的那几句，一个词条页整页进来时能带十几张——全铺在答案
    下面会把正文淹掉。截的是**前**几张：顺序跟着引用走，头几张就是最相关那几个父块里的。
    """
    store = chunk_store(
        GAME,
        make_chunk(
            1,
            content="二郎神怎么打",
            image_urls=tuple(f"images/black_myth/ab12cd/{index}.jpg" for index in range(20)),
        ),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert len(answer.images) == MAX_IMAGES
    assert answer.images[0] == "images/black_myth/ab12cd/0.jpg"


def test_不是对象_key_的地址不进答案():
    """回显是拿地址去对象存储取的，所以只有对象 key 取得到原图。库里留着一条不是 key
    的地址（这一层加上之前导进去的图标外链），答案里就多一条死图——页面只能报
    「没有这张图」，而那是导入时说过的原因。
    """
    store = chunk_store(
        GAME,
        make_chunk(
            1,
            content="二郎神怎么打",
            image_urls=(
                "https://patchwiki.biligame.com/images/wukong/thumb/b/b1/x.png/18px-图标.png",
                "images/a.png",
                "images/black_myth/ab12cd/boss.jpg",
            ),
        ),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, version="1.0")

    assert answer.images == ("images/black_myth/ab12cd/boss.jpg",)


# --- 边界与失败 ---


def test_空问题当场报错():
    """空问题会让检索查出任意一批切片，模型照着它编一段答案——这种失败要能立刻看见。"""
    with pytest.raises(ValueError):
        answerer(chunk_store(GAME, *BOSS_CHUNKS), FakeLlm(REPLY)).answer("   ", game_id=GAME)


def test_生成失败照抛不返回半个答案():
    """没有答案就是没有答案：降级成一段「抱歉我答不上来」比报错更难查。"""
    llm = FakeLlm(LlmTimeout("模型超时"))

    with pytest.raises(LlmTimeout):
        answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(QUESTION, game_id=GAME, version="1.0")


def test_答案里出现范围外的编号时留痕(caplog):
    """模型写了个不存在的 [9]：答案就无从核对了，而它自己不会报错。"""
    llm = FakeLlm("先定身[1]，三阶段有隐藏机制[9]。")

    with caplog.at_level(logging.WARNING):
        answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(QUESTION, game_id=GAME, version="1.0")

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("[9]" in message for message in warnings)


def test_引用标签不重复文档标题():
    """祖先标题路径通常以文档标题开头（一级标题就是它），拼起来时不要念两遍。"""
    assert Citation(1, "二郎神", "二郎神 › 打法").label == "二郎神 › 打法"
    assert Citation(1, "二郎神", "").label == "二郎神"
    assert Citation(1, "二郎神", "打法").label == "二郎神 › 打法"


# --- 流式 ---


def test_流式先给引用再逐字给正文():
    """引用在检索那一步就定下来了，正文要等模型——所以引用能先交出去。"""
    store = chunk_store(GAME, *BOSS_CHUNKS)

    stream = answerer(store, FakeLlm(REPLY)).stream(QUESTION, game_id=GAME, version="1.0")
    deltas = list(stream.deltas)

    assert [citation.doc_title for citation in stream.citations] == ["二郎神"]
    assert len(deltas) > 1  # 一片一片来，不是一次给全
    assert all(len(delta) == 1 for delta in deltas)  # 逐字
    assert "".join(deltas) == REPLY


def test_流式与一次给全用的是同一批引用与图片():
    """两条路各拼一遍引用迟早会分岔——而引用对不上内容这件事，从答案本身看不出来。
    图片同理：命中缓存与否会给出两种结果，说的就是这一条。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content="二郎神怎么打", image_urls=("images/black_myth/ab12cd/boss.jpg",)),
    )

    whole = answerer(store, FakeLlm(REPLY)).answer(QUESTION, game_id=GAME, version="1.0")
    piecewise = answerer(store, FakeLlm(REPLY)).stream(QUESTION, game_id=GAME, version="1.0")
    list(piecewise.deltas)

    assert piecewise.citations == whole.citations
    assert piecewise.images == whole.images == ("images/black_myth/ab12cd/boss.jpg",)


def test_流式时没检索到内容回同一段明确回复():
    """两种情况的呈现一样，只是这一种没有引用——不调模型，脚本一条都不用排。"""
    stream = answerer(InMemoryChunkStore(), FakeLlm()).stream(QUESTION, game_id=GAME)

    assert stream.citations == ()
    assert list(stream.deltas) == [NOT_FOUND]


def test_流式空问题当场报错():
    """改写与检索在调用时跑完，所以这一步的失败在开流之前就报出来。"""
    with pytest.raises(ValueError):
        answerer(chunk_store(GAME, *BOSS_CHUNKS), FakeLlm(REPLY)).stream("   ", game_id=GAME)


def test_流式里越界的编号也留痕(caplog):
    """编号检查要整段正文才做得成，而流式这一路没有累积——所以攒一份，收完再查。"""
    llm = FakeLlm("先定身[1]，三阶段有隐藏机制[9]。")

    with caplog.at_level(logging.WARNING):
        stream = answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).stream(
            QUESTION, game_id=GAME, version="1.0"
        )
        list(stream.deltas)

    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("[9]" in message for message in warnings)
