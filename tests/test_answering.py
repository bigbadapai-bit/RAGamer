"""生成：把检索到的切片交给模型，产出带引用的答案。

四组用例：**缝合**（预置一批候选，断言引用的顺序与条数——这是本票的验收口径）、
**检索不到时**（明确回复，不硬编）、**版本过滤**（所选版本与未标注版本一并纳入）、
**边界与失败**（空问题、生成失败、越界的编号）。模型接 `FakeLlm`（一次网络都不发），
检索那一段接内存假件与确定性假件。

引用是**答案可核对的前提**：答案里写了什么，得能顺着编号找回原文。所以这里断言的
不只是「有几条引用」，还有顺序（编号与一份份切片一一对应）与标签（标题 + 祖先标题路径）。
"""

from __future__ import annotations

import logging

import pytest

from ragamer.answering import NOT_FOUND, TEMPERATURE, Answerer, Citation
from ragamer.llm import FakeLlm, LlmTimeout, Message
from ragamer.stores.base import UNVERSIONED
from ragamer.stores.memory import InMemoryChunkStore
from ragamer.vectors.fake import FakeEmbedder, FakeReranker

from .conftest import chunk_store, make_chunk

GAME = "black_myth"
#: 问题里的字全落在这几条正文里，所以它们与问题的词重合度一样、分数并列。
#: 并列时按切片序号定序（见 `ragamer.retrieval`），引用顺序因此是确定的。
QUESTION = "二郎神怎么打"

#: 一段答案。编号指的就是提示词里那批切片的编号。
REPLY = "先定身再贴身输出[1]，二阶段躲开红光[2]。"


#: 四份切片：前三份与问题高度重合，第四份只有「二郎神」三个字重合。
#: 第四份与第三份之间差 0.5 分，是断崖——它不该被交给生成，也就不该出现在引用里。
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


def answerer(store: InMemoryChunkStore, llm) -> Answerer:
    """整条读取链的内存版：假向量、假精排、假模型，一行云端代码都不碰。"""
    return Answerer(chunks=store, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm)


# --- 缝合 ---


def test_预置一批候选时引用的顺序与条数():
    """断崖截断掉的那条不进引用：引用与交给生成的那批是同一批、同一个顺序。"""
    llm = FakeLlm(REPLY)

    answer = answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(
        QUESTION, game_id=GAME, version="1.0"
    )

    assert [citation.index for citation in answer.citations] == [1, 2, 3]
    assert [citation.label for citation in answer.citations] == [
        "二郎神 › 打法",
        "二郎神 › 打法 › 第二阶段",
        "二郎神 › 打法 › 逃课",
    ]
    assert answer.text == REPLY


def test_引用的编号与提示词里的切片一一对应():
    """正文里的 [2] 要能顺着编号找回第二条切片。"""
    llm = FakeLlm(REPLY)

    answerer(chunk_store(GAME, *BOSS_CHUNKS), llm).answer(QUESTION, game_id=GAME, version="1.0")

    system = llm.calls[0].messages[0]
    assert system.role == "system"
    for citation in (
        Citation(1, "二郎神", "二郎神 › 打法"),
        Citation(2, "二郎神", "二郎神 › 打法 › 第二阶段"),
        Citation(3, "二郎神", "二郎神 › 打法 › 逃课"),
    ):
        assert f"[{citation.index}] {citation.label}" in system.content
    assert system.content.index("[1]") < system.content.index("[2]") < system.content.index("[3]")
    assert llm.calls[0].messages[1] == Message("user", QUESTION)


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


# --- 版本过滤 ---


def test_所选版本与未标注版本一起检索到():
    """漏掉「未标注版本」这一支，用户切到历史版本后世界观类问题会全部答不出，
    而且是静默失效（ADR-0004）。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content=QUESTION, ancestor_path="二郎神 › 打法", version="1.0"),
        make_chunk(2, content=QUESTION, ancestor_path="二郎神 › 打法（旧）", version="2.0"),
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

    assert [citation.label for citation in answer.citations] == [
        "二郎神 › 打法",
        "世界观 › 二郎神",
    ]


def test_问题没点名版本时用知识库标的现行版本():
    """`knowledge_bases` 是「当前该用哪个版本」的唯一真相来源，检索不自己维护一份。"""
    store = chunk_store(
        GAME,
        make_chunk(1, content=QUESTION, ancestor_path="二郎神 › 打法 · 初版", version="1.0"),
        make_chunk(2, content=QUESTION, ancestor_path="二郎神 › 打法 · 新版", version="2.0"),
    )
    llm = FakeLlm(REPLY)

    answer = answerer(store, llm).answer(QUESTION, game_id=GAME, current_version="2.0")

    assert [citation.label for citation in answer.citations] == ["二郎神 › 打法 · 新版"]


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
