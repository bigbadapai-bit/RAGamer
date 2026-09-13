"""查询扩展：多查询改写与 HyDE。

两件事做的是同一件事——把一个问题变成几段**可检索的文本**。这里断的是接线与边界：
调了几次模型、扩出来的能不能用、与原问重复的有没有去掉、失败有没有照抛。
改写本身好不好是模型的事，假件证不了（与 `ragamer.query` 那边的分工一样）。

去重那几条尤其要紧：没去掉的表现是**白跑一趟检索**，而它还顺手让原问在 RRF 里投出
两票——两处都不报错，只是候选悄悄变了。
"""

from __future__ import annotations

import logging

import pytest

from ragamer.expanding import REWRITE_COUNT, TEMPERATURE, distinct, hypothetical, rewrite
from ragamer.llm import FakeLlm, LlmTimeout

QUESTION = "二郎神怎么打"


# --- 多查询改写 ---


def test_一次调用拿回若干问法():
    llm = FakeLlm({"queries": ["二郎神 打法", "妖王 怎么打", "二郎神 打法流程"]})

    assert rewrite(QUESTION, llm=llm) == ("二郎神 打法", "妖王 怎么打", "二郎神 打法流程")
    assert len(llm.calls) == 1


def test_温度钉死零():
    """扩出来的问法直接决定候选取回什么：同一个问题两次问出不同的候选池，答案就会飘。"""
    llm = FakeLlm({"queries": ["二郎神 打法"]})

    rewrite(QUESTION, llm=llm)

    assert llm.calls[0].temperature == TEMPERATURE == 0.0


def test_与原问同形的问法被去掉():
    """主检索路已经拿原问查过一遍了，扩出来的再查一遍是白跑一趟。"""
    llm = FakeLlm({"queries": ["二郎神怎么打", "二郎神 打法"]})

    assert rewrite(QUESTION, llm=llm) == ("二郎神 打法",)


def test_归一只压平空白不动字面():
    """比的是 `ragamer.query.normalize_query` 那个口径，与缓存 key 同一套——
    两处对「同一个问法」的判断不能分叉。

    所以多打几个空格算同一个问法，而「二郎神怎么打」与「二郎神 怎么打」算两个：
    后者要合并得先标定语义阈值，那是缓存层的事，这里不做（与 `normalize_query`
    自己的说明同一条口径）。"""
    llm = FakeLlm({"queries": ["  二郎神   怎么打 ", "二郎神 打法"]})

    assert rewrite("二郎神 怎么打", llm=llm) == ("二郎神 打法",)
    assert distinct("二郎神 怎么打", ["二郎神怎么打"]) == ("二郎神怎么打",)


def test_自己之间也不重复():
    llm = FakeLlm({"queries": ["二郎神 打法", "二郎神  打法", "妖王 怎么打"]})

    assert rewrite(QUESTION, llm=llm) == ("二郎神 打法", "妖王 怎么打")


def test_条数有上限():
    llm = FakeLlm({"queries": [f"二郎神 打法{index}" for index in range(9)]})

    assert len(rewrite(QUESTION, llm=llm, count=2)) == 2


def test_要零条就不调模型():
    """空手要一次调用没有意义。"""
    llm = FakeLlm()

    assert rewrite(QUESTION, llm=llm, count=0) == ()
    assert llm.calls == []


def test_空问题不调模型():
    """空问题的毛病与它在检索那一步一样：会查出任意一批切片。"""
    llm = FakeLlm()

    assert rewrite("   ", llm=llm) == ()
    assert hypothetical("   ", llm=llm) == ""
    assert llm.calls == []


def test_什么都没扩出来时留痕(caplog):
    """一次成功的调用却什么也没扩出来：多半是提示词没交代清楚或者模型没照做，
    而它不报错，只是这一路白跑一趟。"""
    llm = FakeLlm({"queries": ["二郎神怎么打"]})

    with caplog.at_level(logging.WARNING, logger="ragamer.expanding"):
        assert rewrite(QUESTION, llm=llm) == ()

    assert any("没扩出" in record.getMessage() for record in caplog.records)


# --- HyDE ---


def test_假想答案原样带回来():
    llm = FakeLlm("二郎神是隐藏 BOSS，血量 8000。")

    assert hypothetical(QUESTION, llm=llm) == "二郎神是隐藏 BOSS，血量 8000。"
    assert llm.calls[0].temperature == TEMPERATURE


def test_一个字都没写出来时留痕(caplog):
    """空串拿去向量化会查出任意一批切片，与空问题是同一个毛病。"""
    llm = FakeLlm("   ")

    with caplog.at_level(logging.WARNING, logger="ragamer.expanding"):
        assert hypothetical(QUESTION, llm=llm) == ""

    assert any("一个字" in record.getMessage() for record in caplog.records)


def test_提示词要求它像资料而不是像回答():
    """写成像回答，模型就会去回答用户；要的是「一段可能出现在攻略里的文字」。
    这一条是提示词的接线：假件证不了模型听不听，只能证明话交代到了。"""
    llm = FakeLlm("二郎神是隐藏 BOSS。")

    hypothetical(QUESTION, llm=llm)

    instruction = llm.calls[0].messages[0].content
    assert "不是回答用户" in instruction


# --- 失败与边界 ---


def test_模型失败照抛():
    """处置由检索那一层的隔离决定：这一层不自己降级成「返回原问」——那样调用方就
    分不清「改写失败」与「改写没扩出东西」了。"""
    with pytest.raises(LlmTimeout):
        rewrite(QUESTION, llm=FakeLlm(LlmTimeout("超时")))
    with pytest.raises(LlmTimeout):
        hypothetical(QUESTION, llm=FakeLlm(LlmTimeout("超时")))


def test_提示词里写明了要几条():
    """不说条数，模型多半只给一条——那这一路就退化成「换一种说法查一遍」。"""
    llm = FakeLlm({"queries": ["二郎神 打法"]})

    rewrite(QUESTION, llm=llm)

    assert str(REWRITE_COUNT) in llm.calls[0].messages[0].content


def test_空白候选一律丢掉():
    assert distinct(QUESTION, ["", "   ", "二郎神 打法"]) == ("二郎神 打法",)
