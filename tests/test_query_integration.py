"""集成测试：真的调模型，看改写与判定在中文上是不是真的有效。默认不跑。

    uv run pytest -m integration

假件覆盖不到的是**效果**：「那它怎么打」是不是真的被补成了带主体的规范问法——这条写在
T13 的验收项里，只有真模型能证。形状与接线在默认测试里已经钉过了。

**「两次问出同一个改写」不在验收项里**：改写不稳定是模型的性质，不是我们能控的，
理由见 `test_改写结果总是可用的形态`。

跑之前需要一份填好的 `.env`，而且**每跑一次都真的花钱**：这几个用例一共发四次请求。
"""

from __future__ import annotations

import pytest

from ragamer.config import ConfigError, load_settings
from ragamer.container import build_container
from ragamer.llm import LlmClient, Message
from ragamer.query import normalize_query, understand

pytestmark = pytest.mark.integration

#: 真实存在的候选。节点只许从这里取，取不到就留空。
GAMES = ["黑神话·悟空", "燕云十六声"]
VERSIONS = ["1.0", "2.0"]

#: 上一轮的话。「那它怎么打」里的「它」指的是谁，只能从这里看出来。
HISTORY = [
    Message("user", "二郎神是谁"),
    Message("assistant", "二郎神是黑神话·悟空里的隐藏 BOSS，在第三章的隐藏区域出现。"),
]


@pytest.fixture(scope="module")
def llm() -> LlmClient:
    try:
        settings = load_settings()
    except ConfigError as exc:
        pytest.skip(f"没有可用的配置，跳过集成测试：{exc}")
    return build_container(settings).llm


def test_指代被补成带主体的规范问法(llm: LlmClient):
    """「那它怎么打」要变成带主体的问法——这是问法归一有效那条验收项的真章。"""
    result = understand("那它怎么打", llm=llm, games=GAMES, versions=VERSIONS, history=HISTORY)

    assert "二郎神" in result.rewritten_query, f"主体名没补上：{result.rewritten_query!r}"
    assert result.rewritten_query != "那它怎么打"


def test_改写结果总是可用的形态(llm: LlmClient):
    """改写**不保证**两次一样，这里只钉我们真正保证得了的那部分。

    原先这条断言的是「两次问出同一个字符串」，理由是「改写结果要能当缓存 key 用」。
    实测它不成立：`deepseek-flash` 这类推理模型在温度 0 下仍不确定，同一个问题问 6 次
    得到 3 种改写（`query.py` 的 `TEMPERATURE = 0` 确实发出去了）。缓存键因此会跟着抖，
    这是权衡之后**接受**的取舍，不是待修的缺陷——见 `CachedAnswerer._key` 的说明。

    能保证的是改写结果本身可用：非空、空白压平、不丢主体。这三条是缓存键与检索都吃的。
    """
    result = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.rewritten_query, "改写不能是空的"
    assert result.rewritten_query == normalize_query(result.rewritten_query), "改写要压平空白"
    assert "二郎神" in result.rewritten_query, f"主体名丢了：{result.rewritten_query!r}"


def test_判出的游戏与版本都落在候选里(llm: LlmClient):
    """候选之外的游戏名——模型编的那一类——一律留空，不会漏到调用方手里。"""
    result = understand("黑神话 2.0 版二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.game in ["", *GAMES]
    assert result.version in ["", *VERSIONS]
