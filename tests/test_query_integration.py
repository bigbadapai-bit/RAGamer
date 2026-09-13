"""集成测试：真的调模型，看改写与判定在中文上是不是真的有效。默认不跑。

    uv run pytest -m integration

假件覆盖不到的是**效果**：「那它怎么打」是不是真的被补成了带主体的规范问法、
同一个问法两次是不是真的问出同一个改写。这两条都写在 T13 的验收项里，
但只有真模型能证——形状与接线在默认测试里已经钉过了。

跑之前需要一份填好的 `.env`，而且**每跑一次都真的花钱**：这几个用例一共发四次请求。
"""

from __future__ import annotations

import pytest

from ragamer.config import ConfigError, load_settings
from ragamer.container import build_container
from ragamer.llm import LlmClient, Message
from ragamer.query import understand

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


def test_同一个问法两次问出同一个改写(llm: LlmClient):
    """改写结果要能当缓存 key 用：温度钉死 0 之后，两次调用该给出同一个字符串。"""
    first = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)
    second = understand("二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert first.rewritten_query == second.rewritten_query


def test_判出的游戏与版本都落在候选里(llm: LlmClient):
    """候选之外的游戏名——模型编的那一类——一律留空，不会漏到调用方手里。"""
    result = understand("黑神话 2.0 版二郎神怎么打", llm=llm, games=GAMES, versions=VERSIONS)

    assert result.game in ["", *GAMES]
    assert result.version in ["", *VERSIONS]
