"""集成测试：真的打一次博查，看请求形状与响应解析对不对得上。默认不跑。

    uv run pytest -m integration

**这一路尤其需要真跑一次**：适配器是按博查的接口文档写的，而形状对不对只有真发一次
请求才证得了——假件证的只是「我按自己以为的形状发出去、也按自己以为的形状读回来」，
两者可以一起错。发出去形状不对时服务端多半给 400，读回来形状不对时这一路静默返回空
（那正是「没配」与「搜了但没结果」最容易混起来的地方）。

跑之前 `.env` 里要有 `RAGAMER_SEARCH_API_KEY`，**每跑一次都真的花钱**（约 ¥0.04/次）。
没配密钥时自动跳过——这一组是整组可选的配置。
"""

from __future__ import annotations

import pytest

from ragamer.config import ConfigError, load_settings
from ragamer.container import build_container
from ragamer.websearch import WebSearch

pytestmark = pytest.mark.integration

#: 一个时效性很强的问法：问的正是本地语料答不了、这一路专门要补的那一类。
QUERY = "黑神话悟空 最新版本更新了什么"


@pytest.fixture(scope="module")
def search() -> WebSearch:
    try:
        settings = load_settings()
    except ConfigError as exc:
        pytest.skip(f"没有可用的配置，跳过集成测试：{exc}")
    if settings.search.api_key is None:
        pytest.skip("没有配 RAGAMER_SEARCH_API_KEY，这一路整组是可选配置")
    built = build_container(settings).search
    assert built is not None  # 配了密钥就一定有这一路
    return built


def test_搜得回结果且每条都能读出标题与地址(search: WebSearch):
    """形状不对的两种表现都在这一条里：发出去不对会被拒（异常），
    读回来不对会返回空（静默）。"""
    found = search.search(QUERY, limit=5)

    assert found, "一条都没搜到：先确认是接口形状变了还是这个问法真的没结果"
    for result in found:
        # 地址一定非空——没有地址的那些在适配器里就被丢掉了（引用要能点开）
        assert result.url.startswith("http"), result
        assert result.title, result


def test_中文问法搜回来的是中文内容(search: WebSearch):
    """这一路的价值就在中文召回上：接一家对中文游戏圈覆盖差的，等于白接。"""
    found = search.search(QUERY, limit=5)

    chinese = [result for result in found if any("一" <= char <= "鿿" for char in result.title)]
    assert chinese, [result.title for result in found]
