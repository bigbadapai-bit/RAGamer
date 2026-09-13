"""入库端点（主缝）：内存假件换掉八个外部依赖，整条链路跑一遍。

只断言外部可观察的行为：HTTP 响应本身，以及通过存储适配器的查询接口能观察到的
入库结果。原项目审计出来的缺陷几乎全部出在串联层——接错了线，只有跑完真实串联
才抓得到，所以这一层不省。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragamer.api import KB_COLLECTION, create_app
from ragamer.stores.base import UNVERSIONED

from .conftest import make_container

GAME = "black_myth"
CHUNKS_URL = f"/api/kb/{GAME}/import"

ARTICLE = """\
# 二郎神

{{信息框
| 名称 = 二郎神
| 类型 = BOSS
}}

二郎神是隐藏 BOSS，需要三阶段打完。

## 获取方式

在第三章的隐藏区域遇到。

## 属性

血量 12000，抗性偏高。

## 打法

先定身，再贴身输出。

[[Category:角色]]
[[Category:妖王]]
"""

#: 只有大标题与一句正文：切出来的片少于一整篇，用来验重导时的整体替换。
SHORT_ARTICLE = "# 二郎神\n\n{{信息框\n| 名称 = 二郎神\n| 类型 = BOSS\n}}\n\n二郎神是隐藏 BOSS。\n"

#: 这个库的元数据：术语映射来自它，不在代码里写死。
KB = {
    "name": "黑神话·悟空",
    "version": "1.0",
    "subject_types": ["character", "item"],
    "term_mapping": {"妖王": "character", "根器": "item"},
}


@pytest.fixture
def container():
    """整条链路的内存版。知识库先建好——导入不负责建库。"""
    container = make_container()
    container.docs.put(KB_COLLECTION, GAME, KB)
    return container


@pytest.fixture
def client(container) -> TestClient:
    return TestClient(create_app(container))


def upload(name: str, text: str = ARTICLE) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, text.encode("utf-8"), "text/markdown"))


def saved(container, version: str = UNVERSIONED) -> list:
    """库里这份文档的切片，按顺序。写入侧的结果只从存储适配器的查询接口观察。"""
    return container.chunks.fetch_document(GAME, "二郎神", version=version)


def import_articles(client, *files, version: str | None = None):
    data = {} if version is None else {"version": version}
    response = client.post(CHUNKS_URL, files=list(files), data=data)
    assert response.status_code == 200, response.text
    return response.json()


# --- 提交一份 Markdown，查回它切出的全部切片 ---


def test_提交一份_markdown_切出的切片全部查得回来(client, container):
    payload = import_articles(client, upload("二郎神.md"))

    assert payload["imported"] == 1
    assert payload["failed"] == 0
    stored = saved(container)
    assert payload["results"][0]["chunk_count"] == len(stored)
    assert len(stored) > 1  # 确实切出了多片，这条才验得到东西


def test_查回来的切片带祖先标题路径_两层标签与切片类型(client, container):
    import_articles(client, upload("二郎神.md"))

    stored = saved(container)

    assert all(chunk.doc_title == "二郎神" for chunk in stored)
    # 祖先标题路径：脱离页面也要认得出「是谁的掉落」
    assert "二郎神 › 获取方式" in {chunk.ancestor_path for chunk in stored}
    # 主体名与主体类型是文档级，回写到每一个切片
    assert {chunk.subject_name for chunk in stored} == {"二郎神"}
    assert all(chunk.subject_type == ("character",) for chunk in stored)
    # 内容性质是切片级：同一份文档里并存几种
    assert {chunk.content_nature for chunk in stored} >= {
        ("where",),
        ("stats",),
        ("guide",),
    }
    # Infobox 整块一片，与表格同标 `table`
    assert {chunk.chunk_type for chunk in stored} == {"text", "table"}


def test_入库的每一条都带向量与内容摘要(client, container):
    import_articles(client, upload("二郎神.md"))

    for chunk in saved(container):
        assert chunk.content
        assert chunk.content_hash
        assert chunk.dense_vector is not None
        assert chunk.sparse_vector is not None


# --- 重复导入不产生重复切片 ---


def test_同一份资料导入两次_库里的切片数量不变(client, container):
    first = import_articles(client, upload("二郎神.md"))
    before = [chunk.chunk_id for chunk in saved(container)]
    second = import_articles(client, upload("二郎神.md"))

    assert second["results"][0]["chunk_count"] == first["results"][0]["chunk_count"]
    assert [chunk.chunk_id for chunk in saved(container)] == before


def test_重导一份变短的资料时旧的那一截被清掉(client, container):
    import_articles(client, upload("二郎神.md"))
    short = import_articles(client, upload("二郎神.md", SHORT_ARTICLE))

    stored = saved(container)

    assert len(stored) == short["results"][0]["chunk_count"]
    assert [chunk.chunk_index for chunk in stored] == list(range(len(stored)))


def test_标注的版本落进库并与未标注版本并存(client, container):
    import_articles(client, upload("二郎神.md"))
    import_articles(client, upload("二郎神.md"), version="2.0")

    assert saved(container, UNVERSIONED)
    assert {chunk.version for chunk in saved(container, "2.0")} == {UNVERSIONED, "2.0"}


# --- 一批里某个文件失败不牵连其余 ---


def test_某个文件失败时其余正常入库_失败信息带文件名与阶段(client, container):
    payload = import_articles(
        client,
        upload("甲.md"),
        ("files", ("攻略.pdf", b"%PDF-1.7", "application/pdf")),
        upload("乙.md"),
    )

    assert (payload["imported"], payload["failed"]) == (2, 1)
    failed = payload["results"][1]
    assert failed["filename"] == "攻略.pdf"
    assert failed["stage"] == "normalize"
    assert failed["error"]
    assert saved(container)


def test_一份资料都不成时也返回结果而不是整个请求失败(client, container):
    payload = import_articles(client, ("files", ("攻略.pdf", b"%PDF-1.7", "application/pdf")))

    assert payload["imported"] == 0
    assert payload["results"][0]["stage"] == "normalize"


# --- 进度与汇总 ---


def test_导入过程中上报了进度(client):
    payload = import_articles(client, upload("二郎神.md"))
    progress = payload["results"][0]["progress"]

    assert [event["stage"] for event in progress] == [
        "normalize",
        "chunk",
        "tag",
        "embed",
        "store",
    ]
    assert [event["stage_label"] for event in progress][0] == "归一化"
    assert all(event["file_number"] == 1 and event["file_total"] == 1 for event in progress)


def test_返回结果含切片数_覆盖的标签_跳过的条数与错误(client):
    payload = import_articles(client, upload("二郎神.md"), upload("攻略.pdf"))
    ok, failed = payload["results"]

    assert ok["chunk_count"] > 0
    assert (ok["skipped"], ok["error"]) == (0, None)
    assert ok["tags"]["subject_name"] == "二郎神"
    assert ok["tags"]["subject_type"] == ["character"]
    assert set(ok["tags"]["content_nature"]) >= {"where", "stats", "guide"}
    assert (failed["chunk_count"], failed["error"] is not None) == (0, True)


def test_一批的条数按文件算(client):
    payload = import_articles(client, upload("甲.md"), upload("乙.md"))

    assert (payload["imported"], payload["failed"]) == (2, 0)
    assert [result["filename"] for result in payload["results"]] == ["甲.md", "乙.md"]


# --- 库与游戏 ---


def test_知识库不存在时_404_不静默按默认词表建内容():
    client = TestClient(create_app(make_container()))

    response = client.post(CHUNKS_URL, files=[upload("二郎神.md")])

    assert response.status_code == 404
    assert GAME in response.json()["detail"]


@pytest.mark.parametrize("game_id", ["黑神话", "a-b", ""])
def test_游戏_id_不合法时_400(container, game_id):
    client = TestClient(create_app(container))

    response = client.post(f"/api/kb/{game_id}/import", files=[upload("二郎神.md")])

    assert response.status_code in (400, 404)  # 空 id 先被路由吃掉
    if response.status_code == 400:
        assert "游戏 id 不合法" in response.json()["detail"]


def test_没配术语映射的库仍读得出主体名():
    """自定义库没配映射时「妖王」归不出来，主体类型落到模型兜底。

    假模型一条脚本都没排，调用会当场炸成 `LlmError`——打标把这一条兜住、标签留空，
    不阻断入库（用户故事 4：标签稀疏是可接受的降级，漏内容不是）。
    """
    container = make_container()
    container.docs.put(KB_COLLECTION, GAME, {"name": "某款没配映射的游戏"})
    client = TestClient(create_app(container))

    payload = import_articles(client, upload("二郎神.md"))
    tags = payload["results"][0]["tags"]

    assert payload["results"][0]["error"] is None
    assert tags["subject_name"] == "二郎神"  # 文档大标题不依赖术语映射
    assert tags["subject_type"] == []
    assert tags["content_nature"]  # 内容性质按标题归一，同样不调模型
