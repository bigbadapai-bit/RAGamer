"""入库端点（主缝）：内存假件换掉全部外部依赖，整条链路跑一遍。

只断言外部可观察的行为：HTTP 响应本身，以及通过存储适配器的查询接口能观察到的
入库结果。原项目审计出来的缺陷几乎全部出在串联层——接错了线，只有跑完真实串联
才抓得到，所以这一层不省。
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from ragamer.api import KB_COLLECTION, create_app
from ragamer.stores.base import UNVERSIONED

from .conftest import FakeCrawler, make_container

GAME = "black_myth"
CHUNKS_URL = f"/api/kb/{GAME}/import"
URLS_URL = f"{CHUNKS_URL}/urls"
#: 网址那一路的假抓取器排的就是这一页。正文与 `ARTICLE` 相同，
#: 于是「两条入口切出同样的片」可以直接两两对起来
CRAWLED_URL = "https://wiki.test/wiki/二郎神"
MISSING_URL = "https://wiki.test/wiki/没有这页"

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
    """整条链路的内存版。知识库先建好——导入不负责建库。

    抓取器是假的，但它在容器里占的位置与真的那个一样：网址那一路从组合根拿到它，
    与三个存储、两个模型同级。
    """
    container = make_container(crawler=FakeCrawler(**{CRAWLED_URL: ARTICLE}))
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


def import_links(client, *urls: str, version: str | None = None):
    body = {"urls": list(urls)}
    if version is not None:
        body["version"] = version
    response = client.post(URLS_URL, json=body)
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
    assert failed["source"] == "攻略.pdf"
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


def test_导入在跑的时候就把进度落进日志(client, caplog):
    """响应里的 `progress` 是跑完才拿得到的账单；一批几十份资料时，那之前靠日志。"""
    with caplog.at_level(logging.INFO, logger="ragamer.api"):
        import_articles(client, upload("二郎神.md"))

    assert "导入 二郎神.md：[1/1] 归一化" in caplog.text
    assert "导入 二郎神.md：[1/1] 入库" in caplog.text


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
    assert [result["source"] for result in payload["results"]] == ["甲.md", "乙.md"]


# --- 库与游戏 ---


def test_知识库配置读不了时报_422_而不是_500():
    """库里配了个不认识的主体类型：是知识库的数据坏了，不是这份资料的错。"""
    container = make_container()
    container.docs.put(KB_COLLECTION, GAME, {"subject_types": ["这不是类目"]})
    client = TestClient(create_app(container))

    response = client.post(CHUNKS_URL, files=[upload("二郎神.md")])

    assert response.status_code == 422
    assert "这不是类目" in response.json()["detail"]


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


# --- 提交网址 ---


def test_提交一个网址_抓回来的内容切成切片入库(client, container):
    payload = import_links(client, CRAWLED_URL)

    assert payload["imported"] == 1
    assert payload["failed"] == 0
    assert container.crawler.requested == [CRAWLED_URL]
    stored = saved(container)
    assert payload["results"][0]["chunk_count"] == len(stored) > 1


def test_抓下来的切片带上来源地址(client, container):
    """答案的引用里要显示的就是它。"""
    import_links(client, CRAWLED_URL)

    assert {chunk.source_url for chunk in saved(container)} == {CRAWLED_URL}


def test_同一份正文从网址来与从文件来切出的片一样(client, container):
    """验收第 6 条：两条入口走的是同一条链路，切分与打标不感知来源。

    两边切出的片除来源地址之外逐字相同——切片主键也在内，因为主键只看
    游戏、文档标题、版本与序号，不看资料从哪来。
    """
    import_articles(client, upload("二郎神.md"))
    from_file = saved(container)

    import_links(client, CRAWLED_URL)
    from_url = saved(container)

    assert len(from_file) == len(from_url) > 1
    for before, after in zip(from_file, from_url, strict=True):
        assert before.chunk_id == after.chunk_id
        assert before.content == after.content
        assert before.ancestor_path == after.ancestor_path
        assert before.content_nature == after.content_nature
        assert before.source_url == ""
        assert after.source_url == CRAWLED_URL


def test_某个地址抓不到时其余照常入库(client, container):
    payload = import_links(client, CRAWLED_URL, MISSING_URL)

    assert (payload["imported"], payload["failed"]) == (1, 1)
    failed = payload["results"][1]
    assert failed["source"] == MISSING_URL
    assert failed["stage"] == "normalize"  # 抓取属于归一化那一步
    assert failed["error"]
    assert saved(container)  # 失败的那一条没有牵连成功的那一条


def test_网址导入的进度一路报到入库(client, caplog):
    """抓取是同步的一整段，日志是它跑的时候唯一看得见的进度窗口。"""
    with caplog.at_level(logging.INFO):
        import_links(client, CRAWLED_URL)

    messages = [record.getMessage() for record in caplog.records]
    assert any(CRAWLED_URL in message and "归一化" in message for message in messages)
    assert any(CRAWLED_URL in message and "入库" in message for message in messages)


def test_网址导入也标注版本(client, container):
    payload = import_links(client, CRAWLED_URL, version="1.0")

    assert payload["version"] == "1.0"
    assert {chunk.version for chunk in saved(container, version="1.0")} == {"1.0"}


def test_一个网址都不给时_422(client):
    """空数组是请求本身不合法，不是「导入了零条」——后者看起来像成功。"""
    response = client.post(URLS_URL, json={"urls": []})

    assert response.status_code == 422


def test_网址那一路同样拦住非法游戏_id(container):
    client = TestClient(create_app(container))

    response = client.post("/api/kb/黑神话/import/urls", json={"urls": [CRAWLED_URL]})

    assert response.status_code in (400, 404)


def test_网址那一路的知识库不存在时_404(container):
    client = TestClient(create_app(container))

    response = client.post("/api/kb/other_game/import/urls", json={"urls": [CRAWLED_URL]})

    assert response.status_code == 404
