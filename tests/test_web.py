"""界面（主缝）：内存假件换掉全部外部依赖，从建库一路点到切分预览。

只断言外部可观察的行为——HTTP 响应本身，以及通过存储适配器的查询接口能观察到的结果。
这一票要的是「浏览器里能看见东西」，所以这里按验收条目逐条走一遍：建库 → 传 md →
预览页列出切片。

知识库管理那几条同样只看外部行为：改了配置之后**下一次导入**落下来的标签对不对，
而不是去翻内存里那个 dict 长什么样——「保存后立即生效」要验的正是这条链路。
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from ragamer.app import create_app
from ragamer.caching import CachedAnswer, cache_key
from ragamer.conversations import CONVERSATIONS
from ragamer.knowledge import (
    KB_COLLECTION,
    KnowledgeBase,
    create_knowledge_base,
    vocabulary_of,
)
from ragamer.stores.base import UNVERSIONED, image_key, image_prefix
from ragamer.tagging import SubjectType

from .conftest import BrokenChunkStore, make_container

GAME = "black_myth"
DOC_TITLE = "二郎神"
IMPORT_URL = "/import"

ARTICLE = """\
# 二郎神

{{信息框
| 名称 = 二郎神
| 类型 = 妖王
}}

二郎神是隐藏 BOSS，需要三阶段打完。

## 获取方式

在第三章的隐藏区域遇到。

## 属性

血量 12000，抗性偏高。

## 打法

先定身，再贴身输出。

[[Category:妖王]]
"""

#: 这个库的元数据。术语映射来自它——「妖王」归到角色，不在代码里写死。
KB = {
    "name": "黑神话·悟空",
    "subject_types": ["character", "item"],
    "term_mapping": {"妖王": "character", "根器": "item"},
}


@pytest.fixture
def container():
    container = make_container()
    create_knowledge_base(
        container.docs,
        KnowledgeBase.new(GAME, "黑神话·悟空", (SubjectType.CHARACTER, SubjectType.ITEM)),
    )
    container.docs.put(KB_COLLECTION, GAME, KB)
    return container


@pytest.fixture
def client(container) -> TestClient:
    return TestClient(create_app(container))


def upload(name: str, text: str = ARTICLE) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, text.encode("utf-8"), "text/markdown"))


def do_import(client, *files, game_id: str = GAME, version: str = "", htmx: bool = False):
    response = client.post(
        IMPORT_URL,
        files=list(files) or [upload("二郎神.md")],
        data={"game_id": game_id, "version": version},
        headers={"HX-Request": "true"} if htmx else None,
    )
    assert response.status_code == 200, response.text
    return response


def stored(container, version: str = UNVERSIONED) -> list:
    """库里这份文档的切片，按顺序。只从存储适配器的查询接口观察。"""
    return container.chunks.fetch_document(GAME, DOC_TITLE, version=version)


def preview(client, doc_title: str = DOC_TITLE, version: str = UNVERSIONED):
    return client.get(f"/kb/{GAME}/preview", params={"doc_title": doc_title, "version": version})


def import_and_preview(client, container):
    do_import(client)
    return preview(client)


# --- 验收：浏览器里能新建一个游戏知识库 ---


def test_新建知识库后出现在列表里并落了库(client, container):
    response = client.post(
        "/kb",
        data={"game_id": "ghost", "name": "对马岛之魂", "subject_types": ["character", "place"]},
        follow_redirects=False,
    )

    # 建完直接进它的导入页——中间没有别的可做
    assert response.status_code == 303
    assert response.headers["location"] == "/import?game_id=ghost"

    page = client.get("/kb").text
    assert "对马岛之魂" in page
    assert "ghost" in page
    assert container.docs.get(KB_COLLECTION, "ghost")["subject_types"] == ["character", "place"]


def test_根路径进知识库管理页(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/kb"


def test_建库表单能改错_游戏_id_不合法时把原因写在页面上(client, container):
    response = client.post("/kb", data={"game_id": "黑神话", "subject_types": ["character"]})

    assert response.status_code == 400
    assert "游戏 id 不合法" in response.text
    # 填过的值留在表单里，不必重新敲一遍
    assert 'value="黑神话"' in response.text
    assert container.docs.get(KB_COLLECTION, "黑神话") is None


def test_建库表单能改错_一个主体类型都没勾时说明原因(client):
    response = client.post("/kb", data={"game_id": "ghost", "name": ""})

    assert response.status_code == 400
    assert "至少要启用一个主体类型" in response.text


def test_同_id_再建一次报错而不是覆盖(client, container):
    before = container.docs.get(KB_COLLECTION, GAME)

    response = client.post("/kb", data={"game_id": GAME, "subject_types": ["item"]})

    assert response.status_code == 400
    assert "已经有一个 id 为" in response.text
    assert container.docs.get(KB_COLLECTION, GAME) == before


# --- 验收：能上传一份 Markdown 并触发导入 ---


def test_传一份_markdown_就完成导入并给出预览入口(client, container):
    response = do_import(client)

    assert "二郎神.md" in response.text
    assert f"入库 {len(stored(container))} 条" in response.text
    assert "看切分结果" in response.text


def test_一批里某个文件失败时其余照常入库_失败的说清卡在哪一步(client, container):
    response = do_import(
        client,
        upload("甲.md"),
        ("files", ("攻略.pdf", b"%PDF-1.7", "application/pdf")),
    )

    assert stored(container)  # 成功的那份确实入库了
    assert "共 2 份，成功 1 份，失败 1 份" in response.text
    assert "卡在「归一化」" in response.text


def test_没选资料时给一句话而不是报错(client):
    response = client.post(IMPORT_URL, data={"game_id": GAME, "version": ""})

    assert response.status_code == 200
    assert "先选一份资料再提交" in response.text


def test_导进不存在的知识库时说清是哪个库(client):
    response = client.post(IMPORT_URL, files=[upload("二郎神.md")], data={"game_id": "zelda"})

    assert response.status_code == 404
    assert "知识库 zelda 不存在" in response.text


# --- 验收：不带 JavaScript 也能用；htmx 在时只换结果那一块 ---


def test_禁用_javascript_时原生提交照样出结果(client):
    """表单同时带 action／method 与 hx-post：没有 htmx 时浏览器自己提交，回整页。"""
    page = client.get("/import").text
    assert 'action="/import"' in page
    assert 'method="post"' in page
    assert 'hx-post="/import"' in page
    assert 'enctype="multipart/form-data"' in page

    response = do_import(client)

    assert "<html" in response.text  # 整页
    assert "导入结果" in response.text


def test_htmx_提交只回结果那一块(client):
    response = do_import(client, htmx=True)

    assert "<html" not in response.text
    assert "<body" not in response.text
    assert "导入结果" in response.text


@pytest.mark.parametrize(
    ("game_id", "with_file", "message"),
    [
        (GAME, False, "先选一份资料再提交"),
        ("zelda", True, "知识库 zelda 不存在"),
        ("黑神话", True, "游戏 id 不合法"),
    ],
)
def test_htmx_提交出错时那句话也换得进去(client, game_id, with_file, message):
    """出错时同样只回片段，而且**一律 200**。

    htmx 默认不换入非 2xx 的响应：回 404 的话片段根本不会进 DOM，人只会对着一个空的
    结果区发呆。整页那条路仍然报真实状态码（另有用例兜着）。
    """
    response = client.post(
        IMPORT_URL,
        data={"game_id": game_id, "version": ""},
        files=[upload("二郎神.md")] if with_file else [],
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert "<html" not in response.text
    assert message in response.text


def test_前端没有本地构建产物(client):
    """页面上的脚本全在 CDN 上：没有 npm、没有打包、没有 /static（ADR-0005）。"""
    page = client.get("/kb").text

    assert re.search(r'<script src="https://', page)
    assert "/static/" not in page
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)="/', page)


# --- 验收：切分预览页 ---


def test_预览页列出该文档的全部切片(client, container):
    page = import_and_preview(client, container).text

    assert page.count("</li>") > 1
    for chunk in stored(container):
        assert chunk.content in page
    assert len(stored(container)) > 1  # 确实切出了多片，这条才验得到东西


def test_每条切片显示祖先标题路径_主体类型与内容性质(client, container):
    page = import_and_preview(client, container).text

    assert f"#0 · {DOC_TITLE}" in page  # 大标题那一段的路径只有标题
    assert f"{DOC_TITLE} › 获取方式" in page
    assert "主体类型：角色" in page
    assert "内容性质：位置与获取" in page
    assert "内容性质：打法流程" in page


def test_切片按文档顺序排列(client, container):
    page = import_and_preview(client, container).text

    rendered = [int(index) for index in re.findall(r"#(\d+) ·", page)]

    assert rendered == [chunk.chunk_index for chunk in stored(container)]


def test_预览页是只读的_一个编辑入口都没有(client, container):
    page = import_and_preview(client, container).text

    for control in ("<form", "<input", "<button", "<select", "<textarea"):
        assert control not in page, f"预览页不该出现 {control}"


def test_预览页报出来源文档与主体名(client, container):
    page = import_and_preview(client, container).text

    assert "来源文档" in page
    assert DOC_TITLE in page
    assert "主体名" in page


def test_预览一个没导过的文档时说清而不是_500(client):
    response = preview(client, doc_title="不存在的东西")

    assert response.status_code == 404
    assert "没有切片" in response.text


def test_预览里带上版本时未标注版本的内容会一并列出并标出来(client, container):
    do_import(client)  # 未标注版本
    do_import(client, version="2.0")

    page = preview(client, version="2.0").text

    assert "未标注版本，不随版本变化" in page


# --- 验收：导航里留出四个位置 ---


@pytest.mark.parametrize("path", ["/kb", "/import", "/chat", "/eval"])
def test_每个页面都有四项导航(client, path):
    page = client.get(path).text

    for label in ("对话", "知识库管理", "导入", "评测"):
        assert label in page, f"{path} 的导航里少了「{label}」"


def test_还没做的页面点进去是一句人话而不是_404(client):
    response = client.get("/eval")

    assert response.status_code == 200
    assert "这个页面还没做" in response.text


# --- 验收：知识库管理页 ---

CONFIG_URL = f"/kb/{GAME}"
TERMS_URL = f"{CONFIG_URL}/terms"
DELETE_URL = f"{CONFIG_URL}/delete"

#: 另一份语料：它的分类「心法」不在 fixture 那个库的映射里，靠界面上加一条才能认出来。
SKILL_ARTICLE = """\
# 七十二变

[[Category:心法]]

变化之术，共七十二般。
"""


def kb_page(client, game_id: str = GAME):
    return client.get(f"/kb/{game_id}")


def save(client, game_id: str = GAME, **data):
    """走一遍「保存基本配置」。缺的字段按 fixture 那个库的样子填。"""
    data.setdefault("name", KB["name"])
    data.setdefault("subject_types", ["character", "item"])
    return client.post(f"/kb/{game_id}", data=data, follow_redirects=False)


def add_term(client, term: str, kind: str, game_id: str = GAME):
    return client.post(
        f"/kb/{game_id}/terms", data={"term": term, "kind": kind}, follow_redirects=False
    )


def test_能改显示名与启用的类目(client, container):
    response = save(client, name="黑神话·悟空（2025）", subject_types=["place"])

    assert response.status_code == 303
    stored = container.docs.get(KB_COLLECTION, GAME)
    assert stored["name"] == "黑神话·悟空（2025）"
    assert stored["subject_types"] == ["place"]
    # 术语映射不在基本配置那张表单里，保存它不该把映射一起冲掉
    assert stored["term_mapping"] == KB["term_mapping"]


def test_能设置这个库的现行版本(client, container):
    response = save(client, version="2.0")

    assert response.status_code == 303
    assert container.docs.get(KB_COLLECTION, GAME)["version"] == "2.0"
    assert "2.0" in kb_page(client).text


def test_预览页不带版本时按现行版本取切片(client, container):
    """设了现行版本却看不见它在哪儿生效，那个设置就只是一行字。"""
    save(client, version="2.0")
    do_import(client)  # 未标注版本
    do_import(client, version="2.0")

    page = client.get(f"/kb/{GAME}/preview", params={"doc_title": DOC_TITLE}).text

    assert "2.0" in page  # 取的是 2.0 那一批
    assert "未标注版本，不随版本变化" in page  # 未标注的那批一并带上（ADR-0004）


def test_预览页带空版本时按未标注版本看(client, container):
    """空串是明确地要看「未标注版本」，不能当成「没说」而回落到现行版本。"""
    save(client, version="2.0")
    do_import(client, version="2.0")

    page = preview(client, version="").text

    assert "没有切片" in page


def test_一个类目都没勾时页面留住填过的值(client):
    response = client.post(CONFIG_URL, data={"name": "新名字", "version": "3.0"})

    assert response.status_code == 400
    assert "至少要启用一个主体类型" in response.text
    # 敲过的不用重来
    assert 'value="新名字"' in response.text
    assert 'value="3.0"' in response.text


def test_加一条术语映射_下一次导入就认得出这个叫法(client, container):
    """「保存后立即生效」验的是这条链路：改完配置再导一份语料，标签落对了。"""
    save(client, subject_types=["character", "item", "skill"])
    add_term(client, "心法", "skill")

    do_import(client, ("files", ("七十二变.md", SKILL_ARTICLE.encode(), "text/markdown")))

    chunks = container.chunks.fetch_document(GAME, "七十二变", version=UNVERSIONED)
    assert chunks
    assert "skill" in chunks[0].subject_type
    assert "心法" in chunks[0].game_terms


def test_能删掉一条术语映射(client, container):
    response = client.post(f"{TERMS_URL}/delete", data={"term": "妖王"}, follow_redirects=False)

    assert response.status_code == 303
    assert "妖王" not in container.docs.get(KB_COLLECTION, GAME)["term_mapping"]
    # 剩下的那条不受影响
    assert container.docs.get(KB_COLLECTION, GAME)["term_mapping"] == {"根器": "item"}


def test_映射里没填叫法时说一句(client, container):
    response = add_term(client, "   ", "skill")

    assert response.status_code == 400
    assert "先填一个" in response.text
    assert "心法" not in container.docs.get(KB_COLLECTION, GAME)["term_mapping"]


def test_映射里写了不认识的主体类型时报错而不是静默丢掉(client, container):
    response = add_term(client, "心法", "hero")

    assert response.status_code == 400
    assert "不认识的主体类型" in response.text
    assert "心法" not in container.docs.get(KB_COLLECTION, GAME)["term_mapping"]


def test_管理页不带脚本也能用(client):
    """每条写入路径都是普通表单：没有 htmx 时浏览器自己提交，走的是同一段处理逻辑。"""
    page = kb_page(client).text

    for action in (CONFIG_URL, TERMS_URL, f"{TERMS_URL}/delete"):
        assert f'action="{action}"' in page
    assert 'method="post"' in page


# --- 验收：自定义库未配置映射时默认启用全部主体类型 ---


def test_没配过任何东西的库在界面与逻辑上都是七类全开(client, container):
    """架构文档 §2.3 给未配置的自定义库安排的降级：标签会稀疏，但不会漏。"""
    container.docs.put(KB_COLLECTION, "custom", {"name": "手工建的库"})

    page = kb_page(client, "custom").text

    assert page.count('name="subject_types"') == len(SubjectType)
    assert page.count("checked>") == len(SubjectType)
    assert vocabulary_of(container.docs, "custom").subject_types == tuple(SubjectType)


def test_打开不存在的库回列表页_那儿正好能建一个(client):
    response = kb_page(client, "zelda")

    assert response.status_code == 404
    assert "知识库 zelda 不存在" in response.text
    assert "新建知识库" in response.text


def test_配置读不了的库点得进去_存一次就修好了(client, container):
    """列表里能看见它，就该点得进去修——这也是它不被藏起来的原因。"""
    container.docs.put(KB_COLLECTION, GAME, {"name": "坏掉的库", "subject_types": ["这不是类目"]})
    assert "读不了" in kb_page(client).text

    response = save(client, name="坏掉的库", subject_types=["character"])

    assert response.status_code == 303
    assert vocabulary_of(container.docs, GAME).subject_types == (SubjectType.CHARACTER,)


def test_配置读不了的库不让改术语映射(client, container):
    """那份映射本来就没读出来，照着内存里的默认值写回去等于把它悄悄清空、
    还顺手把启用的类目改成全开——两件事都不出声。"""
    broken = {
        "name": "坏掉的库",
        "subject_types": ["这不是类目"],
        "term_mapping": {"妖王": "character"},
    }
    container.docs.put(KB_COLLECTION, GAME, broken)

    response = add_term(client, "心法", "skill")

    assert response.status_code == 422
    assert "配置读不了" in response.text
    assert container.docs.get(KB_COLLECTION, GAME) == broken


# --- 验收：删库前有确认，四处一并清理 ---


#: 一个库名下的一条会话，用来验证删库把会话也清了
SESSION_ID = "sessions-of-black-myth"
#: 一条缓存，用来验证删库把缓存（含提问计数）也清了
CACHED_KEY = cache_key(GAME, "1.0", "二郎神怎么打")


def stocked(client, container) -> int:
    """导一份资料、存两张原图、聊过一句、缓存里有东西，返回切片数。

    删库那几条要的就是「四处都有东西可清」——少铺一处，漏清那一处就测不出来。
    """
    do_import(client)
    for name in ("立绘.png", "地图.png"):
        container.objects.put(image_key(GAME, "a1b2", name), b"PNG")
    container.docs.put(CONVERSATIONS, SESSION_ID, {"game_id": GAME, "title": "二郎神怎么打"})
    container.cache.set(CACHED_KEY, CachedAnswer("先定身再贴身输出[1]。"))
    container.cache.record_question(GAME, "二郎神怎么打")
    return len(stored(container))


def test_删库前的确认页列出将要清理的东西与条数(client, container):
    count = stocked(client, container)

    page = client.get(DELETE_URL).text

    assert "将要清理的数据" in page
    assert f"<strong>{count}</strong> 条切片" in page
    assert "<strong>2</strong> 个原图" in page
    assert "<strong>1</strong> 条" in page  # 会话
    assert "conversations" in page
    assert "提问计数" in page  # 缓存那一处连热门问题一起清
    assert "knowledge_bases" in page
    assert "术语映射" in page
    assert "确认删除" in page


def test_没勾确认不会删(client, container):
    stocked(client, container)

    response = client.post(DELETE_URL, data={})

    assert response.status_code == 400
    assert "先把那句确认勾上" in response.text
    assert container.docs.get(KB_COLLECTION, GAME) is not None
    assert stored(container)


def test_删库把各处数据一并清掉_不留孤儿(client, container):
    count = stocked(client, container)

    response = client.post(DELETE_URL, data={"confirm": "yes"}, follow_redirects=False)

    assert response.status_code == 303
    assert parse_qs(urlparse(response.headers["location"]).query) == {
        "deleted": [GAME],
        "chunks": [str(count)],
        "images": ["2"],
        "sessions": ["1"],
    }
    # 四处里都问不到这个库的东西了
    assert container.chunks.count(GAME) == 0
    assert container.objects.list_keys(image_prefix(GAME)) == []
    assert container.docs.find(CONVERSATIONS, {"game_id": GAME}) == []
    assert container.cache.get(CACHED_KEY) is None
    assert container.cache.top_questions(GAME) == ()
    assert container.docs.get(KB_COLLECTION, GAME) is None
    # 列表页上也看不见了
    assert GAME not in client.get("/kb").text


def test_删完回列表页并说清清掉了多少(client, container):
    """确认页写的是「将要」，这一句写的是「已经」——两个数对得上，那份确认才算数过。"""
    count = stocked(client, container)

    response = client.post(DELETE_URL, data={"confirm": "yes"}, follow_redirects=True)

    assert f"已删除知识库 {GAME}" in response.text
    assert f"清掉 {count} 条切片、2 个原图、1 条会话" in response.text
    assert "还没有知识库" in response.text


def test_清库不碰_id_是它前缀的另一个库():
    """`delete_prefix` 比的是字符串前缀：按 `images/black_myth` 去删，`black_myth_2`
    的原图会被一并收走，而且不报错。"""
    sibling = f"{GAME}_2"
    chunks, container, client = broken_store_client()
    chunks.recover()
    do_import(client)
    container.objects.put(image_key(GAME, "a1b2", "立绘.png"), b"PNG")
    container.objects.put(image_key(sibling, "a1b2", "它的立绘.png"), b"PNG")

    client.post(DELETE_URL, data={"confirm": "yes"})

    assert container.objects.list_keys(image_prefix(GAME)) == [
        image_key(sibling, "a1b2", "它的立绘.png")
    ]


def test_配置读不了的库照样删得掉(client, container):
    """读不出来正是要删它的理由之一：修不好就一了百了，别让它把 id 一直占着。"""
    container.docs.put(KB_COLLECTION, GAME, {"name": "坏掉的库", "subject_types": ["这不是类目"]})

    response = client.post(DELETE_URL, data={"confirm": "yes"}, follow_redirects=False)

    assert response.status_code == 303
    assert container.docs.get(KB_COLLECTION, GAME) is None


def broken_store_client():
    """一个向量库连不上的应用。容器是冻结的，所以连客户端一起另造一套。"""
    chunks = BrokenChunkStore()
    container = make_container(chunks=chunks)
    create_knowledge_base(
        container.docs,
        KnowledgeBase.new(GAME, "黑神话·悟空", (SubjectType.CHARACTER, SubjectType.ITEM)),
    )
    container.docs.put(KB_COLLECTION, GAME, KB)
    return chunks, container, TestClient(create_app(container))


def test_删库失败时库还在_并且说清哪一处没清掉():
    chunks, container, client = broken_store_client()
    do_import(client)
    container.objects.put(image_key(GAME, "a1b2", "立绘.png"), b"PNG")

    response = client.post(DELETE_URL, data={"confirm": "yes"})

    assert response.status_code == 500
    assert "向量库" in response.text
    # 配置是重来的凭据，不能跟着一起没
    assert container.docs.get(KB_COLLECTION, GAME) is not None
    # 没清掉的那一处不跳过其余：原图照清
    assert container.objects.list_keys(image_prefix(GAME)) == []

    # 修好了再删一次就成——上一轮已经清掉的那几处不妨碍重来
    chunks.recover()
    assert client.post(DELETE_URL, data={"confirm": "yes"}).status_code == 200
    assert container.docs.get(KB_COLLECTION, GAME) is None


def test_数不出来时不让确认():
    """数都数不出来就别让人闭着眼睛删。"""
    _, _, client = broken_store_client()

    response = client.get(DELETE_URL)

    assert response.status_code == 500
    assert "确认删除" not in response.text
