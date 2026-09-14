"""对话页（T24）：两级导航、会话正文、引用与图片、版本徽章、澄清按钮、热门问题。

只断言外部可观察的行为——页面里看得见什么、存储里落下了什么。页面与 JSON 端点共用同一套
读取侧（`ragamer.conversations.build_chat`），所以这里多数用例走的是**点页面上的表单**
那条路：那正是验收条目说的「在页面上能做完这件事」。

流式那一小块用原生 `EventSource`，测不到浏览器行为；这一层验的是它退回来的那条路
（`POST /chat/{库}/{会话}/ask` 整轮跑完再回整页）——两条路走的是同一个 `Chat.ask`，
所以落库、澄清、版本回落全都一致。
"""

from __future__ import annotations

import html
import re

from fastapi.testclient import TestClient

from ragamer.app import create_app
from ragamer.clarifying import CONFIDENT, UNSURE
from ragamer.conversations import SESSION_PAGE_SIZE
from ragamer.knowledge import KB_COLLECTION
from ragamer.llm import FakeLlm

from .conftest import chunk_store, joint_reply, make_chunk, make_container
from .test_web import nav_urls

GAME = "black_myth"
OTHER = "yanyun"
KB = {"name": "黑神话·悟空", "version": "1.0", "subject_types": ["character"]}
OTHER_KB = {"name": "燕云十六声", "version": "3.0", "subject_types": ["character"]}

QUESTION = "二郎神怎么打"
REPLY = "先定身再贴身输出[1]。"
#: 带着原图的切片。原图地址就是对象 key（§1.2），页面按 `/images/…` 取。
#: **地址在切片自己的字段上，不在正文里**——正文里只有替代文本（`ragamer.chunking`）。
IMAGE = "images/black_myth/abcdef/phase2.jpg"
#: 一张「图」的字节。这一层验的是取图这条缝，不是解码，所以内容随便给。
PNG = b"\x89PNG\r\n\x1a\n pretend-this-is-a-picture"
DOC = make_chunk(
    1,
    content="二郎神掉落三尖两刃刀\n打法",
    doc_title="二郎神",
    ancestor_path="二郎神 › 掉落",
    image_urls=(IMAGE,),
)


def said(rewritten: str, game: str = "", version: str = "") -> dict[str, object]:
    """提问理解那一步的脚本：一次联合输出。**判出来就算确定**，分级本身另有单测。"""
    return joint_reply(
        game=game,
        game_confidence=CONFIDENT if game else 0.0,
        version=version,
        version_confidence=CONFIDENT if version else 0.0,
        rewritten_query=rewritten,
    )


def client_with(
    llm, *chunks, kb: dict | None = None, game_id: str = GAME, other_kb: bool = False
) -> TestClient:
    """整条链路的内存版：知识库建好、资料入库、模型按脚本回话。

    `other_kb` 再建一个库：反问那几条要有两个候选才问得起来（只有一个库时候选已经
    把那款游戏定死了，`ragamer.clarifying._worth_asking` 会判定没什么可问的）。
    """
    container = make_container(llm=llm, chunks=chunk_store(game_id, *chunks))
    container.docs.put(KB_COLLECTION, game_id, kb if kb is not None else KB)
    if other_kb:
        container.docs.put(KB_COLLECTION, OTHER, OTHER_KB)
    return TestClient(create_app(container))


def start(client: TestClient, game_id: str = GAME) -> str:
    """在页面上开一次会话（提交那张表单），返回它的 id。"""
    response = client.post(f"/chat/{game_id}", follow_redirects=False)
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def ask(client: TestClient, session_id: str, question: str = QUESTION, **fields):
    """在页面上问一句（不开脚本时那条路），拿回渲染好的整页。"""
    return client.post(f"/chat/{GAME}/{session_id}/ask", data={"question": question, **fields})


# --- 两级导航 ---


def test_左栏列出的是知识库不是聊过的游戏():
    """建了库一次没聊过也要在里面——它是「我能问什么」的入口，不是历史记录。"""
    client = client_with(FakeLlm(), kb=KB)

    page = client.get("/chat").text

    assert "黑神话·悟空" in page
    assert "这个库还没聊过" not in page  # 没选库时连会话那一段都不该出现


def test_导航在会话里指回这个会话():
    """切到别的页再点「对话」应当回到正在看的这一条，而不是回到「选一个库」。"""
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY))
    session_id = start(client)

    urls = nav_urls(client.get(f"/chat/{GAME}/{session_id}").text)

    assert urls["对话"] == f"/chat/{GAME}/{session_id}"
    assert urls["知识库管理"] == f"/kb/{GAME}"


def test_还没开会话时导航回到那个库():
    """只选到库这一层：回去就是那个库的会话列表，不必再选一次库。"""
    client = client_with(FakeLlm(), kb=KB)

    urls = nav_urls(client.get(f"/chat/{GAME}").text)

    assert urls["对话"] == f"/chat/{GAME}"


def test_选中一个库之后列出它的近期会话():
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY))
    session_id = start(client)
    ask(client, session_id)

    page = client.get(f"/chat/{GAME}").text

    assert QUESTION in page  # 标题取首轮问句
    assert f"/chat/{GAME}/{session_id}" in page


def test_近期会话按最后活跃倒序():
    """继续聊过的会话不该沉到下面去。"""
    llm = FakeLlm(said("二郎神怎么打"), REPLY, said("二郎神掉什么"), REPLY)
    client = client_with(llm)
    first = start(client)
    ask(client, first, "二郎神怎么打")
    second = start(client)
    ask(client, second, "二郎神掉什么")
    ask(client, first, "二郎神掉什么")  # 先开的那个又聊了一句，从此它最活跃

    page = client.get(f"/chat/{GAME}").text

    assert page.index(f"/chat/{GAME}/{first}") < page.index(f"/chat/{GAME}/{second}")


def test_点开会话把那一轮问答读回来():
    """刷新页面、换台机器打开都读得回来——历史在服务端，不在页面里。"""
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY), DOC)
    session_id = start(client)
    ask(client, session_id)

    page = client.get(f"/chat/{GAME}/{session_id}").text

    assert QUESTION in page
    assert REPLY in page


def test_一个会话都没有时列出的是空列表不是_404():
    client = client_with(FakeLlm())

    response = client.get(f"/chat/{GAME}")

    assert response.status_code == 200
    assert "这个库还没聊过" in response.text


# --- 答案上的三样：引用、图片、版本徽章 ---


def test_引用来源带着文档标题与祖先标题路径():
    """只给标题的话，长词条里是「打法」那一段还是「掉落」那一段，读的人仍然对不上。

    短文档整篇交给生成，父块没有小节这一层（`ragamer.retrieval._block_of`）——
    所以这一条用的是一份超长资料，它才会收敛到命中所在的小节。
    """
    long_doc = make_chunk(
        1,
        content="二郎神掉落三尖两刃刀\n" + "填充" * 4500,
        doc_title="二郎神",
        ancestor_path="二郎神 › 掉落",
    )
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY), long_doc)
    session_id = start(client)

    page = ask(client, session_id).text

    assert "来源 1 条" in page
    assert "二郎神" in page
    assert "二郎神 › 掉落" in page


def test_答案按子集排版且来源不挂编号():
    """模型写的是 Markdown，而模板原先直接输出纯文本：`**` 与 `- ` 会原样露在页面上
    （实测一条答案里 `**` 出现 10 处）。排版的规则本身在 `tests/test_web_markdown.py`，
    这里验的是它接在了页面上。

    来源那一行同时不再挂 `[编号]`：正文里已经不出现编号了，再挂一个就成了没人对得上的号。
    """
    answer = "**打法**：\n- 先定身\n- 再贴身输出"
    client = client_with(FakeLlm(said("二郎神怎么打"), answer), DOC)
    session_id = start(client)

    page = ask(client, session_id).text

    assert "<strong>打法</strong>" in page
    assert "**打法**" not in page
    assert "<li>先定身</li>" in page
    assert "来源 1 条" in page
    assert "[1]" not in page


def test_答案相关的图片直接显示在页面里():
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY), DOC)
    session_id = start(client)

    page = ask(client, session_id).text

    assert f'src="/images/{IMAGE.removeprefix("images/")}"' in page


def test_每条答案带版本徽章():
    """标明这一轮是按哪个版本检索的：会话选定的，其次是知识库的现行版本。"""
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY), DOC)
    session_id = start(client)

    page = ask(client, session_id).text

    assert "版本 1.0" in page


def test_原图按对象_key_取():
    """页面上那张图的地址就是对象 key——库里存的是什么，取的就是什么。"""
    container = make_container(llm=FakeLlm(), chunks=chunk_store(GAME, DOC))
    container.docs.put(KB_COLLECTION, GAME, KB)
    container.objects.put(IMAGE, PNG, content_type="image/png")
    client = TestClient(create_app(container))

    response = client.get(f"/images/{IMAGE.removeprefix('images/')}")

    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/jpeg"


def test_库里没有这张图时_404():
    """如实报 404，不静默给一张空图——那种坏图在页面上看不出是坏的。"""
    client = client_with(FakeLlm())

    assert client.get("/images/black_myth/nope/x.jpg").status_code == 404


# --- 澄清按钮 ---


def test_判不准时页面把候选按钮摆出来():
    """反问就是反问：这一页上看不到答案，候选摆出来等用户点。"""
    client = client_with(
        FakeLlm(joint_reply(game="燕云十六声", game_confidence=UNSURE)),
        other_kb=True,
    )
    session_id = start(client)

    page = ask(client, session_id).text

    assert "你想问的是哪款游戏" in page
    assert REPLY not in page
    assert 'value="黑神话·悟空"' in page
    assert 'value="燕云十六声"' in page


def test_点完候选从暂停点继续给出答案():
    llm = FakeLlm(joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    client = client_with(llm, DOC, other_kb=True)
    session_id = start(client)
    page = ask(client, session_id).text
    pending_id = _pending_id(page)

    resumed = client.post(
        f"/chat/{GAME}/{session_id}/resolve",
        data={"pending_id": pending_id, "label": "黑神话·悟空", "question": QUESTION},
    ).text

    assert REPLY in resumed
    assert "你想问的是哪款游戏" not in resumed


def test_恢复之后那一轮才落进会话():
    """反问不算一轮：还没问完的一轮不该在历史里留下一条等不到回复的提问。"""
    llm = FakeLlm(joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    client = client_with(llm, DOC, other_kb=True)
    session_id = start(client)
    page = ask(client, session_id).text

    assert "还没有问过" in page
    client.post(
        f"/chat/{GAME}/{session_id}/resolve",
        data={
            "pending_id": _pending_id(page),
            "label": "黑神话·悟空",
            "question": QUESTION,
        },
    )
    conversation = client.get(f"/api/chat/sessions/{session_id}").json()

    assert [turn["content"] for turn in conversation["turns"]] == [QUESTION, REPLY]


# --- 版本切换 ---


def test_能在页面上切换当前查询的版本():
    container = make_container(
        llm=FakeLlm(said("二郎神怎么打", version="2.0"), REPLY),
        chunks=chunk_store(
            GAME,
            make_chunk(1, content="旧版正文", version="1.0"),
            make_chunk(2, content="新版正文", version="2.0"),
        ),
    )
    container.docs.put(KB_COLLECTION, GAME, KB)
    client = TestClient(create_app(container))
    session_id = start(client)

    client.post(f"/chat/{GAME}/{session_id}/version", data={"version": "2.0"})

    assert client.get(f"/api/chat/sessions/{session_id}").json()["version"] == "2.0"


def test_切到一个库里没有的版本会被拦下来():
    """换成一个库里没有的版本，检索会静默查空，而界面上看不出区别——所以在这里就拦。

    下拉里只摆真有的那些，这一道拦的是绕过页面的请求（改过的表单、直接打接口）。
    """
    client = client_with(FakeLlm(), DOC)  # 这个库里只有 1.0
    session_id = start(client)
    before = client.get(f"/api/chat/sessions/{session_id}").json()["version"]

    page = client.post(f"/chat/{GAME}/{session_id}/version", data={"version": "9.9"}).text

    assert "没有版本 9.9" in page
    assert client.get(f"/api/chat/sessions/{session_id}").json()["version"] == before


def test_版本候选只来自语料里真实有过的版本():
    """输一个库里没有的版本，检索会静默查空，而界面上看不出区别。"""
    chunks = chunk_store(
        GAME,
        make_chunk(1, content="旧版", version="1.0"),
        make_chunk(2, content="新版", version="2.0"),
    )
    container = make_container(llm=FakeLlm(), chunks=chunks)
    container.docs.put(KB_COLLECTION, GAME, KB)
    client = TestClient(create_app(container))
    session_id = start(client)

    page = client.get(f"/chat/{GAME}/{session_id}").text

    assert '<option value="2.0"' in page
    assert '<option value="9.9"' not in page


# --- 热门问题 ---


def test_页面上显示热门问题():
    """同一份 Redis 里加一个 ZSET 就白捡的功能（架构文档 §4）。"""
    container = make_container(llm=FakeLlm())
    container.docs.put(KB_COLLECTION, GAME, KB)
    container.cache.record_question(GAME, "二郎神怎么打")
    container.cache.record_question(GAME, "二郎神怎么打")
    container.cache.record_question(GAME, "二郎神掉什么")
    client = TestClient(create_app(container))
    session_id = start(client)

    page = client.get(f"/chat/{GAME}/{session_id}").text

    assert "热门问题" in page
    assert "二郎神怎么打" in page
    assert "二郎神掉什么" in page


def test_还没开会话时也显示热门问题并且点得动():
    """选中一个库就该看到大家都在问什么，不必先开一个会话；点一下开一个再问。"""
    container = make_container(
        llm=FakeLlm(said("二郎神怎么打"), REPLY), chunks=chunk_store(GAME, DOC)
    )
    container.docs.put(KB_COLLECTION, GAME, KB)
    container.cache.record_question(GAME, "二郎神怎么打")
    client = TestClient(create_app(container))

    listed = client.get(f"/chat/{GAME}").text
    answered = client.post(f"/chat/{GAME}", data={"question": "二郎神怎么打"})

    assert "热门问题" in listed and "二郎神怎么打" in listed
    assert REPLY in answered.text  # 开了一个会话，并把那一轮问了出来


# --- 边界 ---


def test_问一句空问题不留半个答案():
    client = client_with(FakeLlm())
    session_id = start(client)

    page = ask(client, session_id, "   ").text

    assert "问题不能为空" in page
    assert client.get(f"/api/chat/sessions/{session_id}").json()["turns"] == []


def test_打开一个不存在的会话是_404():
    client = client_with(FakeLlm())

    response = client.get(f"/chat/{GAME}/没有这个会话")

    assert response.status_code == 404


def test_打开一个不存在的库是_404():
    client = client_with(FakeLlm())

    response = client.get("/chat/没有这个库")

    assert response.status_code == 404


def test_会话正文那一段能单独取回来():
    """流答完之后 htmx 拿这一段换掉整块：两条路渲染的是同一份数据。"""
    client = client_with(FakeLlm(said("二郎神怎么打"), REPLY), DOC)
    session_id = start(client)
    ask(client, session_id)

    response = client.get(f"/chat/{GAME}/{session_id}/turns")

    assert response.status_code == 200
    assert REPLY in response.text
    assert "<html" not in response.text  # 是片段，不是整页


def _pending_id(page: str) -> str:
    """从渲染好的那一页里取回暂停点 id。它藏在候选按钮所属的表单里。"""
    marker = 'name="pending_id" value="'
    start_at = page.index(marker) + len(marker)
    return page[start_at : page.index('"', start_at)]


# --- 会话列表翻页 ---


def test_一页装不下时列表挂一个取下一页的哨兵():
    """左栏滚到底接着取。哨兵带着上一页最后一条的位置，滚进视口就去取。"""
    client = client_with(FakeLlm())
    for _ in range(SESSION_PAGE_SIZE + 1):
        start(client)

    page = client.get(f"/chat/{GAME}").text

    assert 'hx-trigger="revealed"' in page
    assert "正在取更早的" in page


def test_哨兵指向的那一段能单独取回来并且接得上():
    """片段与整页走同一份数据、同一段模板——所以接上去的那一页跟第一页长得一样。"""
    client = client_with(FakeLlm())
    for _ in range(SESSION_PAGE_SIZE + 1):
        start(client)
    page = client.get(f"/chat/{GAME}").text
    url = html.unescape(re.search(r'hx-get="([^"]+)"', page).group(1))

    fragment = client.get(url, headers={"HX-Request": "true"})

    assert fragment.status_code == 200
    assert "<html" not in fragment.text  # 是片段，不是整页
    assert f"/chat/{GAME}/" in fragment.text


def test_一页装得下时没有哨兵():
    """到底了就不该再挂一个「正在取更早的」——那是骗人往下滚。"""
    client = client_with(FakeLlm())
    start(client)

    page = client.get(f"/chat/{GAME}").text

    assert "正在取更早的" not in page


def test_会话列表有自己的滚动区():
    """**列表要有固定高度**：整页一起滚的话，哨兵一上来就在视口里，会一口气把全部取完。"""
    client = client_with(FakeLlm())
    start(client)

    page = client.get(f"/chat/{GAME}").text

    assert "overflow-y-auto" in page


def test_取下一页的地址不带_hx_头时给整页():
    """浏览器直接打开那个地址（没有 htmx 头）时，该拿到的是能看的整页。"""
    client = client_with(FakeLlm())
    start(client)

    response = client.get(f"/chat/{GAME}?after=2026-09-13T00%3A00%3A00%2B00%3A00%7Cx")

    assert response.status_code == 200
    assert "<html" in response.text
