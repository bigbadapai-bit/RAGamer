"""提问端点：开会话、把历史读回来、逐字问一句（SSE）。

端点这一层要多证的一件事是**跨请求**：会话是上一个请求写下的，下一个请求读到它——
进程里的对象活不过一次请求，所以历史必须真的落在存储里，才谈得上「刷新之后还在」。
这里因此走真正的 HTTP 路径（`TestClient`），只在最外面换成内存假件。

SSE 那一侧证三件事：**逐字**（不是一次给全）、**引用先到**、`done` 收尾。失败时收的是
`error` 而不是 `done`——响应头早就发出去了，状态码改不了，而收不到 `done` 与「答完了」
在客户端看来必须分得开。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ragamer.api import KB_COLLECTION, create_app
from ragamer.llm import FakeLlm

from .conftest import HalfwayLlm, chunk_store, make_chunk, make_container

GAME = "black_myth"
#: 这个库的元数据。`name` 是显示名——它同时是给模型的游戏候选。
KB = {"name": "黑神话·悟空", "version": "1.0", "subject_types": ["character"]}

DOC = make_chunk(
    1, content="二郎神掉落三尖两刃刀", doc_title="二郎神", ancestor_path="二郎神 › 掉落"
)

QUESTION = "二郎神掉什么"
REPLY = "掉的是三尖两刃刀[1]。"


def said(rewritten: str, game: str = "") -> dict[str, str]:
    """提问理解那一步的脚本：一次联合输出。"""
    return {"game": game, "version": "", "rewritten_query": rewritten}


def client_with(llm, *chunks) -> TestClient:
    """整条链路的内存版：知识库建好、资料入库、模型按脚本回话。"""
    container = make_container(llm=llm, chunks=chunk_store(GAME, *chunks))
    container.docs.put(KB_COLLECTION, GAME, KB)
    return TestClient(create_app(container))


def start(client: TestClient) -> str:
    """开一次会话，返回它的 id。"""
    response = client.post("/api/chat/sessions", json={"game_id": GAME})
    assert response.status_code == 201
    return response.json()["session_id"]


def ask(client: TestClient, session_id: str, question: str, **params) -> str:
    """问一句，把 SSE 正文原样取回来。"""
    response = client.get(
        f"/api/chat/sessions/{session_id}/ask", params={"question": question, **params}
    )
    assert response.status_code == 200, response.text
    return response.text


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """SSE 正文 → `[(事件名, 载荷)]`。

    `retry:` 那一行只有字段没有事件名，也不携带内容，跳过它——它是给浏览器看的兜底，
    不是这一轮问答的一部分。
    """
    parsed = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line]
        if not lines or lines[0].startswith("retry:"):
            continue
        payload = json.loads(lines[1].removeprefix("data: "))
        parsed.append((lines[0].removeprefix("event: "), payload))
    return parsed


# --- 会话与历史 ---


def test_刷新之后历史还在():
    """历史在服务端：页面整块重来也不会把它带走。"""
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    session_id = start(client)

    ask(client, session_id, QUESTION)

    again = client.get(f"/api/chat/sessions/{session_id}").json()
    assert again["game_id"] == GAME
    assert [(turn["role"], turn["content"]) for turn in again["turns"]] == [
        ("user", QUESTION),
        ("assistant", REPLY),
    ]
    # 引用连 label 一起回来，正文里的 [1] 刷新之后仍对得上号
    assert [(item["index"], item["label"]) for item in again["turns"][1]["citations"]] == [
        (1, "二郎神")
    ]


def test_同一个会话里接着上一句提问():
    """跨请求那一层：上一个请求写下的历史，下一个请求要真的读得到。"""
    llm = FakeLlm(said("二郎神是谁"), "二郎神是隐藏 BOSS[1]。", said(QUESTION), REPLY)
    client = client_with(llm, DOC)
    session_id = start(client)

    ask(client, session_id, "二郎神是谁")
    ask(client, session_id, QUESTION)

    # 调用顺序是「理解、生成、理解、生成」，所以第 2 次理解在下标 2
    messages = [(message.role, message.content) for message in llm.calls[2].messages]
    assert ("user", "二郎神是谁") in messages
    assert ("assistant", "二郎神是隐藏 BOSS[1]。") in messages


def test_两次会话互不串扰():
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    first = start(client)
    second = start(client)

    ask(client, first, QUESTION)

    assert client.get(f"/api/chat/sessions/{second}").json()["turns"] == []
    assert len(client.get(f"/api/chat/sessions/{first}").json()["turns"]) == 2


# --- 会话列表 ---


def test_列出这个库的会话_按最后活跃倒序():
    """左栏那一份：**最后说过话的排最前**，先建的那个也一样能翻上来。"""
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    first = start(client)
    second = start(client)

    ask(client, first, QUESTION)  # 先建的会话反而最后说话

    listed = client.get("/api/chat/sessions", params={"game_id": GAME}).json()["sessions"]

    assert [item["session_id"] for item in listed] == [first, second]
    assert listed[0]["title"] == QUESTION
    assert listed[1]["title"] == ""  # 还没问过的会话没有标题


def test_列表不带正文():
    """列表只给标题与时间；正文在 `GET /api/chat/sessions/{id}` 那一条上取。"""
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    session_id = start(client)
    ask(client, session_id, QUESTION)

    listed = client.get("/api/chat/sessions", params={"game_id": GAME}).json()["sessions"]

    assert set(listed[0]) == {"session_id", "title", "updated_at"}


def test_没聊过的库回空列表():
    """一条会话都没有是正常状态，与「没有这个知识库」分得开。"""
    client = client_with(FakeLlm())

    listed = client.get("/api/chat/sessions", params={"game_id": GAME}).json()

    assert listed["sessions"] == []


def test_列出不存在的知识库时404():
    client = client_with(FakeLlm())

    response = client.get("/api/chat/sessions", params={"game_id": "another_game"})

    assert response.status_code == 404


# --- 流式 ---


def test_答案逐字流式输出():
    """逐字不是「分成两段给」：一片一片地来，拼起来仍是完整那一句。"""
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    session_id = start(client)

    response = client.get(f"/api/chat/sessions/{session_id}/ask", params={"question": QUESTION})

    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = parse_sse(response.text)
    deltas = [payload["text"] for name, payload in parsed if name == "delta"]
    assert len(deltas) > 1
    assert "".join(deltas) == REPLY
    assert parsed[-1][0] == "done"


def test_每一步之前先报一条进度():
    """提问到第一个字之间隔着两次等待（一次模型往返加一次检索），界面得有的可显示。

    顺序就是这一轮真正干的事：理解 → 检索 → 来源 → 生成 → 正文。
    """
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    session_id = start(client)

    parsed = parse_sse(ask(client, session_id, QUESTION))

    assert [payload["text"] for name, payload in parsed if name == "status"] == [
        "正在理解问题",
        "正在检索资料",
        "正在生成答案",
    ]
    events = [name for name, _ in parsed]
    assert events[:3] == ["status", "status", "citations"]
    assert events[3] == "status"
    assert set(events[4:-1]) == {"delta"}
    assert events[-1] == "done"


def test_引用比正文先到():
    """引用在检索那一步就定下来了，比正文早得多——界面因此不必等正文吐完才列来源。"""
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    session_id = start(client)

    parsed = parse_sse(ask(client, session_id, QUESTION))

    events = [name for name, _ in parsed]
    assert events.index("citations") < events.index("delta")
    citations = parsed[events.index("citations")][1]["citations"]
    assert [(item["index"], item["label"]) for item in citations] == [(1, "二郎神")]


def test_生成中途失败时发错误事件且不写进历史():
    """开流之后失败只能是一条事件，而且**不是一个 `done`**——客户端靠这个分得开。

    那一轮也不该在会话里留下半句答案。
    """
    client = client_with(HalfwayLlm(said(QUESTION)), DOC)
    session_id = start(client)

    parsed = parse_sse(ask(client, session_id, QUESTION))

    events = [name for name, _ in parsed]
    assert events[-1] == "error"
    assert "done" not in events
    assert client.get(f"/api/chat/sessions/{session_id}").json()["turns"] == []


# --- 边界 ---


def test_没有这个会话时404():
    """找不到的会话不能被当成空会话——那样每刷新一次就多一个会话出来。"""
    client = client_with(FakeLlm())

    assert client.get("/api/chat/sessions/nope").status_code == 404
    assert (
        client.get("/api/chat/sessions/nope/ask", params={"question": QUESTION}).status_code == 404
    )


def test_知识库不存在时404():
    client = client_with(FakeLlm())

    response = client.post("/api/chat/sessions", json={"game_id": "another_game"})

    assert response.status_code == 404


def test_空问题当场400():
    """空问题会让检索查出任意一批切片，模型照着它编一段答案——开流之前就拦掉。"""
    client = client_with(FakeLlm(said(QUESTION), REPLY), DOC)
    session_id = start(client)

    response = client.get(f"/api/chat/sessions/{session_id}/ask", params={"question": "   "})

    assert response.status_code == 400
