"""提问端点（读取侧的主缝）：澄清与作答都在这一层看得见。

只断言外部可观察的行为：SSE 流里的事件本身，以及通过存储适配器的查询接口能观察到的
落库结果。**暂停与恢复必须在这一层成立**——反问是流里的一个 `clarification` 事件，
用户点完按钮是另一次请求，中间那座桥（待澄清记录）只能从存储上观察。

模型那一段接 `FakeLlm`（一次网络都不发），检索那段接内存假件与确定性假件。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ragamer.api import create_app
from ragamer.clarifying import GAME, PENDING_COLLECTION, UNSURE, VERSION
from ragamer.knowledge import KB_COLLECTION
from ragamer.llm import FakeLlm

from .conftest import joint_reply, make_chunk, make_container

GAME_ID = "black_myth"
OTHER_ID = "yanyun"
QUESTION = "二郎神怎么打"

BLACK_MYTH = {"name": "黑神话·悟空", "version": "1.0"}
YANYUN = {"name": "燕云十六声", "version": "3.0"}

#: 一段答案。编号指的就是提示词里那批切片的编号。
REPLY = "先定身再贴身输出[1]。"

#: 一次反问。用户点完候选带它回来，那一轮才算走完。
Pending = dict


@pytest.fixture
def container():
    """整条链路的内存版：两个库建好，两个版本各有一批切片。"""
    container = make_container(llm=FakeLlm(joint_reply(), REPLY))
    container.docs.put(KB_COLLECTION, GAME_ID, BLACK_MYTH)
    container.docs.put(KB_COLLECTION, OTHER_ID, YANYUN)
    container.chunks.upsert(
        GAME_ID,
        [
            make_chunk(1, content=QUESTION, version="1.0", doc_title="二郎神"),
            make_chunk(2, content=QUESTION, version="2.0", ancestor_path="二郎神 › 新版"),
        ],
    )
    return container


@pytest.fixture
def client(container) -> TestClient:
    return TestClient(create_app(container))


def parse(body: str) -> list[tuple[str, dict]]:
    """SSE 正文 → `(事件名, 载荷)` 的序列。`retry:` 那一行不是事件，跳过。"""
    parsed: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.startswith("event: "):
            continue
        head, _, data = block.partition("\ndata: ")
        parsed.append((head.removeprefix("event: "), json.loads(data)))
    return parsed


def start(client: TestClient, game_id: str = GAME_ID) -> str:
    """开一次会话。澄清与作答都挂在它上面。"""
    response = client.post("/api/chat/sessions", json={"game_id": game_id})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def ask(client: TestClient, session_id: str, question: str = QUESTION, **params):
    response = client.get(
        f"/api/chat/sessions/{session_id}/ask", params={"question": question, **params}
    )
    assert response.status_code == 200, response.text
    return parse(response.text)


def kinds(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


def only(events: list[tuple[str, dict]], name: str) -> dict:
    """流里那唯一一条某类事件的载荷。有两条或没有都当场报出来。"""
    found = [payload for event, payload in events if event == name]
    assert len(found) == 1, f"{name} 事件应当只有一条，实际 {kinds(events)}"
    return found[0]


def scripted(container, *replies) -> None:
    """换一套模型脚本。断言的是响应，脚本只负责把模型那一侧摆成想要的样子。"""
    container.llm.replies = list(replies)


def clarified(container, client) -> Pending:
    """走一遍「判不准 → 反问」，把那次反问交回来。"""
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    session_id = start(client)
    payload = only(ask(client, session_id), "clarification")
    return {"session_id": session_id, **payload}


# --- 作答 ---


def test_确定时直接作答并带上引用(client):
    """判得出游戏与版本就作答，流里是来源、逐字正文、收尾。"""
    events = ask(client, start(client))

    citations = only(events, "citations")
    assert citations["citations"] == [
        {
            "index": 1,
            "label": "二郎神",
            "doc_title": "二郎神",
            "ancestor_path": "",
        }
    ]
    assert citations["version"] == "1.0"
    assert "".join(payload["text"] for event, payload in events if event == "delta") == REPLY
    assert kinds(events)[-1] == "done"


# --- 反问 ---


def test_拿不准时给候选按钮并且不生成答案(container, client):
    """反问就是反问：流里没有来源也没有正文，候选摆出来等用户点。"""
    pending = clarified(container, client)

    assert pending["dimension"] == GAME
    assert pending["prompt"]
    assert [choice["label"] for choice in pending["choices"]] == ["黑神话·悟空", "燕云十六声"]
    assert [choice["value"] for choice in pending["choices"]] == [GAME_ID, OTHER_ID]


def test_反问那一轮不发_done(container, client):
    """`done` 的意思是「这一轮问完了」。反问还没问完，页面不该据此收尾。"""
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)

    events = ask(client, start(client))

    assert kinds(events) == ["status", "clarification"]


def test_版本拿不准时给的是这个库里真实有过的版本(container, client):
    """版本候选从这个库的切片里读，未标注版本不在其中（它随任何版本一起被检索到）。"""
    scripted(container, joint_reply(version_confidence=UNSURE), REPLY)

    payload = only(ask(client, start(client)), "clarification")

    assert payload["dimension"] == VERSION
    assert [choice["label"] for choice in payload["choices"]] == ["1.0", "2.0"]


def test_反问把暂停点落下来(container, client):
    """用户点的那一下是另一次请求，中间只有这条记录连着。"""
    pending = clarified(container, client)

    assert container.docs.get(PENDING_COLLECTION, pending["pending_id"]) is not None


def test_反问的那一轮不写进会话(container, client):
    """还没问完的一轮不该在历史里留下一条等不到回复的提问。"""
    pending = clarified(container, client)

    conversation = client.get(f"/api/chat/sessions/{pending['session_id']}").json()
    assert conversation["turns"] == []


# --- 从暂停点继续 ---


def test_点完候选从暂停点继续给出答案(container, client):
    pending = clarified(container, client)

    events = ask(
        client, pending["session_id"], pending_id=pending["pending_id"], label="黑神话·悟空"
    )

    assert "".join(payload["text"] for event, payload in events if event == "delta") == REPLY
    assert kinds(events)[-1] == "done"


def test_恢复时不重新问一遍理解(container, client):
    """暂停点里存着已经判出来的东西：恢复只该再生成一次，不该再理解一次。

    多问一次不只是慢：改写结果会变，同一个问题两次点出不同的问法，缓存也就命不中了。
    """
    pending = clarified(container, client)
    before = len(container.llm.calls)

    ask(client, pending["session_id"], pending_id=pending["pending_id"], label="黑神话·悟空")

    assert len(container.llm.calls) == before + 1


def test_恢复多次结果一致且不多写记录(container, client):
    """§3.4 的坑：恢复会把那一步从头重跑，副作用必须外提或幂等。

    这里恢复这条路上一个写操作都没有：待澄清记录只在暂停那一刻写一条。
    """
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY, REPLY)
    session_id = start(client)
    pending = only(ask(client, session_id), "clarification")

    first = ask(client, session_id, pending_id=pending["pending_id"], label="黑神话·悟空")
    second = ask(client, session_id, pending_id=pending["pending_id"], label="黑神话·悟空")

    assert first == second
    assert container.docs.list_ids(PENDING_COLLECTION) == [pending["pending_id"]]


def test_恢复之后那一轮落进会话(container, client):
    """反问不算一轮，恢复之后的那一问一答才算——历史里存的是用户的原话。"""
    pending = clarified(container, client)

    ask(client, pending["session_id"], pending_id=pending["pending_id"], label="黑神话·悟空")

    conversation = client.get(f"/api/chat/sessions/{pending['session_id']}").json()
    assert [turn["content"] for turn in conversation["turns"]] == [QUESTION, REPLY]
    assert conversation["turns"][1]["citations"][0]["doc_title"] == "二郎神"


def test_选的东西不在候选里时流里报错(container, client):
    """按钮之外的值说明这个请求不是那份暂停点发出来的，不能拿它去检索。

    这一条在流里交代而不是状态码：用户点错的那一下发生在一个已经打开的连接上。
    """
    pending = clarified(container, client)

    events = ask(
        client, pending["session_id"], pending_id=pending["pending_id"], label="塞尔达传说"
    )

    assert kinds(events)[-1] == "error"
    assert "塞尔达传说" in only(events, "error")["message"]


def test_恢复一个不存在的暂停点时报错(client):
    events = ask(client, start(client), pending_id="没有这个暂停点", label="黑神话·悟空")

    assert kinds(events)[-1] == "error"


# --- 说不通的几种 ---


def test_空问题报_400(client):
    """空问题会让检索查出任意一批切片，模型照着它编一段答案。开流之前就拦掉。"""
    response = client.get(f"/api/chat/sessions/{start(client)}/ask", params={"question": "   "})

    assert response.status_code == 400


def test_缺问题字段报_422(client):
    """请求少给一个字段是请求本身不合法，与上面几种「找得到但说不通」分开。"""
    response = client.get(f"/api/chat/sessions/{start(client)}/ask")

    assert response.status_code == 422
