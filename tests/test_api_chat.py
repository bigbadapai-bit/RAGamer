"""提问端点（读取侧的主缝）：反问与作答都在这一层看得见。

只断言外部可观察的行为：HTTP 响应本身，以及通过存储适配器的查询接口能观察到的
落库结果。**暂停与恢复必须在这一层成立**——反问是一条 HTTP 响应，用户点完按钮是
另一次 HTTP 请求，中间那座桥（待澄清记录）只能从存储上观察。

模型那一段接 `FakeLlm`（一次网络都不发），检索那段接内存假件与确定性假件。
"""

from __future__ import annotations

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
CHAT_URL = "/api/chat"

BLACK_MYTH = {"name": "黑神话·悟空", "version": "1.0"}
YANYUN = {"name": "燕云十六声", "version": "3.0"}

#: 一段答案。编号指的就是提示词里那批切片的编号。
REPLY = "先定身再贴身输出[1]。"


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


def ask(client: TestClient, question: str = QUESTION, **fields):
    response = client.post(CHAT_URL, data={"question": question, **fields})
    assert response.status_code == 200, response.text
    return response.json()


def scripted(container, *replies) -> None:
    """换一套模型脚本。断言的是响应，脚本只负责把模型那一侧摆成想要的样子。"""
    container.llm.replies = list(replies)


# --- 作答 ---


def test_确定时直接作答并带上引用(client):
    """判得出游戏与版本就作答，响应里是答案与它的来源。"""
    payload = ask(client, game_id=GAME_ID)

    assert payload["kind"] == "answer"
    assert payload["text"] == REPLY
    assert payload["citations"] == [
        {
            "index": 1,
            "label": "二郎神 › 打法",
            "doc_title": "二郎神",
            "ancestor_path": "二郎神 › 打法",
        }
    ]


# --- 反问 ---


def test_拿不准时给候选按钮并且不生成答案(container, client):
    """反问就是反问：响应里没有答案，候选摆出来等用户点。"""
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)

    payload = ask(client, game_id=GAME_ID)

    assert payload["kind"] == "clarification"
    assert payload["dimension"] == GAME
    assert payload["prompt"]
    assert [choice["label"] for choice in payload["choices"]] == ["黑神话·悟空", "燕云十六声"]
    assert [choice["value"] for choice in payload["choices"]] == [GAME_ID, OTHER_ID]


def test_版本拿不准时给的是这个库里真实有过的版本(container, client):
    """版本候选从这个库的切片里读，未标注版本不在其中（它随任何版本一起被检索到）。"""
    scripted(container, joint_reply(version_confidence=UNSURE), REPLY)

    payload = ask(client, game_id=GAME_ID)

    assert payload["kind"] == "clarification"
    assert payload["dimension"] == VERSION
    assert [choice["label"] for choice in payload["choices"]] == ["1.0", "2.0"]


def test_反问把暂停点落下来(container, client):
    """用户点的那一下是另一次请求，中间只有这条记录连着。"""
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)

    payload = ask(client, game_id=GAME_ID)

    assert container.docs.get(PENDING_COLLECTION, payload["pending_id"]) is not None


# --- 从暂停点继续 ---


def test_点完候选从暂停点继续给出答案(container, client):
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    pending = ask(client, game_id=GAME_ID)

    response = client.post(f"{CHAT_URL}/{pending['pending_id']}", data={"label": "黑神话·悟空"})

    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "answer"
    assert response.json()["text"] == REPLY


def test_恢复时不重新问一遍理解(container, client):
    """暂停点里存着已经判出来的东西：恢复只该再生成一次，不该再理解一次。

    多问一次不只是慢：改写结果会变，同一个问题两次点出不同的问法，缓存也就命不中了。
    """
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    pending = ask(client, game_id=GAME_ID)
    before = len(container.llm.calls)

    client.post(f"{CHAT_URL}/{pending['pending_id']}", data={"label": "黑神话·悟空"})

    assert len(container.llm.calls) == before + 1


def test_恢复多次结果一致且不多写记录(container, client):
    """§3.4 的坑：恢复会把那一步从头重跑，副作用必须外提或幂等。

    这里恢复这条路上一个写操作都没有：待澄清记录只在暂停那一刻写一条。
    """
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY, REPLY)
    pending = ask(client, game_id=GAME_ID)
    url = f"{CHAT_URL}/{pending['pending_id']}"

    first = client.post(url, data={"label": "黑神话·悟空"}).json()
    second = client.post(url, data={"label": "黑神话·悟空"}).json()

    assert first == second
    assert container.docs.list_ids(PENDING_COLLECTION) == [pending["pending_id"]]


def test_选的东西不在候选里报_422(container, client):
    """按钮之外的值说明这个请求不是那份暂停点发出来的，不能拿它去检索。"""
    scripted(container, joint_reply(game="燕云十六声", game_confidence=UNSURE), REPLY)
    pending = ask(client)

    response = client.post(f"{CHAT_URL}/{pending['pending_id']}", data={"label": "塞尔达传说"})

    assert response.status_code == 422


def test_恢复一个不存在的暂停点报_404(client):
    response = client.post(f"{CHAT_URL}/没有这个暂停点", data={"label": "黑神话·悟空"})

    assert response.status_code == 404


# --- 说不通的几种 ---


def test_空问题报_400(client):
    """空问题会让检索查出任意一批切片，模型照着它编一段答案。"""
    assert client.post(CHAT_URL, data={"question": "   "}).status_code == 400


def test_一个库都没有时报_404(container, client):
    """没有候选可问、也没有已定的游戏可依：这一问无解，说清是「还没建库」。"""
    container.docs.delete(KB_COLLECTION, GAME_ID)
    container.docs.delete(KB_COLLECTION, OTHER_ID)
    scripted(container, joint_reply(game=""))

    response = client.post(CHAT_URL, data={"question": QUESTION})

    assert response.status_code == 404
    assert "知识库" in response.json()["detail"]


def test_缺问题字段报_422(client):
    """请求少给一个字段是请求本身不合法，与上面几种「找得到但说不通」分开。"""
    assert client.post(CHAT_URL, data={}).status_code == 422
