"""多轮对话：历史进理解、改写后的问题去检索、一轮问答整份落库。

五组用例：**接着上一句**（上一轮进提问理解那一步，检索用的是改写后的问题）、
**刷新之后**（会话从库里读回来，原话与引用都在）、**互不串扰**（两次会话各有各的历史）、
**中途断掉**（客户端丢掉迭代器、生成炸了，都不留半个答案）、
**边界与兜底**（空问题、会话不存在、历史窗口、检索不到时那句回复也进历史）。

模型接 `FakeLlm`，一次网络都不发。改写那一步因此只能靠脚本摆出来——**它证明的是接线**
（历史有没有进到那一步、改写后的问题有没有被拿去检索、判出来的游戏有没有换回 id），
至于模型在中文上真的能不能把「那它掉什么」补回主体名，那是 `test_query_integration.py`
用真模型证的事。
"""

from __future__ import annotations

import pytest

from ragamer.answering import NOT_FOUND
from ragamer.api import KB_COLLECTION
from ragamer.conversations import (
    HISTORY_TURNS,
    Chat,
    Cited,
    Conversation,
    ConversationNotFound,
    Delta,
)
from ragamer.llm import FakeLlm, LlmTimeout
from ragamer.vectors.fake import FakeReranker

from .conftest import (
    HalfwayLlm,
    RecordingChunkStore,
    chunk_store,
    make_chat,
    make_chunk,
    make_container,
)

GAME = "black_myth"
#: 这个库的元数据。`name` 是显示名——用户问句里出现的是它，不是 id。
KB = {"name": "黑神话·悟空", "version": "1.0", "subject_types": ["character"]}
#: 游戏候选：显示名 → 知识库 id。`ragamer.api` 从知识库列表里查出这一对。
GAMES = (("黑神话·悟空", GAME),)

#: 库里唯一一份资料。追问「掉什么」时它随父块一起进来。
DOC = make_chunk(
    1, content="二郎神掉落三尖两刃刀", doc_title="二郎神", ancestor_path="二郎神 › 掉落"
)


def said(rewritten: str, game: str = "", version: str = "") -> dict[str, str]:
    """提问理解那一步的脚本：一次联合输出。"""
    return {"game": game, "version": version, "rewritten_query": rewritten}


def setup_chat(llm, *chunks, reranker=None) -> Chat:
    """对话侧的内存版：知识库建好、资料入库、模型按脚本回话。"""
    container = make_container(llm=llm, chunks=chunk_store(GAME, *chunks), reranker=reranker)
    container.docs.put(KB_COLLECTION, GAME, KB)
    return make_chat(container)


def recording_chat(llm) -> tuple[Chat, RecordingChunkStore]:
    """同上，但切片存储记下每次检索收到的参数——用来看这一轮进的是哪个知识库。"""
    chunks = RecordingChunkStore()
    chunks.upsert(GAME, [DOC])
    container = make_container(llm=llm, chunks=chunks)
    container.docs.put(KB_COLLECTION, GAME, KB)
    return make_chat(container), chunks


def asked(chat: Chat, conversation: Conversation, question: str, **kwargs) -> list:
    """问一句并**把整条流收完**——收完才会落库，这正是多数用例要的前置状态。"""
    return list(chat.ask(conversation.session_id, question, **kwargs))


# --- 接着上一句 ---


def test_接着上一句问时上一轮进了提问理解那一步():
    """「那它掉什么」里的「它」是谁只能从上一轮看出来——所以上一轮要进那一步的提示词。"""
    llm = FakeLlm(
        said("二郎神是谁"),
        "二郎神是隐藏 BOSS[1]。",
        said("二郎神掉什么"),
        "掉的是三尖两刃刀[1]。",
    )
    chat = setup_chat(llm, DOC)
    conversation = chat.start(game_id=GAME)

    asked(chat, conversation, "二郎神是谁")
    asked(chat, conversation, "那它掉什么")

    # 调用顺序是「理解、生成、理解、生成」，所以第 2 次理解在下标 2
    messages = [(message.role, message.content) for message in llm.calls[2].messages]
    assert ("user", "二郎神是谁") in messages
    assert ("assistant", "二郎神是隐藏 BOSS[1]。") in messages


def test_检索用的是改写之后的问题():
    """改写是为了把指代补全——不拿它去检索，这一步就白做了，而且答案会悄悄变差。"""
    reranker = FakeReranker()
    llm = FakeLlm(said("二郎神掉什么"), "掉的是三尖两刃刀[1]。")
    chat = setup_chat(llm, DOC, reranker=reranker)
    conversation = chat.start(game_id=GAME)

    asked(chat, conversation, "那它掉什么")

    assert reranker.calls[0][0] == "二郎神掉什么"


def test_判出来的游戏显示名换回知识库_id():
    """候选是显示名（用户问句里会出现的写法），检索要的是 id（collection 名）。

    会话选的是另一个库，所以「用了判出来的那一个」与「回落会话选的那一个」在结果上分得开。
    """
    llm = FakeLlm(said("二郎神掉什么", game="黑神话·悟空"), "掉的是三尖两刃刀[1]。")
    chat, chunks = recording_chat(llm)
    conversation = chat.start(game_id="another_game")

    asked(chat, conversation, "那它掉什么", games=GAMES)

    assert chunks.searches[0]["game_id"] == GAME


def test_没有候选时回落会话选定的知识库():
    """判不出游戏不是错误：会话建的时候已经选过一个了，那一轮就在它里面查。"""
    chat, chunks = recording_chat(FakeLlm(said("二郎神掉什么"), "掉的是三尖两刃刀[1]。"))
    conversation = chat.start(game_id=GAME)

    asked(chat, conversation, "那它掉什么")

    assert chunks.searches[0]["game_id"] == GAME


# --- 刷新之后 ---


def test_刷新之后历史还在():
    """历史在服务端，不在页面里——换个浏览器打开，或者刷新一下，都还读得回来。"""
    llm = FakeLlm(said("二郎神掉什么"), "掉的是三尖两刃刀[1]。")
    chat = setup_chat(llm, DOC)
    conversation = chat.start(game_id=GAME, version="1.0")

    asked(chat, conversation, "那它掉什么")

    again = chat.open(conversation.session_id)
    assert (again.game_id, again.version) == (GAME, "1.0")
    assert [(turn.role, turn.content) for turn in again.turns] == [
        # 存的是**用户的原话**：改写是给检索用的中间产物，历史要给人看
        ("user", "那它掉什么"),
        ("assistant", "掉的是三尖两刃刀[1]。"),
    ]
    # 引用跟着答案一起回来，正文里的 [1] 因此刷新之后仍对得上号
    assert [citation.label for citation in again.turns[1].citations] == ["二郎神"]
    assert again.turns[1].citations[0].index == 1
    assert again.turns[0].citations == ()


# --- 互不串扰 ---


def test_两次会话各有各的历史():
    llm = FakeLlm(
        said("二郎神是谁"),
        "二郎神是隐藏 BOSS[1]。",
        said("二郎神掉什么"),
        "掉的是三尖两刃刀[1]。",
    )
    chat = setup_chat(llm, DOC)
    first = chat.start(game_id=GAME)
    second = chat.start(game_id=GAME)

    asked(chat, first, "二郎神是谁")
    asked(chat, second, "那它掉什么")

    assert [turn.content for turn in chat.open(first.session_id).turns] == [
        "二郎神是谁",
        "二郎神是隐藏 BOSS[1]。",
    ]
    assert [turn.content for turn in chat.open(second.session_id).turns] == [
        "那它掉什么",
        "掉的是三尖两刃刀[1]。",
    ]
    # 第二次会话的第一次提问，理解那一步里只有它自己那一句（前面那条是系统提示）
    assert [(message.role, message.content) for message in llm.calls[2].messages][1:] == [
        ("user", "那它掉什么")
    ]


# --- 中途断掉 ---


def test_客户端中途断开时不留半个答案():
    """生成到一半客户端断了：已经吐出去的那半句与它那批引用一起不要，历史里不留痕。"""
    llm = FakeLlm(said("二郎神掉什么"), "掉的是三尖两刃刀[1]。")
    chat = setup_chat(llm, DOC)
    conversation = chat.start(game_id=GAME)

    replies = chat.ask(conversation.session_id, "那它掉什么")
    assert isinstance(next(replies), Cited)
    assert isinstance(next(replies), Delta)  # 已经吐了一片出去
    replies.close()  # 客户端在这时候断了

    assert chat.open(conversation.session_id).turns == ()


def test_生成中途失败时不留半个答案():
    """模型炸在生成中间：同样什么都不写——半截答案配一份完整引用，指向的是没写到的来源。"""
    chat = setup_chat(HalfwayLlm(said("二郎神掉什么")), DOC)
    conversation = chat.start(game_id=GAME)

    with pytest.raises(LlmTimeout):
        list(chat.ask(conversation.session_id, "那它掉什么"))

    assert chat.open(conversation.session_id).turns == ()


# --- 边界与兜底 ---


def test_空问题当场报错():
    """空问题会让检索查出任意一批切片，模型照着它编一段答案。"""
    chat = setup_chat(FakeLlm())
    conversation = chat.start(game_id=GAME)

    with pytest.raises(ValueError):
        chat.ask(conversation.session_id, "   ")


def test_没有这个会话时当场报错():
    """找不到的会话不能被当成空会话——那样每刷新一次就多一个会话出来。"""
    chat = setup_chat(FakeLlm())

    with pytest.raises(ConversationNotFound):
        chat.open("没有这个会话")


def test_历史只带最近几轮():
    """会话是不封顶的：整段历史都进提示词，几十轮之后会把模型的上下文顶穿。"""
    script = [item for index in range(5) for item in (said(f"第{index}问"), f"第{index}答。")]
    llm = FakeLlm(*script)
    chat = setup_chat(llm, DOC)
    conversation = chat.start(game_id=GAME)

    for index in range(5):
        asked(chat, conversation, f"第{index}问")

    # 第 5 次提问的理解排在下标 8（一问一答两次调用），掐掉首尾两条系统提示与当前问题
    history = [(message.role, message.content) for message in llm.calls[8].messages][1:-1]
    assert len(history) == HISTORY_TURNS * 2
    # 问第 4 问时前面已有四轮，带进去的是最后三轮（第 1、2、3 问）
    assert history[0] == ("user", "第1问")
    assert history[-1] == ("assistant", "第3答。")


def test_检索不到时那句明确回复也进历史():
    """没找到也是一次完整的回答：用户问了、系统答了，下一轮接着问时它还该在。"""
    llm = FakeLlm(said("不存在的东西怎么打"))
    chat = setup_chat(llm)  # 库里什么都没有
    conversation = chat.start(game_id=GAME)

    replies = list(chat.ask(conversation.session_id, "不存在的东西怎么打"))

    assert [reply.citations for reply in replies if isinstance(reply, Cited)] == [()]
    assert [turn.content for turn in chat.open(conversation.session_id).turns] == [
        "不存在的东西怎么打",
        NOT_FOUND,
    ]
