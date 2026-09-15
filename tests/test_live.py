"""进行中的轮次：一轮问答脱离页面之后归谁管。

这一层要证的是**归属**：那一轮跑在它自己的线程里，页面断开（= 队列没人取）不影响它跑完；
显式取消才让它收手，而且收手走的是取消哨兵，不是结束哨兵——转发那层据此不发 `done`。

线程与队列都用真的，不换替身：这一层的全部内容就是这两样的配合，把线程换成假的，
剩下的东西就没什么可证的了。
"""

from __future__ import annotations

import threading
import time

import pytest

from ragamer.conversations import Delta, Status
from ragamer.live import CANCELLED, END, LiveTurn, TurnCancelled, TurnRegistry, check_cancelled


def drain(turn: LiveTurn, limit: int = 500) -> list[object]:
    """把队列取到结束哨兵为止。返回的东西按进队列的先后。"""
    got: list[object] = []
    for _ in range(limit):
        item = turn.queue.get(timeout=5)
        got.append(item)
        if item is END:
            break
    return got


def test_一轮跑完的东西按顺序进队列():
    turn = TurnRegistry().start("s1", lambda live: iter([Status("一"), Delta("二")]))
    assert turn is not None
    assert drain(turn) == [Status("一"), Delta("二"), END]


def test_没人取队列时那一轮照样跑到底():
    """页面断开就是这个形状：转发那个生成器没了，队列没人取。**那不叫失败**。"""
    finished = threading.Event()

    def one(live):
        yield Status("一")
        finished.set()

    turn = TurnRegistry().start("s1", one)
    assert turn is not None
    assert finished.wait(5)
    assert drain(turn) == [Status("一"), END]


def test_同一个会话上的第二问被挡回来():
    """两轮同时写同一份会话，后写的会把先写的整个盖掉，而两边都真答过。"""
    registry = TurnRegistry()
    held = threading.Event()

    def stuck(live):
        held.wait(10)
        yield Status("一")

    try:
        assert registry.start("s1", stuck) is not None
        assert registry.start("s1", lambda live: iter([])) is None
        assert registry.running("s1")
    finally:
        held.set()


def test_一轮跑完之后这个会话又能起新一轮():
    registry = TurnRegistry()
    turn = registry.start("s1", lambda live: iter([]))
    assert turn is not None
    drain(turn)
    assert turn.thread is not None
    turn.thread.join(5)
    assert not registry.running("s1")
    assert registry.start("s1", lambda live: iter([])) is not None


def test_取消让那一轮收手且收尾放的是取消哨兵():
    registry = TurnRegistry()
    started = threading.Event()

    def long_one(live):
        started.set()
        for index in range(2000):
            check_cancelled(live.cancelled.is_set)
            yield Delta(str(index))
            time.sleep(0.002)

    turn = registry.start("s1", long_one)
    assert turn is not None
    assert started.wait(5)
    assert registry.cancel("s1")
    got = drain(turn)
    assert CANCELLED in got
    # 取消哨兵紧挨着结束哨兵：转发那层读到它就不发 `done`
    assert got[-2:] == [CANCELLED, END]


def test_没有在跑的一轮时取消返回假():
    """用户点「停止」的那一刻那一轮可能刚好答完，那不是错误。"""
    assert TurnRegistry().cancel("s1") is False


def test_那一轮抛出来的异常进队列():
    """失败不能烂在线程里：转发那层要靠它发一条 `error`。"""

    def boom(live):
        raise RuntimeError("炸了")
        yield  # pragma: no cover

    turn = TurnRegistry().start("s1", boom)
    assert turn is not None
    got = drain(turn)
    assert isinstance(got[0], RuntimeError)
    assert got[-1] is END


def test_检查点按信号抛():
    check_cancelled(None)
    check_cancelled(lambda: False)
    with pytest.raises(TurnCancelled):
        check_cancelled(lambda: True)


def test_没有在跑的一轮时快照是空():
    assert TurnRegistry().snapshot("s1") is None


def test_快照带着问的那句与已经吐出来的字():
    """切回来接着看靠它：那一轮没跑完不落库，**「我问了什么」只有这里记得住**。"""
    registry = TurnRegistry()
    held = threading.Event()
    seen = 0

    def watch(turn: LiveTurn, reply: object) -> None:
        nonlocal seen
        with turn.lock:
            turn.snapshot.text += str(reply)
        seen += 1

    def two(live):
        yield "一"
        yield "二"
        held.wait(10)

    try:
        turn = registry.start("s1", two, question="二郎神怎么打", watch=watch)
        assert turn is not None
        while seen < 2:
            time.sleep(0.01)
        state = registry.snapshot("s1")
        assert state is not None
        assert state["question"] == "二郎神怎么打"
        assert state["text"] == "一二"
    finally:
        held.set()
