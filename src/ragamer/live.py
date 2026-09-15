"""进行中的轮次：一轮问答从「页面的连接」解绑，挂到会话上。

对话页那条 SSE 只是一条**转发通道**。这一轮真正跑在哪由这里说了算——页面断开
（切走、刷新、关标签）只停转发，那一轮照跑、照落库；只有显式取消才让它收手。

三个取舍：

- **一轮归属会话，不归属请求**。请求会结束，会话一直在。挂在请求上就会得到
  「切走 = 白问 + 白花钱」——那正是这一层要消掉的东西。
- **同一个会话同时只允许一轮**。两轮各自读同一份会话、各自追加一轮存回去，后写的
  那一份会把先写的整个盖掉，而两边的答案都真实生成过。挡在第二问进来的时候，比
  事后去合并两份历史简单得多，也不会合错。
- **状态在内存里**。服务重启就没了，但那一轮本身也活不过重启（跑它的线程没了），
  与 `ragamer.jobs` 的导入任务同一个姿势。

队列无界是有意的：一轮最多几十条事件加几百片正文，页面断开后没人取也堆不出多少，
而**限流反而会让后台那一轮卡在 put 上**——它正是要跑完才落库的那个。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import TYPE_CHECKING

from ragamer.logging import get_logger

if TYPE_CHECKING:
    # 只为注解：这个模块是 `conversations` 的下游，运行时反过来导入就成环了
    from ragamer.conversations import Reply

logger = get_logger(__name__)

#: 队列里的结束哨兵。与 `Reply` 和异常都区分得开。
END = object()

#: 「这一轮被取消了」的哨兵。**与 :data:`END` 分开**：取消的那一轮没走完，
#: 转发那层据此不发 `done`——发了页面就会以为答完了。
CANCELLED = object()


class TurnCancelled(Exception):
    """这一轮被用户取消了，**不写进会话**：当没问过。

    抛出点是各步之间的检查点，所以它一路上抛时落库那行根本执行不到——「取消不留痕」
    与「断开不留痕」是同一个机制，只是一个由用户点出来、一个由连接断掉触发。

    接住它的是跑这一轮的后台线程（:meth:`TurnRegistry._pump`）：它是取消，不是失败，
    界面上不该出现一条错误，日志里也只记一条 INFO。

    **住在这里而不是 `ragamer.conversations`**：检索与生成那两层也要抛它，而它们
    都在 `conversations` 的上游——定义放在那边就得反过来导入，成环。
    """


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    """问一次要不要收手。答「是」就抛 :class:`TurnCancelled`，不给就是不取消。

    **两个检查点之间隔着的那一段是打不断的**：一次模型往返、一次向量化、一次精排，
    都只能等它自己返回。取消因此最坏要等到当前那一步跑完——检索里那次精排最长
    二十几秒，是这条路上最难等的一段。
    """
    if cancelled is not None and cancelled():
        raise TurnCancelled


@dataclass
class LiveTurn:
    """一个会话上正在跑的那一轮。

    `cancelled` 是**协作式**的取消信号：跑这一轮的线程在每一步之间查它一次，
    查到了就收手（`ragamer.conversations` 的检查点）。它不是抢占——正在跑的
    那一次模型调用或精排要等它自己返回。
    """

    session_id: str
    cancelled: threading.Event = field(default_factory=threading.Event)
    queue: SimpleQueue[object] = field(default_factory=SimpleQueue)
    thread: threading.Thread | None = None


class TurnRegistry:
    """这个进程里正在跑的那几轮，按会话索引。

    线程安全：`start` 与 `_pump` 的收尾都在这把锁里改字典。取事件不走锁——队列本身是
    线程安全的，而拿锁去等一个可能要几十秒才来的事件会把 `cancel` 一起堵住。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: dict[str, LiveTurn] = {}

    def start(
        self, session_id: str, replies: Callable[[LiveTurn], Iterator[Reply]]
    ) -> LiveTurn | None:
        """登记一轮并起一个线程把它跑完。**这个会话已经有一轮在跑时返回 `None`。**

        `replies` 拿到的那一个 `LiveTurn` 是**取消信号唯一的来源**：把
        `turn.cancelled.is_set` 传进这一轮的实现，它才知道该在哪一步收手。信号从参数
        进来而不是从外面闭包里取——工厂是在**新线程**里调用的，闭包捕获的那个名字
        那时可能还没绑定上。

        `replies` 本身是一个**还没开始跑的**生成器工厂，在后台线程里才第一次迭代它：
        传一个已经建好的生成器进来，理解与检索就会提前到调用方那一侧跑。
        """
        with self._lock:
            if session_id in self._turns:
                return None
            turn = LiveTurn(session_id=session_id)
            self._turns[session_id] = turn
        turn.thread = threading.Thread(
            target=self._pump,
            args=(turn, replies),
            name=f"ragamer-turn-{session_id[:8]}",
            daemon=True,
        )
        turn.thread.start()
        return turn

    def cancel(self, session_id: str) -> bool:
        """让这个会话上正在跑的那一轮收手。**没有在跑的返回 `False`。**

        置位是立刻的，收手不是：跑这一轮的线程要到下一个检查点才看得见它，
        而检查点之间可能隔着一次模型调用或一次精排（最长二十几秒）。
        """
        with self._lock:
            turn = self._turns.get(session_id)
        if turn is None:
            return False
        turn.cancelled.set()
        logger.info("会话 %s 的这一轮被要求停下", session_id)
        return True

    def running(self, session_id: str) -> bool:
        """这个会话上有没有正在跑的一轮。"""
        with self._lock:
            return session_id in self._turns

    def _pump(self, turn: LiveTurn, replies: Callable[[LiveTurn], Iterator[Reply]]) -> None:
        """把这一轮跑到底，事件推进队列。

        **这个线程不属于任何请求**：页面断开、刷新、切走都不影响它，跑完照常落库
        （落库在 `ragamer.conversations._replies` 的最后一行）。被取消时不落库——
        收手走的是异常，落库那行根本执行不到。
        """
        try:
            for reply in replies(turn):
                turn.queue.put(reply)
        except TurnCancelled:
            logger.info("会话 %s 的这一轮已取消，什么都没写", turn.session_id)
            turn.queue.put(CANCELLED)
        except Exception as exc:  # noqa: BLE001
            # 失败的处置交给转发那一层：它知道该发哪种 `error` 事件
            turn.queue.put(exc)
        finally:
            turn.queue.put(END)
            with self._lock:
                self._turns.pop(turn.session_id, None)
