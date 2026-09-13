"""多轮对话：一次提问接着上一次，答案逐字流出来。

会话是一份**落在 MongoDB 里的记录**（`conversations` 集合，文档 id 就是会话 id）：
刷新浏览器、换台机器打开，历史都还在；两次会话各有各的历史，互不串扰。
只在客户端存（localStorage）做不到第一条，只放在进程内存里两条都做不到。

四件事在这里定死：

- **指代靠历史补全，检索用的是改写后的问题**。用户问「那它掉什么」，「它」是谁只能
  从上一轮看出来——所以历史进 `ragamer.query.understand`，检索用它吐出来的
  `rewritten_query`（「二郎神掉什么」）。会话里记下的仍是**用户的原话**：历史要给人看，
  改写是给检索用的中间产物。**版本不参与这一步的判定**，理由见 :meth:`Chat.ask`。
- **走哪几路召回由问题类型决定**，类型是 `ragamer.query.understand` 那一次调用的产物
  （多吐一个字段，不多一次调用）。组合本身是知识库里的一份配置（`ragamer.routing`），
  这一层只负责把「判出来的类型」对到「这一类的组合」上再传下去。
- **历史只进提问理解，不进生成**。生成拿到的是一句已经补全的问题加一批原文父块，
  引用因此永远指向语料。把上一轮的**答案**也塞进提示词，模型就有了一处引用不到的
  来源可以顺着往下编，而它看起来与真答案一模一样。
- **整轮一起落库，收完正文才写**。用户的问题与模型的答案是同一条记录：生成中途断掉
  （客户端断开、页面关掉、模型炸了）时什么都不写，历史里不会留下半句答案，也不会留下
  一条等不到回复的提问。引用只跟着**完整的**正文走——半截答案配一份完整的引用列表，
  指向的是正文里根本没写到的来源（`docs/ARCHITECTURE.md` §5 的对话页靠这份记录
  做「刷新后还在」）。
  **代价是那一问也跟着没了**，用户得重问一次；这是有意选的一侧——反过来（先把问题写进去、
  答案回头再补）会留下一个「问了但没答」的中间态，而刷新页面最容易撞上的就是它，
  下一轮的提问理解也会拿到一段半截上下文。
- **一问一答各占一条**，不是一次提问写一份。按轮存的话，展示侧要自己拆轮次，
  而拆法迟早会和 :data:`HISTORY_TURNS` 那条口径打起来。

会话绑定一个知识库（`game_id`：建会话时选的游戏），版本可以落在会话上，也可以每次提问时
临时点名。**这个库是落点不是牢笼**：问题里点名了另一个真实存在的库时，这一轮就去那儿查
（`ragamer.query.understand` 判得出来时优先用它）——那一步本来就在判问的是哪款游戏，
判出来却不用，等于把它接了个空。会话绑的那个不变，下一轮没点名就还回到它上面。

一轮要走完理解、检索、生成三步才吐得出第一个字，所以这三步各先报一条进度
（:class:`Status`）——「不用干等」是这张票要求的一部分。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from ragamer.answering import Answerer, Citation, require_question
from ragamer.llm import LlmClient, Message
from ragamer.logging import get_logger
from ragamer.query import understand
from ragamer.routing import DEFAULT_TABLE, QUERY_TYPE_LABELS, QueryType, RouteTable
from ragamer.stores.base import DocStore

logger = get_logger(__name__)

#: 会话所在的集合。文档 id 就是会话 id。删库要清四处里的「会话」就在这个集合里。
CONVERSATIONS = "conversations"

#: 带进提问理解的历史轮数（一问一答算一轮）。
#:
#: 留一条上限不是优化：会话是不封顶的，几十轮之后整段历史会把提示词顶到模型的上下文
#: 上限上去，而失败方式取决于服务端（截断、报错，或者慢到超时）。数值本身与项目里
#: 其它阈值一样是**没有评测集时的占位**（`docs/ARCHITECTURE.md` §11），这里能讲清的
#: 只有「指代基本指向上一两轮」。落不进窗口的旧轮次仍留在会话里，只是不参与理解。
HISTORY_TURNS = 3

#: 会话里一条消息的角色。**没有 `system`**：提示词是每一步自己拼的，不存进会话。
TurnRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    """会话里的一条消息。

    `citations` 只对模型那一侧非空：用户的话没有来源。它是**正文里 [n] 指的那批**，
    与当次 `ragamer.answering.AnswerStream.citations` 是同一份——刷新之后要靠它把编号
    对回原文。
    """

    role: TurnRole
    content: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Conversation:
    """一次会话：绑一个知识库，外加到目前为止的全部问答。

    不可变，改动都由 :meth:`Chat.ask` 落成新的一份——会话是「一次写整份」的，
    可变对象上做增量改动与它相冲。
    """

    session_id: str
    game_id: str
    #: 这次会话选定的版本。空串表示没选，检索时再回落知识库的现行版本。
    version: str = ""
    turns: tuple[Turn, ...] = ()


class ConversationNotFound(LookupError):
    """会话不存在：id 是编的，或者已经被删了。

    与「会话是空的」分开：空会话是个正常状态（刚建出来还没问过），
    而找不到的会话不该被当成空会话建一个出来——那样每刷新一次就多一个会话。
    """


#: 游戏候选的一项是**一对**：显示名 → 知识库 id（见 :meth:`Chat.ask` 的 `games`）。
#: 两个都是字符串，顺序从签名上看不出来，所以起个名字、每次都带注释地传。
Game = tuple[str, str]


@dataclass(frozen=True)
class Sources:
    """这一轮用到的来源。**流的第一件事**。

    引用在检索那一步就定下来了，比正文早得多。先把它交出去，界面才能一边吐字一边
    把来源列出来，而不是等正文吐完再闪一下。名字取「来源」而不是「引用」：后者是
    `ragamer.answering` 里那个 :class:`~ragamer.answering.Citation` 的名字，
    一个是对外的一批、一个是里面的一条，两个词分开用免得读岔。
    """

    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Status:
    """一条进度提示：这一轮现在走到哪了。

    **存在的理由是「不用干等」**。提问到第一个字之间隔着两次实打实的等待——理解那一步
    一次模型往返，检索那一步向量化加召回加精排——而正文要等它们都过去才开始流。没有这条
    事件，界面在这一段里没有任何东西可显示，用户看到的就是一个没反应的页面。

    文案就是给人看的一句话（`正在理解问题`），界面照原样显示即可，不要在页面里再拼一遍。
    """

    text: str


@dataclass(frozen=True)
class Delta:
    """正文的一小片。拼起来就是完整答案，也是最终写进会话的那一份。"""

    text: str


#: 一次提问流出来的东西，**按发生的先后**：若干 :class:`Status`、一个 :class:`Sources`、
#: 若干 :class:`Delta`。三类之间不是随手排的——状态出现在它描述的那一段**之前**，
#: 来源出现在检索之后、正文之前。
Reply = Status | Sources | Delta


@dataclass(frozen=True)
class Chat:
    """对话侧的唯一入口。外部依赖由组合根注入（见 `ragamer.container`）。

    与 `ragamer.importing.Importer`、`ragamer.answering.Answerer` 是同一个打法：
    做成不可变对象，一次接线反复使用。`docs` 是会话落库的那一处（MongoDB），
    换掉它整条对话链路在测试里就能跑起来。
    """

    docs: DocStore
    answerer: Answerer
    llm: LlmClient

    def start(self, *, game_id: str, version: str = "") -> Conversation:
        """开一次会话。

        id 在这里生成（uuid4），**不由调用方指定**：会话 id 会进 URL 与提示词之外的
        各处，自增数字能被猜、能被顺手下一个人拿到，随机串不能。

        :param game_id: 这次会话问哪个知识库。库里没有它也不会在这里报错——
            那是 HTTP 面该判的事（见 `ragamer.api`），这一层不认 HTTP 状态码。
        :param version: 这次会话选定的版本。空串即不选，检索时回落知识库的现行版本。
        """
        conversation = Conversation(session_id=uuid.uuid4().hex, game_id=game_id, version=version)
        _save(self.docs, conversation)
        logger.info("新建会话 %s：知识库 %s，版本 %r", conversation.session_id, game_id, version)
        return conversation

    def open(self, session_id: str) -> Conversation:
        """把一次会话读回来。刷新页面之后靠它把历史拿回来。

        :raises ConversationNotFound: 没有这个会话。
        """
        payload = self.docs.get(CONVERSATIONS, session_id)
        if payload is None:
            raise ConversationNotFound(f"会话 {session_id} 不存在")
        return _read(session_id, payload)

    def ask(
        self,
        session_id: str,
        question: str,
        *,
        version: str = "",
        current_version: str = "",
        games: Sequence[Game] = (),
        routes: RouteTable = DEFAULT_TABLE,
    ) -> Iterator[Reply]:
        """问一句，逐字拿回答案：若干条 :class:`Status`，一个 :class:`Sources`，
        然后若干个 :class:`Delta`。

        **只有「问得对不对」在调用时判**——问题为空、会话不存在都当场抛出来，调用方
        还能给出正常的状态码。**理解与检索本身推迟到迭代时做**，这样它们各自能先报一条
        进度（「正在理解问题」「正在检索资料」），调用方不必对着一个没有反应的连接干等。
        代价是这两步的失败只能当作流里的一件事；对浏览器原生的 `EventSource` 反而更好——
        非 2xx 时它什么都不告诉你，只有流里的 `error` 带得回原因。

        **「这一轮算不算问完」由消费方决定**：正常收完才写进会话；中途把迭代器丢掉
        （客户端断开、页面关掉）就什么都不写——见模块说明的第三条。

        **版本不在这里判**。`ragamer.query.understand` 的版本候选要的是「库里真实存在
        的版本清单」，而知识库里只存着一个「现行版本」，没有清单可给。拿唯一那个取值
        当候选等于什么也没判（判出来还是它），判错一个不存在的版本却会静默查空——
        所以版本只从会话选定或这次点名来。

        :param session_id: 问在哪次会话里。**每次都重新读一遍**——会话是上一个请求写下
            的东西，调用方手里攥着的那一份多半已经旧了（会话不可变，`ask` 落的是新的
            一份存回去）。传对象进来会让「隔一轮再问」静默丢掉中间那几轮。
        :param question: 用户的原话。写进会话的是它，不是改写之后那一句。
        :param version: 这次点名按哪个版本检索。空串即回落会话选定的那一个。
        :param current_version: 知识库标着的现行版本，会话与点名都没有时才轮到它。
        :param games: 游戏候选（:data:`Game`，显示名与知识库 id 成对）。**显示名是用户
            问句里会出现的那种写法**——知识库 id 是 collection 名，只能是英文标识符，
            拿 id 当候选，模型只会把「黑神话」判成不在候选里。判出来的显示名在这里换回
            id；没有候选、或者判不出来，都回落会话选定的知识库。
        :param routes: 这个知识库的路由表（`ragamer.routing`）。库里配了就传配的那份，
            不传就用默认表——组合是每个库一份的配置，与打标词表同一个姿势。
        :raises ConversationNotFound: 没有这个会话。
        :raises ValueError: 问题为空。空问题会让检索查出任意一批切片。
        """
        require_question(question)  # 拦在理解那一步之前：空问题没得可理解，别白调一次模型
        conversation = self.open(session_id)
        return self._replies(
            conversation,
            question,
            version=version or conversation.version,
            current_version=current_version,
            games=games,
            routes=routes,
        )

    def _replies(
        self,
        conversation: Conversation,
        question: str,
        *,
        version: str,
        current_version: str,
        games: Sequence[Game],
        routes: RouteTable,
    ) -> Iterator[Reply]:
        """把这一轮从头做到尾，**每一步之前先报一条进度**，最后收完正文才落库。

        进度那三条与这一轮真正干的事一一对应，顺序也一致：理解问题 → 检索资料 →
        生成答案。夹在中间的是来源——它比正文早得多，一拿到就先交出去，界面可以
        先列出来再等字。

        选路夹在理解与检索之间，**没有自己的进度条**：它是一次字典查表，不是一段等待。

        迭代器被丢掉时（客户端断开）最后那一行写不进会话——这正是要的效果：
        已经吐出去的那半句与它那批引用一起消失，历史里不留痕迹。
        """
        yield Status("正在理解问题")
        understanding = understand(
            question,
            llm=self.llm,
            games=[name for name, _ in games],
            history=_history(conversation.turns),
        )
        yield Status("正在检索资料")
        route = routes.route_for(understanding.query_type)
        stream = self.answerer.stream(
            understanding.rewritten_query,
            game_id=_game_id(understanding.game, games) or conversation.game_id,
            version=version,
            current_version=current_version,
            route=route,
        )
        logger.info(
            "会话 %s 提问 %r（改写为 %r，类型 %s），走 %s，用上 %d 条来源",
            conversation.session_id,
            question,
            understanding.rewritten_query,
            _type_name(understanding.query_type),
            "、".join(path.value for path in route.paths),
            len(stream.citations),
        )
        yield Sources(stream.citations)
        yield Status("正在生成答案")
        produced: list[str] = []
        for piece in stream.deltas:
            produced.append(piece)
            yield Delta(piece)
        _save(self.docs, _appended(conversation, question, "".join(produced), stream.citations))


def _history(turns: Sequence[Turn]) -> tuple[Message, ...]:
    """交给提问理解的上下文：最近几轮，按时间先后。

    截断只发生在**开头**：这些消息直接拼在系统提示之后，顺序就是发生顺序，
    掐掉最早的那几轮不会破坏「谁在回答谁」。一问一答成对入列，窗口按轮算就不必
    担心切出半轮——会话里本来就是成对写的（见 `_appended`）。
    """
    return tuple(Message(turn.role, turn.content) for turn in turns[-HISTORY_TURNS * 2 :])


def _type_name(query_type: QueryType | None) -> str:
    """日志里那个类型名的写法。

    「判不出」与「判成了事实型」要分得开：两者的走法一样（都按事实型那一行），
    但一个是提示词没判出来、一个是真的判成了这一类，排查时看的是不同的地方。
    """
    return "判不出" if query_type is None else QUERY_TYPE_LABELS[query_type]


def _game_id(picked: str, games: Sequence[Game]) -> str:
    """理解那一步判出的显示名换回知识库 id。判不出来就留空，由调用方回落。

    `understand` 已经把候选之外的取值丢掉了，所以这里对不上只剩一种可能：
    调用方压根没给候选。那种情况下留空是对的——宁可用会话选定的知识库，
    也不要拿一个换不出 id 的名字去检索。
    """
    if not picked:
        return ""
    return next((game_id for name, game_id in games if name == picked), "")


def _appended(
    conversation: Conversation,
    question: str,
    answer: str,
    citations: tuple[Citation, ...],
) -> Conversation:
    """把这一问一答接到末尾。**两条一起接**：中途断掉时不会留下一条等不到回复的提问。"""
    return replace(
        conversation,
        turns=(
            *conversation.turns,
            Turn("user", question),
            Turn("assistant", answer, citations),
        ),
    )


def _save(docs: DocStore, conversation: Conversation) -> None:
    """整份覆盖写。

    会话不大（一条消息一行正文），整份写回去比增量追加简单，也不会留下「问题写进去了、
    答案还没写」的中间态——那个中间态正是刷新页面时最可能撞上的一个。
    """
    docs.put(CONVERSATIONS, conversation.session_id, _payload(conversation))


def _payload(conversation: Conversation) -> dict[str, Any]:
    return {
        "game_id": conversation.game_id,
        "version": conversation.version,
        "turns": [_turn_payload(turn) for turn in conversation.turns],
    }


def _turn_payload(turn: Turn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "content": turn.content,
        # 引用原样存下：正文里的 [n] 指的就是它，刷新之后还要对得上号
        "citations": [asdict(citation) for citation in turn.citations],
    }


def _read(session_id: str, payload: Mapping[str, Any]) -> Conversation:
    return Conversation(
        session_id=session_id,
        game_id=str(payload.get("game_id", "")),
        version=str(payload.get("version", "")),
        turns=tuple(_read_turn(turn) for turn in payload.get("turns", ())),
    )


def _read_turn(payload: Mapping[str, Any]) -> Turn:
    return Turn(
        role=payload["role"],
        content=str(payload["content"]),
        citations=tuple(Citation(**citation) for citation in payload.get("citations", ())),
    )
