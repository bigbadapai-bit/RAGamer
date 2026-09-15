"""多轮对话：一次提问接着上一次，答案逐字流出来。

会话是一份**落在 MongoDB 里的记录**（`conversations` 集合，文档 id 就是会话 id）：
刷新浏览器、换台机器打开，历史都还在；两次会话各有各的历史，互不串扰。
只在客户端存（localStorage）做不到第一条，只放在进程内存里两条都做不到。

四件事在这里定死：

- **指代靠历史补全，检索用的是改写后的问题**。用户问「那它掉什么」，「它」是谁只能
  从上一轮看出来——所以历史进 `ragamer.query.understand`，检索用它吐出来的
  `rewritten_query`（「二郎神掉什么」）。会话里记下的仍是**用户的原话**：历史要给人看，
  改写是给检索用的中间产物。**版本不参与这一步的判定**，理由见 :meth:`Chat.ask`。
- **走哪几路召回由查询类型决定**，类型是 `ragamer.query.understand` 那一次调用的产物
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

from ragamer.answering import Answerer, Citation, ReadSide, require_question
from ragamer.caching import CachedAnswerer
from ragamer.clarifying import Clarification, Clarifier
from ragamer.container import Container
from ragamer.live import check_cancelled
from ragamer.llm import Message
from ragamer.logging import get_logger
from ragamer.query import effective_version
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

#: 会话列表一页给几条。**这不是上限**：左栏滚到底会接着取下一页，更早的会话到得了。
#: 100 是「一打开就有一屏多的历史」这个口径；取多了本来不划算（每页一次全表扫），
#: 那份成本由 :data:`SESSION_INDEX` 兜住。
SESSION_PAGE_SIZE = 100

#: 会话列表要用的复合索引。**键的顺序就是查询的顺序**：先按库过滤，再按最后活跃倒序。
#: 没有它，每次列会话都是「全表扫 + 内存排序」，而翻页会把这份成本乘以页数——
#: 复合索引才能把排序也一并免掉。落点在 `ragamer.app` 的启动那一段。
SESSION_INDEX: tuple[tuple[str, int], ...] = (("game_id", 1), ("updated_at", -1))

#: 会话标题的长度上限（**字符数**，不是显示宽度）。标题只是列表里的一行提示，
#: 全文在会话里；截断处补一个省略号，免得看起来像问句本来就断在那里。
TITLE_CHARS = 24

#: 会话里一条消息的角色。**没有 `system`**：提示词是每一步自己拼的，不存进会话。
TurnRole = Literal["user", "assistant"]


def _utc_now() -> str:
    """当下时刻，ISO-8601 UTC。

    选字符串而不是时间戳：它在 Mongo 的 shell 里直接看得懂，而**字典序就是时间序**，
    列表排序不需要再转一次。带时区（`+00:00`）而不是裸的本地时间——换台机器跑，
    排序结果不该跟着机器的时区变。
    """
    return datetime.now(UTC).isoformat()


def _title(question: str) -> str:
    """首轮问句 → 列表里的一行标题。压平空白再截断。"""
    flat = " ".join(question.split())
    return flat if len(flat) <= TITLE_CHARS else flat[:TITLE_CHARS] + "…"


@dataclass(frozen=True)
class Turn:
    """会话里的一条消息。

    后三个字段**只对模型那一侧非空**：用户的话没有来源、没有图、也没有版本这一说。
    三者都随正文一起存下来，刷新之后这一轮的呈现才与刚答完时一模一样——
    界面上「引用点不开」「图片没了」「版本徽章不见了」都是同一类毛病：落库时少存了一样。
    """

    role: TurnRole
    content: str
    #: **正文里 [n] 指的那批**，与当次 `ragamer.answering.AnswerStream.citations` 同一份。
    citations: tuple[Citation, ...] = ()
    #: 这次交给生成的内容里出现过的原图地址，见 `ragamer.answering.Answer`。
    images: tuple[str, ...] = ()
    #: 这一轮实际按哪个版本检索（`ragamer.query.effective_version`）。徽章照着它显示。
    version: str = ""


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
    #: 列表里显示的一行，取**首轮问句**截断。只在第一轮写一次，之后不动——
    #: 标题跟着最新一句改的话，用户会觉得侧栏里的东西在自己动。
    title: str = ""
    #: 最后一次落库的时刻（ISO-8601 UTC）。列表按它倒序，所以**每次写都要刷新**。
    updated_at: str = ""


@dataclass(frozen=True)
class ConversationSummary:
    """会话列表里的一条。

    **刻意不含 `turns`**：列表要的是标题与时间，而一份会话的正文可能很长。让这个类型
    天生装不下正文，比在端点那层记得「不要序列化正文」可靠——那是靠人记住的约定。
    """

    session_id: str
    title: str
    updated_at: str


@dataclass(frozen=True)
class SessionPage:
    """会话列表的一页。

    `next` 是下一页的游标，**空串即到底了**。「还有没有更多」只留这一种表示，
    免得两个字段各说各话；它是**多取一条**看出来的——请求 `limit + 1` 条，多的那条
    就是信号，不必额外 count 一次（那本身又是一次全表扫）。
    """

    sessions: tuple[ConversationSummary, ...]
    next: str = ""

    @property
    def has_more(self) -> bool:
        """还有更早的会话。"""
        return bool(self.next)


def session_cursor(summary: ConversationSummary) -> str:
    """一条会话在翻页里的位置。**不透明字符串**，取用的人原样带回来即可。

    落在 `(最后活跃, 会话 id)` 两个值上：`updated_at` 是可变的（每落一次库就刷新），
    单靠它在并列值上会漏条或重条——而会话列表恰恰常常并列（同一秒里问的两句）。
    """
    return f"{summary.updated_at}|{summary.session_id}"


def parse_session_cursor(text: str) -> tuple[str, str] | None:
    """把游标解回 `(最后活跃, 会话 id)`；**看不懂就返回 `None`**，由调用方按第一页处理。

    游标在 URL 上，人手改得动。一个改坏的游标不该让整个列表报错——重头给第一页正是
    它该得的。时间戳里不会有 `|`，所以按第一个 `|` 切是安全的。
    """
    updated_at, separator, session_id = text.partition("|")
    if not separator or not session_id:
        return None
    return updated_at, session_id


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
    """这一轮的依据与口径。**流的第一件事**。

    引用与图片在检索那一步就定下来了，比正文早得多。先把它交出去，界面才能一边吐字
    一边把来源列出来，而不是等正文吐完再闪一下；版本徽章同理——等正文收完再补，
    那一行会跳一下。名字取「来源」而不是「引用」：后者是 `ragamer.answering` 里那个
    :class:`~ragamer.answering.Citation` 的名字，一个是对外的一批、一个是里面的一条，
    两个词分开用免得读岔。
    """

    citations: tuple[Citation, ...]
    #: 答案相关的原图地址，见 `ragamer.answering.Answer.images`。
    images: tuple[str, ...] = ()
    #: 这一轮实际按哪个版本检索。空串表示没定下来（不过滤版本）。
    version: str = ""


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
#:
#: :class:`~ragamer.clarifying.Clarification` 是另一条出路：判不准时这一轮**到此为止**，
#: 交回候选等用户点。它出现时后面不会再有来源与正文，会话里也什么都不写——
#: 那一轮还没问完（见 `Chat` 的模块说明）。
Reply = Status | Sources | Delta | Clarification


@dataclass(frozen=True)
class Chat:
    """对话侧的唯一入口。外部依赖由组合根注入（见 `ragamer.container`）。

    与 `ragamer.importing.Importer`、`ragamer.answering.Answerer` 是同一个打法：
    做成不可变对象，一次接线反复使用。`docs` 是会话落库的那一处（MongoDB），
    换掉它整条对话链路在测试里就能跑起来。
    """

    docs: DocStore
    #: 读取侧入口（`ragamer.answering.Answerer` 或挡了缓存的 `CachedAnswerer`）。
    answerer: ReadSide
    #: 澄清反问那一层。判游戏与版本、拿不准时停下来问，都在它里面。
    clarifier: Clarifier
    #: 取当下时刻。做成可注入的，与 `OpenAiLlm(sleep=…)` 同一个理由：时间一进断言，
    #: 测试就得能摆布它，否则「按最后活跃倒序」那条用例会看机器的脸色。
    clock: Callable[[], str] = _utc_now

    def start(self, *, game_id: str, version: str = "") -> Conversation:
        """开一次会话。

        id 在这里生成（uuid4），**不由调用方指定**：会话 id 会进 URL 与提示词之外的
        各处，自增数字能被猜、能被顺手下一个人拿到，随机串不能。

        :param game_id: 这次会话问哪个知识库。库里没有它也不会在这里报错——
            那是 HTTP 面该判的事（见 `ragamer.api`），这一层不认 HTTP 状态码。
        :param version: 这次会话选定的版本。空串即不选，检索时回落知识库的现行版本。
        """
        conversation = Conversation(
            session_id=uuid.uuid4().hex,
            game_id=game_id,
            version=version,
            updated_at=self.clock(),
        )
        _save(self.docs, conversation)
        logger.info("新建会话 %s：知识库 %s，版本 %r", conversation.session_id, game_id, version)
        return conversation

    def list_for_game(
        self,
        game_id: str,
        *,
        after: tuple[str, str] | None = None,
        limit: int = SESSION_PAGE_SIZE,
    ) -> SessionPage:
        """这个知识库下的会话，**按最后活跃倒序**，一页 `limit` 条。

        左栏那一份列表：滚到底就从 `after` 接着往下取，`after` 是上一页最后一条的
        `(最后活跃, 会话 id)`（:func:`parse_session_cursor` 解出来的那个）。不给就是第一页。

        只取标题与时间：`find` 的投影把正文挡在外面——取回 id 再逐条 `get` 是另一条路，
        代价是每次都要读完整份文档，而界面上只显示一行字。

        **一次查询，过滤、排序、翻页都在存储那侧做完**。排序与游标都落在
        `(updated_at, _id)` 上，:data:`SESSION_INDEX` 是这条查询的索引（见它的说明）。

        「最后一次说话」而不是「什么时候建的」：继续聊过的会话不该沉到下面去。
        空库返回一页空的——**「这个库还没聊过」是正常状态**，不是错误。

        ⚠️ 加这两个字段**之前**写下的会话文档没有它们：标题会是空串、排序垫底。
        本项目还没部署过，实际不存在这种文档；真出现就写一次回填。
        """
        found = self.docs.find(
            CONVERSATIONS,
            {"game_id": game_id},
            fields=("title", "updated_at"),
            order_by="updated_at",
            descending=True,
            # 多要一条：它在不在，就是「还有更早的」这个信号
            limit=limit + 1,
            after=after,
        )
        sessions = tuple(
            ConversationSummary(
                session_id=str(document["_id"]),
                title=str(document.get("title", "")),
                updated_at=str(document.get("updated_at", "")),
            )
            for document in found[:limit]
        )
        beyond = bool(found[limit:])
        return SessionPage(sessions, next=session_cursor(sessions[-1]) if beyond else "")

    def set_version(self, session_id: str, version: str) -> Conversation:
        """改这次会话选定的版本。**下一轮起按它走**，直到再改一次。

        `updated_at` 不动：它是「最后一次说话」的时刻，左栏按它倒序——改个版本不该让
        一次会话在列表里往上跳。标题同理，那是这一串问答的招牌。

        :raises ConversationNotFound: 没有这个会话。
        """
        conversation = self.open(session_id)
        updated = replace(conversation, version=version)
        _save(self.docs, updated)
        logger.info("会话 %s 改按版本 %r 检索", session_id, version)
        return updated

    def open(self, session_id: str) -> Conversation:
        """把一次会话读回来。刷新页面之后靠它把历史拿回来。

        :raises ConversationNotFound: 没有这个会话。
        """
        payload = self.docs.get(CONVERSATIONS, session_id)
        if payload is None:
            raise ConversationNotFound(f"会话 {session_id} 不存在")
        return _read(session_id, payload)

    def delete(self, session_id: str) -> None:
        """删掉一次会话。**不可逆**：那一串问答连同它的引用与图片地址一起没了。

        先 `open` 再删：读不到就当场报出来，而不是让一次已经过期的删除静默成功。
        界面上那个按钮是照当前这一页的会话渲染的——报出「没有这个会话」，说的是那一页
        已经不是最新的了，比什么都不说强。

        删的只有会话本身：澄清反问的待答记录不挂在会话上（`ragamer.clarifying`），
        缓存也是按问题存的、不按会话（`ragamer.caching`），两者都不需要连带清理。

        :raises ConversationNotFound: 没有这个会话。
        """
        self.open(session_id)
        self.docs.delete(CONVERSATIONS, session_id)
        logger.info("删除会话 %s", session_id)

    def ask(
        self,
        session_id: str,
        question: str,
        *,
        version: str = "",
        current_version: str = "",
        pending_id: str = "",
        label: str = "",
        games: Sequence[Game] = (),
        routes: RouteTable = DEFAULT_TABLE,
        cancelled: Callable[[], bool] | None = None,
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
        :param pending_id: 从哪个暂停点继续。给了它就不再判一次——判定上一次就做完了，
            用户点的那个候选由 :meth:`ragamer.clarifying.Clarifier.resolve` 补进去。
            这时 `question` 只用来记这一轮的用户原话，检索用的是暂停点里存着的改写问法。
        :param label: 用户点的那个候选项，原样回传。**必须是他当时看到的那些之一**，
            按钮之外的值当场报错而不是拿去检索（错的是请求，不是语料）。
        :param games: 游戏候选（:data:`Game`，显示名与知识库 id 成对）。**显示名是用户
            问句里会出现的那种写法**——知识库 id 是 collection 名，只能是英文标识符，
            拿 id 当候选，模型只会把「黑神话」判成不在候选里。判出来的显示名在这里换回
            id；没有候选、或者判不出来，都回落会话选定的知识库。
        :param routes: 这个知识库的路由表（`ragamer.routing`）。库里配了就传配的那份，
            不传就用默认表——组合是每个库一份的配置，与打标词表同一个姿势。
        :raises ConversationNotFound: 没有这个会话。
        :raises ValueError: 问题为空。空问题会让检索查出任意一批切片。
        :raises ragamer.clarifying.UnknownPending: 没有这个暂停点。
        :raises ragamer.clarifying.NotACandidate: 选的不在那次反问给出的候选里。
        :param cancelled: 这一轮要不要收手。**每一步之间问它一次**，答「是」就抛
            :class:`TurnCancelled`——落库那行因此执行不到，这一轮当没问过。不给就是
            不取消。它是**协作式**的：正在跑的那一次模型调用或精排要等它自己返回，
            见 :class:`ragamer.live.LiveTurn`。
        """
        require_question(question)  # 拦在理解那一步之前：空问题没得可理解，别白调一次模型
        conversation = self.open(session_id)
        return self._replies(
            conversation,
            question,
            version=version or conversation.version,
            current_version=current_version,
            pending_id=pending_id,
            label=label,
            games=games,
            routes=routes,
            cancelled=cancelled,
        )

    def _replies(
        self,
        conversation: Conversation,
        question: str,
        *,
        version: str,
        current_version: str,
        pending_id: str = "",
        label: str = "",
        games: Sequence[Game],
        routes: RouteTable,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[Reply]:
        """把这一轮从头做到尾，**每一步之前先报一条进度**，最后收完正文才落库。

        进度那两条与这一轮真正干的事一一对应，顺序也一致：理解问题 → 检索资料 →
        生成答案。夹在中间的是来源——它比正文早得多，一拿到就先交出去，界面可以
        先列出来再等字。

        **判不准的那一轮到此为止**：交回一次 :class:`~ragamer.clarifying.Clarification`
        就结束，不检索、不生成、也不落库。用户点完候选再发一次请求（带 `pending_id`），
        那一轮才算走完——所以「反问过的提问」在会话里只留下最终那一问一答，
        不会先留一条等不到回复的提问。
        选路夹在理解与检索之间，**没有自己的进度条**：它是一次字典查表，不是一段等待。

        迭代器被丢掉时（客户端断开）最后那一行写不进会话——这正是要的效果：
        已经吐出去的那半句与它那批引用一起消失，历史里不留痕迹。
        """
        if pending_id:
            # 判定上一次就做完了，这里只把它取回来；那一步还会核对用户点的是不是候选之一
            resolved = self.clarifier.resolve(pending_id, label)
        else:
            yield Status("正在理解问题")
            outcome = self.clarifier.decide(
                question,
                game_id=conversation.game_id,
                version=version,
                history=_history(conversation.turns),
            )
            if isinstance(outcome, Clarification):
                yield outcome
                return
            resolved = outcome
        check_cancelled(cancelled)
        yield Status("正在检索资料")
        route = routes.route_for(resolved.query_type)
        stream = self.answerer.stream(
            resolved.rewritten_query,
            game_id=resolved.game_id,
            version=resolved.version,
            current_version=current_version,
            route=route,
            cancelled=cancelled,
        )
        sources = Sources(
            stream.citations,
            stream.images,
            effective_version(resolved.version, current_version=current_version),
        )
        logger.info(
            "会话 %s 提问 %r（改写为 %r，类型 %s），版本 %r，走 %s，用上 %d 条来源",
            conversation.session_id,
            question,
            resolved.rewritten_query,
            _type_name(resolved.query_type),
            sources.version,
            "、".join(path.value for path in route.paths),
            len(stream.citations),
        )
        yield sources
        check_cancelled(cancelled)
        yield Status("正在生成答案")
        produced: list[str] = []
        for piece in stream.deltas:
            produced.append(piece)
            yield Delta(piece)
        _save(
            self.docs,
            _appended(conversation, question, "".join(produced), sources, self.clock()),
        )


@dataclass(frozen=True)
class ChatStack:
    """读取侧接好的那一套：对话、澄清、以及挡在生成前面的缓存。

    **装配只有 `build_chat` 这一处**（JSON 端点与页面都从它拿）：两处各接一遍的话，
    缓存挡没挡上、澄清器有没有接，就可能两边不一样——而那种差别在界面上看不出来，
    只会表现为「页面上会反问的提问，接口上直接作答」。
    """

    #: 多轮对话。开会话、列会话、问一轮都走它。
    chat: Chat
    #: 挡了缓存的读取侧入口。**热门问题要直接问它**——那不是某一次对话的事，
    #: 但它与对话共用同一份缓存连接与同一个键空间，问的也是同一批提问。
    cache: CachedAnswerer


def build_chat(container: Container) -> ChatStack:
    """按组合根里那套依赖接出读取侧。

    缓存挡在检索生成前面（架构文档 §4）：命中就把上次那份结果原样交回，未命中才走
    完整链路。澄清器与生成走的是同一个 `Answerer`——判完就作答与逐字流式生成只差
    正文怎么出来，两处各接一个的话，同一次提问在两条路上会拿到两份不同的引用。
    """
    answers = Answerer(
        chunks=container.chunks,
        embedder=container.embedder,
        reranker=container.reranker,
        llm=container.llm,
        # 联网兜底那一路：没配就是 None，检索侧据此跳过它
        search=container.search,
    )
    cache = CachedAnswerer(answers=answers, cache=container.cache)
    return ChatStack(
        chat=Chat(
            docs=container.docs,
            answerer=cache,
            clarifier=Clarifier(
                chunks=container.chunks,
                docs=container.docs,
                llm=container.llm,
            ),
        ),
        cache=cache,
    )


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
    sources: Sources,
    now: str,
) -> Conversation:
    """把这一问一答接到末尾。**两条一起接**：中途断掉时不会留下一条等不到回复的提问。

    标题只在第一轮定下（`conversation.title or …`）：它是这一串问答的招牌，跟着最新
    一句改会让侧栏里的东西自己动。`updated_at` 反过来，**每次都要刷新**——列表按它排序。
    """
    return replace(
        conversation,
        title=conversation.title or _title(question),
        updated_at=now,
        turns=(
            *conversation.turns,
            Turn("user", question),
            Turn("assistant", answer, sources.citations, sources.images, sources.version),
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
        "title": conversation.title,
        "updated_at": conversation.updated_at,
        "turns": [_turn_payload(turn) for turn in conversation.turns],
    }


def _turn_payload(turn: Turn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "content": turn.content,
        # 引用原样存下：正文里的 [n] 指的就是它，刷新之后还要对得上号
        "citations": [asdict(citation) for citation in turn.citations],
        "images": list(turn.images),
        "version": turn.version,
    }


def _read(session_id: str, payload: Mapping[str, Any]) -> Conversation:
    return Conversation(
        session_id=session_id,
        game_id=str(payload.get("game_id", "")),
        version=str(payload.get("version", "")),
        turns=tuple(_read_turn(turn) for turn in payload.get("turns", ())),
        title=str(payload.get("title", "")),
        updated_at=str(payload.get("updated_at", "")),
    )


def _read_turn(payload: Mapping[str, Any]) -> Turn:
    return Turn(
        role=payload["role"],
        content=str(payload["content"]),
        citations=tuple(Citation(**citation) for citation in payload.get("citations", ())),
        images=tuple(str(url) for url in payload.get("images", ())),
        version=str(payload.get("version", "")),
    )
