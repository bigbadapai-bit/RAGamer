"""提问理解：读取侧的第一道处理，**一次调用**判定游戏、版本、规范问法与路由标签。

四个结果出自同一次结构化调用（`docs/ARCHITECTURE.md` §3.1 的联合输出节点）。
拆成三次调用会各错各的、还不止一次网络往返——节点存在的理由就是不拆。
路由标签尤其如此：它是整个改造里性价比最高的一处，把一个已有调用的输出从三个字段
扩到四个，**没有新增任何模型调用**（§3.1）。

两件容易做错的事，各有各的静默失效方式：

- **改写要稳**。改写后的问题直接当缓存 key 用（`docs/ARCHITECTURE.md` §4），
  所以温度钉死 0，归一化只压平空白。这里做语义归一（去标点、同义合并）就会把
  不同的问法并成一个 key，那是语义缓存要标定的阈值，v1 不做。
- **候选必须是库里真有的**。模型自己编的游戏名在库里不存在，照它去检索只会查空，
  而且不报错。候选由调用方从库里读出来传进来，编出来的取值在这里丢掉——
  与澄清反问「候选必须来自语料中真实存在的选项」是同一条约束。
- **确定度与取值成对回来**。确定不了时要反问而不是猜（§3.4），而「确定不了」不是
  非黑即白：判得出取值但不太稳，与压根没判出来，是两种不同的处境，得靠确定度分开
  （`ragamer.clarifying` 那两档阈值就架在它上面）。确定度**来自模型自报**——
  它是这里唯一拿得到的连续信号，也还是个没有评测集校准的经验值（§11）。

**这一步失败有降级路径**：模型挂了就按原问法继续，游戏与版本留空交回给调用方
（会话里已经选定的那两个）。整个提问不该因为第一道处理失败而失败。唯一按 ERROR
报出来的是被服务端拒绝（密钥、模型名配错）——那种情况下之后每条提问都会这样降级，
只留一条 WARNING 会让人看不出根因。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ragamer.llm import LlmClient, LlmError, LlmRejected, LlmRequest, Message
from ragamer.logging import get_logger
from ragamer.routing import (
    QUERY_TYPE_HINTS,
    QUERY_TYPE_LABELS,
    QueryType,
    parse_query_type,
)
from ragamer.stores.base import ChunkFilter

logger = get_logger(__name__)

#: 改写的温度。钉死 0 是为了改写结果稳定可复现——缓存 key 直接建立在它上面。
TEMPERATURE = 0.0


@dataclass(frozen=True)
class Understanding:
    """一次联合输出的结果。

    游戏与版本判不出来时是空串，由调用方回落到会话里已选定的那一个。
    **空串是「没判出来」，与库里 `version` 字段的「未标注版本」不是一回事**：
    后者是一个必须被一并检索到的真实取值，前者只说明这一步没得出结论。
    """

    #: 问的是哪款游戏。判不出、或模型给的不在候选里，都是空串。
    game: str
    #: 问的是哪个版本。判不出、或模型给的不在候选里，都是空串。
    version: str
    #: 改写后的规范问法。补上了指代的主体名，语义与原问题一致。
    rewritten_query: str
    #: 这个游戏判断有多稳，0~1，模型自报。**取值是空串时它一定是 0**：
    #: 没有取值就谈不上对这个取值有多确定。
    game_confidence: float = 0.0
    #: 这个版本判断有多稳，0~1，模型自报。同上。
    version_confidence: float = 0.0
    #: 这个问题属于哪一类，决定这次走哪几路召回（`ragamer.routing`）。
    #: **`None` 是「没判出来」**，不是某一类：路由侧按事实型那一行回落，不阻断作答。
    query_type: QueryType | None = None


#: 备忘最多记多少条。**又是一个没有依据的占位**（§11）：能讲清的只有它必须有个上界
#: ——键里带着整段历史，不封顶就会随着「问过多少种问法」一直长下去。
MEMO_SIZE = 256

#: 备忘的键：`understand` 除了模型客户端之外的**全部**入参。
#:
#: 候选（游戏、版本）也进来是有原因的，不是保险起见：`_pick` 会把候选之外的取值丢掉，
#: 所以算出来的 `Understanding` 里那个游戏**一定曾是候选里的一个**。冻住之后知识库被
#: 改名或删掉，那份取值就成了一个候选里根本没有的标签——而 `ragamer.clarifying._game`
#: 拿 `_choice_of` 认它，认不出是**当场抛 `NotACandidate`**，一条本来问得通的提问直接
#: 报错。版本那侧轻一些但同样是错的：它不校验，会照一个已经不在库里的版本去检索。
#: 把候选放进键里，这两种都退化成「候选变了就重新算一次」，不需要另做失效。
_MemoKey = tuple[str, tuple[Message, ...], tuple[str, ...], tuple[str, ...]]


class UnderstandingMemo:
    """`understand` 的记忆：同一句问话、同一段历史、同一批候选，只算一次。

    存在的理由是**改写会抖**。温度已经钉死在 0，但模型在 0 下仍会给出不同的改写
    （实测同一个问题问 6 次得到 3 种），而答案缓存的键正建立在改写之上
    （`ragamer.caching.cache_key`）——于是同一个问题问两次**有可能算成两个键、
    双双未命中**，各花掉一次完整检索。`CachedAnswerer._key` 把这条取舍写在明处，
    并指明「真嫌命中率低，要动的是这个取舍本身」：记下理解结果就是动它，
    而且**不动 `normalize_query`**（那个函数同时被用来比候选，收紧它会顺带收紧候选）。

    记下来之后，改写对同一组入参就成了确定的：缓存键跟着稳定。顺带每次提问还省掉
    这一次模型调用。

    **只记成功的那些。** 这个判断在 :func:`understand` 里做，不在这里——降级那条路
    交回的是「按原问法继续、游戏与版本留空」，把它记进来的话，一次模型抖动会被
    钉成整个进程生命周期里的固定行为，而且看不出来。

    键里带历史与候选都是有意的：「那它怎么打」配上不同上文本来就是不同的问题，
    只按问法记会让后一个会话拿到前一个会话的指代补全结果；候选那一条的理由见
    :data:`_MemoKey`。**模型客户端不在键里**：一次接线只有它一个，换客户端等于换
    一套接线，那时连这个备忘对象也一起换了。

    线程安全：界面后端把同步端点丢进线程池，两个人同时问同一句话是常事
    ——`ragamer.lazy.LazyModel` 出于同一个理由加了锁。满了丢**最久没用过**的那条。
    """

    def __init__(self, size: int = MEMO_SIZE) -> None:
        if size < 1:
            raise ValueError(f"备忘的容量至少是 1，给的是 {size}")
        self._size = size
        self._lock = threading.Lock()
        self._items: OrderedDict[_MemoKey, Understanding] = OrderedDict()

    def get(self, key: _MemoKey) -> Understanding | None:
        """取一条；取到就把它挪到最新，于是丢的永远是最久没用过的那条。"""
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
            return found

    def set(self, key: _MemoKey, value: Understanding) -> None:
        """记一条，按容量丢最久没用过的那些。重复记同一个键就是刷新它的位置。"""
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def understand(
    question: str,
    *,
    llm: LlmClient,
    games: Sequence[str] = (),
    versions: Sequence[str] = (),
    history: Sequence[Message] = (),
    memo: UnderstandingMemo | None = None,
) -> Understanding:
    """读一个问题：问的是哪款游戏、哪个版本、规范问法是什么、属于哪一类。
    **只调一次模型。**

    **不抛异常**：模型失败时按原问法降级，游戏与版本留空。

    :param games: 真实存在的游戏候选（知识库列表）。空表示调用方拿不出来，
        此时游戏一律留空——不许模型自己编一个。
    :param versions: 该游戏真实存在的版本候选。空表示判不出，理由同上。
    :param history: 上一轮及更早的对话。指代（「那它怎么打」）要靠它才能补成主体名。
    :param memo: 记下这次的判定的地方（:class:`UnderstandingMemo`）。给了就在算之前
        先查、算完再记；不给就每次都问模型。**降级的结果不记**——理由见那个类。
    """
    asked = normalize_query(question)
    degraded = Understanding("", "", asked)
    if not question.strip():
        return degraded  # 空问题没有可理解的，也别白调一次模型
    # 键照算不误（都是几个短字符串），只在没给备忘时才不查不记——两处判断合成一处，
    # 不给「查了这个键、记的却是另一个」留出机会
    key = (asked, tuple(history), tuple(games), tuple(versions))
    if memo is not None:
        remembered = memo.get(key)
        if remembered is not None:
            return remembered
    try:
        guess = llm.complete_structured(_request(question, games, versions, history), _JointOutput)
    except LlmRejected as exc:
        # 密钥、模型名、参数配错这类问题重试无用，之后每一条提问都会栽在这里。
        # 按 ERROR 报出来：降级成本身是对的，但根因不该只留一条 WARNING 让人去猜
        logger.error("提问理解被模型服务拒绝，之后的提问也会一直降级：%s", exc)
        return degraded
    except LlmError as exc:
        logger.warning("提问理解失败（%s），按原问法继续：%s", type(exc).__name__, exc)
        return degraded
    game = _pick(guess.game, games, what="游戏")
    version = _pick(guess.version, versions, what="版本")
    kind = parse_query_type(guess.route)
    if kind is None and guess.route.strip():
        # 词表以外的标签与「按提示留了空串」不是一回事：后者是判不出，前者是提示词
        # 与 schema 没对上或者模型跑偏了。判不出不报，跑偏要留痕——留痕的判据见
        # `_JointOutput.route`：这个字段只在这一层失手时回落，不该连累另外三个。
        logger.warning("模型判出的查询类型不在词表里，本次按默认组合走：%r", guess.route)
    elif kind is None and "route" not in guess.model_fields_set:
        logger.warning("模型没给查询类型（这个字段整个缺席），本次按默认组合走")
    resolved = Understanding(
        game=game,
        version=version,
        rewritten_query=normalize_query(guess.rewritten_query) or degraded.rewritten_query,
        # 取值丢掉时确定度一并归零：留着一个高确定度配一个空取值，调用方会照着
        # 确定度走进「确定」那一支，然后拿着空游戏去检索
        game_confidence=guess.game_confidence if game else 0.0,
        version_confidence=guess.version_confidence if version else 0.0,
        query_type=kind,
    )
    # 记在**这一条**返回路径上，不记上面那两个 `return degraded`：降级不是判定结果
    if memo is not None:
        memo.set(key, resolved)
    return resolved


def normalize_query(text: str) -> str:
    """问法的归一形式。**缓存层复用它算 key**（`docs/ARCHITECTURE.md` §4）。

    只压平空白，不动字面：同一个问法多打几个空格仍要命中同一条缓存，
    而「怎么打」与「打法」合并成一件事就得先标定阈值，v1 不做语义缓存。

    模型给的取值对候选也是按它比的（见 `_pick`）：同一个归一形式因此有两个用处，
    要收紧规则时得两处一起想清楚——收紧缓存那边会顺带收紧候选的比对。
    """
    return " ".join(text.split())


def effective_version(version: str, *, current_version: str) -> str:
    """这次实际按哪个版本检索：问题里点名的那个，其次是知识库的现行版本。

    `knowledge_bases` 是「当前该用哪个版本」的唯一真相来源，检索不自己维护一份。
    两个参数的空串都表示「没判出来」，返回空串。

    缓存键要用它（`ragamer.caching`）：键里的版本必须是**实际生效**的那一个。放问题
    点名的那个，问题没点名时同一个问题在知识库换了现行版本之后仍会命中旧版本的答案；
    放知识库标的那个，用户点名问历史版本时会与问现行版本撞成同一条。
    """
    return version or current_version


def version_filter(version: str, *, current_version: str) -> ChunkFilter:
    """检索用的版本过滤条件：「**所选版本或未标注版本**」（ADR-0004）。

    **这里对 ADR-0004 的「版本过滤必须默认开启」有一处有意收窄**：两处都判不出来时
    不做过滤。过滤成「只留未标注版本」会把标了版本的资料整批漏掉，而漏是静默的；
    不过滤的代价只是近重复之间互相稀释分数，看得见。这条在日志里留痕，
    不静默——知识库没标现行版本是该被修掉的配置问题。
    """
    chosen = effective_version(version, current_version=current_version)
    if not chosen:
        logger.warning("问题与知识库都给不出版本，本次检索不按版本过滤")
    return ChunkFilter(version=chosen or None)


def _pick(value: str, candidates: Sequence[str], *, what: str) -> str:
    """把模型给的取值对到候选表里的那一个，对不上就留空。

    候选为空即无解：宁可不判，也不接受一个库里不存在的取值——这时连提醒都不发，
    提示词已经交代过「一律留空串」，是调用方没给候选，不是模型判错了。
    """
    wanted = normalize_query(value)
    if not wanted or not candidates:
        return ""
    for candidate in candidates:
        if normalize_query(candidate) == wanted:
            return candidate
    logger.warning("模型判出的%s不在候选里，已丢掉：%r", what, value)
    return ""


def _request(
    question: str,
    games: Sequence[str],
    versions: Sequence[str],
    history: Sequence[Message],
) -> LlmRequest:
    """一次调用的请求。历史排在当前问题之前，指代才有得可依。"""
    return LlmRequest(
        messages=[
            Message("system", _instruction(games, versions)),
            *history,
            Message("user", question),
        ],
        temperature=TEMPERATURE,
    )


def _instruction(games: Sequence[str], versions: Sequence[str]) -> str:
    """系统提示。可选值写在这里，字段描述写在 schema 里，两处一起进提示词。"""
    return (
        "你在理解一个游戏问题。读用户的问题，判断他问的是哪款游戏、哪个版本，"
        "并把问题改写成一句完整、规范的问法。\n"
        "改写时把指代替换成明确的游戏内名称——知道上下文时，「那它怎么打」写成"
        "「二郎神怎么打」——但不要改变原意，不要回答问题，也不要补充问题里没有的限定。\n"
        f"{_candidates('游戏', games)}\n"
        f"{_candidates('版本', versions)}\n"
        "游戏与版本各自还要给一个 0 到 1 的确定度：问题里明确写了、没有第二种可能给 1 附近；"
        "只是从上下文猜的、也可能不对，给 0.5 附近；判不出来时取值留空串、确定度给 0。\n"
        f"{_query_type_options()}"
    )


def _query_type_options() -> str:
    """查询类型的可选值。叫法与典型问法都写上——只给名字，模型会在「事实型」与
    「表格型」之间猜，而这两类问的都是数值。"""
    options = "；".join(
        f"{QUERY_TYPE_LABELS[kind]}（{QUERY_TYPE_HINTS[kind]}）" for kind in QueryType
    )
    return f"查询类型只能从这些里原样取一个：{options}；都不符就留空串。"


def _candidates(what: str, values: Sequence[str]) -> str:
    if not values:
        return f"现在拿不到{what}的候选，{what}一律留空串。"
    return f"{what}只能从这些里原样取一个：{'、'.join(values)}；都不符就留空串。"


class _JointOutput(BaseModel):
    """一次联合输出的全部字段。字段描述会随 schema 一起进提示词。

    `route` 是路由标签（`ragamer.routing.QueryType` 的一个）。它**有默认值，但缺席要
    留痕**：整个返不回来时，联合输出那一步会以 `LlmInvalidOutput` 重试一次（错误带回去，
    见 `ragamer.llm`），重试用尽就整套降级——为一个**本来就定义了回落**的字段赔上
    游戏、版本、改写问法这三个没得回落的，不划算。

    所以缺字段只有这一个字段自己吃亏，代价是「缺席」与「按提示留了空串」在 schema
    层面分不开，得靠 `model_fields_set` 认——判据在 `understand` 里。
    """

    game: str = Field(
        description="用户问的是哪款游戏，只能从候选里原样取一个；判断不出或候选里没有就留空串"
    )
    game_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="上面那个游戏判断的确定度，0 到 1；游戏留空串时给 0",
    )
    version: str = Field(
        description="用户问的是哪个版本，只能从候选里原样取一个；判断不出或候选里没有就留空串"
    )
    version_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="上面那个版本判断的确定度，0 到 1；版本留空串时给 0",
    )
    rewritten_query: str = Field(
        description="改写后的规范问法：补齐指代的主体名，语义与原问题一致，不回答问题"
    )
    route: str = Field(
        default="",
        description="问题属于哪一类，只能从系统提示列出的几类里原样取一个；判断不出就留空串",
    )
