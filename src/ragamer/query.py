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

**这一步失败有降级路径**：模型挂了就按原问法继续，游戏与版本留空交回给调用方
（会话里已经选定的那两个）。整个提问不该因为第一道处理失败而失败。唯一按 ERROR
报出来的是被服务端拒绝（密钥、模型名配错）——那种情况下之后每条提问都会这样降级，
只留一条 WARNING 会让人看不出根因。
"""

from __future__ import annotations

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
    #: 这个问题属于哪一类，决定这次走哪几路召回（`ragamer.routing`）。
    #: **`None` 是「没判出来」**，不是某一类：路由侧按事实型那一行回落，不阻断作答。
    query_type: QueryType | None = None


def understand(
    question: str,
    *,
    llm: LlmClient,
    games: Sequence[str] = (),
    versions: Sequence[str] = (),
    history: Sequence[Message] = (),
) -> Understanding:
    """读一个问题：问的是哪款游戏、哪个版本、规范问法是什么、属于哪一类。
    **只调一次模型。**

    **不抛异常**：模型失败时按原问法降级，游戏与版本留空。

    :param games: 真实存在的游戏候选（知识库列表）。空表示调用方拿不出来，
        此时游戏一律留空——不许模型自己编一个。
    :param versions: 该游戏真实存在的版本候选。空表示判不出，理由同上。
    :param history: 上一轮及更早的对话。指代（「那它怎么打」）要靠它才能补成主体名。
    """
    degraded = Understanding("", "", normalize_query(question))
    if not question.strip():
        return degraded  # 空问题没有可理解的，也别白调一次模型
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
    kind = parse_query_type(guess.route)
    if kind is None and guess.route.strip():
        # 词表以外的标签与「按提示留了空串」不是一回事：后者是判不出，前者是提示词
        # 与 schema 没对上或者模型跑偏了。判不出不报，跑偏要留痕——留痕的判据见
        # `_JointOutput.route`：这个字段只在这一层失手时回落，不该连累另外三个。
        logger.warning("模型判出的查询类型不在词表里，本次按默认组合走：%r", guess.route)
    elif kind is None and "route" not in guess.model_fields_set:
        logger.warning("模型没给查询类型（这个字段整个缺席），本次按默认组合走")
    return Understanding(
        game=_pick(guess.game, games, what="游戏"),
        version=_pick(guess.version, versions, what="版本"),
        rewritten_query=normalize_query(guess.rewritten_query) or degraded.rewritten_query,
        query_type=kind,
    )


def normalize_query(text: str) -> str:
    """问法的归一形式。**缓存层复用它算 key**（`docs/ARCHITECTURE.md` §4）。

    只压平空白，不动字面：同一个问法多打几个空格仍要命中同一条缓存，
    而「怎么打」与「打法」合并成一件事就得先标定阈值，v1 不做语义缓存。

    模型给的取值对候选也是按它比的（见 `_pick`）：同一个归一形式因此有两个用处，
    要收紧规则时得两处一起想清楚——收紧缓存那边会顺带收紧候选的比对。
    """
    return " ".join(text.split())


def version_filter(version: str, *, current_version: str) -> ChunkFilter:
    """检索用的版本过滤条件：「**所选版本或未标注版本**」（ADR-0004）。

    版本按序取第一个判得出来的：问题里点名的那个，其次是知识库的现行版本
    ——`knowledge_bases` 是「当前该用哪个版本」的唯一真相来源，检索不自己维护一份。
    两个参数的空串都表示「没判出来」。

    **这里对 ADR-0004 的「版本过滤必须默认开启」有一处有意收窄**：两处都判不出来时
    不做过滤。过滤成「只留未标注版本」会把标了版本的资料整批漏掉，而漏是静默的；
    不过滤的代价只是近重复之间互相稀释分数，看得见。这条在日志里留痕，
    不静默——知识库没标现行版本是该被修掉的配置问题。
    """
    chosen = version or current_version
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
    """一次联合输出的四个字段。字段描述会随 schema 一起进提示词。

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
    version: str = Field(
        description="用户问的是哪个版本，只能从候选里原样取一个；判断不出或候选里没有就留空串"
    )
    rewritten_query: str = Field(
        description="改写后的规范问法：补齐指代的主体名，语义与原问题一致，不回答问题"
    )
    route: str = Field(
        default="",
        description="问题属于哪一类，只能从系统提示列出的几类里原样取一个；判断不出就留空串",
    )
