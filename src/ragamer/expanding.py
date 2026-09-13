"""查询扩展：把一个问法变成**几个可检索的文本**，供两路召回去查。

两件事，各对应一路召回：

- **多查询改写**（:func:`rewrite`）：把一个问题扩成若干条等价的问法，各自检索一遍。
  用户问「二郎神怎么打」，语料里写的可能是「妖王 打法」「二郎神 打法流程」——
  一种说法查不全，换个说法就查得到。
- **HyDE**（:func:`hypothetical`）：先让模型写一段**假想答案**，拿它去检索。
  口语化、指代不清的问法（「那个很难的 BOSS 怎么过」）与语料的用词差得远，
  而假想答案的用词更接近语料本身。

三件事在这里定死：

- **扩出来的只用于检索，绝不进上下文。** HyDE 那段是模型编的，把它当资料交给生成
  就是让模型照着自己编的东西回答——而且看起来与真答案一模一样。这一段在
  `ragamer.retrieval` 里用完即弃，连日志都不落原文。
- **改写要真的不一样**。主检索路已经用原问检索过一遍了，扩出来的问法若与原问相同
  就是白跑一趟（多一次检索、多一批重复候选）。所以这里按
  :func:`ragamer.query.normalize_query` 同一套归一去掉重复——与缓存 key 用的是同一个
  口径，两处对「同一个问法」的判断不会分叉。
- **温度钉死 0**。扩出来的问法直接决定候选取回什么，同一个问题两次问出不同的候选池，
  答案就会跟着飘。与 `ragamer.answering` 钉温度是同一个理由。

**失败照抛**（`ragamer.llm.LlmError`）：这两步各是一路召回里的一环，失败该怎么处置
由 `ragamer.retrieval` 的隔离那一层决定——那一路记下来，其余路继续。这一层不自己
降级成「返回原问」，那样会让「改写失败」与「改写没扩出东西」在调用方看来一样。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ragamer.llm import LlmClient, LlmRequest, Message
from ragamer.logging import get_logger
from ragamer.query import normalize_query

logger = get_logger(__name__)

#: 扩出来的问法条数（**不含原问**）。每条都要各发一次检索，条数直接乘在检索成本上。
#: 数值本身与项目里其它阈值一样是**没有评测集时的占位**（`docs/ARCHITECTURE.md` §11）。
REWRITE_COUNT = 3

#: 扩展用的温度。钉死 0 的理由见模块说明——它决定候选取回什么。
TEMPERATURE = 0.0


def rewrite(question: str, *, llm: LlmClient, count: int = REWRITE_COUNT) -> tuple[str, ...]:
    """把一个问题扩成若干条**等价而不同**的问法。**只调一次模型。**

    返回的条数可能少于 `count`（模型给得少），也可能为空（模型给的都与原问重复）。
    两种都不是错误：这一路只是少贡献一点候选，其余路照跑。

    :param count: 最多要几条。给 0 就不调模型了——空手要一次调用没有意义。
    :raises ragamer.llm.LlmError: 模型失败。处置见模块说明。
    """
    if count <= 0 or not question.strip():
        return ()
    guess = llm.complete_structured(_rewrite_request(question, count), _Rewrites)
    kept = distinct(question, guess.queries)[:count]
    if not kept:
        # 一次成功的调用却什么也没扩出来：多半是提示词没交代清楚或模型没照做，
        # 而它不报错，只是这一路白跑一趟
        logger.warning("多查询改写没扩出与原问不同的问法，这一路本次不贡献候选")
    return kept


def hypothetical(question: str, *, llm: LlmClient) -> str:
    """写一段**假想答案**，用来检索（HyDE）。**只调一次模型。**

    返回空串表示模型没给出可用文本——由调用方当作这一路没贡献候选处理。

    :raises ragamer.llm.LlmError: 模型失败。处置见模块说明。
    """
    if not question.strip():
        return ""
    written = llm.complete(_hypothetical_request(question)).strip()
    if not written:
        logger.warning("假想答案一个字都没写出来，HyDE 这一路本次不贡献候选")
    return written


def distinct(question: str, candidates: Sequence[str]) -> tuple[str, ...]:
    """去掉与原问重复、以及自己之间重复的那些问法，保留原顺序。

    比的是 `ragamer.query.normalize_query` 那个归一形式——与缓存 key 同一套口径。
    归一之后为空的一律丢掉（模型偶尔会吐空串）。
    """
    seen = {normalize_query(question)}
    kept: list[str] = []
    for candidate in candidates:
        wanted = normalize_query(candidate)
        if not wanted or wanted in seen:
            continue
        seen.add(wanted)
        kept.append(candidate.strip())
    return tuple(kept)


def _rewrite_request(question: str, count: int) -> LlmRequest:
    return LlmRequest(
        messages=[
            Message("system", _REWRITE_INSTRUCTION.format(count=count)),
            Message("user", question),
        ],
        temperature=TEMPERATURE,
    )


def _hypothetical_request(question: str) -> LlmRequest:
    return LlmRequest(
        messages=[
            Message("system", _HYPOTHETICAL_INSTRUCTION),
            Message("user", question),
        ],
        temperature=TEMPERATURE,
    )


#: 多查询改写的系统提示。三条要求各对应一种失败：改成了回答、语义变了、
#: 扩出来的与原问一样（那样这一路就是白跑）。
_REWRITE_INSTRUCTION = (
    "你在帮检索系统改写检索词。把用户的问题改写成 {count} 条**意思相同但说法不同**的"
    "检索式，每条一行。\n"
    "要求：只给检索式，不要回答问题；不要改变原意，不要添问题里没有的限定；"
    "沿用问题里的游戏内名称，不要换成别的叫法。\n"
)

#: HyDE 的系统提示。**输出只用于检索**，所以要求它像一篇资料正文，而不是像回答用户。
_HYPOTHETICAL_INSTRUCTION = (
    "你在写一段用于检索的假想资料。针对用户的问题，写一段**可能出现在游戏攻略资料里**"
    "的文字：直接给出可能的相关内容，用资料里会用的说法。\n"
    "这是在代替查不到的原文，不是回答用户——不要问我、不要解释你在做什么、"
    "不要提「根据资料」之类的话。写三到五句话即可。\n"
)


class _Rewrites(BaseModel):
    """一次多查询改写的输出。字段描述会随 schema 一起进提示词。"""

    queries: list[str] = Field(description="若干条意思相同、说法不同的检索式，每条一行")
