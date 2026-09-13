"""澄清反问：系统拿不准时停下来问，而不是猜。

读取侧在 `ragamer.answering` 之上多了一层：提问进来，先看问的是哪款游戏、哪个版本，
判不准就先问一句，用户点完再从暂停点继续。**候选必须来自语料里真实存在的取值，
绝不由模型生成**（§3.4）——让模型自由生成澄清选项，它会给出库里根本没有的选项，
用户选了也检索不到，而「选了没东西」与「选了查得到」在界面上长得一样。

四件事在这里定死：

- **确定度分两档**（§3.4 坑 #13）：模型判得出取值、确定度过 :data:`CONFIDENT` 就直接用；
  落在 :data:`UNSURE` 与 :data:`CONFIDENT` 之间是「接近但不肯定」，停下来问。
  两档中间的这段正是这个模块存在的理由——**猜错的成本远高于反问**。
  两个阈值是**经验值**，量纲是模型自报的 [0, 1]（见 `ragamer.query`），
  本项目还没有评测集，照 §11 的口径：任何调参都应先有评测集。
- **调用方给的游戏是默认值，不是判断**。页面或会话里已经选定的那个库在没别的依据时兜底；
  模型真在问题里读出了另一款游戏，以问题为准。
- **恢复这条路上一个写操作都没有**（§3.4 的坑：原项目用 LangGraph 的 `interrupt()`，
  恢复时节点从头重执行，节点内的落库因此必须外提或幂等）。这里没有那个机制，
  但同一条约束换了个位置成立：**写只发生在暂停的那一刻**，待澄清记录写一条，
  之后恢复多少次都是拿它算一遍，结果一致、记录也不会多出来。
  将来在恢复路径上接缓存（§4）时要注意：那条写也必须幂等。
- **一次只问一个维度，且先问游戏**。版本候选是按游戏取的（见下），游戏没定就无从谈版本。

**版本候选只在游戏已经定下来时才有**：它是从那个库的切片里读出来的（`ChunkStore.versions`），
而不知道是哪款游戏就不知道该读哪个库。因此这一版里，只带着问题来（不传 `game_id`）
的提问判不出版本——它按知识库的现行版本走，这是 `ragamer.query.version_filter` 既有的回落。
反过来，模型判出的游戏与候选来源那个库不是同一个时，它那条版本判断**作废**：
拿 A 库的版本列表去认 B 库的版本，认出来也不作数，照它检索只会查空。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from ragamer.answering import require_question
from ragamer.knowledge import list_knowledge_bases
from ragamer.llm import LlmClient, Message
from ragamer.logging import get_logger
from ragamer.query import Understanding, understand
from ragamer.stores.base import ChunkStore, DocStore

logger = get_logger(__name__)

#: 待澄清的问题落在哪儿。**文档 id 就是暂停点的 id**——用户点完按钮拿它回来继续。
PENDING_COLLECTION = "pending_questions"

#: 两个维度的名字。它们同时是接口上的取值，改字面就是改协议。
GAME = "game"
VERSION = "version"

#: 确定度的两档阈值（§3.4 坑 #13）。**经验值**：本项目还没有评测集（§11），
#: 这两个数是从原项目继承下来的标定，量纲是模型自报的 [0, 1]。
#: 到阈值**含**在本档（`>=`）：改边界等于改分级，别顺手四舍五入。
CONFIDENT = 0.65
UNSURE = 0.50

#: 反问时摆在按钮上方的那句话。v1 只有中文这一份，界面直接用它。
_PROMPTS = {GAME: "你想问的是哪款游戏？", VERSION: "你想问的是哪个版本？"}


class UnknownPending(LookupError):
    """没有这个暂停点：id 打错了，或者那次提问的记录已经不在了。

    与「选的东西不在候选里」分开：那个是这次请求带错了值，这个是这次请求问错了对象。
    """

    def __init__(self, pending_id: str) -> None:
        super().__init__(f"没有 id 为 {pending_id!r} 的待澄清问题")


class NotACandidate(ValueError):
    """选的东西不在这次反问给出的候选里。

    **必须报错而不是照它去检索**：按钮之外的值说明这个请求不是那份暂停点发出来的，
    而它多半来自模型或人手编的一个取值——库里有没有它，这一层无从知道。
    """

    def __init__(self, label: str, dimension: str) -> None:
        self.label = label
        self.dimension = dimension
        super().__init__(f"{label!r} 不在这次反问给出的{dimension}候选里")


class NoKnowledgeBase(LookupError):
    """一个知识库都没有，问题里的游戏又判不出来。

    没有候选可问，也没有已定的游戏可依：这一问无解。让它走到检索那一步才炸的话，
    报出来的是 collection 名不合法之类的实现细节，而真正的原因是「还没建过库」。
    """

    def __init__(self) -> None:
        super().__init__("还没有任何知识库。先在知识库管理里建一个，再提问")


@dataclass(frozen=True)
class Choice:
    """一个候选项：**摆给人看的字面** + 它对应到库里的取值。

    两者分开是因为游戏这两个常常不同：库里叫 `black_myth`（它还得是合法的 collection 名），
    界面上叫「黑神话·悟空」。模型与用户认的都是标签，检索用的是取值。
    版本上两者相同。
    """

    label: str
    value: str


@dataclass(frozen=True)
class Resolved:
    """一次提问判完之后的落脚点：拿它去检索与生成。

    三个字段正是检索侧要的那三样（`ragamer.answering.Answerer` 的入参），
    所以「决定」与「生成」之间传的就是它，不必在各处拆开重拼。
    """

    #: 改写后的问法。检索与生成都用它——用原话会丢掉指代。
    rewritten_query: str
    #: 这一轮在哪个库里查。
    game_id: str
    #: 这一轮按哪个版本过滤。空串表示没定下来，检索回落知识库的现行版本。
    version: str


@dataclass(frozen=True)
class Clarification:
    """一次反问：问哪个维度、候选有哪些，以及回到哪一步继续。

    `pending_id` 是暂停点的凭据——用户选完带着它与选中的标签回来（`Clarifier.resolve`）。
    """

    pending_id: str
    dimension: str
    prompt: str
    choices: tuple[Choice, ...]


@dataclass(frozen=True)
class Pending:
    """停在半路的一次提问。

    存下来的是**恢复所需的最小一集**：改写后的问法（恢复时拿它去检索与生成，
    不必再问一次模型）、已经定下来的取值、等的是哪个维度，以及当时给出的候选。
    候选一并存下来而不是恢复时重读一遍语料：用户点的必须是他当时看到的那些之一
    ——两次请求之间语料变了的话，重读会认不出他的选择。
    """

    pending_id: str
    rewritten_query: str
    dimension: str
    #: 这一轮的取值是照哪个库取的：问版本时是已经定下来的游戏，问游戏时是调用方给的那个。
    #: 用户点的若不是这个库，记录里那条版本判断就不作数（见 `Clarifier.resolve`）。
    game_id: str
    #: 已经定下来的版本。只有确定度过线的那种才带得走；问版本时它是要问的东西，留空。
    version: str
    choices: tuple[Choice, ...]

    def payload(self) -> dict[str, Any]:
        """落库的形态。键名与 `_from_payload` 一一对应，两边一起改。"""
        return {
            "rewritten_query": self.rewritten_query,
            "dimension": self.dimension,
            "game_id": self.game_id,
            "version": self.version,
            "choices": [{"label": choice.label, "value": choice.value} for choice in self.choices],
        }


def game_choices(docs: DocStore) -> tuple[Choice, ...]:
    """库里真实存在的游戏。标签是显示名，取值是游戏 id（它同时是 collection 名）。

    **显示名重名时标签带上 id**：标签是模型与用户认的依据，两个库同名时标签分不开，
    模型选的那个就会被认到另一个库上去——查出来的是另一款游戏的资料，而且不报错。

    **建了库却一份资料都没导的库也在候选里**。它确实「在语料里没有内容」，但把它藏起来
    的代价更大：用户看得到自己建过这个库，点了却没有任何反应，那才是真正说不清的状态。
    摆出来，点进去得到的是「知识库里没有找到与这个问题相关的资料」——那句话正是实情。
    """
    bases = list_knowledge_bases(docs)
    names = Counter(base.name for base in bases)
    return tuple(
        Choice(
            label=base.name if names[base.name] == 1 else f"{base.name}（{base.game_id}）",
            value=base.game_id,
        )
        for base in bases
    )


def version_choices(chunks: ChunkStore, game_id: str) -> tuple[Choice, ...]:
    """这个库里真实存在过的版本。**未标注版本不在候选里**（见 `ChunkStore.versions`）。"""
    return tuple(Choice(label=version, value=version) for version in chunks.versions(game_id))


@dataclass(frozen=True)
class Clarifier:
    """读取侧的入口：提问进，答案或一次反问出。外部依赖由组合根注入。

    与 `ragamer.answering.Answerer` 是同一个打法：不可变对象，一次接线反复使用。
    """

    chunks: ChunkStore
    docs: DocStore
    llm: LlmClient

    def decide(
        self,
        question: str,
        *,
        game_id: str = "",
        version: str = "",
        history: Sequence[Message] = (),
    ) -> Resolved | Clarification:
        """读一个问题，**只判不定答案**：判得出给 :class:`Resolved`，判不准给一次反问。

        这是「决定」与「生成」之间那道缝：这一层只判，生成归 `ragamer.answering`。
        `ragamer.conversations.Chat` 拿着判定的落脚点去逐字流式地生成、并把这一轮连同
        引用与图片落进会话。**判定只有这一处**——各写一遍迟早分岔，而分岔的那一次
        表现为「同一个问题在页面上被反问、在接口上直接作答」。

        :param game_id: 这次提问所在的游戏知识库（页面或会话里选定的那一个）。
            空串表示没有上下文可依——那就只能从库里读候选来问。
        :param version: 会话里选定的版本。空串表示没选定，按知识库的现行版本走。
        :param history: 上一轮及更早的对话。指代（「那它掉什么」）要靠它才补得全，
            而指代补不全时判出来的游戏与版本多半也是错的。
        :raises NoKnowledgeBase: 一个库都没有。**拦在理解之前**：候选为空时模型判不出
            任何游戏（`ragamer.query._pick` 不会接受候选外的取值），这一次理解注定白调。
        :raises ValueError: 问题为空（由 :func:`ragamer.answering.require_question` 报出来）。
            **拦在理解之前**：空问题本来就没什么可理解的，而照它问下去会得到一次
            「你问的是哪款游戏」的反问。
        :raises ragamer.llm.LlmError: 理解失败。
        """
        require_question(question)
        games = game_choices(self.docs)
        if not games:
            raise NoKnowledgeBase
        # 版本候选按游戏取。不知道是哪款游戏时给不出候选，模型也就判不出版本（见模块说明）
        versions = version_choices(self.chunks, game_id) if game_id else ()
        understanding = understand(
            question,
            llm=self.llm,
            games=[choice.label for choice in games],
            versions=[choice.label for choice in versions],
            history=history,
        )
        resolved_game = self._game(understanding, games, game_id)
        if resolved_game is None:
            # 已经判出来的版本一并带上：用户点的还是同一个库时它照旧算数（见 `resolve`）
            return self._ask(
                understanding, GAME, games, game_id=game_id, version=_settled(understanding)
            )
        if resolved_game != game_id:
            # 版本候选是照调用方那个库给的，模型说的却是另一个库：它那条版本判断没有依据
            understanding = replace(understanding, version="", version_confidence=0.0)
        resolved_version = self._version(understanding, versions, version)
        if resolved_version is None:
            return self._ask(understanding, VERSION, versions, game_id=resolved_game)
        return Resolved(understanding.rewritten_query, resolved_game, resolved_version)

    def resolve(self, pending_id: str, label: str) -> Resolved:
        """从暂停点继续：用户点的那一项补进那次判定，交回这一次的落脚点。

        **这一步只读不写**（见模块说明）：重复调用得到同一份判定，也不会多出一条记录。

        :raises UnknownPending: 没有这个暂停点。
        :raises NotACandidate: 选的不在这次反问给出的候选里。
        """
        pending = self._load_pending(pending_id)
        choice = _choice_of(pending.choices, label, pending.dimension)
        if pending.dimension == GAME:
            game_id = choice.value
            # 记录里那条版本判断是照 `pending.game_id` 那个库给的：用户点的若不是它，
            # 拿 A 库的版本去检索 B 库只会查空，所以换库就作废
            version = pending.version if choice.value == pending.game_id else ""
        else:
            game_id, version = pending.game_id, choice.value
        logger.info("从暂停点 %s 继续：%s 取 %r", pending_id, pending.dimension, choice.value)
        return Resolved(pending.rewritten_query, game_id, version)

    def _game(
        self, understanding: Understanding, choices: Sequence[Choice], game_id: str
    ) -> str | None:
        """定下问的是哪款游戏。定不下来返回 `None`，由调用方去问。

        两档在这里分开（见模块说明）：确定度够就用模型判的那个；落在中间那档才考虑问。
        调用方给的游戏是**默认值**：模型判出的那个与它一致时不必再问——问了也只有同一个
        答案，白白打断一次；**对不上才是真的拿不准**，那种才值得把候选摆出来。
        `choices` 已经由 `start` 保证非空——空的那一档进不了这里。
        """
        if understanding.game and understanding.game_confidence >= CONFIDENT:
            return _choice_of(choices, understanding.game, GAME).value
        judged = _choice_of(choices, understanding.game, GAME).value if understanding.game else ""
        if game_id and judged in ("", game_id):
            return game_id
        if _worth_asking(choices):
            return None
        # 只有一个库时没有可问的：候选已经把它定死了
        return judged or game_id or choices[0].value

    def _version(
        self, understanding: Understanding, choices: Sequence[Choice], version: str
    ) -> str | None:
        """定下问的是哪个版本。定不下来返回 `None`，由调用方去问。

        与游戏那档有一点不同：**没判出版本不构成反问**。版本是有默认值的
        （知识库的现行版本，ADR-0004），而游戏没有——问「哪个版本」之前先得有依据。
        """
        if understanding.version and understanding.version_confidence >= CONFIDENT:
            return understanding.version
        if (
            understanding.version
            and understanding.version_confidence >= UNSURE
            and _worth_asking(choices)
        ):
            return None
        return version

    def _ask(
        self,
        understanding: Understanding,
        dimension: str,
        choices: Sequence[Choice],
        *,
        game_id: str = "",
        version: str = "",
    ) -> Clarification:
        """记下暂停点，把候选交回去。

        **写只发生在这一处**：待澄清记录写一条，恢复那条路上没有再写任何东西
        （见模块说明）。
        """
        pending = Pending(
            pending_id=uuid4().hex,
            rewritten_query=understanding.rewritten_query,
            dimension=dimension,
            game_id=game_id,
            version=version,
            choices=tuple(choices),
        )
        self.docs.put(PENDING_COLLECTION, pending.pending_id, pending.payload())
        logger.info(
            "提问 %r 的%s判不准，停下来问（候选 %d 个，暂停点 %s）",
            understanding.rewritten_query,
            dimension,
            len(choices),
            pending.pending_id,
        )
        return Clarification(
            pending_id=pending.pending_id,
            dimension=dimension,
            prompt=_PROMPTS[dimension],
            choices=pending.choices,
        )

    def _load_pending(self, pending_id: str) -> Pending:
        payload = self.docs.get(PENDING_COLLECTION, pending_id)
        if payload is None:
            raise UnknownPending(pending_id)
        return _from_payload(pending_id, payload)


def _settled(understanding: Understanding) -> str:
    """这一轮已经定下来的版本，带得走的那种。

    只有确定度过 :data:`CONFIDENT` 的才算：夹在两档之间的那个还等着问，不是结论。
    没有判出版本时是空串——调用方给的那个版本属于它自己的库，跟着游戏一起换库就错了。
    """
    if understanding.version and understanding.version_confidence >= CONFIDENT:
        return understanding.version
    return ""


def _worth_asking(choices: Sequence[Choice]) -> bool:
    """只有一个候选时没什么可问的：答案已经被候选定死了，问一句只是多一次点击。

    游戏那一档还可能是「这个库里没有」，但那种情况给一个单按钮也解决不了
    ——`ragamer.answering` 的「没找到」回复才是它该得到的。
    """
    return len(choices) >= 2


def _choice_of(choices: Sequence[Choice], label: str, dimension: str) -> Choice:
    """标签 → 候选。两处都从这里走：模型判出来的游戏，与用户点的那一个。

    取不到时宁可当场报出来，也不退回空串——空串意味着换一个游戏去检索，
    查出来的是别的游戏的资料，而且不报错。
    """
    for choice in choices:
        if choice.label == label:
            return choice
    raise NotACandidate(label, dimension)


def _from_payload(pending_id: str, payload: Mapping[str, Any]) -> Pending:
    """库里那份文档 → 这个对象。"""
    return Pending(
        pending_id=pending_id,
        rewritten_query=str(payload["rewritten_query"]),
        dimension=str(payload["dimension"]),
        game_id=str(payload["game_id"]),
        version=str(payload["version"]),
        choices=tuple(
            Choice(label=str(choice["label"]), value=str(choice["value"]))
            for choice in payload["choices"]
        ),
    )
