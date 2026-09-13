"""知识库元数据：一个游戏知识库在库里长什么样。

**写入侧（界面上的「新建知识库」）与读取侧（导入端点要的打标词表）共用这一份定义。**
两边各自拼一遍字典的话，键名写岔了既不报错也不崩——只会静默少读一个字段，而打标从此
按默认词表跑，标签稀疏得看不出是配置没读到。

元数据存 MongoDB 的 `knowledge_bases` 集合，**文档 id 就是游戏 id**：它同时是 Milvus 的
collection 名（ADR-0002），所以合法性校验直接复用 `collection_name`，不另立一套规则。

「配置读不了」与「库不存在」是两件事，这里分成两个异常：前者的数据坏了，后者是游戏选错了。
`ragamer.api` 把它们映射成 422 与 404，界面把它们渲染成两句话——判断只在这一处做。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragamer.logging import get_logger
from ragamer.stores.base import DocStore, collection_name
from ragamer.tagging import SubjectType, TagVocabulary

logger = get_logger(__name__)

#: 知识库元数据所在的集合，文档 id 就是游戏 id。
KB_COLLECTION = "knowledge_bases"


class KnowledgeBaseError(Exception):
    """知识库本身的问题——不是这次导入的资料的问题。"""


class UnknownKnowledgeBase(KnowledgeBaseError):
    """没有这个知识库：游戏选错了，或者还没建。

    与「配置读不了」分开，是因为处理方式不同：这一个要人去建库，那一个是库的数据坏了。
    """

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        super().__init__(f"知识库 {game_id} 不存在。先在知识库管理里建一个，再导入资料")


class BrokenKnowledgeBase(KnowledgeBaseError):
    """库在，但配置读不了——比如配了个不认识的主体类型。"""

    def __init__(self, game_id: str, reason: str) -> None:
        self.game_id = game_id
        self.reason = reason
        super().__init__(f"知识库 {game_id} 的配置读不了：{reason}")


@dataclass(frozen=True)
class KnowledgeBase:
    """一个游戏知识库。字段就是 `knowledge_bases` 里那份文档的内容。

    `problem` 非空表示这份配置读不出来。**列表里不把它藏起来**——藏了界面上就看不见它，
    而 id 又被占着：用户既建不了同 id 的新库，也不知道该去修哪个。
    """

    game_id: str
    name: str
    vocabulary: TagVocabulary
    #: 配置读不出来时的原因；正常时是空串。
    problem: str = ""

    @classmethod
    def new(
        cls,
        game_id: str,
        name: str = "",
        subject_types: Sequence[SubjectType] | None = None,
    ) -> KnowledgeBase:
        """新建一个库。术语映射留空——那要等知识库管理页来配。

        显示名留空就回落到游戏 id：界面上认得出是哪个库就够了，不必逼人再想一个名字。
        勾选的类目留空则是**错**，不是「默认全开」：没勾任何一个多半是漏了，
        静默全开会让标签多出一批用户以为自己关掉了的。

        :raises ValueError: 游戏 id 不合法，或一个主体类型都没启用。
        """
        collection_name(game_id)
        return cls(
            game_id=game_id,
            name=name.strip() or game_id,
            vocabulary=TagVocabulary(
                tuple(SubjectType) if subject_types is None else tuple(subject_types)
            ),
        )

    @classmethod
    def from_payload(cls, game_id: str, payload: Mapping[str, Any]) -> KnowledgeBase:
        """从库里的文档读回来。配置里没写的项走 `TagVocabulary` 的默认值。

        读不出来不抛异常而是记进 `problem`：列表页要能把它显示出来（见类文档）。
        真要拿它打标时，`vocabulary_of` 会照着这个字段报错。
        """
        try:
            vocabulary = TagVocabulary.from_mapping(payload)
        except ValueError as exc:
            return cls(
                game_id=game_id,
                name=str(payload.get("name") or game_id),
                vocabulary=TagVocabulary(),
                problem=str(exc),
            )
        return cls(game_id=game_id, name=str(payload.get("name") or game_id), vocabulary=vocabulary)

    def payload(self) -> dict[str, Any]:
        """存进库里的形态。键名由 `TagVocabulary.from_mapping` 那一侧定，两边必须对得上。"""
        return {
            "name": self.name,
            "subject_types": [kind.value for kind in self.vocabulary.subject_types],
            "term_mapping": {
                term: kind.value for term, kind in sorted(self.vocabulary.term_mapping.items())
            },
        }


def list_knowledge_bases(docs: DocStore) -> list[KnowledgeBase]:
    """全部知识库，按游戏 id 字典序（`DocStore.list_ids` 的次序）。"""
    bases: list[KnowledgeBase] = []
    for game_id in docs.list_ids(KB_COLLECTION):
        payload = docs.get(KB_COLLECTION, game_id)
        if payload is None:  # 列出来之后被删了
            continue
        bases.append(KnowledgeBase.from_payload(game_id, payload))
    return bases


def create_knowledge_base(docs: DocStore, knowledge_base: KnowledgeBase) -> None:
    """建一个库，**id 已被占用时报错而不是覆盖**。

    覆盖会连着这个库的术语映射一起换掉，而界面上看起来只是"又建了一个"。

    :raises ValueError: 游戏 id 不合法，或该 id 已经有一个库了。
    """
    # 再校一遍：这个对象也可能是从别处（比如 `from_payload`）造出来的，
    # 而一个不合法的 id 会让 collection 名与库里的文档 id 对不上
    collection_name(knowledge_base.game_id)
    if docs.get(KB_COLLECTION, knowledge_base.game_id) is not None:
        raise ValueError(
            f"已经有一个 id 为 {knowledge_base.game_id} 的知识库了。"
            "换一个 id，或者直接往已有的那个里导入资料"
        )
    docs.put(KB_COLLECTION, knowledge_base.game_id, knowledge_base.payload())
    logger.info("新建知识库 %s（%s）", knowledge_base.game_id, knowledge_base.name)


def vocabulary_of(docs: DocStore, game_id: str) -> TagVocabulary:
    """这个知识库的打标词表。

    配置里没写的项一律走 `TagVocabulary` 的默认值（全部主体类型、映射为空）——用户自定义库
    没配映射时就是这条降级路径，标签会稀疏但不会漏（docs/ARCHITECTURE.md §2.3）。

    :raises UnknownKnowledgeBase: 没有这个库。
    :raises BrokenKnowledgeBase: 库在，但配置里的类目认不出来。
    """
    payload = docs.get(KB_COLLECTION, game_id)
    if payload is None:
        raise UnknownKnowledgeBase(game_id)
    knowledge_base = KnowledgeBase.from_payload(game_id, payload)
    if knowledge_base.problem:
        raise BrokenKnowledgeBase(game_id, knowledge_base.problem)
    return knowledge_base.vocabulary
