"""知识库元数据：一个游戏知识库在库里长什么样，以及建它、改它、删干净它。

**写入侧（界面上的「新建知识库」）与读取侧（导入端点要的打标词表）共用这一份定义。**
两边各自拼一遍字典的话，键名写岔了既不报错也不崩——只会静默少读一个字段，而打标从此
按默认词表跑，标签稀疏得看不出是配置没读到。

元数据存 MongoDB 的 `knowledge_bases` 集合，**文档 id 就是游戏 id**：它同时是 Milvus 的
collection 名（ADR-0002），所以合法性校验直接复用 `collection_name`，不另立一套规则。
文档里还存着**现行版本**——检索与聚合父块都从这一处取，不各自维护一份。

「配置读不了」与「库不存在」是两件事，这里分成两个异常：前者的数据坏了，后者是游戏选错了。
`ragamer.api` 把它们映射成 422 与 404，界面把它们渲染成两句话——判断只在这一处做。

**删库要清四处**（架构文档 §5）：向量库 collection、对象存储前缀、Mongo 里的配置与会话、
缓存。今天接上的存储只有前三处里的前两项加配置本体，会话与缓存还没有存储承载
（T16 / T21），等它们接上时在 `purge_knowledge_base` 那一处补一行。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ragamer.logging import get_logger
from ragamer.stores.base import (
    UNVERSIONED,
    ChunkStore,
    DocStore,
    ObjectStore,
    collection_name,
    image_folder,
)
from ragamer.tagging import SubjectType, TagVocabulary

logger = get_logger(__name__)

#: 知识库元数据所在的集合，文档 id 就是游戏 id。
KB_COLLECTION = "knowledge_bases"


class KnowledgeBaseError(Exception):
    """知识库本身的问题——不是这次导入的资料的问题。

    两个 HTTP 面（JSON 端点与页面）都要把它翻成状态码，**翻法必须一致**：各自维护一份
    映射，迟早会变成「同一个问题在接口上是 404、在页面上是 400」。所以状态码跟着异常走，
    两边都从这里取。
    """

    #: 该报哪个 HTTP 状态码。由子类给。
    status = 500


class UnknownKnowledgeBase(KnowledgeBaseError):
    """没有这个知识库：游戏选错了，或者还没建。

    与「配置读不了」分开，是因为处理方式不同：这一个要人去建库，那一个是库的数据坏了。
    """

    status = 404

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        super().__init__(f"知识库 {game_id} 不存在。先在知识库管理里建一个，再导入资料")


class BrokenKnowledgeBase(KnowledgeBaseError):
    """库在，但配置读不了——比如配了个不认识的主体类型。

    这是库自己的数据坏了：不该长成一个 500，那样界面上只会看见「服务器错误」，查无可查。
    """

    status = 422

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
    #: 现行版本。检索默认按它过滤（架构文档 §2.4），空串即未标注版本。
    version: str = UNVERSIONED
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
                version=_version(payload),
                problem=str(exc),
            )
        return cls(
            game_id=game_id,
            name=str(payload.get("name") or game_id),
            vocabulary=vocabulary,
            version=_version(payload),
        )

    def payload(self) -> dict[str, Any]:
        """存进库里的形态。键名由 `TagVocabulary.from_mapping` 那一侧定，两边必须对得上。"""
        return {
            "name": self.name,
            "version": self.version,
            "subject_types": [kind.value for kind in self.vocabulary.subject_types],
            "term_mapping": {
                term: kind.value for term, kind in sorted(self.vocabulary.term_mapping.items())
            },
        }


def _version(payload: Mapping[str, Any]) -> str:
    """配置里的现行版本。老库里没有这个键——回落到未标注版本，不是随手挑一个。"""
    return str(payload.get("version") or UNVERSIONED)


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


def update_knowledge_base(docs: DocStore, knowledge_base: KnowledgeBase) -> None:
    """改一个已有的库：名称、启用的主体类型、术语映射、现行版本一起换掉。

    **库不存在时报错，不静默新建**：界面上点的是「保存」，凭空冒出一个库不是它要的结果，
    而且新库的 id 多半是打错的那个。

    :raises ValueError: 游戏 id 不合法。
    :raises UnknownKnowledgeBase: 没有这个库。
    """
    game_id = knowledge_base.game_id
    # 与建库同一个理由：id 同时是 collection 名，改错了后面的数据会落到别处
    collection_name(game_id)
    if docs.get(KB_COLLECTION, game_id) is None:
        raise UnknownKnowledgeBase(game_id)
    docs.put(KB_COLLECTION, game_id, knowledge_base.payload())
    logger.info("更新知识库 %s（%s）", game_id, knowledge_base.name)


@dataclass(frozen=True)
class PurgeInventory:
    """这个库在各处占着多少东西。

    确认页照着它逐条列出「将要清理什么」，清完之后回话的也是它——两处**数的必须是同一批
    东西**（整个 collection 的切片、整个前缀下的对象）：页面上写「12 张原图」而真删掉
    15 张，那份确认就成了摆设。
    """

    chunk_count: int
    image_count: int


class PurgeError(KnowledgeBaseError):
    """删库时某几处没清干净。

    **知识库配置没有被删**——它是「这个库还在」的凭据：先删配置，剩下的切片与原图就再也
    看不见了，只能靠人去翻 Milvus 与 MinIO。留着它，界面上还能再点一次；已经清掉的那几处
    再清一次也不会出错。
    """

    status = 500

    def __init__(self, game_id: str, failures: Sequence[str]) -> None:
        self.game_id = game_id
        self.failures = tuple(failures)
        super().__init__(
            f"知识库 {game_id} 没有清干净，已经停下："
            + "；".join(failures)
            + "。这个库还留在列表里，可以再删一次"
        )


def purge_inventory(chunks: ChunkStore, objects: ObjectStore, game_id: str) -> PurgeInventory:
    """数一遍这个库在各处占着多少东西。**只读**，不动任何数据。

    确认页在人按下「确认删除」之前调它。数不出来的话，那份确认就只剩一句「会删掉一些
    东西」——而删除是不可逆的，说清楚要删什么正是这一步的用处。
    """
    return PurgeInventory(
        chunk_count=chunks.count(game_id),
        # 前缀带尾随斜杠：少了它会把 id 是它前缀的另一个库的原图也数进来（见 `image_folder`）
        image_count=len(objects.list_keys(image_folder(game_id))),
    )


def purge_knowledge_base(
    chunks: ChunkStore, docs: DocStore, objects: ObjectStore, game_id: str
) -> PurgeInventory:
    """把这个库在各处的数据清干净，**最后才删它自己的配置**。

    前三处各自兜住失败，一处清了不跳过其余：一次就能看清还差什么，而不是修一处再删一次。
    配置排在最后（且只有前面全清了才动它），失败时它就是重来的凭据，见 `PurgeError`。

    **兜住的是全部异常，不只是 `StoreError`**：三个适配器只把连不上包成 `StoreError`，
    连上之后操作失败（Mongo 掉线、表被并发删掉）漏出来的是供应商自己的异常类型。这一处
    要保证的是「不论哪儿出岔子，都别把库删成看不见的孤儿」——只认一种异常类型，
    这条保证就漏了一大半。

    :raises PurgeError: 有哪一处没清干净；此时知识库配置原样留着。

    ⚠️ 这里多清一处，`knowledge_base_delete.html` 里那张「将要清理的数据」清单就要跟着加一条：
    确认页上少一行，人按下确认时看到的就不是全部。
    """
    chunk_count = 0
    image_count = 0
    failures: list[str] = []

    try:
        # 先数再删：collection 一 drop，条数就再也问不出来了
        chunk_count = chunks.count(game_id)
        chunks.drop(game_id)
    except Exception as exc:
        failures.append(f"向量库：{exc}")

    try:
        # `delete_prefix` 把删掉的个数报回来，不必先列一遍再数一遍
        image_count = objects.delete_prefix(image_folder(game_id))
    except Exception as exc:
        failures.append(f"对象存储：{exc}")

    if not failures:
        try:
            docs.delete(KB_COLLECTION, game_id)
        except Exception as exc:
            failures.append(f"MongoDB：{exc}")

    if failures:
        raise PurgeError(game_id, failures)
    logger.info("删除知识库 %s：清掉 %d 条切片、%d 个原图", game_id, chunk_count, image_count)
    return PurgeInventory(chunk_count=chunk_count, image_count=image_count)


def knowledge_base_of(docs: DocStore, game_id: str) -> KnowledgeBase:
    """读一个库。

    配置读不出来时**照常返回**（`problem` 里带着原因）：坏掉的库也要能在界面上显示、
    也点得进它的配置页去修——列表里藏起来的话，id 被占着，人连该修哪个都不知道。

    :raises ValueError: 游戏 id 不合法。
    :raises UnknownKnowledgeBase: 没有这个库。
    """
    collection_name(game_id)
    payload = docs.get(KB_COLLECTION, game_id)
    if payload is None:
        raise UnknownKnowledgeBase(game_id)
    return KnowledgeBase.from_payload(game_id, payload)


def readable_knowledge_base(docs: DocStore, game_id: str) -> KnowledgeBase:
    """读一个库，并要求它的配置是读得出来的。

    配置读不出来的库只能被显示、被修、被删，**不能被拿来干活**：那份词表本来就没读出来，
    照着内存里的默认值用下去，标签会按「七类全开、映射为空」落——看起来一切正常，
    错的全在标签里。

    :raises ValueError: 游戏 id 不合法。
    :raises UnknownKnowledgeBase: 没有这个库。
    :raises BrokenKnowledgeBase: 库在，但配置读不了。
    """
    knowledge_base = knowledge_base_of(docs, game_id)
    if knowledge_base.problem:
        raise BrokenKnowledgeBase(game_id, knowledge_base.problem)
    return knowledge_base


def vocabulary_of(docs: DocStore, game_id: str) -> TagVocabulary:
    """这个知识库的打标词表。

    配置里没写的项一律走 `TagVocabulary` 的默认值（全部主体类型、映射为空）——用户自定义库
    没配映射时就是这条降级路径，标签会稀疏但不会漏（docs/ARCHITECTURE.md §2.3）。

    :raises UnknownKnowledgeBase: 没有这个库。
    :raises BrokenKnowledgeBase: 库在，但配置里的类目认不出来。
    """
    return readable_knowledge_base(docs, game_id).vocabulary


def set_term(docs: DocStore, game_id: str, term: str, kind: SubjectType) -> None:
    """给这个库的术语映射加一条；叫法与表里已有的完全一样时改掉它的归类。

    「读出来 → 改映射 → 原样写回去」整段放在这里，不放到页面上：那边要自己拼
    `term_mapping`、还要记得把启用的主体类型一并带回去，抄一遍就是两处各错一半的机会。

    :raises ValueError: 叫法归一之后与表里另一条撞上了（见 `TagVocabulary`）。
    :raises BrokenKnowledgeBase: 库的配置读不了——先修好它再来配映射。
    """
    knowledge_base = readable_knowledge_base(docs, game_id)
    update_knowledge_base(
        docs,
        replace(
            knowledge_base,
            vocabulary=TagVocabulary(
                knowledge_base.vocabulary.subject_types,
                {**knowledge_base.vocabulary.term_mapping, term: kind},
            ),
        ),
    )


def remove_term(docs: DocStore, game_id: str, term: str) -> None:
    """从术语映射里去掉一条。表里没有这个叫法就什么都不做。

    去掉之后语料里的这个叫法就只剩模型兜底那条路了——标签稀疏，但不会漏。
    """
    knowledge_base = readable_knowledge_base(docs, game_id)
    remaining = {
        name: kind for name, kind in knowledge_base.vocabulary.term_mapping.items() if name != term
    }
    update_knowledge_base(
        docs,
        replace(
            knowledge_base,
            vocabulary=TagVocabulary(knowledge_base.vocabulary.subject_types, remaining),
        ),
    )
