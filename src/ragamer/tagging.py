"""打标：给切片补上两层正交的标签——主体（文档级）与内容性质（切片级）。

粒度不同是有意的：「背景故事」和「打法」在同一个页面里，文档级统一就废了——
理由见 [ADR-0003](../docs/adr/0003-two-orthogonal-tag-fields.md)。生产顺序是
**结构优先、模型兜底**：

- **主体名 / 主体类型**（文档级）：语料有结构就读 MediaWiki 分类与 Infobox 类型，
  **不调模型**；没有才让模型读开头若干切片总结，回写全文档。
- **内容性质**（切片级）：语料有结构就按该切片的标题归一，**不调模型**；
  没有才让模型逐切片判断，带上切片位置作弱上下文。

两条路各自独立降级：主体结构读不出来只影响主体字段，标题认不出的切片只问它自己
那一片。**打标失败一律不阻断入库**——标签留空、正文照常返回，见每一处 `except`。
这里只接 `LlmError`：语言模型客户端对外承诺的就是它，宽泛地吞异常会把真 bug 也吞掉
（旧项目的反向清单里正有一条「按批吞异常」）。

已知的限：结构读取按行扫，围栏代码块里的 `[[Category:…]]` 会被当成真的分类。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ragamer.chunking import PATH_SEPARATOR, Chunk
from ragamer.llm import LlmClient, LlmError, LlmRequest, Message
from ragamer.logging import get_logger

logger = get_logger(__name__)

#: 语料没结构时，喂给模型总结主体信息的切片数。多了费上下文，少了看不出主体。
SUBJECT_SAMPLE_CHUNKS = 5


class SubjectType(StrEnum):
    """主体所属的类别。七类，**文档级**——一份文档归属一个主体（架构文档 §2.3）。

    中文叫法见 `SUBJECT_TYPE_LABELS`，只是同一套类的另一面，不是第二套词表。
    """

    CHARACTER = "character"
    ITEM = "item"
    PLACE = "place"
    QUEST = "quest"
    SKILL = "skill"
    BACKGROUND = "background"
    SYSTEM = "system"


#: 每一类覆盖哪些说法。取自架构文档 §2.3 的原话，同时进提示词。
SUBJECT_TYPE_LABELS: Mapping[SubjectType, str] = {
    SubjectType.CHARACTER: "角色／怪物／BOSS／组织势力",
    SubjectType.ITEM: "物品／装备／道具／素材",
    SubjectType.PLACE: "地图／场景／关卡",
    SubjectType.QUEST: "任务／副本／活动",
    SubjectType.SKILL: "技能／法术／招式／心法",
    SubjectType.BACKGROUND: "世界观／剧情／设定",
    SubjectType.SYSTEM: "机制／规则／玩法",
}


class ContentNature(StrEnum):
    """一个切片回答的是哪一类问题。五类，**切片级**——同一文档里可以并存。

    🔴 绝不能文档级统一，理由见 ADR-0003。
    """

    INTRO = "intro"
    WHERE = "where"
    STATS = "stats"
    GUIDE = "guide"
    REVIEW = "review"


#: 每一类回答什么问题。取自架构文档 §2.3 的原话，同时进提示词。
CONTENT_NATURE_LABELS: Mapping[ContentNature, str] = {
    ContentNature.INTRO: "是什么／是谁",
    ContentNature.WHERE: "在哪／怎么获得／掉落",
    ContentNature.STATS: "数值属性",
    ContentNature.GUIDE: "怎么打／流程步骤",
    ContentNature.REVIEW: "好不好／哪个强",
}

#: 一级词表收的通用叫法（架构文档 §2.3 里每一类后面列的那几个词）。
#: 它让**没配术语映射的自定义库**也能直接读中文 wiki 的分类，不必先问一遍模型。
#: 键是 `_normalize` 之后的形态，查表前也要过一遍 `_normalize`。
GLOBAL_TERMS: Mapping[str, SubjectType] = {
    "角色": SubjectType.CHARACTER,
    "怪物": SubjectType.CHARACTER,
    "boss": SubjectType.CHARACTER,
    "组织势力": SubjectType.CHARACTER,
    "物品": SubjectType.ITEM,
    "装备": SubjectType.ITEM,
    "道具": SubjectType.ITEM,
    "素材": SubjectType.ITEM,
    "地图": SubjectType.PLACE,
    "场景": SubjectType.PLACE,
    "关卡": SubjectType.PLACE,
    "任务": SubjectType.QUEST,
    "副本": SubjectType.QUEST,
    "活动": SubjectType.QUEST,
    "技能": SubjectType.SKILL,
    "法术": SubjectType.SKILL,
    "招式": SubjectType.SKILL,
    "心法": SubjectType.SKILL,
    "世界观": SubjectType.BACKGROUND,
    "剧情": SubjectType.BACKGROUND,
    "设定": SubjectType.BACKGROUND,
    "机制": SubjectType.SYSTEM,
    "规则": SubjectType.SYSTEM,
    "玩法": SubjectType.SYSTEM,
}

#: 内容性质的判定词表：标题里含哪个词就算哪一类，含几个算几个。
#: 词之间用空格分开。表里的次序就是结果里各类的先后——越具体的越靠前，
#: 「属性说明」这类两头都沾的标题，数值排在介绍前头。
_NATURE_KEYWORDS: tuple[tuple[ContentNature, str], ...] = (
    (ContentNature.WHERE, "获取 获得 掉落 出处 来源 位置 地点 在哪 坐标 购买 兑换 合成 刷新 解锁"),
    (
        ContentNature.STATS,
        "属性 数值 面板 数据 参数 加成 增益 消耗 冷却 倍率 伤害 血量 抗性 需求",
    ),
    (
        ContentNature.GUIDE,
        "打法 攻略 流程 步骤 怎么打 应对 通关 技巧 配队 阵容 连招 加点 阶段 逃课",
    ),
    (ContentNature.REVIEW, "评价 推荐 强度 排行 排名 值得 优缺点 评测 选择 对比 梯队"),
    (ContentNature.INTRO, "介绍 简介 概述 背景 故事 设定 说明 是什么 档案 资料 图鉴 词条"),
)

#: MediaWiki 分类：`[[Category:角色]]` / `[[分类:角色|排序键]]`。
_CATEGORY_LINK = re.compile(r"\[\[\s*(?:Category|分类)\s*:\s*([^\]|]+)")
#: 裸写一行的分类，架构文档 §1.4 把它列为结构信号之一。
_CATEGORY_LINE = re.compile(r"^\s*(?:Category|分类)\s*:\s*(.+?)\s*$")

#: 模板块里的一项 `| 键 = 值`。值到行尾或下一个竖线为止。
_TEMPLATE_FIELD = re.compile(r"\|\s*([^|=\n]+?)\s*=\s*([^|\n]*)")
#: Infobox 里表示「名称」与「类型」的字段名，都按 `_normalize` 之后的形态比。
_NAME_KEYS: tuple[str, ...] = ("name", "title", "名称", "名字", "标题")
_TYPE_KEYS: tuple[str, ...] = ("type", "category", "类型", "分类", "类别", "种类")

#: 文档大标题。MediaWiki 页面的条目名就在这里，主体名优先取它。
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TagVocabulary:
    """一个游戏知识库的打标词表：启用的主体类型 + 该游戏的术语映射。

    两者都来自知识库自己的配置（存 MongoDB、GUI 可编辑），不来自环境变量——
    它是每个库一份的数据，不是进程级的配置。

    不传就是**全部主体类型 + 一级词表的通用叫法**：用户自定义库没配映射时默认全开，
    标签会稀疏但不会漏，是安全的降级路径（架构文档 §2.3）。
    """

    #: 这个库启用的主体类型，是七类的一个子集。没启用的类一律不落标。
    subject_types: tuple[SubjectType, ...] = tuple(SubjectType)
    #: 术语映射：该游戏对一类对象的本地叫法 → 通用主体类型，如「妖王」→ character。
    term_mapping: Mapping[str, SubjectType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject_types:
            # 空集会让每个切片都漏标，而且一路静默到检索才发现——在这里就拦下
            raise ValueError("至少要启用一个主体类型，否则全部切片都会漏标")
        # 叫法的比对形式在构造时定下来，查表时才不必指望调用方先洗干净
        normalized = {_normalize(term): kind for term, kind in self.term_mapping.items()}
        object.__setattr__(self, "term_mapping", normalized)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TagVocabulary:
        """从知识库的配置读入。配置里没写的项一律走默认值。

        `{"subject_types": ["character", "skill"], "term_mapping": {"妖王": "character"}}`
        """
        raw_types = payload.get("subject_types")
        subject_types = (
            tuple(SubjectType)
            if raw_types is None
            else tuple(_subject_type(item) for item in raw_types)
        )
        raw_terms = payload.get("term_mapping") or {}
        term_mapping = {term: _subject_type(kind) for term, kind in raw_terms.items()}
        return cls(subject_types, term_mapping)

    def resolve(self, term: str) -> SubjectType | None:
        """把一种叫法归到主体类型：先查该游戏的术语映射，再查一级词表的通用叫法。"""
        key = _normalize(term)
        found = self.term_mapping.get(key)
        return found if found is not None else GLOBAL_TERMS.get(key)

    def is_game_term(self, term: str) -> bool:
        """这个叫法是不是该游戏自己的术语，而不是一级词表的通用叫法。"""
        return _normalize(term) in self.term_mapping

    def enabled(self, kind: SubjectType) -> bool:
        return kind in self.subject_types


@dataclass(frozen=True)
class DocumentStructure:
    """从语料自身结构读出的文档级主体信息。读不出来的字段是空值。"""

    #: 主体名。文档大标题优先，其次是 Infobox 的名称字段。
    subject_name: str
    #: 主体类型，去重后按读到的次序。
    subject_types: tuple[SubjectType, ...]
    #: 命中的游戏术语原文，供 `chunks.game_terms` 用。
    game_terms: tuple[str, ...]


@dataclass(frozen=True)
class TaggedChunk:
    """打完标的切片。

    字段名与 `chunks` collection 的 schema（架构文档 §2.2）一一对应，入库时直接落。
    切片本体用组合而非抄一遍字段——`Chunk` 还会长（表格原子化那一票就要加
    `content_meta` 与 `chunk_type`），抄字段就得跟着改两处。
    """

    chunk: Chunk
    #: 主体名，文档级，回写到该文档的每一个切片。
    subject_name: str
    #: 主体类型，文档级。可以多个——「二郎神的技能」同时属于角色与技能。
    subject_type: tuple[SubjectType, ...]
    #: 内容性质，切片级。同一文档里可以并存多种。
    content_nature: tuple[ContentNature, ...]
    #: 该游戏的术语标签。
    game_terms: tuple[str, ...]


def tag_document(
    markdown: str,
    chunks: Sequence[Chunk],
    *,
    vocabulary: TagVocabulary | None = None,
    llm: LlmClient | None = None,
) -> list[TaggedChunk]:
    """给一份文档的全部切片打标。顺序与正文原样保留。

    :param markdown: 源文档，用来读 Category / Infobox / 大标题这些结构信号。
    :param chunks: 切分器的产物，顺序即文档顺序。
    :param vocabulary: 该知识库的词表；不传则全部主体类型 + 一级词表的通用叫法。
    :param llm: 兜底用的模型客户端。不传时结构读不出来的那些标签就留空，
        不报错——打标失败不阻断入库。
    """
    chunks = list(chunks)
    if not chunks:
        return []  # 没有切片就没有标签要打，别为它白调一次模型
    vocabulary = vocabulary if vocabulary is not None else TagVocabulary()
    structure = read_document_structure(markdown, vocabulary)
    tagger = _Tagger(chunks, vocabulary, llm, _document_title(markdown))
    subject_name, subject_types = tagger.subject(structure)

    return [
        TaggedChunk(
            chunk=chunk,
            subject_name=subject_name,
            subject_type=subject_types,
            content_nature=tagger.nature(chunk),
            game_terms=structure.game_terms,
        )
        for chunk in chunks
    ]


def read_document_structure(markdown: str, vocabulary: TagVocabulary) -> DocumentStructure:
    """读语料自身的结构：MediaWiki 分类、Infobox 的类型与名称、文档大标题。

    认出的叫法按术语映射归到通用主体类型，映射里没有的再查一级词表；**两者都认不
    出、或归到的类型这个库没启用，就丢掉**——宁缺勿错，模型兜底那条路还在。

    值必须能归到某个主体类型，那个模板才算数：`{{cite|type=web}}` 里的 `web`
    归不出来，于是连它的 `title` 也不会被当成主体名。
    """
    terms = list(_categories(markdown))
    name = ""
    for block in _template_blocks(markdown):
        fields = _fields(block)
        kind_values = [fields[key] for key in _TYPE_KEYS if key in fields]
        if not any(vocabulary.resolve(value) is not None for value in kind_values):
            continue
        terms.extend(kind_values)
        if not name:
            name = next((fields[key] for key in _NAME_KEYS if key in fields), "")
    if not name:
        name = _document_title(markdown)

    subject_types, game_terms = _classify(terms, vocabulary)
    return DocumentStructure(name, subject_types, game_terms)


def _document_title(markdown: str) -> str:
    """文档大标题。MediaWiki 页面的条目名就在这里。"""
    title = _TITLE.search(markdown)
    return title.group(1).strip() if title is not None else ""


def natures_for_heading(path: str) -> tuple[ContentNature, ...]:
    """按切片的标题归一内容性质。认不出返回空元组，交给模型。

    从最靠里的一段往外找：「二郎神 › 打法 › 第二阶段」里末段没有信息量，要能落到
    父标题「打法」上。一段里命中几类就留几类——「掉落与属性」确实是两类。
    """
    segments = [segment.strip() for segment in path.split(PATH_SEPARATOR) if segment.strip()]
    for segment in reversed(segments):
        matched = tuple(
            nature
            for nature, keywords in _NATURE_KEYWORDS
            if any(keyword in segment for keyword in keywords.split())
        )
        if matched:
            return matched
    return ()


@dataclass(frozen=True)
class _Tagger:
    """一份文档的打标过程。结构读不出来的部分才落到模型上。"""

    chunks: Sequence[Chunk]
    vocabulary: TagVocabulary
    llm: LlmClient | None
    #: 文档大标题。用来认出词条页的开篇——它的路径只有标题这一段。
    title: str = ""

    def subject(self, structure: DocumentStructure) -> tuple[str, tuple[SubjectType, ...]]:
        """主体名与主体类型。读到结构就直接用，读不到才问模型。"""
        if structure.subject_types:
            return structure.subject_name, structure.subject_types
        return self._ask_subject()

    def nature(self, chunk: Chunk) -> tuple[ContentNature, ...]:
        """内容性质。标题认得出来就归一，认不出才问模型。"""
        matched = natures_for_heading(chunk.ancestor_path)
        if matched:
            return matched
        if chunk.ancestor_path and chunk.ancestor_path == self.title:
            # 路径只有文档大标题这一段：这是词条页的开篇，答的正是「是什么」
            return (ContentNature.INTRO,)
        return self._ask_nature(chunk)

    def _ask_subject(self) -> tuple[str, tuple[SubjectType, ...]]:
        if self.llm is None:
            logger.warning("读不出主体结构，也没有可用的模型客户端：主体标签留空")
            return "", ()
        sample = self.chunks[:SUBJECT_SAMPLE_CHUNKS]
        try:
            guess = self.llm.complete_structured(self._subject_request(sample), _SubjectGuess)
        except LlmError as exc:
            logger.warning("主体信息打标失败，标签留空：%s", exc)
            return "", ()
        return guess.subject_name.strip(), self._enabled(guess.subject_types)

    def _ask_nature(self, chunk: Chunk) -> tuple[ContentNature, ...]:
        if self.llm is None:
            return ()
        try:
            guess = self.llm.complete_structured(self._nature_request(chunk), _NatureGuess)
        except LlmError as exc:
            logger.warning("第 %d 个切片的内容性质打标失败，标签留空：%s", chunk.chunk_index, exc)
            return ()
        return tuple(dict.fromkeys(guess.content_nature))

    def _enabled(self, kinds: Sequence[SubjectType]) -> tuple[SubjectType, ...]:
        """只留这个库启用的主体类型，去重并保持原次序。"""
        kept = tuple(dict.fromkeys(kind for kind in kinds if self.vocabulary.enabled(kind)))
        dropped = [kind.value for kind in kinds if not self.vocabulary.enabled(kind)]
        if dropped:
            logger.info("模型给出了这个库没启用的主体类型，已丢掉：%s", "、".join(dropped))
        return kept

    def _subject_request(self, sample: Sequence[Chunk]) -> LlmRequest:
        return LlmRequest(
            messages=[
                Message(
                    "system",
                    "你在给游戏资料打标。读下面这份文档开头的几个切片，判断它讲的是哪个"
                    f"主体、属于哪几类主体类型。可选的主体类型只有这些：{_subject_options(self.vocabulary)}。"
                    "主体类型可以多选——「二郎神的技能」同时属于角色与技能。判断不出的字段留空。",
                ),
                Message("user", "\n\n".join(self._sample_text(chunk) for chunk in sample)),
            ]
        )

    def _nature_request(self, chunk: Chunk) -> LlmRequest:
        return LlmRequest(
            messages=[
                Message(
                    "system",
                    "你在判断一段游戏资料回答的是哪一类问题。可选的内容性质只有这些："
                    f"{_nature_options()}。可以多选，判断不出就给空数组。",
                ),
                Message("user", self._sample_text(chunk)),
            ]
        )

    def _sample_text(self, chunk: Chunk) -> str:
        """一个切片连同它在文档中的位置。位置只是弱上下文，切片本身才是判据。"""
        note = f"第 {chunk.chunk_index + 1}/{len(self.chunks)} 个切片"
        if chunk.ancestor_path:
            note = f"{note}，标题路径：{chunk.ancestor_path}"
        return f"{note}\n{chunk.content}"


class _SubjectGuess(BaseModel):
    """无结构时问模型要的主体信息。字段描述会随 schema 一起进提示词。"""

    subject_name: str = Field(
        description="这条文档描述的主体在游戏里的具体名称，如「二郎神」；判断不出就留空串"
    )
    subject_types: list[SubjectType] = Field(
        description="主体所属的类别，可多选；判断不出就给空数组"
    )


class _NatureGuess(BaseModel):
    """无结构时问模型要的内容性质。"""

    content_nature: list[ContentNature] = Field(
        description="这段内容回答的是哪一类问题，可多选；判断不出就给空数组"
    )


def _subject_options(vocabulary: TagVocabulary) -> str:
    return "；".join(
        f"{kind.value}（{SUBJECT_TYPE_LABELS[kind]}）" for kind in vocabulary.subject_types
    )


def _nature_options() -> str:
    return "；".join(
        f"{nature.value}（{CONTENT_NATURE_LABELS[nature]}）" for nature in ContentNature
    )


def _classify(
    terms: Sequence[str], vocabulary: TagVocabulary
) -> tuple[tuple[SubjectType, ...], tuple[str, ...]]:
    """把读到的一串叫法归到主体类型，并挑出其中的游戏术语。"""
    subject_types: list[SubjectType] = []
    game_terms: list[str] = []
    for term in terms:
        kind = vocabulary.resolve(term)
        if kind is None or not vocabulary.enabled(kind):
            continue
        if kind not in subject_types:
            subject_types.append(kind)
        if vocabulary.is_game_term(term) and term not in game_terms:
            game_terms.append(term)
    return tuple(subject_types), tuple(game_terms)


def _categories(markdown: str) -> Iterator[str]:
    """扫出这条文档声明的分类。链接形式与裸写一行都认。"""
    for line in markdown.splitlines():
        yield from _CATEGORY_LINK.findall(line)
        bare = _CATEGORY_LINE.match(line)
        if bare is not None:
            yield bare.group(1)


def _template_blocks(markdown: str) -> Iterator[str]:
    """把 `{{…}}` 整段扫出来，嵌套按计数配对，一行写完的也算。"""
    depth = 0
    buffered: list[str] = []
    for line in markdown.splitlines():
        if depth == 0 and "{{" not in line:
            continue
        buffered.append(line)
        depth += line.count("{{") - line.count("}}")
        if depth <= 0:
            yield "\n".join(buffered)
            buffered = []
            depth = 0


def _fields(block: str) -> dict[str, str]:
    """把模板块里的 `| 键 = 值` 收成字典，键按 `_normalize` 对齐。同名取先出现的那个。"""
    found: dict[str, str] = {}
    for key, value in _TEMPLATE_FIELD.findall(block):
        found.setdefault(_normalize(key), value.strip())
    return found


def _normalize(term: str) -> str:
    """叫法的比对形式：去掉首尾空白，ASCII 折叠大小写。中文原样。"""
    return term.strip().casefold()


def _subject_type(value: Any) -> SubjectType:
    try:
        return SubjectType(value)
    except ValueError as exc:
        known = "、".join(kind.value for kind in SubjectType)
        raise ValueError(f"不认识的主体类型 {value!r}，可用的有：{known}") from exc
