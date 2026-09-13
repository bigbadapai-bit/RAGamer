"""打标：结构优先、模型兜底，两层标签各自独立降级。

与切分器一样，结构读取与标题归一都是纯函数、零假件，直接调；
只有「模型兜底」那一段接 `FakeLlm`——它一次网络都不发。
"""

from __future__ import annotations

import pytest

from ragamer.chunking import Chunk, ChunkRules, chunk_document
from ragamer.llm import FakeLlm, LlmError, LlmRequest
from ragamer.tagging import (
    SUBJECT_SAMPLE_CHUNKS,
    ContentNature,
    SubjectType,
    TagVocabulary,
    natures_for_heading,
    read_document_structure,
    tag_document,
)
from ragamer.wikitext import CATEGORY_NAMESPACES

RULES = ChunkRules(max_chars=200, min_chars=40, heading_density=0.02)

#: 一份 MediaWiki 风格的词条页：结构齐全——大标题、Infobox、分类。
WIKI_ARTICLE = """\
# 二郎神

{{信息框
| 名称 = 二郎神
| 类型 = BOSS
}}

二郎神是隐藏 BOSS，需要三阶段打完。

## 获取方式

在第三章的隐藏区域遇到。

## 属性

血量 12000，抗性偏高。

## 打法

先定身，再贴身输出。

[[Category:角色]]
[[Category:妖王]]
"""

#: 只有大标题、没有分类与 Infobox：主体结构读不出来，内容性质却够用。
HEADINGS_ONLY = """\
# 二郎神

二郎神是隐藏 BOSS。

## 获取方式

在第三章的隐藏区域遇到。
"""

#: 扁平文档：没有标题层级也没有分类，两条兜底路都得走。一段一片。
FLAT_PARAGRAPHS = (
    "二郎神是隐藏 BOSS，躲在第三章的隐藏区域最深处。",
    "血量很高，建议先练到 60 级再来，被打中会掉半管血。",
    "第一阶段的定身符很关键，看到抬手立刻往侧面翻滚。",
    "推荐配装是双刀加轻甲，打不过就先去做三尖两刃刀。",
)

FLAT_ARTICLE = "\n\n".join(FLAT_PARAGRAPHS)

#: 该游戏的术语映射：「妖王」「根器」是黑神话自己的叫法。
BLACK_MYTH = TagVocabulary(
    subject_types=tuple(SubjectType),
    term_mapping={"妖王": SubjectType.CHARACTER, "根器": SubjectType.ITEM},
)


def _chunks(markdown: str = WIKI_ARTICLE, rules: ChunkRules = RULES) -> list[Chunk]:
    return chunk_document(markdown, rules)


def _flat_chunks() -> list[Chunk]:
    """扁平文档的切片，一段一片，直接构造。

    打标这一层要验的是「每片各自判性质」，不该被切分器的合并策略带偏——
    真实文档能不能切出四片由切分器的测试管。
    """
    return [Chunk(text, index, "") for index, text in enumerate(FLAT_PARAGRAPHS)]


def _subject_reply(name: str, *kinds: SubjectType) -> dict:
    return {"subject_name": name, "subject_types": [kind.value for kind in kinds]}


def _nature_reply(*natures: ContentNature) -> dict:
    return {"content_nature": [nature.value for nature in natures]}


class _打不通的模型(FakeLlm):
    """每次调用都失败。用来验「打标失败不阻断入库」这条。

    用假件排一串 `LlmError` 脚本也行，但调用几次就得排几条，调用次数一变脚本就
    对不上了——这样写调用几次都一样。
    """

    def _reply(self, request: LlmRequest) -> str:
        self.calls.append(request)
        raise LlmError("模型服务连不上")


# --- 读语料自身的结构 ---


def test_分类与大标题读出主体名与主体类型():
    structure = read_document_structure(WIKI_ARTICLE, BLACK_MYTH)

    assert structure.subject_name == "二郎神"
    assert structure.subject_types == (SubjectType.CHARACTER,)


def test_术语映射把游戏本地叫法归到通用主体类型():
    structure = read_document_structure(WIKI_ARTICLE, BLACK_MYTH)

    # 「妖王」在黑神话里就是角色，靠术语映射归过去；原文另记进 game_terms
    assert "妖王" in structure.game_terms
    assert structure.game_terms == ("妖王",)


def test_没配映射的库认不出叫法_交给模型兜底():
    """自定义库不配映射就读不出结构，这是 §2.3 给它安排的降级，不是漏洞。"""
    vocabulary = TagVocabulary()

    assert read_document_structure("[[Category:物品]]", vocabulary).subject_types == ()
    assert vocabulary.resolve("物品") is None


def test_Infobox_的类型字段也算结构():
    markdown = "# 三尖两刃刀\n\n{{物品信息框\n| 名称 = 三尖两刃刀\n| 类型 = 装备\n}}\n"

    structure = read_document_structure(
        markdown, TagVocabulary(term_mapping={"装备": SubjectType.ITEM})
    )

    assert structure.subject_name == "三尖两刃刀"
    assert structure.subject_types == (SubjectType.ITEM,)


def test_认不出的分类被丢掉而不是猜():
    structure = read_document_structure("[[Category:需要整理的页面]]", TagVocabulary())

    assert structure.subject_types == ()


def test_转换器认得的每一种分类写法这里都读得出来():
    """两份清单会漂，漂掉的那一边不报错：转换器把 `[[分類:角色]]` 原样留着，
    打标器却不认它，繁体站的词条页标签就悄无声息地空了。
    """
    vocabulary = TagVocabulary(term_mapping={"角色": SubjectType.CHARACTER})

    for namespace in CATEGORY_NAMESPACES:
        structure = read_document_structure(f"[[{namespace}:角色]]", vocabulary)
        assert structure.subject_types == (SubjectType.CHARACTER,), namespace


def test_归不出来时连它的名称字段也不当主体名():
    """`{{cite|type=web|title=…}}` 里的 web 不是主体类型，那个 title 也就不是主体名。"""
    markdown = "{{cite\n| type = web\n| title = 某个网页\n}}\n"

    structure = read_document_structure(markdown, TagVocabulary())

    assert structure.subject_name == ""


def test_没启用的主体类型不落标():
    only_skill = TagVocabulary(subject_types=(SubjectType.SKILL,))

    structure = read_document_structure(WIKI_ARTICLE, only_skill)

    assert structure.subject_types == ()
    assert structure.game_terms == ()


def test_扁平文档读不出主体结构():
    structure = read_document_structure(FLAT_ARTICLE, BLACK_MYTH)

    assert structure.subject_types == ()
    assert structure.subject_name == ""


# --- 按标题归一内容性质 ---


def test_按标题归一内容性质():
    assert natures_for_heading("二郎神 › 获取方式") == (ContentNature.WHERE,)
    assert natures_for_heading("二郎神 › 属性") == (ContentNature.STATS,)
    assert natures_for_heading("二郎神 › 打法") == (ContentNature.GUIDE,)


def test_末段没有信息量时落到祖先路径的上一段():
    assert natures_for_heading("二郎神 › 打法 › 第二阶段") == (ContentNature.GUIDE,)


def test_一段标题里命中几类就留几类():
    assert natures_for_heading("掉落与属性") == (ContentNature.WHERE, ContentNature.STATS)


def test_认不出的标题没有性质():
    assert natures_for_heading("二郎神 › 琐事") == ()


# --- 打标 ---


def test_语料有结构时不调用模型():
    llm = FakeLlm()  # 一条脚本都没排：真调了就会炸

    tagged = tag_document(WIKI_ARTICLE, _chunks(), vocabulary=BLACK_MYTH, llm=llm)

    assert llm.calls == []
    assert tagged  # 确实打了标，不是空跑


def test_主体名与主体类型回写到该文档的每一个切片():
    tagged = tag_document(WIKI_ARTICLE, _chunks(), vocabulary=BLACK_MYTH)

    assert [item.subject_name for item in tagged] == ["二郎神"] * len(tagged)
    assert all(item.subject_type == (SubjectType.CHARACTER,) for item in tagged)


def test_内容性质逐切片判定_同一文档里两种性质并存():
    tagged = tag_document(WIKI_ARTICLE, _chunks(), vocabulary=BLACK_MYTH)

    assert [item.content_nature for item in tagged] == [
        (ContentNature.INTRO,),  # Infobox 整块一片，路径只有大标题
        (ContentNature.INTRO,),  # 大标题下的引子正文，路径同样只有大标题
        (ContentNature.WHERE,),
        (ContentNature.STATS,),
        (ContentNature.GUIDE,),
    ]


def test_整篇只有一个大标题时逐切片问模型而不是一律盖成介绍():
    """ADR-0003 的红线：同一文档里「数值」和「打法」并存，不许文档级统一。

    整篇没有小节时每个切片的路径都等于文档大标题，那不是「开篇」，是一整篇没有结构
    可依的正文——盖成 `intro` 就正好踩中「读完前几块、回写全文档」那条禁令。
    """
    body = "\n\n".join(
        (
            "血量 12000，抗性偏高，打之前先把等级练够，不然第一阶段就会被秒。" * 4,
            "先定身再贴身输出，横扫之后有硬直，看到收刀就往侧面翻滚。" * 4,
        )
    )
    markdown = f"# 二郎神\n\n{body}\n"
    chunks = _chunks(markdown)

    assert len(chunks) > 1  # 确实切出了多片，这条才验得到东西
    assert {chunk.ancestor_path for chunk in chunks} == {"二郎神"}  # 全挂在大标题下

    llm = FakeLlm(
        _subject_reply("二郎神", SubjectType.CHARACTER),  # 没有分类，主体类型得问
        *[_nature_reply(ContentNature.STATS)] * len(chunks),
    )

    tagged = tag_document(markdown, chunks, llm=llm)

    # 每片各问一次。少一次就说明有切片被那条捷径盖成了 intro
    assert len(llm.calls) == 1 + len(chunks)
    assert not any(ContentNature.INTRO in item.content_nature for item in tagged)


def test_切片顺序与正文原样保留():
    chunks = _chunks()

    tagged = tag_document(WIKI_ARTICLE, chunks, vocabulary=BLACK_MYTH)

    assert [item.chunk.chunk_index for item in tagged] == list(range(len(chunks)))
    assert [item.chunk.content for item in tagged] == [chunk.content for chunk in chunks]


def test_没有切片时什么都不做():
    llm = FakeLlm()

    # 扁平文档也一样：没有切片就没有标签要打，不该为它白调一次模型
    assert tag_document(FLAT_ARTICLE, [], llm=llm) == []
    assert llm.calls == []


# --- 无结构时的模型兜底 ---


def test_无结构时模型读开头若干切片总结主体():
    chunks = _flat_chunks()
    llm = FakeLlm(
        _subject_reply("二郎神", SubjectType.CHARACTER),
        *[_nature_reply(ContentNature.GUIDE)] * len(chunks),
    )

    tag_document(FLAT_ARTICLE, chunks, llm=llm)

    # 一次拿回主体名与主体类型，不是分两次问
    prompt = llm.calls[0].messages[-1].content
    assert "二郎神是隐藏 BOSS" in prompt
    # 带上切片位置作弱上下文
    assert f"第 1/{len(chunks)} 个切片" in prompt


def test_无结构时喂给模型的是文档开头那几个切片():
    # 每段都长过半个上限，相邻两段就并不进同一片——九段正好九片
    many = "\n\n".join(
        f"第 {index} 段正文，扯得够长好让每一段各成一片，不至于被并到邻居那里去。" * 4
        for index in range(9)
    )
    chunks = _chunks(many)
    llm = FakeLlm(
        _subject_reply("二郎神", SubjectType.CHARACTER),
        *[_nature_reply(ContentNature.INTRO)] * len(chunks),
    )

    tag_document(many, chunks, llm=llm)

    prompt = llm.calls[0].messages[-1].content
    assert len(chunks) > SUBJECT_SAMPLE_CHUNKS  # 切片够多，截断才看得出来
    assert prompt.count("个切片") == SUBJECT_SAMPLE_CHUNKS
    assert f"第 {len(chunks)}/" not in prompt


def test_无结构时逐切片问模型并带上切片位置():
    chunks = _flat_chunks()
    llm = FakeLlm(
        _subject_reply("二郎神", SubjectType.CHARACTER),
        *[_nature_reply(ContentNature.INTRO)] * len(chunks),
    )

    tagged = tag_document(FLAT_ARTICLE, chunks, llm=llm)

    # 一次问主体，之后每个切片各问一次
    assert len(llm.calls) == 1 + len(chunks)
    assert all(item.subject_name == "二郎神" for item in tagged)
    assert "标题路径" not in llm.calls[1].messages[-1].content  # 扁平文档没有路径


def test_模型给出的没启用的主体类型被丢掉():
    only_skill = TagVocabulary(subject_types=(SubjectType.SKILL,))
    chunks = _flat_chunks()
    llm = FakeLlm(
        _subject_reply("二郎神", SubjectType.CHARACTER, SubjectType.SKILL),
        *[_nature_reply(ContentNature.INTRO)] * len(chunks),
    )

    tagged = tag_document(FLAT_ARTICLE, chunks, vocabulary=only_skill, llm=llm)

    assert tagged[0].subject_name == "二郎神"  # 名留下了
    assert tagged[0].subject_type == (SubjectType.SKILL,)  # 没启用的类没留下


def test_两个字段都能装多个值():
    """「二郎神的技能」同时属于角色与技能，一段内容也可以既是数值又是获取。"""
    markdown = "# 二郎神\n\n[[Category:角色]]\n[[Category:技能]]\n\n## 琐事\n\n不知道该算哪一类。\n"
    vocabulary = TagVocabulary(
        term_mapping={"角色": SubjectType.CHARACTER, "技能": SubjectType.SKILL}
    )
    llm = FakeLlm(_nature_reply(ContentNature.STATS, ContentNature.WHERE))

    tagged = tag_document(markdown, _chunks(markdown), vocabulary=vocabulary, llm=llm)

    assert tagged[0].subject_type == (SubjectType.CHARACTER, SubjectType.SKILL)
    # 开篇按结构归成介绍；问模型的那一片才拿回两个性质
    assert tagged[0].content_nature == (ContentNature.INTRO,)
    assert tagged[1].content_nature == (ContentNature.STATS, ContentNature.WHERE)


# --- 失败不阻断入库 ---


def test_打标失败不阻断_标签留空但正文还在():
    chunks = _flat_chunks()

    tagged = tag_document(FLAT_ARTICLE, chunks, llm=_打不通的模型())

    assert len(tagged) == len(chunks)
    assert [item.chunk.content for item in tagged] == [chunk.content for chunk in chunks]
    assert all(item.subject_name == "" for item in tagged)
    assert all(item.subject_type == () for item in tagged)
    assert all(item.content_nature == () for item in tagged)


def test_没给模型客户端时结构读不出的标签留空():
    tagged = tag_document(FLAT_ARTICLE, _flat_chunks())

    assert all(item.content_nature == () for item in tagged)
    assert tagged[0].chunk.content  # 正文照常返回


def test_标题读出的主体名不因类型没读到就被扔掉():
    """逐字段降级：类型问模型失败了，结构读到手的名字照留。"""
    tagged = tag_document(HEADINGS_ONLY, _chunks(HEADINGS_ONLY), llm=_打不通的模型())

    assert all(item.subject_name == "二郎神" for item in tagged)
    assert all(item.subject_type == () for item in tagged)
    assert [item.content_nature for item in tagged] == [
        (ContentNature.INTRO,),  # 开篇
        (ContentNature.WHERE,),  # 靠标题归一，没碰模型
    ]


def test_只有分类没有大标题时主体名问模型补上():
    """结构给了类型、给不出名字，缺的那一个字段单独问回来。"""
    markdown = "[[Category:妖王]]\n\n二郎神是隐藏 BOSS。\n"
    llm = FakeLlm(
        _subject_reply("二郎神", SubjectType.CHARACTER),
        _nature_reply(ContentNature.INTRO),  # 没有标题可归一，性质那一问也落回模型
    )

    tagged = tag_document(markdown, _chunks(markdown), vocabulary=BLACK_MYTH, llm=llm)

    assert tagged[0].subject_name == "二郎神"
    assert tagged[0].subject_type == (SubjectType.CHARACTER,)  # 结构给的，没被模型覆盖


# --- 词表从配置读入 ---


def test_自定义库不配映射时默认启用全部主体类型():
    vocabulary = TagVocabulary.from_mapping({})

    assert vocabulary.subject_types == tuple(SubjectType)
    assert vocabulary.term_mapping == {}


def test_术语映射从配置读入():
    vocabulary = TagVocabulary.from_mapping(
        {"subject_types": ["character", "skill"], "term_mapping": {"妖王": "character"}}
    )

    assert vocabulary.subject_types == (SubjectType.CHARACTER, SubjectType.SKILL)
    assert vocabulary.resolve("妖王") is SubjectType.CHARACTER
    assert vocabulary.resolve("没配过的叫法") is None
    assert vocabulary.is_game_term("妖王")


def test_配置里写了不认识的主体类型要报错():
    with pytest.raises(ValueError, match="不认识的主体类型"):
        TagVocabulary.from_mapping({"subject_types": ["hero"]})


def test_一个主体类型都不启用要报错():
    with pytest.raises(ValueError, match="至少要启用一个主体类型"):
        TagVocabulary(subject_types=())
