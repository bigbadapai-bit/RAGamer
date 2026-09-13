"""wikitext → Markdown：纯函数、零网络、零假件。

这一层的取舍全在「什么该转、什么该原样留着」上，所以断言也分两类：
转成 Markdown 的（标题、表格、列表、图片）逐字钉住；
留给下游的（`[[Category:…]]`、`{{模板块}}`）断言它**没被动过**——
它们被洗掉的后果不报错，只是打标器读不到主体类型、切分器认不出词条页。
"""

from __future__ import annotations

import pytest

from ragamer.wikitext import WikiPage, file_titles, normalize_file_name, to_markdown

#: 一张真页面的骨架。模板块、内链、分类、表格都在这，后面几处断言共用它。
ARTICLE = """\
{{Infobox character
| 名称 = 二郎神
| 类型 = 角色
}}
'''二郎神'''是《黑神话：悟空》中的[[妖王]]，出自[[黑神话：悟空|本作]]。

== 打法 ==
=== 第二阶段 ===
* 闪避
** 侧闪

{| class="wikitable"
|+ 属性
|-
! 名称 !! 数值
|-
| 血量 || 1000
|-
| style="color:red" | 攻击 || 200
|}

[[Category:角色]]
"""


def test_标题写成一级标题():
    """文档标题同时是重导时的替换键，它必须来自页面本身，不能靠文件名回落。"""
    assert to_markdown("正文", title="二郎神").markdown == "# 二郎神\n\n正文\n"


def test_不给标题时不凭空造一级标题():
    assert to_markdown("正文").markdown == "正文\n"


@pytest.mark.parametrize(
    ("wikitext", "expected"),
    [
        ("== 打法 ==", "## 打法"),
        ("=== 第二阶段 ===", "### 第二阶段"),
        ("====== 深 ======", "###### 深"),
        ("== 带 空格 ==", "## 带 空格"),
    ],
)
def test_等号层数原样映射成井号(wikitext, expected):
    assert to_markdown(wikitext).markdown == f"{expected}\n"


def test_加粗与斜体():
    assert to_markdown("'''粗'''与''斜''").markdown == "**粗**与*斜*\n"


def test_内链只留显示文字():
    """正文是给生成用的，`[[目标|显示]]` 留着会让答案里出现一层方括号。"""
    assert to_markdown("见[[二郎神]]与[[黑神话：悟空|本作]]").markdown == "见二郎神与本作\n"


def test_内链没有显示文字时拿目标当文字():
    assert to_markdown("见[[二郎神|]]").markdown == "见二郎神\n"


def test_分类原样留着():
    """打标器从 `[[Category:角色]]` 读主体类型——转掉它，标签会静默变空。"""
    assert "[[Category:角色]]" in to_markdown(ARTICLE).markdown


def test_繁体分类也原样留着():
    assert to_markdown("[[分類:角色]]").markdown == "[[分類:角色]]\n"


def test_模板块原样留着():
    """打标器读 Infobox 的 `| 类型 = 角色`；切分器`{{` 开头的块整块不切。"""
    markdown = to_markdown(ARTICLE).markdown

    assert "{{Infobox character\n| 名称 = 二郎神\n| 类型 = 角色\n}}" in markdown


def test_表格转成_GFM_表格():
    wikitext = """\
{| class="wikitable"
|-
! 名称 !! 数值
|-
| 血量 || 1000
|}
"""

    assert to_markdown(wikitext).markdown == ("| 名称 | 数值 |\n| --- | --- |\n| 血量 | 1000 |\n")


def test_单元格属性被去掉():
    wikitext = """\
{|
|-
! 名称 !! 数值
|-
| style="color:red" | 攻击 || 200
|}
"""

    assert "| 攻击 | 200 |" in to_markdown(wikitext).markdown


def test_没有表头行时补一行空的而不是把数据提上来当表头():
    """提上来那一行就重复了：表头和数据各出现一次。"""
    wikitext = """\
{|
|-
| 甲 || 乙
|}
"""

    assert to_markdown(wikitext).markdown == "|  |  |\n| --- | --- |\n| 甲 | 乙 |\n"


def test_行长短不齐时补空格():
    """GFM 表格必须是个矩形，少的那几格补空——丢了整行才是真的丢内容。"""
    wikitext = """\
{|
|-
! 甲 !! 乙 !! 丙
|-
| 1
|}
"""

    assert to_markdown(wikitext).markdown == (
        "| 甲 | 乙 | 丙 |\n| --- | --- | --- |\n| 1 |  |  |\n"
    )


def test_格里的竖线转义掉():
    """`{{lang|zh|二郎神}}` 这种模板在格子里很常见，不转义就会多切出一列。

    切分器认 `\\|`（`chunking._CELL_SEPARATOR`），列数因此不会错位。
    """
    wikitext = """\
{|
|-
! 甲 !! 乙
|-
| {{lang|zh|二郎神}} || 200
|}
"""

    assert "{{lang\\|zh\\|二郎神}}" in to_markdown(wikitext).markdown


def test_表格里的一格可以写成好几行():
    wikitext = """\
{|
|-
! 甲 !! 乙
|-
| 第一行
第二行 || 200
|}
"""

    assert "| 第一行 第二行 | 200 |" in to_markdown(wikitext).markdown


def test_表外的裸竖线不会被当成表格():
    assert to_markdown("甲 | 乙").markdown == "甲 | 乙\n"


def test_图片转成_Markdown_引用():
    page = to_markdown(
        "[[File:Erlang.jpg|thumb|200px|二郎神立绘]]",
        image_urls={"Erlang.jpg": "https://img/erlang.jpg"},
    )

    assert page.markdown == "![二郎神立绘](https://img/erlang.jpg)\n"
    assert page.images == ("https://img/erlang.jpg",)


def test_图片没有说明时拿文件名当替代文字():
    page = to_markdown(
        "[[File:Erlang.jpg|thumb]]", image_urls={"Erlang.jpg": "https://img/erlang.jpg"}
    )

    assert page.markdown == "![Erlang](https://img/erlang.jpg)\n"


def test_图片用_alt_参数当替代文字():
    page = to_markdown(
        "[[File:Erlang.jpg|thumb|alt=三眼]]", image_urls={"Erlang.jpg": "https://img/erlang.jpg"}
    )

    assert page.markdown == "![三眼](https://img/erlang.jpg)\n"


def test_查不到地址的图片原样留着():
    """编一条空地址的链接比留着原始写法更坏：前者看不出问题。"""
    page = to_markdown("[[File:Erlang.jpg|缩略图|说明]]")

    assert page.markdown == "[[File:Erlang.jpg|缩略图|说明]]\n"
    assert page.images == ()


def test_图片命名空间的大小写与别名都认():
    urls = {"Erlang.jpg": "https://img/erlang.jpg"}
    for syntax in ("[[file:Erlang.jpg]]", "[[Image:Erlang.jpg]]", "[[文件:Erlang.jpg]]"):
        assert to_markdown(syntax, image_urls=urls).images == ("https://img/erlang.jpg",)


def test_gallery_里的每一张图都算():
    wikitext = """\
<gallery>
Erlang.jpg|二郎神
Weapon.png
</gallery>
"""
    page = to_markdown(
        wikitext, image_urls={"Erlang.jpg": "https://img/a.jpg", "Weapon.png": "https://img/b.png"}
    )

    assert page.images == ("https://img/a.jpg", "https://img/b.png")


def test_引注整段丢掉():
    """留着它在正文里没有标记，读的人对不上号；脚注列表本身又不在 wikitext 里。"""
    assert to_markdown('甲<ref name="a"/>乙<ref>注</ref>丙').markdown == "甲乙丙\n"


def test_代码块转成围栏块_里面的引号不当粗体():
    wikitext = """\
<syntaxhighlight lang="lua">
local x = '''a'''
</syntaxhighlight>
"""

    assert to_markdown(wikitext).markdown == "```lua\nlocal x = '''a'''\n```\n"


def test_代码里有反引号时围栏加长():
    wikitext = """\
<syntaxhighlight lang="lua">
local s = ```a```
</syntaxhighlight>
"""

    assert "````lua" in to_markdown(wikitext).markdown


def test_nowiki_里的标记原样留着():
    """`<nowiki>` 的意思就是「别解释它」，屏蔽之后再放回去。"""
    assert to_markdown("<nowiki>'''不是粗体'''</nowiki>").markdown == "'''不是粗体'''\n"


def test_简繁转换标记取简体():
    assert to_markdown("-{zh-hans:简体;zh-hant:繁體;}-").markdown == "简体\n"


def test_没有变体的简繁标记去掉壳():
    assert to_markdown("-{正文}-").markdown == "正文\n"


def test_列表按层缩进():
    """Markdown 只用缩进表达层级：`**` 该落成 `  - `，不是 `-   - `。"""
    assert to_markdown("* 闪避\n** 侧闪\n* 翻滚").markdown == "- 闪避\n  - 侧闪\n- 翻滚\n"


def test_有序列表不落成标题():
    """MediaWiki 的 `#` 是有序列表，照抄过去就是 Markdown 的一级标题。"""
    assert to_markdown("# 第一步\n## 细分").markdown == "1. 第一步\n  1. 细分\n"


def test_定义列表退化成普通行():
    """`;` 与 `:` 在正文里分不开：中文 wiki 拿 `:` 缩进引文是常态，一律转成列表会凭空多出列表项。"""
    assert to_markdown("; 血量\n: 一千").markdown == "**血量**\n一千\n"


def test_重定向那一行不被当成有序列表():
    assert to_markdown("#REDIRECT [[二郎神]]").markdown == "REDIRECT 二郎神\n"


def test_分隔线转成三个短横():
    assert to_markdown("甲\n\n----\n\n乙").markdown == "甲\n\n---\n\n乙\n"


def test_外部链接转成_Markdown():
    assert (
        to_markdown("见 [https://example.com 官网]").markdown == "见 [官网](https://example.com)\n"
    )


def test_没有说明文字的外部链接用尖括号形式():
    assert to_markdown("见 [https://example.com]").markdown == "见 <https://example.com>\n"


def test_换行标签变成真的换行():
    assert to_markdown("甲<br/>乙").markdown == "甲\n乙\n"


def test_实体转义还原():
    assert to_markdown("甲&amp;乙&lt;丙").markdown == "甲&乙<丙\n"


def test_一篇词条该有的结构都还在():
    """整篇走一遍：这几样缺任何一样，后面的切分与打标都会走另一条路。"""
    markdown = to_markdown(ARTICLE, title="二郎神").markdown

    assert markdown.startswith("# 二郎神\n")
    assert "## 打法" in markdown and "### 第二阶段" in markdown
    assert "| 名称 | 数值 |" in markdown
    assert "**二郎神**是《黑神话：悟空》中的妖王，出自本作。" in markdown
    assert "{{Infobox character" in markdown
    assert "[[Category:角色]]" in markdown


def test_同一份输入转两次结果逐字相同():
    """结果要稳定：它后面是切片主键的一部分，漂一次就是一片查不回来的旧数据。"""
    assert to_markdown(ARTICLE, title="二郎神") == to_markdown(ARTICLE, title="二郎神")


def test_空输入不炸():
    assert to_markdown("").markdown == "\n"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Erlang.jpg", "Erlang.jpg"),
        ("file:foo_bar.JPG", "Foo bar.JPG"),
        ("File:Foo bar.JPG", "Foo bar.JPG"),
        ("文件:二郎神.png", "二郎神.png"),
        ("  二郎神.png  ", "二郎神.png"),
        ("", ""),
    ],
)
def test_文件名按_MediaWiki_的规则归一(raw, expected):
    assert normalize_file_name(raw) == expected


def test_挑出的图片名去重且保持出现顺序():
    wikitext = "[[File:B.jpg]] [[File:A.png|缩略图]] [[File:B.jpg|说明]]"

    assert file_titles(wikitext) == ("B.jpg", "A.png")


def test_挑图片名时把_gallery_也算上():
    assert file_titles("<gallery>\nB.jpg|说明\nA.png\n</gallery>") == ("B.jpg", "A.png")


def test_转出来的东西自带图片地址():
    page = to_markdown("[[File:A.png]]", image_urls={"A.png": "https://img/a.png"})

    assert isinstance(page, WikiPage)
    assert page.images == ("https://img/a.png",)
    assert page.markdown == "![A](https://img/a.png)\n"
