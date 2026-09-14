"""答案正文的排版：认一个**很小的 Markdown 子集**，渲染成 HTML。

页面原先是一段纯文本，模型写的 `**加粗**`、`- ` 列表、`#` 标题全原样露出来——
一段答案里十几个星号，比不排版还乱（实测一条答案里 `**` 出现 10 处）。

**不引第三方库**：要认的就那么几种，而通用 Markdown 库带来的是一整套语法外加一个
必须自己保证的清洗环节。这里的做法是**先整段转义，再放行白名单**：`<`、`>`、`&`
在第一步就没了，输出里能当标签用的只有本模块自己插的那几种。模型输出按不可信对待
是有理由的——网络检索的内容会进提示词，能诱导它吐 HTML（`ragamer.websearch`）。

**认不出的记号原样留着**（表格、链接、行内代码、围栏代码块）：排版认一半比不认更糟，
留成原文至少看得出它想写什么。要加一种记号，就在这里加一条规则与一条测试。

**单个换行按换行渲染**（`<br>`），不按 Markdown 的软换行合成空格：中文里合成空格
会凭空多出一个空格，而模型写下的换行是有意的。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from markupsafe import Markup

#: 标题行：一至六个 `#` 加空格。
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

#: 列表项：缩进 + 记号 + 空格 + 正文。`-`／`*`／`+` 是无序，`1.`／`1)` 是有序。
#: 记号后面**必须有空格**：`**粗体**` 开头的行不能被当成列表项。
_ITEM = re.compile(r"^(?P<indent>[ \t]*)(?:(?P<num>\d+[.)])|[-*+])\s+(?P<text>.*)$")

#: 行内粗体。不跨行（`.+?` 不含换行），也不认斜体与行内代码——见模块说明。
_BOLD = re.compile(r"\*\*(.+?)\*\*")

#: `#` 换算成几级标题：页面上面还有页面标题与「回答」那一行，`#` 直接当 `h1` 太重。
_HEADING_OFFSET = 2


@dataclass
class _List:
    """一个打开的列表。`item_open` 说的是它最后那一项还开着没有。

    **项是开着写的**（`<li>` 与它要装的东西一起收尾），因为子列表要落在父项**里面**：
    立刻把 `<li>…</li>` 合上的话，下一层列表只能挂在父项外面。
    """

    indent: int
    tag: str
    item_open: bool = False


def to_html(text: str) -> Markup:
    """答案正文 → 一段可以直接放进模板的 HTML。"""
    out: list[str] = []
    paragraph: list[str] = []
    lists: list[_List] = []

    def close_item() -> None:
        if lists and lists[-1].item_open:
            out.append("</li>")
            lists[-1].item_open = False

    def close_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{'<br>'.join(paragraph)}</p>")
            paragraph.clear()

    def close_lists() -> None:
        while lists:
            close_item()
            out.append(f"</{lists.pop().tag}>")

    def open_item(indent: int, tag: str) -> None:
        """按缩进把列表调到这一项该在的那一层，然后开一条 `<li>`。"""
        # 比它深的层先收掉：每一层收两条 `</li>`——最深那条是这一项自己，
        # 另一条是装着这个子列表的那个父项
        while lists and indent < lists[-1].indent:
            close_item()
            out.append(f"</{lists.pop().tag}>")
            close_item()
        if lists and indent == lists[-1].indent:
            close_item()
        else:
            lists.append(_List(indent, tag))
            out.append(f"<{tag}>")
        out.append("<li>")
        lists[-1].item_open = True

    for line in html.escape(text).splitlines():
        if not line.strip():
            close_paragraph()
            close_lists()
            continue
        heading = _HEADING.match(line)
        if heading is not None:
            close_paragraph()
            close_lists()
            level = min(len(heading.group(1)) + _HEADING_OFFSET, 6)
            out.append(f"<h{level}>{_bold(heading.group(2).strip())}</h{level}>")
            continue
        item = _ITEM.match(line)
        if item is not None:
            close_paragraph()
            open_item(len(item.group("indent")), "ol" if item.group("num") else "ul")
            out.append(_bold(item.group("text").strip()))
            continue
        if lists and lists[-1].item_open:
            # 列表项自己换行：接着它往下写，不另起一段
            out.append(f"<br>{_bold(line.strip())}")
            continue
        paragraph.append(_bold(line.strip()))

    close_paragraph()
    close_lists()
    # 拼的时候不加分隔：块级标签之间不需要空白，加了反而把 `<li>` 与它的正文拆到两行去
    return Markup("".join(out))


def _bold(escaped: str) -> str:
    """行内粗体。**收的是转义过的文本**：放行的是标签，不是输入。"""
    return _BOLD.sub(r"<strong>\1</strong>", escaped)
