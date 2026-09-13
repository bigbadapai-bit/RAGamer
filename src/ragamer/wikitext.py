"""MediaWiki 的 wikitext → Markdown。

网页这条来源有两条路（见 `ragamer.crawl`），这一条走站点的开放接口，拿回来的是
**wikitext 原文**，不是渲染后的 HTML。选原文而不是渲染结果，是因为下游读的正是这些
标记：切分器靠 `[[内链]]`／`Category:`／`{{模板}}` 判断这是不是词条页
（docs/ARCHITECTURE.md §1.4），打标器靠 `[[Category:角色]]` 与 Infobox 的
`| 类型 = 角色` 读主体类型（`ragamer.tagging`）。先转 HTML 再转 Markdown 会把这两处
信号一起洗掉——而那两处**不会报错**，只会让词条页的标签悄悄变空。

于是这个转换器有一条明确的分界：**该像 Markdown 的转成 Markdown，该留给下游的原样留着**。

| wikitext | 转成 | 为什么 |
| --- | --- | --- |
| `== 打法 ==` | `## 打法` | 切分器按 `#` 认层级 |
| `{| … |}` 表格 | GFM 表格 | 切分器按 GFM 表格做原子化（§2.5） |
| `[[Category:角色]]` | 原样 | 打标器从它读主体类型 |
| `{{Infobox …}}` | 原样 | 打标器读字段；切分器当原子块 |
| `[[File:X.jpg|缩略图|说明]]` | `![说明](地址)` | 地址由调用方查好传进来 |
| `[[二郎神|他]]` | `他` | 正文要能读 |

**几处已知的取舍**（都是数据到齐前不拍阈值的那一类，与 §2.5 末尾同一个态度）：

- **`;` 定义列表与 `:` 缩进都退化成普通行**。两者在正文里分不开——中文 wiki 拿 `:` 缩进
  引文和注释是常态，一律转成列表会凭空多出一堆列表项，而切分器看得见列表。
- **`{{模板}}` 一律原样留着**，包括 `{{cite}}` 这类纯引用模板。按名字挑该留哪些是猜，
  等真实语料里看到噪音再定规则。
- **`<ref>` 整段丢掉**：留着它在正文里没有标记，读的人对不上号；脚注列表本身又不在
  wikitext 里。
- **`colspan`／`rowspan` 丢掉**，行按最大列数补空——GFM 表达不了合并格。补空而不是
  丢掉整行，是因为格里的内容还在。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import unescape

#: 图片命名空间的别名。中文站写「文件」，繁体站写「檔案」，都要认。
_IMAGE_NAMESPACES = ("File", "Image", "文件", "图像", "圖片", "档案", "檔案")
#: 分类命名空间。**输出必须原样留着**——打标器从它读主体类型。
_CATEGORY_NAMESPACES = ("Category", "分类", "分類")

_IMAGE_NAMESPACES_LOWER = {name.lower() for name in _IMAGE_NAMESPACES}
_CATEGORY_NAMESPACES_LOWER = {name.lower() for name in _CATEGORY_NAMESPACES}

#: 占位符：`\\x00序号\\x00`。正文里不会出现空字符，拿它当掩码是安全的。
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# 自闭合的写在前面：`<ref name="a" />` 不能被长的那条吞成「从它到下一个 </ref>」
_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_GALLERY = re.compile(r"<gallery\b[^>]*>(.*?)</gallery>", re.DOTALL | re.IGNORECASE)
_CODE = re.compile(
    r"<(?:syntaxhighlight|source)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:syntaxhighlight|source)>",
    re.DOTALL | re.IGNORECASE,
)
_LANG_ATTR = re.compile(r'\blang\s*=\s*"?([\w+#-]+)"?')
_BACKTICKS = re.compile(r"`+")
_NOWIKI = re.compile(r"<nowiki\b[^>]*>(.*?)</nowiki>", re.DOTALL | re.IGNORECASE)
_MATH = re.compile(r"<math\b[^>]*>(.*?)</math>", re.DOTALL | re.IGNORECASE)
_STRIKE = re.compile(
    r"<(?:s|strike|del)\b[^>]*>(.*?)</(?:s|strike|del)>", re.DOTALL | re.IGNORECASE
)
_TAG = re.compile(r"</?(?P<name>[a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")
_MAGIC_WORD = re.compile(r"__[A-Z]+__")
#: 简繁转换标记 `-{zh-hans:简;zh-hant:繁;}-`。中文 wiki 上遍布正文。
_LANG_CONV = re.compile(r"-\{(.*?)\}-", re.DOTALL)

_HEADING = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$")
_HRULE = re.compile(r"^-{4,}\s*$")
_LIST = re.compile(r"^([*#;:]+)\s*(.*)$")
_REDIRECT = re.compile(r"^(?:REDIRECT|重定向)\b", re.IGNORECASE)

_BOLD_ITALIC = re.compile(r"'''''(.*?)'''''", re.DOTALL)
_BOLD = re.compile(r"'''(.*?)'''", re.DOTALL)
_ITALIC = re.compile(r"''(.*?)''", re.DOTALL)
_EXTERNAL_LINK = re.compile(r"\[(https?://\S+?)(?:\s+([^\]]+))?\]")

_FILE_LINK = re.compile(
    r"\[\[\s*(?:" + "|".join(_IMAGE_NAMESPACES) + r")\s*:\s*([^\]|]+)",
    re.IGNORECASE,
)

#: 图片参数里不是说明文字的那些：显示方式、对齐、尺寸、`alt=` 之类。
_IMAGE_OPTION = re.compile(
    r"^(?:thumb(?:nail)?|frame(?:less)?|border|right|left|centre?r?|none|baseline|top|"
    r"middle|bottom|text-top|text-bottom|sub|super|upright|\d*x?\d*px|"
    r"(?:alt|link|lang|class|page|upright)=.*)$",
    re.IGNORECASE,
)

#: 单元格属性：`style="…" | 内容`、`colspan="2" | 内容`。
#: 要求属性段里有 `=`——没有它，「`| 甲 | 乙`」这种普通两格会被当成属性切掉前半截。
_CELL_ATTRIBUTE = re.compile(r"^\s*[^|\[\]{}]*=[^|\[\]{}]*\|")


@dataclass(frozen=True)
class WikiPage:
    """一份 wikitext 转出来的东西。"""

    markdown: str
    #: 正文里引用到的图片地址，按出现顺序、去重前原样。
    #: 口径与 :class:`ragamer.sources.NormalizedDoc.images` 一致，直接就能交过去。
    images: tuple[str, ...] = ()


def normalize_file_name(name: str) -> str:
    """文件名的比较用形态：去掉命名空间、下划线当空格、首字母大写。

    MediaWiki 自己就是这么归一的（`file:foo_bar.JPG` 与 `File:Foo bar.JPG` 是同一张图）。
    两处不按同一个规则比，查回来的地址就对不上——而那种错法不报错，图片静默变成纯文本。
    中文名首字母大写是空操作，这里是给英文文件名准备的。
    """
    prefix, separator, bare = name.partition(":")
    if separator and prefix.strip().lower() in _IMAGE_NAMESPACES_LOWER:
        name = bare
    cleaned = " ".join(name.replace("_", " ").split())
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def file_titles(wikitext: str) -> tuple[str, ...]:
    """正文引用到的图片名，按出现顺序去重。

    抓取层拿它去问接口要地址（`prop=imageinfo`），再喂回 :func:`to_markdown` 的
    `image_urls`。**先要名字再要地址，是因为地址只有接口知道**：wikitext 里写的是
    文件名，真实 URL 在服务端算。
    """
    names: list[str] = []
    for match in _FILE_LINK.finditer(wikitext):
        _remember(names, normalize_file_name(match.group(1)))
    for match in _GALLERY.finditer(wikitext):
        for entry in match.group(1).splitlines():
            entry = entry.strip()
            if entry and not entry.startswith("#"):
                _remember(names, normalize_file_name(_split_top(entry)[0]))
    return tuple(names)


def to_markdown(
    wikitext: str,
    *,
    title: str = "",
    image_urls: Mapping[str, str] | None = None,
) -> WikiPage:
    """wikitext → Markdown。

    :param title: 页面标题。给了就写成正文的一级标题——文档标题同时是重导时的替换键，
        它必须来自页面本身，不能靠文件名回落（见 `ragamer.importing.document_title`）。
    :param image_urls: 图片名（:func:`normalize_file_name` 的形态）→ 地址。
        查不到的那张图**原样留着**，不编一条空地址的链接。
    """
    urls = image_urls or {}
    images: list[str] = []
    shield = _Shield()
    lines = _strip(wikitext, shield).splitlines()

    out: list[str] = [f"# {title}", ""] if title else []
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith("{|"):
            table, index = _table(lines, index, images=images, image_urls=urls)
            if table:
                # 表格两侧各留一个空行：紧贴着上一段时，`|` 会被当成上一段的一部分
                out.extend(["", *table, ""])
            continue
        out.append(_line(lines[index], images=images, image_urls=urls))
        index += 1
    return WikiPage(markdown=shield.restore(_tidy(out)), images=tuple(images))


def _remember(names: list[str], name: str) -> None:
    if name and name not in names:
        names.append(name)


class _Shield:
    """把一段文本换成占位符，整体转换完再放回去。

    `<nowiki>` 与代码块里的内容是**字面量**：不换掉，里面的 `'''` 会被当成粗体、
    行首的 `{|` 会被当成表格开头——把一段代码读成一张表，后半篇就全乱了。
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def hide(self, text: str) -> str:
        self._parts.append(text)
        return f"\x00{len(self._parts) - 1}\x00"

    def restore(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda match: self._parts[int(match.group(1))], text)


def _strip(wikitext: str, shield: _Shield) -> str:
    """先去掉那些「不是正文结构」的东西，剩下的逐行转换。"""
    text = _COMMENT.sub("", wikitext)
    text = _REF.sub("", text)
    text = _GALLERY.sub(_gallery_lines, text)
    text = _CODE.sub(lambda match: _fenced(match, shield), text)
    text = _NOWIKI.sub(lambda match: shield.hide(match.group(1)), text)
    text = _MATH.sub(lambda match: shield.hide(f" ${match.group(1).strip()}$ "), text)
    text = _STRIKE.sub(r"~~\1~~", text)
    text = _TAG.sub(_tag, text)
    text = _MAGIC_WORD.sub("", text)
    return _LANG_CONV.sub(_variant, text)


def _gallery_lines(match: re.Match[str]) -> str:
    """`<gallery>` 里的每一行都是一张图，转成普通的内链交给后面的图片处理。"""
    lines = []
    for entry in match.group(1).splitlines():
        entry = entry.strip()
        if entry and not entry.startswith("#"):
            lines.append(f"[[File:{entry}]]")
    return "\n".join(lines)


def _fenced(match: re.Match[str], shield: _Shield) -> str:
    """代码块转成围栏块。围栏取比代码里最长的一段反引号还长一档。"""
    lang = _LANG_ATTR.search(match.group("attrs"))
    code = match.group("body").strip("\n")
    fence = "`" * max(3, max((len(run) for run in _BACKTICKS.findall(code)), default=0) + 1)
    return f"\n{fence}{lang.group(1) if lang else ''}\n{shield.hide(code)}\n{fence}\n"


def _tag(match: re.Match[str]) -> str:
    """剩下那些 HTML 标签：`<br>` 是换行，其余连标签一起丢掉。"""
    return "\n" if match.group("name").lower() == "br" else ""


def _variant(match: re.Match[str]) -> str:
    """`-{zh-hans:简;zh-hant:繁;}-` 取一个变体。

    优先简体：本项目面向中文用户，而简繁转换标记在中文 wiki 的正文里到处都是，
    整段留着会让同一个意思在切片里出现三遍。
    """
    body = match.group(1)
    pairs = [part.partition(":") for part in body.split(";")]
    if not any(separator for _, separator, _ in pairs):
        return body
    for code in ("zh-hans", "zh-cn", "zh-sg", "zh-my", "zh"):
        for key, separator, value in pairs:
            if separator and key.strip().lower() == code:
                return value.strip()
    return next((value for _, separator, value in pairs if separator), body)


def _segments(text: str) -> list[tuple[bool, str]]:
    """把一行拆成「内链／模板块」与「其余文字」两种段。

    先拆段再各自处理，是因为两边的规则不一样：模板块要原样留着，而正文里的
    `''` 与 `[http://…]` 要转成 Markdown。混在一起处理，模板参数里的 `''` 会被当粗体。
    """
    segments: list[tuple[bool, str]] = []
    index = 0
    plain_start = 0
    while index < len(text):
        for opener, closer in (("[[", "]]"), ("{{", "}}")):
            if not text.startswith(opener, index):
                continue
            if index > plain_start:
                segments.append((False, text[plain_start:index]))
            end = _closing(text, index, opener, closer)
            if end < 0:
                # 没配平：把剩下的原样交出去，不做猜测
                segments.append((True, text[index:]))
                return segments
            segments.append((True, text[index : end + 2]))
            index = end + 2
            plain_start = index
            break
        else:
            index += 1
    if plain_start < len(text):
        segments.append((False, text[plain_start:]))
    return segments


def _closing(text: str, start: int, opener: str, closer: str) -> int:
    """从 `start` 处的开括号配对找收尾，返回收尾符的下标；没配平返回 `-1`。"""
    depth = 0
    index = start
    while index < len(text):
        if text.startswith(opener, index):
            depth += 1
            index += 2
            continue
        if text.startswith(closer, index):
            depth -= 1
            if depth == 0:
                return index
            index += 2
            continue
        index += 1
    return -1


def _inline(text: str, *, images: list[str], image_urls: Mapping[str, str]) -> str:
    """一行里的行内标记。"""
    parts: list[str] = []
    for protected, segment in _segments(text):
        if not protected:
            parts.append(_plain(segment))
        elif segment.startswith("[["):
            parts.append(_link(segment[2:-2], images=images, image_urls=image_urls))
        else:
            parts.append(segment)
    return "".join(parts)


def _plain(text: str) -> str:
    """没有内链与模板的那部分。"""
    text = _BOLD_ITALIC.sub(r"***\1***", text)
    text = _BOLD.sub(r"**\1**", text)
    text = _ITALIC.sub(r"*\1*", text)
    text = _EXTERNAL_LINK.sub(_external, text)
    return unescape(text)


def _external(match: re.Match[str]) -> str:
    url, label = match.group(1), match.group(2)
    return f"[{label.strip()}]({url})" if label else f"<{url}>"


def _link(inner: str, *, images: list[str], image_urls: Mapping[str, str]) -> str:
    """`[[…]]` 里的内容。图片、分类、普通内链三种走向。"""
    parts = _split_top(inner)
    target = parts[0]
    prefix, separator, _ = target.partition(":")
    namespace = prefix.strip().lower() if separator else ""
    if namespace in _CATEGORY_NAMESPACES_LOWER:
        return f"[[{inner}]]"
    if namespace in _IMAGE_NAMESPACES_LOWER:
        return _image(target, parts[1:], images=images, image_urls=image_urls)
    # `[[目标|显示]]`：显示文字要能读；`[[目标]]` 直接拿目标当文字。
    # 目标带 `#小节` 时整段留着——那一段本身就是读者要找的锚点
    return parts[1].strip() if len(parts) > 1 and parts[1].strip() else target.strip()


def _image(
    target: str,
    params: Sequence[str],
    *,
    images: list[str],
    image_urls: Mapping[str, str],
) -> str:
    """`[[File:X.jpg|缩略图|说明]]` → `![说明](地址)`。

    地址查不到时**原样留着**：编一条空地址的链接比留着原始写法更坏——前者看不出问题，
    后者一眼就知道这张图没取到。
    """
    name = target.partition(":")[2]
    url = image_urls.get(normalize_file_name(name), "")
    if not url:
        return f"[[{target}{''.join(f'|{param}' for param in params)}]]"
    caption = ""
    alternative = ""
    for param in params:
        text = param.strip()
        if text.lower().startswith("alt="):
            alternative = text[4:].strip()
        elif text and not _IMAGE_OPTION.match(text):
            caption = text
    images.append(url)
    # 说明文字优先于 `alt=`：前者是图上看得见的那行字，检索时它比 alt 更像正文
    alt = caption or alternative or name.rsplit(".", 1)[0]
    return f"![{_inline(alt, images=images, image_urls=image_urls)}]({url})"


def _line(line: str, *, images: list[str], image_urls: Mapping[str, str]) -> str:
    """一行（表格之外）。"""
    heading = _HEADING.match(line)
    if heading is not None:
        title = _inline(heading.group(2), images=images, image_urls=image_urls).strip()
        return f"{'#' * len(heading.group(1))} {title}"
    if _HRULE.match(line):
        # 夹在空行里：`----` 紧贴上一段时是 setext 标题，不是分隔线
        return "\n---\n"
    listed = _LIST.match(line)
    if listed is not None:
        return _list_item(listed.group(1), listed.group(2), images=images, image_urls=image_urls)
    return _inline(line, images=images, image_urls=image_urls)


def _list_item(
    markers: str, content: str, *, images: list[str], image_urls: Mapping[str, str]
) -> str:
    """`*` 与 `#` 转成 Markdown 列表；`;` 与 `:` 只去掉记号（见模块开头那两条取舍）。"""
    if markers.strip("#") == "" and _REDIRECT.match(content.strip()):
        # `#REDIRECT [[目标]]`：接口带 `redirects=1` 时不会有它，留着是防没带上那一趟
        return _inline(content, images=images, image_urls=image_urls)
    text = _inline(content, images=images, image_urls=image_urls)
    if markers.strip(";") == "":
        return f"**{text}**" if text else ""
    if markers.strip(":") == "":
        return text
    # Markdown 只用缩进表达层级，记号不重复：MediaWiki 的 `**` 该落成 `  - `，
    # 不是 `-   - `——后者会被读成「一个里面写着 '- 侧闪' 的列表项」
    prefix = "  " * (len(markers) - 1) + ("1. " if markers[-1] == "#" else "- ")
    return f"{prefix}{text}"


def _table(
    lines: Sequence[str],
    start: int,
    *,
    images: list[str],
    image_urls: Mapping[str, str],
) -> tuple[list[str], int]:
    """从 `{|` 起到配平的 `|}` 止，返回渲染好的 Markdown 表格行与下一个下标。

    嵌套表（`{|` 里面还有 `{|`）很少见，做法是把内层的标记丢掉、内容并进当前格：
    整块丢掉会缺一截正文，而按两层结构还原要多写一个递归——数据到齐前不值得。
    """
    caption = ""
    header: list[str] = []
    rows: list[list[str]] = []
    current: list[str] | None = None
    current_is_header = False
    depth = 0
    index = start

    def flush() -> None:
        nonlocal current, header, current_is_header
        if current:
            if current_is_header and not header:
                header = current
            else:
                rows.append(current)
        current = None
        current_is_header = False

    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith("{|"):
            depth += 1
            continue
        if line.startswith("|}"):
            depth -= 1
            if depth == 0:
                break
            continue
        if depth > 1:
            if current and line and not line.startswith(("{", "|")):
                current[-1] = f"{current[-1]} {line}".strip()
            continue
        if line.startswith("|+"):
            caption = line[2:].strip()
        elif line.startswith("|-"):
            flush()
        elif line.startswith("!"):
            if current is None:
                current, current_is_header = [], True
            current += _cells(line[1:], "!!")
        elif line.startswith("|"):
            if current is None:
                current = []
            current += _cells(line[1:], "||")
        elif current and line:
            # 格的续行：MediaWiki 允许一格写成好几行。续行里还能再起新格
            # （`第二行 || 200`），按本行的分隔符接着切，不整行并进上一格
            parts = _split_top(line, "!!" if current_is_header else "||")
            current[-1] = f"{current[-1]} {parts[0]}".strip()
            current += [_cell(part) for part in parts[1:]]
    flush()
    return _render_table(caption, header, rows, images=images, image_urls=image_urls), index


def _cells(text: str, separator: str) -> list[str]:
    return [_cell(part) for part in _split_top(text, separator)]


def _cell(raw: str) -> str:
    """一格的内容：去掉属性，格里的竖线转义掉。

    转义是必须的：`{{lang|zh|二郎神}}` 这类模板在格子里很常见，不转义就会多切出一列，
    切分器按列做的长文本列降级与表头重复随之全部错位。切分器认 `\\|`（`_CELL_SEPARATOR`）。
    """
    attribute = _CELL_ATTRIBUTE.match(raw)
    content = raw[attribute.end() :] if attribute is not None else raw
    # 格里的换行（`<br>` 来的）并成空格：GFM 表格的格子不能跨行
    return " ".join(content.replace("|", "\\|").split())


def _render_table(
    caption: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    images: list[str],
    image_urls: Mapping[str, str],
) -> list[str]:
    width = max([len(header), *(len(row) for row in rows)], default=0)
    if width == 0:
        return []

    def render(row: Sequence[str]) -> str:
        cells = [_inline(cell, images=images, image_urls=image_urls) for cell in row]
        cells += [""] * (width - len(cells))
        return "| " + " | ".join(cells) + " |"

    lines = []
    if caption:
        lines.append(f"**{_inline(caption, images=images, image_urls=image_urls)}**")
    # 没有表头行时补一行空的，而不是把第一行数据提上来当表头——提上来那一行就重复了
    lines.append(render(header if header else [""] * width))
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend(render(row) for row in rows)
    return lines


def _split_top(text: str, separator: str = "|") -> list[str]:
    """按分隔符切，但跳过 `[[…]]` 与 `{{…}}` 里面的。

    模板参数里再嵌模板是常态（`{{甲|{{乙|x}}}}`），不按层数跳过去，会在内层那个 `|`
    上切一刀——而这一刀切出来的参数看着还挺像回事，不报错。
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair in ("[[", "{{"):
            depth += 1
        elif pair in ("]]", "}}"):
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(separator, index):
            parts.append("".join(current))
            current = []
            index += len(separator)
            continue
        if pair in ("[[", "{{", "]]", "}}"):
            current.append(pair)
            index += 2
            continue
        current.append(text[index])
        index += 1
    parts.append("".join(current))
    return parts


def _tidy(lines: Sequence[str]) -> str:
    """收掉多余的空行，让结果稳定——同一份 wikitext 转两次必须逐字相同。"""
    text = re.sub(r"[ \t]+\n", "\n", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
