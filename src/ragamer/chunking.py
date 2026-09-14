"""切分器：把 Markdown 切成切片。纯函数，不碰网络与数据库。

全项目边角最多的一段，也是规格 #1 的测试决策里定下「缝一」的理由：纯函数、零假件，
测试直接调它，失败就落在出错的那一行。四件事：

- **结构探测**：有可用的标题结构就按标题层级切，**每个切片带完整祖先标题路径**
  （`二郎神 › 打法 › 第二阶段`，不是只记一层父标题）；标题稀疏的文档退化为递归切分
  加语义边界。同一条 md 内部两种切法并存——标题底下的超长正文按语义边界继续切，
  标题之外的引子本就按语义切。
- **围栏状态机**：` ``` ` 翻转，代码块里的 `#` 不会被当成标题；MinerU 塞进
  `<details>` 的图内文字，标记去掉、文字留下（见架构文档 1.2）。
- **表格原子化**：Markdown 表格整表一块，装不下按行组切、**每块重复表头**；
  **长文本列整列降级进 `content_meta`，不进正文**（见 `_table_chunks`）；
  Infobox 与模板块整块保留，与表格同标 `table`。
- **图片地址摘走**：正文里只留替代文本，**取得到原图**的那些地址进切片自己的
  `image_urls`（见 `_piece_chunks`）。它唯一用处是答案里显示原图，既不参与向量化
  也不交给生成；而留在正文里既占满字数预算、又会被切分从中间切开，存下一条取不到
  原图的坏地址。
- **顺序**：`chunk_index` 从 0 起连续，聚合父块靠它还原文档顺序。

只做切分。打标与图片补全不在这一层。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ragamer.sources import ImageRef, strip_image_refs
from ragamer.stores.base import is_image_key

#: 祖先标题路径的分隔符，写法以 CONTEXT.md 的同名词条为准。
PATH_SEPARATOR = " › "

#: 标题行：一至六个 `#` 加空格。`#话题` 这种没有空格的不是标题。
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

#: 围栏标记：三个及以上的反引号或波浪线。
_FENCE = re.compile(r"^(`{3,}|~{3,})")

#: MinerU 折叠块的标记。只去标记，块内的图内文字留着——按 MinerU 产物的形态换：
#: 正文里若真的在讲 `<details>` 这个元素，那几个字符也会一并去掉。
_FOLD_TAG = re.compile(r"</?(?:details|summary)(?:\s[^>]*)?>", re.IGNORECASE)

#: 表格的分隔行。有它才说明这是个表格块——结构探测的一个信号。
_TABLE_RULE = re.compile(
    # 两列及以上：`| --- | --- |`
    r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$"
    # 单列表格也是表格，但它那一行必须带竖线：光一条 `---` 与 Setext 标题分不开
    r"|^\s*\|\s*:?-{2,}:?\s*\|?\s*$"
)

#: 表格行里分隔单元格的竖线。转义的（`\|`）是单元格内容，不是分隔符。
_CELL_SEPARATOR = re.compile(r"(?<!\\)\|")

#: 模板块的开头：行首的 `{{`。行内出现的 `{{` 不算——正文里提一句模板语法很常见。
_TEMPLATE_START = re.compile(r"^\s*\{\{")

#: MediaWiki 特征：内链、分类、模板。有这些的文档即使标题稀疏也是词条页。
_MEDIAWIKI = re.compile(r"\[\[[^\]]+\]\]|Category:|\{\{[^}]+\}\}")

#: 递归切分的边界，从粗到细。中文没有词间空格，最后两手是逗号与硬切。
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "；", "…", "，", ", ", " ")


#: 单个切片的字数软上限（默认值）。检索侧推父块上限时也按它算量级，所以提成模块级常量：
#: 同一个数在两处各写一遍，改了一处另一处就静默地不同步了。
DEFAULT_MAX_CHARS = 800


@dataclass(frozen=True)
class ChunkRules:
    """切分参数。都是经验值——要调先有评测集，否则只是把一个猜测换成另一个猜测。"""

    #: 单个切片的字数软上限。按字数而非 token 算：这一层没有分词器，也不该为它引入。
    max_chars: int = DEFAULT_MAX_CHARS
    #: 切片字数的硬下限。短于它的片会尽量与相邻片合并，免得留下读不成句的碎片。
    min_chars: int = 200
    #: 标题行占正文行的比。低于它就不再认为这份文档有可用的标题结构。
    heading_density: float = 0.02
    #: 单元格超过这个字数就算长文本列，整列降级进 `content_meta`（见 `_table_chunks`）。
    long_cell_chars: int = 100

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars 必须为正数，否则正文会被静默丢光")
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars 不能大于 max_chars")
        if self.long_cell_chars < 1:
            raise ValueError("long_cell_chars 必须为正数，否则整张表都会被当成长列")


@dataclass(frozen=True)
class Chunk:
    """检索与生成的最小内容单元。"""

    #: 正文。参与向量化。
    content: str
    #: 在源文档中的顺序，从 0 起。
    chunk_index: int
    #: 祖先标题路径，形如「二郎神 › 打法 › 第二阶段」。不在任何标题底下时是空串。
    ancestor_path: str
    #: 不参与向量化的附加文本：表格里的长文本列。随结果返回但没进 embedding，
    #: 也就不存在「超长被截断」这回事。取值与 `chunks` schema 的 `content_meta` 一致。
    content_meta: str = ""
    #: 这片正文里出现过、**且取得到原图**的图片地址，按出现顺序、去重前原样。
    #: **它不在正文里**——地址在切分之前就被摘走了（`strip_image_refs`），正文里留下的是
    #: 替代文本。唯一的用处是答案里能显示原图（用户故事 52），既不参与向量化、
    #: 也不随正文交给生成。取不到原图的不带进来（`_servable`）。
    image_urls: tuple[str, ...] = ()
    #: `text` 正文 · `table` 结构化块（Markdown 表格、Infobox 与模板块）· `image` 图片。
    #: v1 的切分器只产出前两种：补图那一层是把图内文字**写回正文**（`ragamer.enriching`），
    #: 不另外产出图片切片——图片引用连同它的文字落在同一片正文里，答案里还能展示原图。
    chunk_type: Literal["text", "table", "image"] = "text"


@dataclass(frozen=True)
class StructureProbe:
    """结构探测的结果。

    判定依据一并带出来：这几个门槛都是经验值（见 `ChunkRules`），判定不合预期时
    要能看见是哪个信号把它推过去的，否则只能对着一个布尔值猜。
    """

    #: 标题行数。
    heading_count: int
    #: 正文行数。围栏代码块按它的行数计——代码也是正文，见 `_probe`。
    body_line_count: int
    #: 标题行占正文行的比，也就是主信号。
    density: float
    #: 有没有表格块。
    has_table: bool
    #: 有没有 MediaWiki 特征（`[[内链]]` / `Category:` / `{{模板}}`）。
    mediawiki: bool
    #: 最终判定：按标题层级切，还是退化为扁平切分。
    structured: bool


def chunk_document(markdown: str, rules: ChunkRules | None = None) -> list[Chunk]:
    """把一份 Markdown 切成切片。纯函数：不联网、不落库、不读配置。"""
    rules = rules or ChunkRules()
    blocks = _scan(markdown)
    pieces = _hierarchy_pieces(blocks) if _probe(blocks, rules).structured else _flat_pieces(blocks)

    chunks: list[Chunk] = []
    for piece in pieces:
        chunks.extend(_piece_chunks(piece, len(chunks), rules))
    return chunks


def _piece_chunks(piece: _Piece, index: int, rules: ChunkRules) -> list[Chunk]:
    """把一段待切内容切成切片。

    三种走法：表格自成一套切法（它按单元格摘地址）；模板块整块留一片；其余按语义边界切。
    降级切分（`_flat_pieces`）只产出 `text` 片段——表格与模板本身就把文档判成有结构的。

    图片地址在这一层就摘走，正文里只留替代文本（`strip_image_refs`）：地址留着会占满
    字数预算，还会被下面的切分从中间切开——那会存下一条取不到原图的坏地址，不报错。
    摘下来的地址再过一道 `_servable`：取不到原图的不跟着切片走。
    """
    if piece.kind == "table":
        return _table_chunks(piece, index, rules)
    text, refs = strip_image_refs(piece.text)
    refs = _servable(refs)
    if piece.kind == "template":
        # 切开就不是 Infobox 了，长度再超也不动它
        return [Chunk(text, index, piece.path, chunk_type="table", image_urls=_urls(refs))]
    return [
        Chunk(text, index + offset, piece.path, image_urls=urls)
        for offset, (text, urls) in enumerate(_spread(text, _split(text, rules), refs))
    ]


def _urls(refs: Sequence[ImageRef]) -> tuple[str, ...]:
    return tuple(ref.ref for ref in refs)


def _servable(refs: Sequence[ImageRef]) -> tuple[ImageRef, ...]:
    """只留我们真取回了原图的那几处引用。

    正文里的引用不都取得到原图：行内图标不下载，下载失败的与相对路径的原样留着
    （`ragamer.sources.fetch_images`）。它们没有对象 key，而答案里的图是拿地址去
    对象存储取的——带进 `image_urls` 只会在答案里多一条取不到的死图，页面上报一句
    「没有这张图」，而取不到的原因在导入时就说过了。

    取不到就不带着走，与补图那一层同一条口径（`is_image_key`）。
    """
    return tuple(ref for ref in refs if is_image_key(ref.ref))


def _spread(
    text: str, texts: Sequence[str], refs: Sequence[ImageRef]
) -> list[tuple[str, tuple[str, ...]]]:
    """把这一段里的图片地址分派给切出来的那几片正文。

    分派看引用在这段正文里的**落点**：几片正文是这段的连续几刀，顺着往下走，
    落点进了哪一片，地址就归哪一片。落点取自 `strip_image_refs`（在摘完地址的正文上算），
    所以一条地址不可能横跨两片。

    落点落在两片之间的空白上时归**前**一片：那处空白本来就在前一片的边界上。

    一段切不出任何正文（整段都是图片、替代文本又都空着）时返回空列表，
    地址跟着一起没有落点——那种段本来就没有可检索的正文，进不了库（见 `ragamer.importing`）。
    """
    if not texts:
        return []
    starts: list[int] = []
    cursor = 0
    for piece in texts:
        cursor = max(text.find(piece, cursor), 0)
        starts.append(cursor)
        cursor += len(piece)
    owned: list[list[str]] = [[] for _ in texts]
    for ref in refs:
        owned[_slot(starts, ref.end, len(texts))].append(ref.ref)
    return [(piece, tuple(urls)) for piece, urls in zip(texts, owned, strict=True)]


def _slot(starts: Sequence[int], position: int, count: int) -> int:
    """落点落在第几片里。落点在整段开头的空白上时归第一片（`bisect_right` 会算出 -1）。"""
    return min(max(bisect_right(starts, position) - 1, 0), count - 1)


def probe_structure(markdown: str, rules: ChunkRules | None = None) -> StructureProbe:
    """探测这份文档有没有可用的标题结构。

    判定信号按架构文档 §1.4 的权重：**标题密度为主**，表格块与 MediaWiki 特征为辅——
    带表格或 `[[内链]]` / `Category:` / `{{模板}}` 的文档，即使标题稀疏也是词条页，
    按结构切比按长度切更贴它的本来面目。
    架构文档列的第四个信号「平均段落长度」还没用上：它与密度同向变化，
    在没有真实语料能标定阈值之前加进来只会多一个拍出来的数。
    """
    return _probe(_scan(markdown), rules or ChunkRules())


def _probe(blocks: Sequence[_Block], rules: ChunkRules) -> StructureProbe:
    headings = sum(1 for block in blocks if block.kind == "heading")
    body_lines = sum(_body_lines(block) for block in blocks)
    has_table = any(block.kind == "table" for block in blocks)
    # 围栏里的不算：代码里出现 `[[链接]]` 说明不了这份文档是词条页
    mediawiki = any(
        _MEDIAWIKI.search(block.text) for block in blocks if block.kind in ("text", "template")
    )
    # 有标题就先把分母垫到 1：只有标题的文档不该因为没有正文而被判成扁平
    density = headings / max(body_lines, 1)
    return StructureProbe(
        heading_count=headings,
        body_line_count=body_lines,
        density=density,
        has_table=has_table,
        mediawiki=mediawiki,
        # 标题密度是主信号；另两个信号说明这是有结构的词条页，标题少也按结构切
        structured=density >= rules.heading_density or has_table or mediawiki,
    )


def _body_lines(block: _Block) -> int:
    """这一块算几行正文。标题不算；空行不算；整块（围栏、表格、模板）按它的行数算。

    围栏与表格里的行也算正文行：一份正文全是代码的文档照样有标题结构，漏掉它们会让
    密度恒为 0，整篇被误判成扁平——于是标题丢了路径，`#` 还漏回正文里去。
    """
    if block.kind == "heading":
        return 0
    if block.kind == "text":
        return 1 if block.text.strip() else 0
    return len(block.text.splitlines())


@dataclass(frozen=True)
class _Block:
    """扫出来的一行，或一整个围栏／表格／模板块。"""

    kind: Literal["heading", "text", "fence", "table", "template"]
    #: 标题的正文（不含 `#`）；整块的是它的全部行（含首尾标记）；其余情况是原始行。
    text: str
    #: 标题层级，一至六级；其余情况是 0。
    level: int = 0


@dataclass(frozen=True)
class _Piece:
    """一段待切的正文，以及它在源文档标题目录中的位置。"""

    #: 正文。同一个祖先标题路径底下的行连成一段。
    text: str
    #: 祖先标题路径。不在任何标题底下时是空串。
    path: str
    #: `text` 按语义边界继续切；`table` 与 `template` 自成一片，不按长度切。
    kind: Literal["text", "table", "template"] = "text"


def _scan(markdown: str) -> list[_Block]:
    """逐行扫，认围栏、表格与模板块：围栏翻进去之后 `#` 就只是代码里的字符。"""
    lines = markdown.splitlines()
    blocks: list[_Block] = []
    index = 0

    while index < len(lines):
        span = (
            _fence_span(lines, index) or _table_span(lines, index) or _template_span(lines, index)
        )
        if span is not None:
            kind, end = span
            blocks.append(_Block(kind, "\n".join(lines[index:end])))
            index = end
            continue

        heading = _HEADING.match(lines[index])
        if heading is not None:
            blocks.append(_Block("heading", heading.group(2).strip(), len(heading.group(1))))
            index += 1
            continue

        unfolded = _FOLD_TAG.sub("", lines[index])
        # 整行就是个折叠块标记：去掉标记之后什么都不剩，这一行也就没有内容
        if not lines[index].strip() or unfolded.strip():
            blocks.append(_Block("text", unfolded))
        index += 1

    return blocks


def _fence_span(lines: Sequence[str], index: int) -> tuple[Literal["fence"], int] | None:
    """从这一行起是不是围栏代码块。是就返回它的块类型与结束位置（不含）。"""
    marker = _opening_fence(lines[index])
    if marker is None:
        return None
    end = index + 1
    while end < len(lines) and not _closes(lines[end], marker):
        end += 1
    # 围栏没闭合：剩下的整段原样收下，既不当标题也不丢内容
    return "fence", min(end + 1, len(lines))


def _table_span(lines: Sequence[str], index: int) -> tuple[Literal["table"], int] | None:
    """从这一行起是不是一张 Markdown 表格。表头行、分隔行，然后一直接着数据行。

    表头行里有竖线不算数，下一行是分隔行才算——`|` 在正文里太常见了。
    """
    if "|" not in lines[index] or index + 1 >= len(lines):
        return None
    if not _TABLE_RULE.match(lines[index + 1]):
        return None
    end = index + 2
    while end < len(lines) and "|" in lines[end]:
        end += 1
    return "table", end


def _template_span(lines: Sequence[str], index: int) -> tuple[Literal["template"], int] | None:
    """从这一行起是不是一个模板块：行首 `{{` 起，配平 `}}` 收。

    没配平的不当模板——正文里一个孤零零的 `{{` 会把后面整篇吞进一个原子块里，
    既切不开也认不出。
    """
    if not _TEMPLATE_START.match(lines[index]):
        return None
    depth = 0
    for end in range(index, len(lines)):
        depth += lines[end].count("{{") - lines[end].count("}}")
        if depth <= 0:
            return "template", end + 1
    return None


def _opening_fence(line: str) -> str | None:
    """这一行是不是围栏的开头。是就返回它的标记（``` 或 ~~~）。

    缩进超过三格的是正文里的缩进代码，不算围栏；标记后面还带着同种标记的
    （` ```code``` ` 这种一行写完的）也不当围栏开头。
    """
    stripped = line.lstrip()
    if len(line) - len(stripped) > 3:
        return None
    match = _FENCE.match(stripped)
    if match is None:
        return None
    marker = match.group(1)
    return marker if marker[0] not in stripped[len(marker) :] else None


def _closes(line: str, marker: str) -> bool:
    """这一行收不收这个围栏。收尾只认同一种字符，且不短于开头的标记。"""
    stripped = line.strip()
    return len(stripped) >= len(marker) and set(stripped) == {marker[0]}


def _hierarchy_pieces(blocks: Sequence[_Block]) -> list[_Piece]:
    """按标题层级切：每个标题底下的正文是一段，路径带上它的全部祖先标题。

    表格与模板块自成一片，两侧的正文各自收口——不然表会跟前后正文挤进同一片，
    再由 `_split` 按长度从中间切开。
    """
    pieces: list[_Piece] = []
    buffer: list[str] = []
    # 还开着的标题，从根往下。栈上放的就是那些标题块本身
    opened: list[_Block] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            pieces.append(_Piece(text, _path(opened)))
        buffer.clear()

    for block in blocks:
        if block.kind == "heading":
            flush()
            # 回过头去到同级的上一层：`## 打法` 之后再出现 `## 掉落`，它不该认打法做父亲
            while opened and opened[-1].level >= block.level:
                opened.pop()
            opened.append(block)
            continue
        if block.kind in ("table", "template"):
            flush()
            pieces.append(_Piece(block.text, _path(opened), block.kind))
            continue
        buffer.append(block.text)
    flush()
    return pieces


def _path(opened: Sequence[_Block]) -> str:
    """还开着的标题连成祖先标题路径。"""
    return PATH_SEPARATOR.join(heading.text for heading in opened)


def _flat_pieces(blocks: Sequence[_Block]) -> list[_Piece]:
    """没有可用的标题结构：整篇按语义边界切，标题行只是普通正文。"""
    text = "\n".join(_source_line(block) for block in blocks).strip()
    return [_Piece(text, "")] if text else []


def _source_line(block: _Block) -> str:
    """还原成源文档里的那一行。降级切分时标题保留 `#`，正文不做改动。"""
    return f"{'#' * block.level} {block.text}" if block.kind == "heading" else block.text


# --- 表格 ---


@dataclass(frozen=True)
class _Table:
    """解析后的表格。单元格已去空白、已摘掉图片地址，转义的竖线留在原处。

    **结构归它自己**（`render`），**怎么切归下面的自由函数**：哪几列算长文本列、
    按什么分组都是切分策略，要跟着 `ChunkRules` 走，不该长在数据结构上。

    `render` 收的是**行号**不是行本身：行里的图片地址要和它所在的那一片正文对上，
    传行号才认得出是哪些格。
    """

    header: list[str]
    aligns: list[str]
    rows: list[list[str]]
    #: 每一格的图片地址：`[0]` 是表头、`[1:]` 是数据行，形状与 `header` / `rows` 对齐。
    #: 与单元格分开存，是因为渲染会把若干格拼成一片正文，而地址要跟着那片正文走。
    cell_urls: list[list[tuple[str, ...]]]

    def render(self, columns: Sequence[int], rows: Sequence[int]) -> str:
        """按列投影渲回一张 Markdown 表格。

        单元格之间统一成一个空格：源表里那些对齐空格是给人看的，重排一次反而整齐。
        """
        head = [self.header[column] for column in columns]
        rule = [self.aligns[column] for column in columns]
        body = [[_cell(self.rows[index], column) for column in columns] for index in rows]
        return "\n".join("| " + " | ".join(line) + " |" for line in [head, rule, *body])


def _table_chunks(piece: _Piece, index: int, rules: ChunkRules) -> list[Chunk]:
    """把一张表切成切片：整表装得下就一片，装不下按行组切、**每块重复表头**。

    **长文本列整列降级进 `content_meta`，不进正文。** 这不是优化：超长行会在
    向量化阶段被静默截断，内容永久丢失且不报错，而 `content_meta` 不参与向量化，
    随结果返回即可（架构文档 §2.5）。

    两处取舍：

    - **整列一起降级**，而不是只挪走超长的那几格——一列里长短不齐时，行的内容会被
      拆到两个地方，读的人对不上号。
    - **每一列都是长文本列时，正文只留表头骨架**，数据行全数进 meta：正文不能是空的
      （空了就检索不回来，meta 也就没机会随结果返回），但也不能拿长文本凑数。

    图片地址按**格**摘（`_parse_table`），一片正文拿到的是它自己那些格里的地址——
    留在正文里的话，一格只有一个图标也顶得上一篇正文的长度，长文本列还会判错。
    """
    table = _parse_table(piece.text)
    long_columns = _long_columns(table, rules)
    kept_columns = [column for column in range(len(table.header)) if column not in long_columns]
    # meta 里带上第一列：长列单拎出来之后，得知道哪一格属于哪一行
    meta_columns = sorted(long_columns | {0}) if long_columns else []
    # 一列都没剩下时不再分组：正文只剩骨架，分几组都一样，数据行整份进 meta
    groups = (
        _group_rows(table, kept_columns, rules) if kept_columns else [list(range(len(table.rows)))]
    )

    chunks: list[Chunk] = []
    for rows in groups:
        chunks.append(
            Chunk(
                content=table.render(kept_columns, rows) if kept_columns else _skeleton(table),
                chunk_index=index + len(chunks),
                ancestor_path=piece.path,
                content_meta=table.render(meta_columns, rows) if meta_columns else "",
                image_urls=_table_urls(table, rows),
                chunk_type="table",
            )
        )
    return chunks


def _skeleton(table: _Table) -> str:
    """只剩表头的一张空表——列名是全表都是长文本列时仅剩的检索锚点。"""
    return table.render(range(len(table.header)), [])


def _table_urls(table: _Table, rows: Sequence[int]) -> tuple[str, ...]:
    """这一片正文里的图片地址：表头那一行 + 这几行数据行，按出现顺序去重。

    **降级进 `content_meta` 的那些列也算**：meta 随结果一起交给生成，里面同样可以有图，
    漏掉它答案里就会少显示几张。留下的列与降级的列合起来正好是全部列
    （见 `_table_chunks`），所以这里按整行取即可。
    """
    lines = (0, *(index + 1 for index in rows))
    return tuple(
        dict.fromkeys(url for line in lines for cell in table.cell_urls[line] for url in cell)
    )


def _parse_table(text: str) -> _Table:
    """把表格块拆成单元格，顺带把每格的图片地址摘出来。

    数据行比表头宽时把表头补到那么宽——多出来的格也是内容。**先补宽再摘地址**：
    补出来的空格本就该与别的格一样处理。
    """
    head, rule, *rows = text.splitlines()
    header = _cells(head)
    body = [_cells(row) for row in rows]
    width = max(len(header), max((len(row) for row in body), default=0))
    padded = [_pad(header, width), *(_pad(row, width) for row in body)]
    stripped = [_strip_cells(row) for row in padded]
    aligns = [align or "---" for align in _cells(rule)]
    return _Table(
        header=stripped[0][0],
        aligns=[*aligns, *(["---"] * width)][:width],
        rows=[cells for cells, _ in stripped[1:]],
        cell_urls=[urls for _, urls in stripped],
    )


def _pad(cells: Sequence[str], width: int) -> list[str]:
    return [*cells, *([""] * (width - len(cells)))]


def _strip_cells(cells: Sequence[str]) -> tuple[list[str], list[tuple[str, ...]]]:
    """一格一格地摘图片地址：返回摘完的格与每格的地址（顺序与格一一对应）。

    地址与正文那条路一样过一道 `_servable`：表格里的图标比正文里还多
    （bwiki 的物品表整列都是 `18px-图标-…`）。
    """
    stripped: list[str] = []
    urls: list[tuple[str, ...]] = []
    for cell in cells:
        text, refs = strip_image_refs(cell)
        stripped.append(text)
        urls.append(_urls(_servable(refs)))
    return stripped, urls


def _cells(line: str) -> list[str]:
    """把一行拆成单元格。首尾的竖线是表格的边框，不是空单元格。"""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in _CELL_SEPARATOR.split(stripped)]


def _cell(row: Sequence[str], column: int) -> str:
    """取第 `column` 格。行比表头窄时补空串——渲染出来的表要保持矩形。"""
    return row[column] if column < len(row) else ""


def _long_columns(table: _Table, rules: ChunkRules) -> set[int]:
    """哪几列算长文本列：表头或任一单元格超过 `long_cell_chars`。

    一格超长就整列降级：只挪走超长的那几格，同一列会在正文与 `content_meta` 里各留一半。

    量的是**摘掉图片地址之后**的格（`_parse_table`）：一格只有一个图标时，地址比那一格
    真正的文字还长，按原样量会把整整一列判成长文本列——列里的说明文字因此全被挪出正文，
    而那正是这一列最该被检索到的东西。
    """
    long_columns = {
        column for column, name in enumerate(table.header) if len(name) > rules.long_cell_chars
    }
    for row in table.rows:
        long_columns.update(
            column for column, cell in enumerate(row) if len(cell) > rules.long_cell_chars
        )
    return long_columns


def _group_rows(table: _Table, columns: Sequence[int], rules: ChunkRules) -> list[list[int]]:
    """把数据行分组（组里放的是**行号**），每组配上表头之后不超过 `max_chars`。

    量的是**渲染之后的正文**，只算留下的那几列：`content_meta` 不参与向量化，
    长短与这一片的上限无关。行号而不是行本身：图片地址要跟着行走到它落的那一片，
    见 :meth:`_Table.render`。

    两个与正文切分不同的地方：单行自己就超长时让它独占一组，不把一行掰到两组里去；
    也不为了凑够下限把两组并回去——每块都带着表头，只剩一行也是读得懂的。
    """
    groups: list[list[int]] = []
    for index in range(len(table.rows)):
        if groups and len(table.render(columns, [*groups[-1], index])) <= rules.max_chars:
            groups[-1].append(index)
        else:
            groups.append([index])
    # 没有数据行也留一片：表头本身是内容，丢掉就等于把这张表删了
    return groups or [[]]


def _split(text: str, rules: ChunkRules) -> list[str]:
    """把一段正文切成若干片，每片不超过上限，也不留下读不成句的碎片。

    本来就装得下的一段原样留成一片；超长的按语义边界递归切开，再把过短的相邻片
    并回去（见 `_cut_by` 与 `_merge`）。
    """
    if len(text) <= rules.max_chars:
        return [text] if text.strip() else []
    return _merge(_cut_by(text, rules, _SEPARATORS), rules)


def _cut_by(text: str, rules: ChunkRules, separators: tuple[str, ...]) -> list[str]:
    """按 `separators` 从粗到细地切；切完仍旧超长的那些候选片，换更细的边界再切。

    边界留在前一片的尾巴上（`_split_keep`），所以拼回去与原文一致——
    合并时直接相接即可，不必再补分隔符。
    """
    if len(text) <= rules.max_chars:
        return [text] if text.strip() else []
    if not separators:
        # 一处边界都没有（比如一整串没有标点的长数字）：只能按字数硬切
        return [
            text[start : start + rules.max_chars] for start in range(0, len(text), rules.max_chars)
        ]

    separator, rest = separators[0], separators[1:]
    pieces: list[str] = []
    for piece in _split_keep(text, separator):
        if not piece.strip():
            continue
        pieces.extend(_cut_by(piece, rules, rest) if len(piece) > rules.max_chars else [piece])
    return pieces


def _split_keep(text: str, separator: str) -> list[str]:
    """按分隔符切开，并把分隔符留在前一片的尾巴上。"""
    parts = text.split(separator)
    return [part + separator for part in parts[:-1]] + [parts[-1]]


def _merge(pieces: Sequence[str], rules: ChunkRules) -> list[str]:
    """把相邻的短片并起来，别留下读不成句的碎片。

    `min_chars` 是硬下限，`max_chars` 是软上限：装不下时，只要当前这片还没到下限就
    继续并进去——多占一点上下文，好过丢掉一整句话。整段切完剩下的尾巴再与前一片
    重新分一次，免得末尾孤零零留下几个字。
    """
    groups: list[list[str]] = []
    for piece in pieces:
        if groups and _fits(groups[-1], piece, rules):
            groups[-1].append(piece)
        else:
            groups.append([piece])
    if len(groups) > 1 and _width(groups[-1]) < rules.min_chars:
        groups[-2:] = _rebalance(groups[-2], groups[-1], rules)
    return [_text(group) for group in groups]


def _fits(group: Sequence[str], piece: str, rules: ChunkRules) -> bool:
    """这一片还装得下吗。装不下，但当前这片自己还没到下限，那也并进去。"""
    width = _width(group)
    return width + len(piece) <= rules.max_chars or width < rules.min_chars


def _rebalance(prev: Sequence[str], tail: Sequence[str], rules: ChunkRules) -> list[list[str]]:
    """把太短的尾巴与前一片重新分一次，从中间往外找第一个两边都合法的一刀。

    直接并进前一片会让那一片超出上限，重新分一次则两边都落回区间里。
    实在分不出（合起来还不够两片的下限），并成一片了事。
    """
    combined = [*prev, *tail]
    total = _width(combined)
    for cut in sorted(
        range(1, len(combined)),
        key=lambda cut: abs(_width(combined[:cut]) * 2 - total),
    ):
        head, rest = combined[:cut], combined[cut:]
        if _in_range(head, rules) and _in_range(rest, rules):
            return [head, rest]
    return [combined]


def _in_range(group: Sequence[str], rules: ChunkRules) -> bool:
    return rules.min_chars <= _width(group) <= rules.max_chars


def _width(group: Sequence[str]) -> int:
    return len(_text(group))


def _text(group: Sequence[str]) -> str:
    """把若干片接成一片。边界本来就留在前一片的尾巴上，直接相接即可。"""
    return "".join(group).strip()
