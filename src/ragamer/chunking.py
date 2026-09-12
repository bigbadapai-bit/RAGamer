"""切分器：把 Markdown 切成切片。纯函数，不碰网络与数据库。

全项目边角最多的一段，也是唯一值得逐条钉死的地方（架构文档「缝一」）。三件事：

- **结构探测**：有可用的标题结构就按标题层级切，**每个切片带完整祖先标题路径**
  （`二郎神 › 打法 › 第二阶段`，不是只记一层父标题）；标题稀疏的文档退化为递归切分
  加语义边界。同一条 md 内部两种切法并存——标题底下的超长正文按语义边界继续切，
  标题之外的引子本就按语义切。
- **围栏状态机**：` ``` ` 翻转，代码块里的 `#` 不会被当成标题；MinerU 塞进
  `<details>` 的图内文字，标记去掉、文字留下（见架构文档 1.2）。
- **顺序**：`chunk_index` 从 0 起连续，聚合父块靠它还原文档顺序。

只做切分。表格原子化与 `content_meta` 不在这一层，打标与图片补全也不在。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

#: 祖先标题路径的分隔符，写法以 CONTEXT.md 的同名词条为准。
PATH_SEPARATOR = " › "

#: 标题行：一至六个 `#` 加空格。`#话题` 这种没有空格的不是标题。
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

#: 围栏标记：三个及以上的反引号或波浪线。
_FENCE = re.compile(r"^(`{3,}|~{3,})")

#: MinerU 折叠块的标记。只去标记，块内的图内文字留着。
_FOLD_TAG = re.compile(r"</?(?:details|summary)(?:\s[^>]*)?>", re.IGNORECASE)

#: Markdown 表格的分隔行。有它才说明这是个表格块——结构探测的一个信号。
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

#: MediaWiki 特征：内链、分类、模板。有这些的文档即使标题稀疏也是词条页。
_MEDIAWIKI = re.compile(r"\[\[[^\]]+\]\]|Category:|\{\{[^}]+\}\}")

#: 递归切分的边界，从粗到细。中文没有词间空格，最后两手是逗号与硬切。
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "；", "…", "，", ", ", " ")


@dataclass(frozen=True)
class ChunkRules:
    """切分参数。都是经验值——要调先有评测集，否则只是把一个猜测换成另一个猜测。"""

    #: 单个切片的字数软上限。按字数而非 token 算：这一层没有分词器，也不该为它引入。
    max_chars: int = 800
    #: 切片字数的硬下限。短于它的片会尽量与相邻片合并，免得留下读不成句的碎片。
    min_chars: int = 200
    #: 标题行占正文行的比。低于它就不再认为这份文档有可用的标题结构。
    heading_density: float = 0.02

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars 必须为正数，否则正文会被静默丢光")
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars 不能大于 max_chars")


@dataclass(frozen=True)
class Chunk:
    """检索与生成的最小内容单元。"""

    #: 正文。参与向量化。
    content: str
    #: 在源文档中的顺序，从 0 起。
    chunk_index: int
    #: 祖先标题路径，形如「二郎神 › 打法 › 第二阶段」。不在任何标题底下时是空串。
    ancestor_path: str


@dataclass(frozen=True)
class StructureProbe:
    """结构探测的结果。判定依据一并带出来，排障与切分预览都用得上。"""

    headings: int
    body_lines: int
    density: float
    has_table: bool
    mediawiki: bool
    structured: bool


def chunk_document(markdown: str, rules: ChunkRules | None = None) -> list[Chunk]:
    """把一份 Markdown 切成切片。纯函数：不联网、不落库、不读配置。"""
    rules = rules if rules is not None else ChunkRules()
    blocks = _scan(markdown)
    pieces = _hierarchy_pieces(blocks) if _probe(blocks, rules).structured else _flat_pieces(blocks)

    chunks: list[Chunk] = []
    for piece in pieces:
        texts = (
            [piece.text]
            if len(piece.text) <= rules.max_chars
            else _split_semantically(piece.text, rules)
        )
        for text in texts:
            if text.strip():
                chunks.append(Chunk(text, len(chunks), piece.path))
    return chunks


def probe_structure(markdown: str, rules: ChunkRules | None = None) -> StructureProbe:
    """探测这份文档有没有可用的标题结构。

    判定信号按架构文档的权重：**标题密度为主**，表格块与 MediaWiki 特征为辅——
    带表格或 `[[内链]]` / `Category:` / `{{模板}}` 的文档，即使标题稀疏也是词条页，
    按结构切比按长度切更贴它的本来面目。
    架构文档列的第四个信号「平均段落长度」还没用上：它与密度同向变化，
    在没有真实语料能标定阈值之前加进来只会多一个拍出来的数。
    """
    return _probe(_scan(markdown), rules if rules is not None else ChunkRules())


def _probe(blocks: Sequence[_Block], rules: ChunkRules) -> StructureProbe:
    headings = sum(1 for block in blocks if block.kind == "heading")
    body = [block.text for block in blocks if block.kind == "text" and block.text.strip()]
    has_table = any(_TABLE_RULE.match(line) for line in body)
    mediawiki = any(_MEDIAWIKI.search(line) for line in body)
    density = headings / len(body) if body else 0.0
    return StructureProbe(
        headings=headings,
        body_lines=len(body),
        density=density,
        has_table=has_table,
        mediawiki=mediawiki,
        # 标题密度是主信号；另两个信号说明这是有结构的词条页，标题少也按结构切
        structured=density >= rules.heading_density or has_table or mediawiki,
    )


@dataclass(frozen=True)
class _Block:
    """扫出来的一行，或一整个围栏代码块。"""

    kind: Literal["heading", "text", "fence"]
    #: 标题的正文（不含 `#`）；其余情况是原始行。
    text: str
    #: 标题层级，一至六级；其余情况是 0。
    level: int = 0


@dataclass(frozen=True)
class _Piece:
    """一段待切的正文，以及它在源文档标题目录中的位置。"""

    text: str
    path: str


def _scan(markdown: str) -> list[_Block]:
    """逐行扫，用翻转状态机认围栏：翻进去之后 `#` 就只是代码里的字符。"""
    blocks: list[_Block] = []
    fence: str | None = None
    buffered: list[str] = []

    for line in markdown.splitlines():
        if fence is not None:
            buffered.append(line)
            if _closes(line, fence):
                blocks.append(_Block("fence", "\n".join(buffered)))
                buffered = []
                fence = None
            continue

        marker = _opening_fence(line)
        if marker is not None:
            fence = marker
            buffered = [line]
            continue

        heading = _HEADING.match(line)
        if heading is not None:
            blocks.append(_Block("heading", heading.group(2).strip(), len(heading.group(1))))
            continue

        unfolded = _FOLD_TAG.sub("", line)
        # 整行就是个折叠块标记：去掉标记之后什么都不剩，这一行也就没有内容
        if line.strip() and not unfolded.strip():
            continue
        blocks.append(_Block("text", unfolded))

    if buffered:
        # 围栏没闭合：剩下的整段原样收下，既不当标题也不丢内容
        blocks.append(_Block("fence", "\n".join(buffered)))
    return blocks


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
    """按标题层级切：每个标题底下的正文是一段，路径带上它的全部祖先标题。"""
    pieces: list[_Piece] = []
    buffer: list[str] = []
    stack: list[tuple[int, str]] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            pieces.append(_Piece(text, PATH_SEPARATOR.join(title for _, title in stack)))
        buffer.clear()

    for block in blocks:
        if block.kind != "heading":
            buffer.append(block.text)
            continue
        flush()
        # 回过头去到同级的上一层：`## 打法` 之后再出现 `## 掉落`，它不该认打法做父亲
        while stack and stack[-1][0] >= block.level:
            stack.pop()
        stack.append((block.level, block.text))
    flush()
    return pieces


def _flat_pieces(blocks: Sequence[_Block]) -> list[_Piece]:
    """没有可用的标题结构：整篇按语义边界切，标题行只是普通正文。"""
    text = "\n".join(_source_line(block) for block in blocks).strip()
    return [_Piece(text, "")] if text else []


def _source_line(block: _Block) -> str:
    """还原成源文档里的那一行。降级切分时标题保留 `#`，正文不做改动。"""
    return f"{'#' * block.level} {block.text}" if block.kind == "heading" else block.text


def _split_semantically(text: str, rules: ChunkRules) -> list[str]:
    """按语义边界递归切开一段超长正文，再把过短的相邻片并回去。"""
    return _merge(_cut_by(text, rules, _SEPARATORS), rules)


def _cut_by(text: str, rules: ChunkRules, separators: tuple[str, ...]) -> list[str]:
    """按 `separators` 从粗到细地切；切完仍旧超长的片段，换更细的边界再切。

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
