"""归一化：把四条来源的原始资料变成同一种形态（ADR-0006）。

**来源差异只在这一层存在。** PDF 与图片走 MinerU（`ragamer.mineru`）、md／txt 直接读
（`MarkdownParser`）、网页走爬虫，出了这一层只剩 `NormalizedDoc` 一种形态——
切分与打标不必知道内容从哪来，新增一种来源也只动它自己的适配器。适配器由
`ParserRouter` 按扩展名挑；网页没有扩展名可挑，它走另一个入口（`PageCrawler`）。

归一化的**最后两步都在收拢图片地址**：`fetch_images` 把正文里的外链图下载成附件
（解析产物自带的附件早在手上），`publish_assets` 把附件发进对象存储、正文里的引用
改指对象 key。出了这一层，图片地址只有对象 key 一种形态——这句话今天才真的成立：
网页与 md 来源的图一直是外链，而补图那一层是按对象 key 取原图的，那些图因此
一张都补不上。

`Enricher` 的缝也开在这里：补图吃一份 `NormalizedDoc`、吐一份 `NormalizedDoc`，
位置在归一化与切分之间。它要处理的三件事（VLM 摘要进 alt、展开 MinerU 的 `<details>`
折叠块、二次 OCR 回填）见 docs/ARCHITECTURE.md §1.2 与 §1.3，实现是
`ragamer.enriching.ImageEnricher`。**协议与实现刻意不同名**，与本仓别处一致
（`SourceParser` 对 `MarkdownParser`、`LlmClient` 对 `OpenAiLlm`）。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlsplit

from ragamer.logging import get_logger
from ragamer.stores.base import IMAGE_PREFIX, ObjectStore, image_key

logger = get_logger(__name__)

#: 图片引用的两种形式。取地址，供补图那一层去取原图。
#: **两种都要认**：MinerU 把表格内嵌的图片导成 HTML 的 `<img>`，只认 Markdown 那种
#: 会把它们静默漏掉——补图那一层于是取不到这些原图，而且不报错。
#: 两种都分出了替代文本那一段，因为补图要往那里写摘要（见 `set_image_alt`）：
#: 空着才写，作者已经写好的不动。
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\((\s*)([^)\s]+)")
#: HTML 那种取**整个标签**：`alt` 可以在 `src` 前面也可以在后面，只截到 `src` 结尾
#: 就看不到它了。`src` 与 `alt` 再各自从标签里取。
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_SRC = re.compile(r"\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_HTML_ALT = re.compile(r"\balt\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


@dataclass(frozen=True)
class SourceDocument:
    """一份待导入的原始资料：文件名 + 字节。

    收字节而不是路径：界面上传拿到的是字节，本地文件读进来也是字节，
    两条路在这一层汇合，解析适配器不必区分，也不必碰文件系统。
    """

    filename: str
    data: bytes


@dataclass(frozen=True)
class SourceAsset:
    """一份随资料来的二进制附件：MinerU 结果包里的原图，或从外链拉下来的图。

    `name` 是**正文引用它时用的那个字面**（`images/xxx.jpg`，或网页来源的一整条地址）
    ——它是「正文里的哪一处引用该改指哪个对象」的依据，所以必须与正文里逐字一致，
    不能顺手规整（解码百分号转义、去掉查询串都会让那一处引用对不上）。
    """

    name: str
    data: bytes
    content_type: str = "application/octet-stream"
    #: 它在对象存储里的名字。空即从 `name` 推（MinerU 的产物名去掉那层 `images/`）。
    #: 外链那条路必须单独给：`name` 是一整条地址，拿它当对象名会把一串 URL
    #: 原样写进 key 里。
    key_name: str = ""


@dataclass(frozen=True)
class NormalizedDoc:
    """归一之后的文档。四条来源在这一层之后不再有区别。"""

    #: 正文。全部输入最终都变成它。
    markdown: str
    #: 正文里引用到的图片地址，按出现顺序、去重前原样。
    #: 补图那一层按它取原图做二次 OCR 与摘要。`publish_assets` 之后这里是对象 key。
    #: 认的是「正文怎么写」而不是「取不取得到」：md 来源里指向外网的 `<img src="http…">`
    #: 也会进来，补图那一层按自己能取到的那种处理。
    images: tuple[str, ...] = ()
    #: 解析适配器附带的条目级结构（MinerU 的 `content_list.json` 就是它）。
    #: 切分用不到它——正文已经在 `markdown` 里了；补图的二次 OCR 回填要用，
    #: 因为「哪个条目是图片」只有条目级结构说得清。md 来源留空。
    content_list: tuple[Mapping[str, Any], ...] = ()
    #: 解析产物里附带的二进制附件，还没进对象存储。`publish_assets` 把它们发出去，
    #: 之后这里就空了——归一化那条路上只有它一处拿得到这些字节。
    assets: tuple[SourceAsset, ...] = ()
    #: 这份资料从哪来的地址。**只有网页来源有**：本地文件没有地址可填。
    #: 它一路跟着切片进库，答案的引用里显示的就是它。
    source_url: str = ""


class SourceError(Exception):
    """一份资料读不成 Markdown。信息里已点名是哪个文件。"""


class UnsupportedSourceError(SourceError):
    """这个格式还没有对应的解析适配器。"""


@runtime_checkable
class SourceParser(Protocol):
    """一份原始资料 → 归一后的文档。换来源只换这里。"""

    #: 这个适配器认得的扩展名，全小写、带点。`ParserRouter` 按它挑适配器，
    #: 各组之间不许重叠——重叠了先命中的那个赢，另一条路就成了摆设。
    SUFFIXES: tuple[str, ...]

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        """读不了时抛 :class:`SourceError`。"""
        ...


@runtime_checkable
class PageCrawler(Protocol):
    """一个网址 → 归一后的文档。与解析适配器同一个位置、同一种产物。

    它自己不碰切分与打标：抓回来的东西和本地文件在 `NormalizedDoc` 这一层汇合，
    下游那条链路分不出手上这份资料是从哪来的（验收要求「走完全相同的后续链路」）。
    """

    def crawl(self, url: str) -> NormalizedDoc:
        """抓不了时抛 :class:`SourceError`。"""
        ...

    def image(self, url: str) -> bytes:
        """取一张图的原图字节。取不了时抛 :class:`SourceError`。

        与 :meth:`crawl` 分开、而不是让它顺手把图也带回来：一份资料里几十张图，
        抓哪些、要不要抓由归一化那一层定（见 :func:`fetch_images`），抓取层只管出网。

        **必须与抓页面受同一套约束**（robots、每主机限频、标明身份）：图常在另一台主机上
        （bwiki 的正文在 `wiki.biligame.com`、图在 `patchwiki.biligame.com`），
        对那一台就是一次新的请求——绕过这层等于悄悄多抓了一台站，而且不报错。
        """
        ...


@runtime_checkable
class Enricher(Protocol):
    """补图：给图片补一段可检索的文字、展开折叠块、二次 OCR 回填。

    进出都是 :class:`NormalizedDoc`——它在归一化与切分之间，两侧只认这一种形态。
    """

    def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
        """补完还是同一份 :class:`NormalizedDoc`。

        单张图补不上（取不到原图、识别失败、摘要调用失败）留一条日志接着走，
        不断整份资料的路；**整层用不了**（OCR 依赖没装）才抛异常，
        由导入编排器按文件兜住——那种情况下每一张图都会补不上。
        """
        ...


class MarkdownParser:
    """md／txt：读文本、挑出图片引用。

    标题不在这里定：文档标题由导入编排器从正文与文件名两头取（见 `ragamer.importing`），
    解析器只管把字节读成正文。
    """

    #: 认得的扩展名。
    SUFFIXES = (".md", ".markdown", ".txt")

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        if Path(source.filename).suffix.lower() not in self.SUFFIXES:
            raise _unsupported(source, self.SUFFIXES)
        markdown = _decode(source)
        return NormalizedDoc(markdown=markdown, images=image_refs(markdown))


class ParserRouter:
    """按扩展名把一份资料交给唯一一个适配器。

    四条来源各有自己的适配器，导入编排器只认 `SourceParser` 一个协议，挑适配器
    这件事收在这里——新增一种来源只往这组里加一个，编排器一行不动。
    """

    #: 认得的扩展名，构造时按那组适配器算出来（协议上要求它是类属性，
    #: 这里每一份实例的取值不同，所以是实例属性）。
    SUFFIXES: tuple[str, ...] = ()

    def __init__(self, parsers: Sequence[SourceParser]) -> None:
        if not parsers:
            raise ValueError("至少要有一个解析适配器，否则什么格式都读不了")
        suffixes = [suffix for parser in parsers for suffix in parser.SUFFIXES]
        # 重叠当场报出来：先命中的那个赢，另一条来源就成了摆设，而且不会有任何提示
        repeated = sorted({suffix for suffix in suffixes if suffixes.count(suffix) > 1})
        if repeated:
            raise ValueError(
                f"扩展名 {'、'.join(repeated)} 被两个解析适配器认领了：先命中的那个会把另一个盖掉"
            )
        self._parsers = tuple(parsers)
        self.SUFFIXES = tuple(suffixes)

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        suffix = Path(source.filename).suffix.lower()
        for parser in self._parsers:
            if suffix in parser.SUFFIXES:
                return parser.parse(source)
        raise _unsupported(source, self.SUFFIXES)


def _unsupported(source: SourceDocument, suffixes: Sequence[str]) -> UnsupportedSourceError:
    """不认识这个格式。认得的扩展名由调用方给：单独用某个解析器时认的是它自己那几个，
    经 `ParserRouter` 分发时认的是全部适配器的那几个。
    """
    return UnsupportedSourceError(
        f"{source.filename}：这个格式还没有对应的解析适配器"
        f"（现在只认 {'、'.join(suffixes)}）。"
        "PDF 与图片走 MinerU、网页走爬虫那一条入口"
    )


@dataclass(frozen=True)
class ImageRef:
    """正文里的一处图片引用。补图那一层按它取原图、按位置落笔。"""

    #: 引用地址：Markdown 的 `![](…)` 取括号里那段，HTML 的 `<img src="…">` 取 `src`。
    ref: str
    #: 这段引用在正文里**开始**的位置（`![` 的那个感叹号）。
    #: 补图要取它前面的那段文字当上下文，见 `ragamer.enriching`。
    start: int
    #: 这段引用在正文里**结束**的位置。二次 OCR 的文字插在它之后。
    end: int
    #: 替代文本。Markdown 的 `![alt](…)` 取方括号里那段，HTML 的 `<img>` 取 `alt`
    #: 属性；两边都可能没有，取不到就是空串。**空着就是「还没有人说明过这张图」**，
    #: 补图往那里写摘要；已经有内容的不覆盖。
    alt: str = ""


def image_refs_in(markdown: str) -> tuple[ImageRef, ...]:
    """正文里的图片引用，按出现顺序。两种形式都认。"""
    found = [
        ImageRef(match.group(3), match.start(), match.end(), match.group(1))
        for match in _MD_IMAGE.finditer(markdown)
    ]
    found += [
        html
        for match in _HTML_IMAGE.finditer(markdown)
        if (html := _html_image_ref(match)) is not None
    ]
    return tuple(sorted(found, key=lambda item: item.end))


def _html_image_ref(match: re.Match[str]) -> ImageRef | None:
    """一个 `<img>` 标签 → 一处引用。没有 `src` 的不是引用——取不到原图。"""
    tag = match.group(0)
    src = _HTML_SRC.search(tag)
    if src is None:
        return None
    alt = _HTML_ALT.search(tag)
    return ImageRef(
        src.group(1), match.start(), match.end(), alt.group(1) if alt is not None else ""
    )


def image_refs(markdown: str) -> tuple[str, ...]:
    """正文里引用到的图片地址，按出现顺序、去重前原样。"""
    return tuple(item.ref for item in image_refs_in(markdown))


#: MediaWiki 缩略图地址里的**显示宽度**：`…/thumb/<a>/<b>/<哈希>.png/18px-图标-丹药.png`。
#: 它是页面上那张图实际显示的宽度，一个行内图标与一张立绘的差别就落在这一段上。
_THUMB_WIDTH = re.compile(r"/(\d+)px-")

#: 显示宽度不超过它的图当行内图标：不下载、不占对象存储、也不调视觉模型。
#:
#: 实测 bwiki 一个词条的 127 张图里绝大多数落在 18／25／30px——那是表格与正文里的
#: 行内图标（`18px-图标-丹药.png`）；60px 与 130px 的是物品图，算正经内容，阈值取在
#: 两者之间。**这只是个经验值，要调先有评测集**（§11），所以留成一个常量。
ICON_MAX_PX = 32


def is_icon(ref: str) -> bool:
    """这个地址指的是不是一张行内图标。

    **只有带缩略宽度标记的地址判得了**，也就是 MediaWiki 那条路（wikitext 转换出来的
    外链）。别的来源没有尺寸信息，一律不当图标——MinerU 那条路上每个 `type == "image"`
    的条目都是真截图（T10 实测），那里也没有图标成灾的问题。
    """
    found = _THUMB_WIDTH.search(urlsplit(ref).path)
    return found is not None and int(found.group(1)) <= ICON_MAX_PX


def strip_image_refs(text: str) -> tuple[str, tuple[ImageRef, ...]]:
    """把图片地址从正文里摘走，只留替代文本；返回摘完的正文与各处引用的落点。

    地址留在正文里是有代价的：它占满切片的字数预算，还会进向量化——一个词条页上百个
    图标时，正文里大半是地址（实测 black_myth 库里 77.6% 的字符）。而地址唯一的用处是
    「答案里能显示原图」，那件事由切片自己的图片地址字段单独带着走
    （`ragamer.stores.base.Chunk.image_urls`）。

    **替代文本留下**：它是这张图唯一的可检索文本（`ragamer.enriching` 的视觉摘要正写在
    这里），连它一起去掉，图里没有文字的那些图就再也搜不到了。

    **在切分之前调用**：切完再摘的话，一条比切片上限还长的地址会被切分器从中间切开，
    两半各自留在相邻的两片正文里，存下一条取不到原图的坏地址，而且不报错。
    地址现在不进正文，切分器也就没有机会切到它。

    返回的 `ImageRef.end` 是在**结果正文**里的落点——与 :func:`image_refs_in` 的 `end`
    不同，那个是在入参正文里的。落点供切分器把地址分派到切出来的那几片上，
    见 `ragamer.chunking`。
    """
    found: list[tuple[int, re.Match[str], bool]] = [
        (match.start(), match, True) for match in _MD_IMAGE.finditer(text)
    ]
    found += [(match.start(), match, False) for match in _HTML_IMAGE.finditer(text)]
    found.sort(key=lambda item: item[0])

    out: list[str] = []
    refs: list[ImageRef] = []
    cursor = 0
    written = 0
    for start, match, markdown_ref in found:
        # 两种形式叠在同一处时以先出现的那个为准：剩下的半个已经不是引用了
        if start < cursor:
            continue
        if markdown_ref:
            alt, ref = match.group(1), match.group(3)
        else:
            src = _HTML_SRC.search(match.group(0))
            if src is None:
                # 没有 `src` 的 `<img>` 不是引用（`_html_image_ref` 同一条口径），原样留着
                continue
            ref = src.group(1)
            written_alt = _HTML_ALT.search(match.group(0))
            alt = written_alt.group(1) if written_alt is not None else ""
        out.append(text[cursor:start])
        written += start - cursor
        cursor = _end_of_ref(text, match, markdown_ref)
        out.append(alt)
        refs.append(ImageRef(ref, written, written + len(alt), alt))
        written += len(alt)
    out.append(text[cursor:])
    return "".join(out), tuple(refs)


def _end_of_ref(text: str, match: re.Match[str], markdown_ref: bool) -> int:
    """一处引用到哪一列为止。

    Markdown 那种的匹配到地址就停了（`_MD_IMAGE` 不收右括号），收尾的那个 `)` 因此要
    自己咽掉——不咽它就会留在正文里，`![](images/1.jpg)` 被摘成孤零零一个 `)`。
    `[![说明](图.jpg)](页面)` 这种嵌在链接里的图咽掉之后正好剩 `[说明](页面)`，
    是一个正常的链接，不必另外处理。
    """
    if markdown_ref and text[match.end() : match.end() + 1] == ")":
        return match.end() + 1
    return match.end()


def set_image_alt(markdown: str, alt_by_ref: Mapping[str, str]) -> str:
    """给替代文本空着的图片引用写上替代文本。两种形式都写。

    **不覆盖已经写好的替代文本**：那是作者或解析器给的说明，比模型现补的一段准，
    覆盖掉等于拿一个可能更差的描述换掉一个已经能用的。

    认不出的引用原样留着：一张图缺了不该让整份资料进不了库。
    """

    def markdown_ref(match: re.Match[str]) -> str:
        alt, spaces, ref = match.group(1), match.group(2), match.group(3)
        written = alt_by_ref.get(ref)
        if alt.strip() or not written:
            return match.group(0)
        # 地址在这一段匹配的末尾之后，`![…](` 与空格原样拼回去
        return f"![{written}]({spaces}{ref}"

    def html_ref(match: re.Match[str]) -> str:
        tag = match.group(0)
        src = _HTML_SRC.search(tag)
        written = None if src is None else alt_by_ref.get(src.group(1))
        if written is None:
            return tag
        existing = _HTML_ALT.search(tag)
        if existing is None:
            return _with_alt_attribute(tag, written)
        if existing.group(1).strip():
            return tag
        return f"{tag[: existing.start(1)]}{written}{tag[existing.end(1) :]}"

    return _HTML_IMAGE.sub(html_ref, _MD_IMAGE.sub(markdown_ref, markdown))


def _with_alt_attribute(tag: str, alt: str) -> str:
    """给一个没有 `alt` 的 `<img>` 标签补上它，位置在标签收尾之前。

    自己拼而不是找库：只多一个属性，为它引一个 HTML 解析器不划算。自闭合的要留一个
    空格与斜杠收尾，不把原来的标签改成另一种写法。
    """
    body = tag[:-1].rstrip()
    closing = ">"
    if body.endswith("/"):
        body, closing = body[:-1].rstrip(), " />"
    return f'{body} alt="{alt}"{closing}'


def rewrite_image_refs(markdown: str, mapping: Mapping[str, str]) -> str:
    """把正文里的图片地址换成对象 key，两种形式都换。认不出的原样留着，并留一条痕。

    留一条痕而不是抛错：一张图缺了不该让整份资料进不了库；但静默留着也不行——
    答案里会是一条坏图，而且没人知道为什么。
    """

    def markdown_ref(match: re.Match[str]) -> str:
        ref = match.group(3)
        key = mapping.get(ref)
        if key is None:
            return _unmapped(match.group(0), ref)
        # 地址是这一段匹配的末尾，砍掉它换成 key
        return match.group(0)[: -len(ref)] + key

    def html_ref(match: re.Match[str]) -> str:
        tag = match.group(0)
        src = _HTML_SRC.search(tag)
        if src is None:
            return tag
        ref = src.group(1)
        key = mapping.get(ref)
        if key is None:
            return _unmapped(tag, ref)
        # 只换地址那一段，标签的其余部分（alt、宽高）原样留着
        return f"{tag[: src.start(1)]}{key}{tag[src.end(1) :]}"

    return _HTML_IMAGE.sub(html_ref, _MD_IMAGE.sub(markdown_ref, markdown))


def _unmapped(matched: str, ref: str) -> str:
    """这一处引用没有对应的原图，原样留着。

    **只留 debug**：走到这里说明它没进 `mapping`，而「为什么没进来」只有
    :func:`fetch_images` 说得清（它才是出网取图的那一处）。两处各报一条 warning，
    同一张图会被说两遍，后一遍还说不出原因。
    """
    logger.debug("正文里的图片引用 %s 没有对应的原图，原样留着", ref)
    return matched


def fetch_images(doc: NormalizedDoc, *, crawler: PageCrawler) -> NormalizedDoc:
    """把正文里的外链图下载下来，变成附件交给 :func:`publish_assets` 发出去。

    **归一化那一层的契约靠它兑现**：模块文档写着「出了这一层，图片地址只有对象 key
    一种形态」，而网页与 md 这两种来源的图一直是外链，从来没兑现过——补图那一层是按
    对象 key 取原图的，这些图因此一张都没被补过。MinerU 的附件本来就到手了，
    这里补的是外链那一条。

    三种引用不动它：

    - **相对路径**（`![](images/a.png)`）：没有基准地址可取，下载不了。它与「下载失败」
      是同一种处境，处置也就一样——原样留着并留一条痕。
    - **行内图标**（:func:`is_icon`）：不值得为它占一份对象存储，也不值得为它调一次
      视觉模型。它在正文里仍是外链，答案里照样显示得出来。
    - **已经在附件里的**（MinerU 的产物）：跳过，别重复下载。

    一张图取不到不让整份资料失败（与补图那一层同一条规矩）：留一条痕接着走。

    这里是**图片地址的唯一一处出网**，所以「取不到」的痕也都在这一处：跳到的那张图为什么
    没有原图，只有这里说得清（相对路径、被站点挡了、超了字节上限、还是它本来就是个图标）。
    `rewrite_image_refs` 那边因此只留一条 debug——同一张图报两遍，第二遍还说不出原因。
    """
    known = {asset.name for asset in doc.assets}
    taken = {_key_name(asset) for asset in doc.assets}
    assets = list(doc.assets)
    for ref in dict.fromkeys(item.ref for item in image_refs_in(doc.markdown)):
        if ref in known:
            continue  # 解析产物自带的（MinerU 那条路），附件已经到手
        if is_icon(ref):
            logger.debug("%s：行内图标，不下载、也不补摘要", ref)
            continue
        if not _is_external(ref):
            logger.warning("%s：相对路径的图没有基准地址可取，正文里的引用原样留着", ref)
            continue
        try:
            data = crawler.image(ref)
        except SourceError as exc:
            logger.warning("%s：这张图取不到（%s），正文里的引用原样留着", ref, exc)
            continue
        if not data:
            logger.warning("%s：这张图是空的，正文里的引用原样留着", ref)
            continue
        name = _external_name(ref, taken)
        taken.add(name)
        assets.append(SourceAsset(name=ref, data=data, key_name=name))
    return replace(doc, assets=tuple(assets)) if len(assets) != len(doc.assets) else doc


def _is_external(ref: str) -> bool:
    return urlsplit(ref).scheme in ("http", "https")


#: 对象名的长度上限。地址最后一段可以是任意长，而对象名要能一眼看出是什么图。
_MAX_STEM_CHARS = 120

#: 对象名里排掉的字符：它们要么是路径分隔符、要么在 URL 片段里有特殊含义。
_UNSAFE_IN_NAME = re.compile(r"""[\s/\\?#%"'<>|:*]+""")


def _external_name(ref: str, taken: set[str]) -> str:
    """外链图在对象存储里的名字：地址最后一段（解码过），重名时再并上地址的短摘要。

    重名**必须在这里解决**：`image_key` 那层来源摘要只隔开不同资料之间的重名；
    同一份资料里两张不同的图落到同一个名字上时，后写的那张会把前一张原地覆盖掉，
    而且不报错。名字由地址算出来，所以重导同一份资料仍落在同一批 key 上。
    """
    raw = unquote(PurePosixPath(urlsplit(ref).path).name)
    cleaned = _UNSAFE_IN_NAME.sub("_", raw).strip("._") or "image"
    stem, dot, suffix = cleaned.rpartition(".")
    if not dot:  # 没有后缀：整个名字都是主干
        stem, dot, suffix = cleaned, "", ""
    stem = stem[:_MAX_STEM_CHARS]
    if f"{stem}{dot}{suffix}" not in taken:
        return f"{stem}{dot}{suffix}"
    digest = hashlib.blake2b(ref.encode("utf-8"), digest_size=4).hexdigest()
    return f"{stem}.{digest}{dot}{suffix}"


def _key_name(asset: SourceAsset) -> str:
    """这份附件在对象存储里的名字。"""
    return asset.key_name or _attachment_name(asset.name)


def publish_assets(
    doc: NormalizedDoc, objects: ObjectStore, *, game_id: str, digest: str
) -> NormalizedDoc:
    """把附件存进对象存储，正文与条目级结构的引用一并改指对象 key。

    这是归一化的最后一步：出了这一层，「图片地址」只有对象 key 一种形态，
    补图与展示都按 key 取。附件来自两头——解析产物自带的（MinerU），
    以及 :func:`fetch_images` 从外链拉下来的。两条都没有就是一次也不碰对象存储。

    **每个附件的对象名由 `_key_name` 一处定**：写入与按前缀清理要对得上，
    两处各推一遍名字，推岔了就是把图写进一个谁也清不掉的地方。
    """
    if not doc.assets:
        return doc
    mapping: dict[str, str] = {}
    for asset in doc.assets:
        key = image_key(game_id, digest, _key_name(asset))
        objects.put(key, asset.data, content_type=asset.content_type)
        mapping[asset.name] = key
    markdown = rewrite_image_refs(doc.markdown, mapping)
    return replace(
        doc,
        markdown=markdown,
        images=image_refs(markdown),
        content_list=tuple(_with_image_key(entry, mapping) for entry in doc.content_list),
        assets=(),
    )


def _attachment_name(name: str) -> str:
    """附件在对象 key 里的名字：去掉解析产物里那层 `images/`。

    对象本身已经放在 `images/<游戏>/…` 下面了，留着会拼成 `images/…/images/xxx.jpg`。
    """
    return name.removeprefix(f"{IMAGE_PREFIX}/")


def _with_image_key(entry: Mapping[str, Any], mapping: Mapping[str, str]) -> Mapping[str, Any]:
    """条目级结构里的 `img_path` 同样改指对象 key。

    二次 OCR 要按它取原图，留成解析产物里的相对路径就取不到了。
    """
    ref = entry.get("img_path")
    if not isinstance(ref, str) or ref not in mapping:
        return entry
    return {**entry, "img_path": mapping[ref]}


def _decode(source: SourceDocument) -> str:
    """按 UTF-8 读。带 BOM 的也认——Windows 记事本另存 UTF-8 会带 BOM。

    读不了就当场报出来：按 errors="replace" 硬读会把乱码当正文送进切分与向量化，
    整份资料从此查不出来，而且不报错。
    """
    try:
        return source.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceError(
            f"{source.filename} 不是 UTF-8 编码（{exc.reason}，第 {exc.start} 字节）。"
            "本项目一律按 UTF-8 读——Windows 记事本选「UTF-8」另存即可"
        ) from exc
