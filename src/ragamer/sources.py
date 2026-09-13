"""归一化：把四条来源的原始资料变成同一种形态（ADR-0006）。

**来源差异只在这一层存在。** PDF 与图片走 MinerU（`ragamer.mineru`）、md／txt 直接读
（`MarkdownParser`）、网页走爬虫，出了这一层只剩 `NormalizedDoc` 一种形态——
切分与打标不必知道内容从哪来，新增一种来源也只动它自己的适配器。适配器由
`ParserRouter` 按扩展名挑，爬虫那条在后面的票里接上。

归一化的**最后一步是发布附件**（`publish_assets`）：解析产物里的原图进对象存储，
正文里的引用改指对象 key。出了这一层，图片地址只有对象 key 一种形态。

`ImageEnricher` 的缝也开在这里：补图吃一份 `NormalizedDoc`、吐一份 `NormalizedDoc`，
位置在归一化与切分之间。它要处理的三件事（VLM 摘要进 alt、展开 MinerU 的 `<details>`
折叠块、二次 OCR 回填）见 docs/ARCHITECTURE.md §1.2 与 §1.3，真实实现由那一票接上。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragamer.logging import get_logger
from ragamer.stores.base import IMAGE_PREFIX, ObjectStore, image_key

logger = get_logger(__name__)

#: 图片引用的两种形式。取地址，供补图那一层去取原图。
#: **两种都要认**：MinerU 把表格内嵌的图片导成 HTML 的 `<img>`，只认 Markdown 那种
#: 会把它们静默漏掉——补图那一层于是取不到这些原图，而且不报错。
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")
_HTML_IMAGE = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)


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
    """解析产物里附带的二进制附件。MinerU 结果包里的原图就是它。

    `name` 是正文引用它时用的那个相对路径（`images/xxx.jpg`）——它同时是
    「正文里的哪一处引用该改指哪个对象」的依据。
    """

    name: str
    data: bytes
    content_type: str = "application/octet-stream"


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
class ImageEnricher(Protocol):
    """补图：给图片补一段可检索的文字、展开折叠块、二次 OCR 回填。

    进出都是 :class:`NormalizedDoc`——它在归一化与切分之间，两侧只认这一种形态。
    """

    def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
        """读不了某张图时抛 :class:`SourceError`，由导入编排器按文件兜住。"""
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
        "PDF 与图片走 MinerU、网页走爬虫；后者还没接上"
    )


def image_refs(markdown: str) -> tuple[str, ...]:
    """正文里引用到的图片地址，按出现顺序、去重前原样。"""
    found = [(match.start(), match.group(1)) for match in _MD_IMAGE.finditer(markdown)]
    found += [(match.start(), match.group(2)) for match in _HTML_IMAGE.finditer(markdown)]
    return tuple(ref for _, ref in sorted(found))


def rewrite_image_refs(markdown: str, mapping: Mapping[str, str]) -> str:
    """把正文里的图片地址换成对象 key，两种形式都换。认不出的原样留着，并留一条痕。

    留一条痕而不是抛错：一张图缺了不该让整份资料进不了库；但静默留着也不行——
    答案里会是一条坏图，而且没人知道为什么。
    """

    def markdown_ref(match: re.Match[str]) -> str:
        ref = match.group(1)
        key = mapping.get(ref)
        if key is None:
            return _unmapped(match.group(0), ref)
        # 地址是这一段匹配的末尾，砍掉它换成 key
        return match.group(0)[: -len(ref)] + key

    def html_ref(match: re.Match[str]) -> str:
        ref = match.group(2)
        key = mapping.get(ref)
        if key is None:
            return _unmapped(match.group(0), ref)
        return f"{match.group(1)}{key}{match.group(3)}"

    return _HTML_IMAGE.sub(html_ref, _MD_IMAGE.sub(markdown_ref, markdown))


def _unmapped(matched: str, ref: str) -> str:
    logger.warning("正文里的图片引用 %s 在解析产物里找不到对应文件，原样留着", ref)
    return matched


def publish_assets(
    doc: NormalizedDoc, objects: ObjectStore, *, game_id: str, digest: str
) -> NormalizedDoc:
    """把解析产物里的附件存进对象存储，正文与条目级结构的引用一并改指对象 key。

    这是归一化的最后一步：出了这一层，「图片地址」只有对象 key 一种形态，
    补图与展示都按 key 取。md／txt 没有附件，原样返回，一次也不碰对象存储。
    """
    if not doc.assets:
        return doc
    mapping: dict[str, str] = {}
    for asset in doc.assets:
        key = image_key(game_id, digest, _attachment_name(asset.name))
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
