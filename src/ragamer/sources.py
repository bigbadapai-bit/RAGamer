"""归一化：把四条来源的原始资料变成同一种形态（ADR-0006）。

**来源差异只在这一层存在。** PDF 与图片走 MinerU、网页走爬虫、md／txt 直接读，
出了这一层只剩 `NormalizedDoc` 一种形态——切分与打标不必知道内容从哪来，
新增一种来源也只动它自己的适配器。本模块现在只有 md／txt 一条路（`MarkdownParser`），
另外两条在后面的票里接上。

`ImageEnricher` 的缝也开在这里：补图吃一份 `NormalizedDoc`、吐一份 `NormalizedDoc`，
位置在归一化与切分之间。它要处理的三件事（VLM 摘要进 alt、展开 MinerU 的 `<details>`
折叠块、二次 OCR 回填）见 docs/ARCHITECTURE.md §1.2 与 §1.3，真实实现由那一票接上。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: 图片引用：`![alt](地址)`。取地址，供补图那一层去取原图。
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")


def image_refs(markdown: str) -> tuple[str, ...]:
    """正文里的图片引用地址（`![alt](地址)`），按出现顺序，**不去重**。

    补图那一层据此去取原图，生成那一层据此把原图随答案带回去
    （`ragamer.answering`）——正则只有这一处，两边的口径不会各漂各的。
    """
    return tuple(_IMAGE_REF.findall(markdown))


@dataclass(frozen=True)
class SourceDocument:
    """一份待导入的原始资料：文件名 + 字节。

    收字节而不是路径：界面上传拿到的是字节，本地文件读进来也是字节，
    两条路在这一层汇合，解析适配器不必区分，也不必碰文件系统。
    """

    filename: str
    data: bytes


@dataclass(frozen=True)
class NormalizedDoc:
    """归一之后的文档。四条来源在这一层之后不再有区别。"""

    #: 正文。全部输入最终都变成它。
    markdown: str
    #: 正文里引用到的图片地址，按出现顺序、去重前原样。
    #: 补图那一层按它取原图做二次 OCR 与摘要。
    images: tuple[str, ...] = ()
    #: 解析适配器附带的条目级结构（MinerU 的 `content_list.json` 就是它）。
    #: 切分用不到它——正文已经在 `markdown` 里了；补图的二次 OCR 回填要用，
    #: 因为「哪个条目是图片」只有条目级结构说得清。md 来源留空。
    content_list: tuple[Mapping[str, Any], ...] = ()


class SourceError(Exception):
    """一份资料读不成 Markdown。信息里已点名是哪个文件。"""


class UnsupportedSourceError(SourceError):
    """这个格式还没有对应的解析适配器。"""


@runtime_checkable
class SourceParser(Protocol):
    """一份原始资料 → 归一后的文档。换来源只换这里。"""

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
        suffix = Path(source.filename).suffix.lower()
        if suffix not in self.SUFFIXES:
            supported = "、".join(self.SUFFIXES)
            raise UnsupportedSourceError(
                f"{source.filename}：这个格式还没有对应的解析适配器（现在只认 {supported}）。"
                "PDF 与图片走 MinerU、网页走爬虫，两条路都还没接上"
            )
        markdown = _decode(source)
        return NormalizedDoc(markdown=markdown, images=image_refs(markdown))


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
