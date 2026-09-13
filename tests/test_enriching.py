"""补图：折叠块展开、二次 OCR 回填、视觉摘要。

三件事的范围都按 T10 的实验结论定（`docs/experiments/mineru-ocr.md`）：每个
`type == "image"` 的条目都做、不挑大小图；要能对付「整页一张大图」。

引擎与模型都换成假件：真 OCR 要装 `ocr` 组、真视觉模型要发网络请求，两者都不该
进默认测试。真实效果由 `tests/test_ocr.py` 与集成测试分别兜。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from ragamer.chunking import chunk_document
from ragamer.enriching import ImageEnricher, unfold
from ragamer.importing import Importer
from ragamer.llm import FakeLlm, LlmTimeout
from ragamer.ocr import OcrError, OcrUnavailable
from ragamer.sources import NormalizedDoc, SourceAsset, SourceDocument, image_refs
from ragamer.stores.memory import InMemoryChunkStore, InMemoryObjectStore
from ragamer.vectors.fake import FakeEmbedder

from .conftest import FakeOcr

#: 对象 key 的样子：`publish_assets` 之后，正文与条目级结构里的引用都是它。
A = "images/blackmyth/ab12/a.jpg"
B = "images/blackmyth/ab12/b.png"

#: 一份 MinerU 产物：一张 Markdown 引用的图（alt 空着，正是产物的样子——
#: 实验里 `image_caption` 全是空的），一张 HTML 引用的图（表格内嵌的图片就是这种）。
MARKDOWN = f"""\
# 二郎神攻略

![]({A})

<table><tr><td><img src="{B}"/></td></tr></table>
"""


def make_doc(
    markdown: str = MARKDOWN,
    *,
    content_list: Sequence[Mapping[str, Any]] | None = None,
    images: tuple[str, ...] | None = None,
) -> NormalizedDoc:
    entries = (
        [
            {"type": "text", "text": "二郎神攻略"},
            {"type": "image", "img_path": A, "bbox": [0, 0, 400, 800]},
            {"type": "image", "img_path": B, "bbox": [0, 0, 30, 30]},
        ]
        if content_list is None
        else content_list
    )
    return NormalizedDoc(
        markdown=markdown,
        images=image_refs(markdown) if images is None else images,
        content_list=tuple(entries),
    )


def make_enricher(
    *,
    objects: InMemoryObjectStore | None = None,
    ocr: FakeOcr | None = None,
    vision: Any = None,
    put: Mapping[str, bytes] | None = None,
) -> ImageEnricher:
    store = objects if objects is not None else InMemoryObjectStore()
    store.ensure_bucket()
    for key, data in (put if put is not None else {A: b"png-a", B: b"png-b"}).items():
        store.put(key, data)
    return ImageEnricher(
        objects=store,
        ocr=ocr if ocr is not None else FakeOcr("图内文字一", "图内文字二"),
        vision=vision,
    )


# ── 折叠块 ──


def test_折叠块的标记与摘要都不进正文():
    """`text_image` 是「这一块是图内文字」这句标签，不是内容——留着就是一个假词。"""
    folded = f"![立绘]({A})\n<details>\n<summary>text_image</summary>\n图内文字\n</details>\n"

    unfolded = unfold(folded)

    assert "图内文字" in unfolded
    for junk in ("<details>", "</details>", "<summary>", "text_image"):
        assert junk not in unfolded


def test_摘要里是真的正文时留着():
    """别的工具产出的 `<details>` 里，摘要是真的小节名——那丢了就找不回来。"""
    folded = "<details>\n<summary>安装步骤</summary>\n先装 Java\n</details>\n"

    unfolded = unfold(folded)

    assert "安装步骤" in unfolded
    assert "先装 Java" in unfolded
    assert "<summary>" not in unfolded


def test_没有条目级结构的产物原样返回():
    """md／txt 与网页的图要么是外链、要么作者自己写好了说明，不归这一层管。

    少了这个门，一次 md 导入会把每张外链图都当成「取不到的原图」报一遍警，
    而且会把用户正文里真的在讲 `<details>` 的那段拆掉。
    """
    doc = make_doc(
        markdown="<details>\n<summary>text_image</summary>\n正文\n</details>\n",
        content_list=[],
    )

    enriched = make_enricher().enrich(doc)

    assert enriched is doc


# ── 二次 OCR 回填 ──


def test_每个图片条目都做二次_OCR_不挑大小图():
    """比例是 100%，没有「小图跳过」的余地；跳过任何一张都是静默丢信息。

    第二个条目的 bbox 只有 30×30——正是架构文档里 MinerU 会跳过的那个尺寸。
    """
    ocr = FakeOcr("大图的文字", "小图的文字")

    enriched = make_enricher(ocr=ocr).enrich(make_doc())

    assert ocr.images == [b"png-a", b"png-b"]
    assert "大图的文字" in enriched.markdown
    assert "小图的文字" in enriched.markdown


def test_OCR_文字插在该图引用之后():
    enriched = make_enricher(ocr=FakeOcr("图内文字一", "图内文字二")).enrich(make_doc())
    lines = enriched.markdown.splitlines()

    assert lines[lines.index(f"![]({A})") + 1] == ""
    assert lines[lines.index(f"![]({A})") + 2] == "图内文字一"


def test_原图引用仍然留在正文里():
    """图本身要留着，答案里得能把原图展示出来（ADR-0006 的代价那一节）。"""
    enriched = make_enricher().enrich(make_doc())

    assert f"![]({A})" in enriched.markdown
    assert f'<img src="{B}"/>' in enriched.markdown
    assert enriched.images == (A, B)


def test_HTML_引用的图也回填():
    """MinerU 把表格内嵌的图片导成 `<img>`，只认 Markdown 那种会把它们静默漏掉。"""
    enriched = make_enricher(ocr=FakeOcr("图内文字一", "图内文字二")).enrich(make_doc())
    lines = enriched.markdown.splitlines()
    table = f'<table><tr><td><img src="{B}"/></td></tr></table>'

    assert lines[lines.index(table) + 2] == "图内文字二"


def test_条目自带文字时不再做_OCR():
    """MinerU 哪天真的把图内文字填进条目，就没必要再识别一遍。"""
    ocr = FakeOcr("不该被用到")
    doc = make_doc(
        content_list=[
            {"type": "image", "img_path": A, "content": "条目自带的文字"},
            {"type": "image", "img_path": B, "content": ""},
        ]
    )

    enriched = make_enricher(ocr=ocr).enrich(doc)

    assert "条目自带的文字" in enriched.markdown
    assert ocr.images == [b"png-b"]


def test_同一张图被两个条目指着时只识别一次():
    ocr = FakeOcr("图内文字")
    doc = make_doc(
        markdown=f"![立绘]({A})\n",
        content_list=[
            {"type": "image", "img_path": A},
            {"type": "image", "img_path": A},
        ],
    )

    enriched = make_enricher(ocr=ocr).enrich(doc)

    assert ocr.images == [b"png-a"]
    assert enriched.markdown.count("图内文字") == 1


def test_取不到原图时留痕不炸():
    """一张图缺了不该让整份资料进不了库；但也不能装作没这回事。"""
    enricher = make_enricher(ocr=FakeOcr("只有第二张有文字"), put={B: b"png-b"})

    enriched = enricher.enrich(make_doc())

    assert f"![]({A})" in enriched.markdown
    assert "只有第二张有文字" in enriched.markdown


def test_单张图识别失败不牵连其余():
    ocr = FakeOcr(OcrError("这张图坏了"), "第二张的文字")

    enriched = make_enricher(ocr=ocr).enrich(make_doc())

    assert "第二张的文字" in enriched.markdown
    assert f"![]({A})" in enriched.markdown


def test_引擎用不了时抛出去():
    """依赖没装是整份资料的事：在这里静默跳过，等于每张图的文字都悄悄没了。"""
    ocr = FakeOcr(OcrUnavailable("uv sync --extra ocr"))

    with pytest.raises(OcrUnavailable):
        make_enricher(ocr=ocr).enrich(make_doc())


def test_正文里找不到引用的文字接到文末():
    """走到这一步的每个字都是 MinerU 少给的那部分——检索得到好过只留在日志里。"""
    doc = make_doc(
        markdown="# 二郎神攻略\n\n正文\n",
        content_list=[{"type": "image", "img_path": A}],
    )

    enriched = make_enricher(ocr=FakeOcr("找不到落点的文字")).enrich(doc)

    assert enriched.markdown.rstrip().endswith("找不到落点的文字")
    assert "正文" in enriched.markdown


# ── 视觉摘要 ──


def test_视觉摘要写进空的_alt():
    vision = FakeLlm("二郎神立绘，全身正面")

    enriched = make_enricher(vision=vision).enrich(make_doc())

    assert f"![二郎神立绘，全身正面]({A})" in enriched.markdown
    # 一次调用一张图，图跟着消息发出去
    assert len(vision.calls) == 1
    assert vision.calls[0].messages[-1].images[0].data == b"png-a"


def test_已有_alt_的图不覆盖也不调模型():
    """别人已经写好的说明比模型现补的一段准，覆盖等于拿更差的换掉能用的。"""
    vision = FakeLlm("不该被用到")
    doc = make_doc(markdown=f"![二郎神立绘]({A})\n")

    enriched = make_enricher(vision=vision).enrich(doc)

    assert f"![二郎神立绘]({A})" in enriched.markdown
    assert vision.calls == []


def test_HTML_引用的图不写摘要():
    """`<img>` 没有可写摘要的位置——硬塞一个 alt 属性会把原来的标签改得残缺。"""
    vision = FakeLlm("不该被用到")
    doc = make_doc(markdown=f'<table><tr><td><img src="{B}"/></td></tr></table>\n')

    enriched = make_enricher(vision=vision).enrich(doc)

    assert f'<img src="{B}"/>' in enriched.markdown
    assert vision.calls == []


def test_没接视觉模型时只做二次_OCR():
    """没配视觉模型是常态（它要另有一个多模态模型），不该因此缺了图内文字。"""
    enriched = make_enricher(ocr=FakeOcr("图内文字一", "图内文字二")).enrich(make_doc())

    assert "图内文字一" in enriched.markdown
    assert f"![]({A})" in enriched.markdown


def test_摘要调用失败时留痕不炸():
    vision = FakeLlm(LlmTimeout("超时了"), "第二张的摘要")
    doc = make_doc(markdown=f"![]({A})\n\n![]({B})\n")

    enriched = make_enricher(vision=vision).enrich(doc)

    assert f"![]({A})" in enriched.markdown
    assert f"![第二张的摘要]({B})" in enriched.markdown


def test_摘要里的方括号被去掉():
    """替代文本夹在 `![` 与 `]` 之间，里面再出现方括号会把这一段 Markdown 拆掉。"""
    vision = FakeLlm("二郎神[立绘]\n全身正面")

    enriched = make_enricher(vision=vision).enrich(make_doc())

    assert f"![二郎神立绘 全身正面]({A})" in enriched.markdown


# ── 端到端：从归一化产物到切片 ──


class StubParser:
    """直接给一份归一化文档的假解析器（真去造 MinerU 的字节在 test_mineru.py 里验）。"""

    SUFFIXES = (".png",)

    def __init__(self, doc: NormalizedDoc) -> None:
        self.doc = doc

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        return self.doc


def test_独立上传的截图从解析到切片全走一遍():
    """票里的验收标准：独立上传的攻略截图，图内文字能被检索到。

    这条路上不该藏住任何接线错误——附件真的发进对象存储、真的按 key 取回来做二次
    OCR、文字真的进了切片的正文。
    """
    objects = InMemoryObjectStore()
    objects.ensure_bucket()
    # 解析器给的那份产物：正文、条目级结构与附件，三样都照 MinerU 的样子
    parsed = NormalizedDoc(
        markdown="# 论坛帖\n\n![](images/shot.png)\n",
        images=("images/shot.png",),
        content_list=({"type": "image", "img_path": "images/shot.png", "bbox": [0, 0, 1, 1]},),
        assets=(SourceAsset("images/shot.png", b"\x89PNG-shot"),),
    )
    chunks = InMemoryChunkStore()
    importer = Importer(
        chunks=chunks,
        embedder=FakeEmbedder(),
        parser=StubParser(parsed),
        objects=objects,
        enricher=ImageEnricher(
            objects=objects,
            ocr=FakeOcr("二郎神三阶段打法：第一阶段躲横扫"),
            vision=FakeLlm("一张论坛帖截图"),
        ),
    )

    result = importer.import_one(SourceDocument("shot.png", b"png"), game_id="blackmyth")

    assert result.ok, result.error
    stored = chunks.fetch_document("blackmyth", "论坛帖", version="")
    content = "\n".join(chunk.content for chunk in stored)
    # 原图换成对象 key 之后仍然留在正文里，答案里展示得出来
    assert f"![一张论坛帖截图]({image_refs(content)[0]})" in content
    assert image_refs(content)[0].startswith("images/blackmyth/")
    # 图内文字进了切片，检索得到
    assert chunk_document(content), "正文切不出切片"
    assert any("躲横扫" in chunk.content for chunk in chunk_document(content))
