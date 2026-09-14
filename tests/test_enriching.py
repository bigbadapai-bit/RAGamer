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


def test_没有条目级结构时不展开折叠块():
    """md／txt 与网页里那对标记可能是作者真的在讲这个元素，拆掉就是改用户写的东西。

    **只有这一件事仍按来源分**：图片的补全三条来源一起管（见
    `test_外链来源的图也补摘要`）。
    """
    doc = make_doc(
        markdown="<details>\n<summary>text_image</summary>\n正文\n</details>\n",
        content_list=[],
    )

    enriched = make_enricher().enrich(doc)

    assert "<details>" in enriched.markdown
    assert "text_image" in enriched.markdown


def test_外链来源的图也补摘要():
    """网页与 md 的图以前是外链，按对象 key 取不到，一张都补不上。

    归一化那一层现在把外链也收进了对象存储（`ragamer.sources.fetch_images`），
    到了这里两者的输入是同一种形态：引用就是对象 key。
    """
    vision = FakeLlm("二郎神立绘，全身正面")

    enriched = make_enricher(vision=vision).enrich(
        make_doc(markdown=f"# 二郎神\n\n![]({A})\n", content_list=[])
    )

    assert f"![二郎神立绘，全身正面]({A})" in enriched.markdown
    assert len(vision.calls) == 1


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


def test_摘要带上图片前后的原文():
    """给上下文是为了让它认出「这是谁、这是哪一处」——没上下文时它只能猜，
    而猜错的游戏名会混进语料（实测同一张燕云截图被猜成过《永劫无间》与《逆水寒手游》）。"""
    vision = FakeLlm("血条 12000")
    doc = make_doc(
        markdown=f"# 二郎神\n\n第二阶段会先蓄力，\n\n![]({A})\n\n然后横扫。\n",
        content_list=[],
    )

    make_enricher(vision=vision).enrich(doc)

    prompt = vision.calls[0].messages[-1].content
    assert "第二阶段会先蓄力" in prompt
    assert "然后横扫" in prompt
    assert A not in prompt  # 地址自己不进提示词，它没有可读的信息


def test_上下文里不留别的图片地址():
    """wiki 页面的图是成片的，前后 100 字符里往往夹着邻居的地址。

    不摘的话那段上下文有一大半是地址，等于把「地址混进正文」这件事从提示词这条路
    放回来——而它正是这一层旁边刚摘掉的东西。
    """
    vision = FakeLlm("血条 12000")
    doc = make_doc(markdown=f"前面 ![邻居]({B}) 中间 ![]({A}) 结尾\n", content_list=[])

    make_enricher(vision=vision).enrich(doc)

    prompt = vision.calls[0].messages[-1].content
    assert B not in prompt
    # 邻居的**替代文本**留着：那是可读的字，有用
    assert "邻居" in prompt


def test_窗口边界切到的半截地址也不进上下文():
    """窗口是硬切的 100 字符，边界会从中间切断邻居的地址。

    在**原始正文**上取窗口再摘地址的话，半截地址认不出来就留下了——真跑出来过：
    给模型的是 `thumb/b/b1/77u19lle…png/18px-%E5%9B%BE%E6%A0%87.png` 这种半截 URL
    加一串百分号转义。上下文改在摘完地址的正文上取，被切断的地址就不可能露出来。
    """
    long_url = "https://cdn.test/" + "x" * 200 + ".png"
    vision = FakeLlm("血条 12000")
    # 邻居那张图够长，窗口的左边界正好落在它的地址当中
    doc = make_doc(markdown=f"![邻居]({long_url})![]({A})\n", content_list=[])

    make_enricher(put={A: b"png-a"}, vision=vision).enrich(doc)

    prompt = vision.calls[0].messages[-1].content
    assert "xxxx" not in prompt
    assert "https" not in prompt


def test_视觉摘要写进空的_alt():
    vision = FakeLlm("二郎神立绘，全身正面", "物品图标")

    enriched = make_enricher(vision=vision).enrich(make_doc())

    assert f"![二郎神立绘，全身正面]({A})" in enriched.markdown
    assert f'<img src="{B}" alt="物品图标" />' in enriched.markdown
    # 一次调用一张图，图跟着消息发出去
    assert len(vision.calls) == 2
    assert vision.calls[0].messages[-1].images[0].data == b"png-a"


def test_已有_alt_的图不覆盖也不调模型():
    """别人已经写好的说明比模型现补的一段准，覆盖等于拿更差的换掉能用的。"""
    vision = FakeLlm("不该被用到")
    doc = make_doc(markdown=f"![二郎神立绘]({A})\n")

    enriched = make_enricher(vision=vision).enrich(doc)

    assert f"![二郎神立绘]({A})" in enriched.markdown
    assert vision.calls == []


def test_HTML_引用的图把摘要写进_alt_属性():
    """表格内嵌的图正是「图里没有文字」概率最高的一类，不补就一点可检索的文本都没有。

    标签的其余部分不动：自闭合的斜杠要留在最后。
    """
    vision = FakeLlm("不该被用到")
    doc = make_doc(markdown=f'<table><tr><td><img src="{B}"/></td></tr></table>\n')

    enriched = make_enricher(vision=vision).enrich(doc)

    assert f'<img src="{B}" alt="不该被用到" />' in enriched.markdown


def test_HTML_引用已有_alt_属性时不覆盖():
    vision = FakeLlm("不该被用到")
    doc = make_doc(markdown=f'<img src="{B}" alt="物品图标" width="32">\n')

    enriched = make_enricher(vision=vision).enrich(doc)

    assert '<img src="' in enriched.markdown
    assert 'alt="物品图标"' in enriched.markdown
    assert 'width="32"' in enriched.markdown
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
    # 原图换成对象 key 之后落在切片的图片地址字段上，答案里展示得出来；正文里不再有它
    urls = [url for chunk in stored for url in chunk.image_urls]
    assert len(urls) == 1
    assert urls[0].startswith("images/blackmyth/")
    assert "![" not in content
    # 视觉摘要写进替代文本，摘走地址之后它是这张图唯一的可检索文本
    assert "一张论坛帖截图" in content
    # 图内文字进了切片，检索得到
    assert chunk_document(content), "正文切不出切片"
    assert any("躲横扫" in chunk.content for chunk in chunk_document(content))
