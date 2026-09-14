"""归一化：四条来源在这一层之后不再有区别。

纯函数、零假件——把字节读成正文、认出图片引用、认不出的格式当场报出来。
PDF／图片与网页两条路在后面的票里接上，这里的断言到时候对它们同样成立。
"""

from __future__ import annotations

import logging

import pytest

from ragamer.sources import (
    Enricher,
    MarkdownParser,
    NormalizedDoc,
    ParserRouter,
    SourceAsset,
    SourceDocument,
    SourceError,
    SourceParser,
    UnsupportedSourceError,
    fetch_images,
    image_refs,
    image_refs_in,
    is_icon,
    publish_assets,
    rewrite_image_refs,
    set_image_alt,
    strip_image_refs,
)
from ragamer.stores.base import image_key, image_prefix, is_image_key
from ragamer.stores.memory import InMemoryObjectStore

PARSER = MarkdownParser()

ARTICLE = """\
# 二郎神

![二郎神立绘](images/erlang.png)

二郎神是隐藏 BOSS。

![三尖两刃刀](images/weapon.png)
"""


def markdown(text: str = ARTICLE, filename: str = "二郎神.md") -> SourceDocument:
    return SourceDocument(filename=filename, data=text.encode("utf-8"))


def test_解析器与补图各自满足自己的协议():
    assert isinstance(PARSER, SourceParser)
    assert isinstance(NormalizedDoc("正文"), NormalizedDoc)


@pytest.mark.parametrize("filename", ["a.md", "a.markdown", "a.txt"])
def test_认得的扩展名大小写不敏感(filename):
    assert PARSER.parse(markdown("正文", filename.upper())).markdown == "正文"


def test_读出来的是正文_不解析标题():
    doc = PARSER.parse(markdown())

    assert doc.markdown == ARTICLE


def test_挑出正文里引用到的图片():
    doc = PARSER.parse(markdown())

    assert doc.images == ("images/erlang.png", "images/weapon.png")


def test_没有图片时图片列表为空():
    assert PARSER.parse(markdown("光有正文")).images == ()


def test_行内的图片引用也认():
    """`![alt](地址)` 出现在段落中间时同样要挑出来。"""
    doc = PARSER.parse(markdown("先看图 ![图](a.png) 再看后面"))

    assert doc.images == ("a.png",)


def test_条目级结构留给解析适配器填():
    """切分用不到它——正文已经在 `markdown` 里了；补图的二次 OCR 回填要用。"""
    assert PARSER.parse(markdown()).content_list == ()


def test_不认识这个格式时点名文件():
    with pytest.raises(UnsupportedSourceError) as excinfo:
        PARSER.parse(SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7"))

    assert "攻略.pdf" in str(excinfo.value)


def test_带_BOM_的也认():
    """Windows 记事本另存 UTF-8 会带 BOM，BOM 不该混进正文。"""
    source = SourceDocument(filename="a.md", data=b"\xef\xbb\xbf" + "# 二郎神".encode())

    assert PARSER.parse(source).markdown == "# 二郎神"


def test_不是_UTF8_时点名文件而不是读成乱码():
    """按 errors="replace" 硬读会把乱码当正文送进切分与向量化，整份资料从此查不出来。"""
    source = SourceDocument(filename="a.md", data="# 二郎神".encode("gbk"))

    with pytest.raises(SourceError) as excinfo:
        PARSER.parse(source)

    assert "a.md" in str(excinfo.value)
    assert "UTF-8" in str(excinfo.value)


def test_补图的协议进出都是归一化文档():
    """实现是 `ragamer.enriching.ImageEnricher`；这里的假件钉住缝的形状。"""

    class Passthrough:
        def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
            return doc

    assert isinstance(Passthrough(), Enricher)


# --- 图片引用的两种形式 ---


def test_HTML_形式引用的图片也认():
    """MinerU 把表格内嵌的图片导成 `<img>`，只认 Markdown 会静默漏掉它们。"""
    doc = PARSER.parse(markdown('<table><tr><td><img src="images/info.png"/></td></tr></table>'))

    assert doc.images == ("images/info.png",)


def test_两种形式的引用按出现顺序排():
    text = '![甲](a.png)\n\n<img src="b.png">\n\n![乙](c.png)\n'

    assert image_refs(text) == ("a.png", "b.png", "c.png")


def test_没有_src_的_img_不算引用():
    assert image_refs('<img class="icon" alt="图">') == ()


def test_两种形式的替代文本都读得出来():
    refs = image_refs_in('![甲](a.png)\n\n<img src="b.png" alt="乙">\n\n![](c.png)\n')

    assert [(item.ref, item.alt) for item in refs] == [
        ("a.png", "甲"),
        ("b.png", "乙"),
        ("c.png", ""),
    ]


def test_alt_在_src_前面也读得出来():
    """属性顺序不固定，只截到 `src` 结尾就看不到写在它前面的 `alt`。"""
    refs = image_refs_in('<img alt="甲" class="icon" src="a.png">')

    assert [(item.ref, item.alt) for item in refs] == [("a.png", "甲")]


# --- 行内图标 ---

#: bwiki 那种缩略图地址：路径最后一段带显示宽度。同一个哈希目录下，
#: 18px 那份是正文里的行内图标，130px 那份是物品图。
_THUMB = (
    "https://patchwiki.biligame.com/images/wukong/thumb/b/b1/abc.png/{}px-%E5%9B%BE%E6%A0%87.png"
)


def test_缩略图按显示宽度认出图标():
    assert is_icon(_THUMB.format(18)) is True
    assert is_icon(_THUMB.format(32)) is True


def test_物品图不算图标():
    """60／130px 是物品与武器的图，算正经内容——阈值取在 32，两者之间。"""
    assert is_icon(_THUMB.format(60)) is False
    assert is_icon(_THUMB.format(130)) is False


def test_没有尺寸信息的地址不当图标():
    """只有 MediaWiki 那条路判得了。判不了就不判——MinerU 那些条目都是真截图。"""
    assert is_icon("https://img1.gamersky.com/image2024/08/1303_S.jpg") is False
    assert is_icon("images/black_myth/ab12/a.jpg") is False


# --- 外链图收进对象存储 ---


class StubFetcher:
    """一个只会回字节的抓取器。没排过的地址当场炸——与 `FakeCrawler` 同一条口径。"""

    def __init__(self, **images: bytes) -> None:
        self._images = images
        self.requested: list[str] = []

    def image(self, url: str) -> bytes:
        self.requested.append(url)
        if url not in self._images:
            raise SourceError(f"{url}：假件里没有排这张图")
        return self._images[url]


def test_外链图下载下来变成本份资料的附件():
    doc = NormalizedDoc(markdown="正文 ![](https://cdn.example.com/a.png) 继续\n")
    fetcher = StubFetcher(**{"https://cdn.example.com/a.png": b"PNG"})

    collected = fetch_images(doc, crawler=fetcher)

    assert [asset.name for asset in collected.assets] == ["https://cdn.example.com/a.png"]
    assert [asset.data for asset in collected.assets] == [b"PNG"]
    assert [asset.key_name for asset in collected.assets] == ["a.png"]


def test_附件名与正文里的引用逐字一致():
    """`publish_assets` 靠这个名字对上正文里的那一处引用：规整过（解码百分号、
    去掉查询串）就对不上，图会连着引用一起悬空。"""
    url = "https://cdn.example.com/thumb/%E5%9B%BE.png?cb=123"
    doc = NormalizedDoc(markdown=f"![]({url})\n")

    collected = fetch_images(doc, crawler=StubFetcher(**{url: b"PNG"}))

    assert collected.assets[0].name == url
    # 对象名是解码过、去掉查询串的那一份：能读，也不会把 `?` 带进对象 key
    assert collected.assets[0].key_name == "图.png"


def test_图标不下载():
    """行内图标不值得占一份对象存储，也不值得为它调一次视觉模型。"""
    doc = NormalizedDoc(markdown=f"![]( {_THUMB.format(18)} )\n".replace(" ", ""))
    fetcher = StubFetcher(**{_THUMB.format(18): b"PNG"})

    collected = fetch_images(doc, crawler=fetcher)

    assert collected.assets == ()
    assert fetcher.requested == []


def test_相对路径的图下载不了_原样留着并留一条痕(caplog):
    doc = NormalizedDoc(markdown="![](images/a.png)\n")

    with caplog.at_level("WARNING"):
        collected = fetch_images(doc, crawler=StubFetcher())

    assert collected.assets == ()
    assert collected.markdown == doc.markdown
    assert "相对路径" in caplog.text


def test_一张图取不到不让整份资料失败(caplog):
    """与补图那一层同一条规矩：留一条痕接着走，其余照常收。"""
    doc = NormalizedDoc(
        markdown="![](https://cdn.example.com/a.png)\n![](https://cdn.example.com/b.png)\n"
    )
    fetcher = StubFetcher(**{"https://cdn.example.com/b.png": b"PNG"})

    with caplog.at_level("WARNING"):
        collected = fetch_images(doc, crawler=fetcher)

    assert [asset.key_name for asset in collected.assets] == ["b.png"]
    assert "a.png" in caplog.text


def test_已经带来附件的引用不重复下载():
    """MinerU 那条路的原图早在手上，名字对得上就跳过。"""
    doc = NormalizedDoc(
        markdown="![](images/a.png)\n",
        assets=(SourceAsset("images/a.png", b"PNG"),),
    )
    fetcher = StubFetcher()

    collected = fetch_images(doc, crawler=fetcher)

    assert collected.assets == doc.assets
    assert fetcher.requested == []


def test_同一份资料里两张不同的图重名时不会互相覆盖():
    """`image_key` 那层来源摘要只隔开不同资料之间的重名，资料内部的重名得在这里解决。"""
    first = "https://a.example.com/x/icon.png"
    second = "https://b.example.com/y/icon.png"
    doc = NormalizedDoc(markdown=f"![]({first})\n![]({second})\n")

    collected = fetch_images(doc, crawler=StubFetcher(**{first: b"ONE", second: b"TWO"}))
    names = [asset.key_name for asset in collected.assets]

    assert len(set(names)) == 2, names
    assert names[0] == "icon.png"  # 先来的用本名，后来的并上地址的短摘要
    assert names[1].startswith("icon.") and names[1].endswith(".png")


def test_重名时算出来的名字是稳定的():
    """名字由地址算出来，重导同一份资料落在同一批对象 key 上（幂等重导靠它）。"""
    first = "https://a.example.com/x/icon.png"
    second = "https://b.example.com/y/icon.png"
    doc = NormalizedDoc(markdown=f"![]({first})\n![]({second})\n")
    fetcher = StubFetcher(**{first: b"ONE", second: b"TWO"})

    once = fetch_images(doc, crawler=fetcher)
    again = fetch_images(doc, crawler=fetcher)

    assert [asset.key_name for asset in once.assets] == [asset.key_name for asset in again.assets]


def test_认得出对象_key_也认得出不是的():
    """补图那一层靠它把「收进来的原图」与「故意没收的、取不到的」分开。

    按段数判而不是按前缀：解析产物里的相对路径（`images/a.png`）也以同一个前缀
    开头，但它不是 key——真 key 的游戏与来源摘要那两层是必有的。
    """
    assert is_image_key(image_key("black_myth", "a1b2", "立绘.png")) is True

    assert is_image_key("images/a.png") is False  # 相对路径
    assert is_image_key("https://cdn.test/a.png") is False  # 外链
    assert is_image_key("images") is False


def test_外链与自带的附件一起发出去():
    """两步合起来的出口：正文里的引用全变成对象 key。"""
    objects = InMemoryObjectStore()
    url = "https://cdn.example.com/a.png"
    doc = NormalizedDoc(
        markdown=f"![](images/local.png)\n\n![]({url})\n",
        assets=(SourceAsset("images/local.png", b"LOCAL"),),
    )

    collected = fetch_images(doc, crawler=StubFetcher(**{url: b"REMOTE"}))
    published = publish_assets(collected, objects, game_id="black_myth", digest="ab12")

    keys = objects.list_keys(image_prefix("black_myth"))
    assert len(keys) == 2
    assert all(key in published.markdown for key in keys)
    assert "http" not in published.markdown
    assert published.assets == ()


# --- 摘走地址 ---


def test_摘地址时留下替代文本():
    """地址的用处只有回显（由切片自己的字段带着走），替代文本是这张图唯一的
    可检索文本——补图那一层的视觉摘要正写在这里。"""
    text, refs = strip_image_refs("先看图 ![三阶段立绘](a.png) 再往下打。\n")

    assert text == "先看图 三阶段立绘 再往下打。\n"
    assert [(item.ref, item.alt) for item in refs] == [("a.png", "三阶段立绘")]


def test_摘地址时连收尾的右括号一起摘():
    """`_MD_IMAGE` 匹配到地址就停，不收那个 `)`——不咽它正文里会留一个孤零零的括号。"""
    text, refs = strip_image_refs("![](a.png)")

    assert text == ""
    assert [item.ref for item in refs] == ["a.png"]


def test_摘地址时两种形式都认():
    text, refs = strip_image_refs('甲 ![乙](b.png) 丙 <img src="c.png" alt="丁"> 戊')

    assert text == "甲 乙 丙 丁 戊"
    assert [item.ref for item in refs] == ["b.png", "c.png"]


def test_嵌在链接里的图摘完还剩一个正常的链接():
    """攻略站常见写法：缩略图外面套一层指向大图的链接。"""
    text, refs = strip_image_refs("[![大图](https://x/a_S.jpg)](https://p/show?https://x/a.jpg)")

    assert text == "[大图](https://p/show?https://x/a.jpg)"
    assert [item.ref for item in refs] == ["https://x/a_S.jpg"]


def test_摘地址后的落点是结果正文里的位置():
    """切分器靠它把地址分派到切出来的那几片上，所以量的必须是**摘完**的正文。"""
    text, refs = strip_image_refs("前面 ![甲](a.png) 中间")

    assert text[: refs[0].end] == "前面 甲"


def test_没有图片时正文一字不动():
    text, refs = strip_image_refs('光有正文。\n\n<img class="icon">\n')

    assert text == '光有正文。\n\n<img class="icon">\n'
    assert refs == ()


# --- 写替代文本 ---


def test_给替代文本空着的引用写摘要():
    text = '![](a.png)\n\n<img src="b.png">\n'

    written = set_image_alt(text, {"a.png": "甲", "b.png": "乙"})

    assert written == '![甲](a.png)\n\n<img src="b.png" alt="乙">\n'


def test_已经写好的替代文本不动():
    """作者或解析器给的说明比模型现补的一段准，覆盖等于拿更差的换掉能用的。"""
    text = '![甲](a.png)\n\n<img src="b.png" alt="乙">\n'

    assert set_image_alt(text, {"a.png": "新甲", "b.png": "新乙"}) == text


def test_补_alt_时保留标签的其余部分():
    """宽高之类的属性不动；自闭合的斜杠要留在最后。"""
    text = '<img src="a.png" width="32" />'

    assert set_image_alt(text, {"a.png": "甲"}) == '<img src="a.png" width="32" alt="甲" />'


def test_没有摘要的引用原样留着():
    assert set_image_alt("![](a.png)", {}) == "![](a.png)"


def test_换掉两种形式的引用():
    text = '![甲](a.png)\n\n<img src="b.png">\n'

    assert rewrite_image_refs(text, {"a.png": "K1", "b.png": "K2"}) == (
        '![甲](K1)\n\n<img src="K2">\n'
    )


def test_认不出的引用原样留着并留一条痕(caplog):
    """静默留着就是答案里一条坏图，还没人知道为什么；抛错又太重——一张图不该毁一份资料。

    **痕只留 debug**：「为什么没有原图」只有 `fetch_images` 说得清（它才是出网取图的那
    一处），这里再报一条 warning 就是同一张图说两遍，后一遍还说不出原因。
    """
    text = "![甲](a.png)\n\n![乙](b.png)\n"

    # 级别点名给这个 logger：`setup_logging` 把项目根 logger 钉在 INFO，
    # 不点名的话这条 debug 根本不会被发出来（跑全量时上一条用例刚调用过它）
    with caplog.at_level(logging.DEBUG, logger="ragamer.sources"):
        rewritten = rewrite_image_refs(text, {"a.png": "K1"})

    assert rewritten == "![甲](K1)\n\n![乙](b.png)\n"
    assert "b.png" in caplog.text


# --- 按扩展名挑适配器 ---


class StubParser:
    """只认一种扩展名的假适配器：回什么由构造时给。"""

    def __init__(self, suffixes: tuple[str, ...], markdown: str) -> None:
        self.SUFFIXES = suffixes
        self.markdown = markdown

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        return NormalizedDoc(markdown=f"{self.markdown}:{source.filename}")


def test_按扩展名把资料交给对应的适配器():
    router = ParserRouter((StubParser((".md",), "md"), StubParser((".pdf",), "pdf")))

    assert router.parse(markdown("正文", "a.md")).markdown == "md:a.md"
    assert router.parse(SourceDocument("a.pdf", b"%PDF")).markdown == "pdf:a.pdf"


def test_扩展名大小写不敏感():
    router = ParserRouter((StubParser((".md",), "md"),))

    assert router.parse(markdown("正文", "A.MD")).markdown == "md:A.MD"


def test_不认识的格式点名全部认得的扩展名():
    router = ParserRouter((MarkdownParser(), StubParser((".pdf",), "pdf")))

    with pytest.raises(UnsupportedSourceError) as excinfo:
        router.parse(SourceDocument("a.docx", b"x"))

    message = str(excinfo.value)
    assert "a.docx" in message
    assert ".md" in message
    assert ".pdf" in message


def test_一个适配器都没有的_router_当场报错():
    with pytest.raises(ValueError):
        ParserRouter(())


def test_两个适配器认领同一个扩展名时当场报错():
    """先命中的那个赢，另一条来源就成了摆设——而且不会有任何提示。"""
    with pytest.raises(ValueError) as excinfo:
        ParserRouter((StubParser((".md", ".txt"), "甲"), StubParser((".txt",), "乙")))

    assert ".txt" in str(excinfo.value)
    assert ".md" not in str(excinfo.value)  # 只点重叠的那个


def test_路由器本身也是解析器():
    assert isinstance(ParserRouter((MarkdownParser(),)), SourceParser)


# --- 附件发布 ---


def scanned_doc() -> NormalizedDoc:
    """一份带附件的归一化文档：正文里 Markdown 与 HTML 各引用一张图。"""
    return NormalizedDoc(
        markdown='# 二郎神\n\n![立绘](images/a.png)\n\n<img src="images/b.png">\n',
        images=("images/a.png", "images/b.png"),
        content_list=(
            {"type": "text", "text": "二郎神"},
            {"type": "image", "img_path": "images/a.png"},
        ),
        assets=(
            SourceAsset("images/a.png", b"A", "image/png"),
            SourceAsset("images/b.png", b"B", "image/jpeg"),
        ),
    )


def test_附件进对象存储_引用改指对象_key():
    objects = InMemoryObjectStore()

    published = publish_assets(scanned_doc(), objects, game_id="black_myth", digest="d1")

    first = image_key("black_myth", "d1", "a.png")
    second = image_key("black_myth", "d1", "b.png")
    assert published.images == (first, second)
    assert f"![立绘]({first})" in published.markdown
    assert f'src="{second}"' in published.markdown
    assert published.content_list[1]["img_path"] == first
    # 附件已经发出去了，归一化文档里不再留着字节
    assert published.assets == ()
    assert objects.get(first) == b"A"
    assert objects.get(second) == b"B"
    # 对象名两级：游戏一级、来源文件一级——删库时按游戏那一级清一次就够
    assert objects.list_keys(image_prefix("black_myth")) == sorted([first, second])
    assert image_prefix("black_myth", "d1").startswith(image_prefix("black_myth"))


def test_没有附件时一次也不碰对象存储():
    class ExplodingStore:
        """碰一下就炸：没有附件时连一次 put 都不该发生。"""

        def put(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("没有附件的文档不该碰对象存储")

    doc = NormalizedDoc("光有正文")

    assert publish_assets(doc, ExplodingStore(), game_id="g", digest="d") is doc
