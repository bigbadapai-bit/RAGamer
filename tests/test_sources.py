"""归一化：四条来源在这一层之后不再有区别。

纯函数、零假件——把字节读成正文、认出图片引用、认不出的格式当场报出来。
PDF／图片与网页两条路在后面的票里接上，这里的断言到时候对它们同样成立。
"""

from __future__ import annotations

import pytest

from ragamer.sources import (
    ImageEnricher,
    MarkdownParser,
    NormalizedDoc,
    SourceDocument,
    SourceError,
    SourceParser,
    UnsupportedSourceError,
)

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
    """真实实现要等二次 OCR 那一票；这里的假件钉住缝的形状。"""

    class Passthrough:
        def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
            return doc

    assert isinstance(Passthrough(), ImageEnricher)
