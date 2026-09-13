"""二次 OCR 适配器：认出来了、没认出字、引擎用不了，三条路各是什么行为。

跑这里**不需要装 rapidocr**——适配器只在真的要用引擎时才 import 它，测试注入一个
假引擎就绕开了。真实引擎认中文攻略截图的效果要在装了 `ocr` 组之后另跑，
那部分不进默认测试。
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from ragamer.ocr import OcrError, OcrUnavailable, RapidOcrEngine, load_rapidocr


class _Result:
    """假引擎的返回值。形状照 `RapidOCROutput` 的样子，只用得到 `txts` 一个字段。

    **什么都没认出来时 `txts` 是 `None`**，不是空元组——这是 RapidOCR 的行为。
    """

    def __init__(self, txts: Any) -> None:
        self.txts = txts


class _Engine:
    """假 RapidOCR：返回固定的一批文本，并记下每一次喂进来的字节。"""

    def __init__(self, txts: Any) -> None:
        self.txts = txts
        self.images: list[bytes] = []

    def __call__(self, image: bytes) -> _Result:
        self.images.append(image)
        return _Result(self.txts)


class _Loader:
    """记下加载了几次。"""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        return self.engine


def _engine(txts: Any = ("第一行", "第二行")) -> tuple[RapidOcrEngine, _Engine, _Loader]:
    engine = _Engine(txts)
    loader = _Loader(engine)
    return RapidOcrEngine(loader=loader), engine, loader


def test_识别结果按检测顺序拼成文本():
    ocr, engine, _ = _engine(("寒江雪", "攻击力 120"))

    assert ocr.read(b"png-bytes") == "寒江雪\n攻击力 120"
    assert engine.images == [b"png-bytes"]


def test_图里一个字都没认出来时返回空串():
    """RapidOCR 没认出来时给的是 `None`；不能让它漏成字符串 "None" 进正文。"""
    ocr, _, _ = _engine(None)

    assert ocr.read(b"png-bytes") == ""


def test_空白结果不进正文():
    """识别出一串空格没有意义，进了正文还会把图片那一行顶走。"""
    ocr, _, _ = _engine(("", "   ", "正文"))

    assert ocr.read(b"png-bytes") == "正文"


def test_引擎只加载一次():
    """一份资料几十张图，加载一次几百毫秒，每次都重来会把导入拖垮。"""
    ocr, _, loader = _engine()

    ocr.read(b"a")
    ocr.read(b"b")

    assert loader.calls == 1


def test_引擎认不出这张图时抛_OcrError():
    class _Boom:
        def __call__(self, image: bytes) -> Any:
            raise ValueError("cannot identify image file")

    ocr = RapidOcrEngine(loader=lambda: _Boom())

    with pytest.raises(OcrError, match="识别不了"):
        ocr.read(b"not-an-image")


def test_引擎用不了时抛_OcrUnavailable():
    """依赖没装是**引擎级**的失败，不是「这一张图认不出来」。

    两者的处置完全不同：单张图失败留一条痕接着走，引擎用不了则这一份资料的每一张图
    都拿不到文字——那正是二次 OCR 要防的静默丢失，必须抛出去。
    """

    def loader() -> Any:
        raise OcrUnavailable("没装 rapidocr")

    with pytest.raises(OcrUnavailable):
        RapidOcrEngine(loader=loader).read(b"png-bytes")


def test_没装_rapidocr_时报出装法(monkeypatch: pytest.MonkeyPatch):
    """光秃秃的 ImportError 会让人自己猜要装什么。"""
    monkeypatch.setitem(sys.modules, "rapidocr", None)

    with pytest.raises(OcrUnavailable, match="uv sync --extra ocr"):
        load_rapidocr()
