"""二次 OCR：把 MinerU 判成「图片区域」的地方再识别一遍。

MinerU 不提取图片区域里的文字，这是它的产品决策（docs/ARCHITECTURE.md §1.2）。
T10 的实测把代价量了出来：22 张真实截图里 56 个图片条目**没有一个**给出文字，
其中 8 张几乎整页被判成一个图片区域——整页论坛帖那张抽出 0 字，而它的内容全在图里。
所以这一层不是可选项：不做，那些内容一个也进不了库，而且**不报错**。

**引擎必须是本地的，而且必须换一个。** 再走 MinerU 没有意义——它不识别图片区域
正是问题本身，把图原样再传一次只会得到同一张白纸。RapidOCR 带 PP-OCR 的中文模型、
随包装好、离线跑、不按次计费：一份资料几十张图（长图尤其如此），按次计费的云端接口
在这里既慢又贵。

**不给阈值留配置。** T10 的实验定下的是范围——每个 `type == "image"` 的条目都做、
不挑大小图——没有定下任何阈值。照 RapidOCR 自己的默认值走：等有评测集再谈调参，
先拍一个数只是把一个猜测换成另一个猜测（docs/ARCHITECTURE.md §11）。

`rapidocr` 与 `onnxruntime` 放在可选的 `ocr` 组里（`uv sync --extra ocr`）。
本模块因此对它做懒导入：没装也能 import，只在真的要用引擎时报错并说清怎么装。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ragamer.lazy import LazyModel
from ragamer.logging import get_logger

logger = get_logger(__name__)

#: 缺依赖时的提示。
_INSTALL_HINT = "二次 OCR 在可选的 ocr 组里，装法：uv sync --extra ocr"


class OcrError(Exception):
    """这张图识别不了。信息里说清了是哪一类失败。"""


class OcrUnavailable(OcrError):
    """引擎根本用不了：依赖没装、模型加载失败。

    **与「这张图认不出来」必须分开**：这一条成立时，一份资料里每一张图都拿不到文字，
    而那正是二次 OCR 要防的静默丢失——它该让整份资料带着原因失败，不是留一条痕接着走。
    """


@runtime_checkable
class OcrEngine(Protocol):
    """一张图 → 一段文字。换引擎只换这里。"""

    def read(self, data: bytes) -> str:
        """识别图里的文字，按检测顺序、每行一段。图里没有文字时返回空串。

        单张图识别不了时抛 :class:`OcrError`，引擎用不了时抛 :class:`OcrUnavailable`。
        """
        ...


class _Engine(Protocol):
    """`RapidOCR` 实例里我们用到的那一部分。"""

    def __call__(self, image: bytes) -> Any:
        """返回带 `txts` 字段的结果对象。"""
        ...


#: 引擎加载器。默认是真实加载，测试换掉它就不必装 onnxruntime。
OcrLoader = Callable[[], _Engine]


def load_rapidocr() -> _Engine:
    """真实加载 RapidOCR。模型随包带，不联网下载。"""
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OcrUnavailable(f"没有装 rapidocr。{_INSTALL_HINT}") from exc
    return RapidOCR()


class RapidOcrEngine:
    """本地 RapidOCR（PP-OCR 中文模型）。

    引擎在第一次真的要用时才加载，之后整个进程复用同一个实例——加载要几百毫秒，
    而一份长图资料会喂进来几十张图。
    """

    def __init__(self, *, loader: OcrLoader | None = None) -> None:
        load = loader if loader is not None else load_rapidocr
        self._lazy = LazyModel("二次 OCR 引擎", load)

    def read(self, data: bytes) -> str:
        result = self._run(data)
        return "\n".join(text for text in _texts(result) if text.strip())

    def _run(self, data: bytes) -> Any:
        """跑一次识别，把引擎的异常翻成项目自己的类型。

        引擎级与单张图级在这里分不开（同一个 `RapidOCRError` 两边都会抛），所以一律
        翻成 :class:`OcrError`，由调用方按「哪一张图」记下来——一份资料里绝大多数
        图片是正常的，为一张坏图让整份资料失败不划算。**依赖没装不在此列**：
        它在加载时就抛，是 `OcrUnavailable`，原样穿过去。
        """
        try:
            return self._lazy.get()(data)
        except OcrError:
            raise
        except Exception as exc:
            raise OcrError(f"这张图识别不了（{type(exc).__name__}：{exc}）") from exc


def _texts(result: Any) -> tuple[str, ...]:
    """取识别结果里的文本。

    RapidOCR 一个字都没认出来时给的是 `None`（不是空元组），结果对象本身也不保证
    有 `txts` 这个属性——两者都按「没有文字」处理，不让 `None` 漏成字符串进正文。
    """
    texts = getattr(result, "txts", None)
    if texts is None:
        return ()
    return tuple(text for text in texts if isinstance(text, str))
