"""实验口径：二次 OCR 这一轮怎么记账。

这张票的验收标准之一是「范围与强度依据上一张票的实验结论确定」——依据就是这一轮量
出来的数。记错账就是拿着一个错的数去定范围，所以这几个口径单独测。
脚本本身在 `tools/` 下（不进包），由 pyproject 里的 `pythonpath` 加进导入路径。
"""

from __future__ import annotations

from pathlib import Path

from second_pass_ocr import ImageStat, report, run, summarize

from .conftest import FakeOcr


def test_没认出字的单独数出来():
    """「一个字都没认出来」和「整批都很好」是两回事，混进平均值里就看不出来了。"""
    summary = summarize([ImageStat("a", chars=300), ImageStat("b", chars=0)])

    assert (summary.total, summary.empty) == (2, 1)
    assert summary.median_chars == 150
    assert summary.empty_ratio == "50.0%"


def test_识别失败的算失败不算没认出来():
    """这两件事要分开：一个是引擎压根没跑成，一个是跑成了但图里真没字。"""
    summary = summarize([ImageStat("a", chars=300), ImageStat("b", error="识别不了")])

    assert (summary.failed, summary.empty) == (1, 0)
    assert summary.median_chars == 300
    # 分母只算跑成的那些，失败的不该把「没认出来」的比例冲淡
    assert summary.empty_ratio == "0.0%"


def test_一张都没跑成时不给比例():
    """分母是 0 时报 `0.0%` 会把「全挂了」读成「全都认出来了」。"""
    summary = summarize([ImageStat("a", error="引擎用不了")])

    assert summary.failed == 1
    assert summary.median_chars == 0
    assert "—" in summary.empty_ratio


def test_报告里有逐张明细与总账():
    text = report([ImageStat("a.png", chars=521, seconds=1.5)])

    assert "| a.png | 521 | 1.5 |" in text
    assert "| 1 | 0 | 0（0.0%） | 521 | 1.5 |" in text


def test_失败的那张在明细里点名而不是消失():
    text = report([ImageStat("a.png", error="这张图坏了")])

    assert "a.png" in text
    assert "这张图坏了" in text


def test_目录里的每一张图都跑_不挑大小图(tmp_path: Path):
    """范围就是每一个图片条目（T10 的结论），这一轮的口径与它一致。"""
    for name in ("a.png", "b.PNG", "note.txt"):
        (tmp_path / name).write_bytes(b"x")

    stats = run(tmp_path, FakeOcr("甲", "乙"))

    assert [stat.name for stat in stats] == ["a.png", "b.PNG"]
    assert [stat.chars for stat in stats] == [1, 1]
