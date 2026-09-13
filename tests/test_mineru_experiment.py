"""实验口径：哪种图片条目算「有图但没提取出文字」。

那个比例决定 T12 的范围，口径算错就是拿着一个错数字去定范围——所以这一段单独测。
脚本本身在 `tools/` 下（不进包），由 pyproject 里的 `pythonpath` 加进导入路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mineru_ocr_experiment import FileStat, images_in, measure, report

from ragamer.mineru import MineruBundle

TEXT_ENTRY = {"type": "text", "text": "二郎神是隐藏 BOSS"}
IMAGE_ENTRY = {"type": "image", "img_path": "images/a.jpg", "image_caption": []}


def bundle(markdown: str, entries: list[dict]) -> MineruBundle:
    return MineruBundle(markdown=markdown, content_list=tuple(entries), images=())


def test_条目自带文字就算有文字():
    """pipeline 后端若能直接抽到图内文字，走的是这一路。"""
    stat = measure(
        "a.jpg", "pipeline", bundle("![](images/a.jpg)", [{**IMAGE_ENTRY, "text": "血量 12000"}])
    )

    assert (stat.image_entries, stat.with_entry_text, stat.missing) == (1, 1, 0)


def test_正文里的折叠块也算有文字():
    """vlm 后端走的是这一路：图内文字落在紧跟图片引用的 `<details>` 里。"""
    markdown = (
        "![](images/a.jpg)\n<details>\n<summary>text_image</summary>\n血量 12000\n</details>\n"
    )
    stat = measure("a.jpg", "vlm", bundle(markdown, [IMAGE_ENTRY]))

    assert (stat.with_details, stat.missing) == (1, 0)


def test_两处都没有才算有图无字():
    stat = measure("a.jpg", "vlm", bundle("![](images/a.jpg)\n\n后面是正文。", [IMAGE_ENTRY]))

    assert (stat.image_entries, stat.missing) == (1, 1)


def test_正文里根本没有这张图的引用也算有图无字():
    """条目说有这张图，正文里却找不到它——信息一样是取不到的。"""
    stat = measure("a.jpg", "vlm", bundle("只有正文，一张图都没有。", [IMAGE_ENTRY]))

    assert stat.missing == 1


def test_窗口之外的折叠块不算在这张图头上():
    """下一节的折叠块离得太远，算进来就会把比例压低，正是最坏的那种错。"""
    far = "![](images/a.jpg)\n" + "正文。" * 300 + "\n<details>\n<summary>text_image</summary>\n"
    stat = measure("a.jpg", "vlm", bundle(far, [IMAGE_ENTRY]))

    assert stat.missing == 1


def test_只数图片条目():
    stat = measure(
        "a.jpg", "vlm", bundle("![](images/a.jpg)", [TEXT_ENTRY, IMAGE_ENTRY, {"type": "table"}])
    )

    assert stat.image_entries == 1


def test_汇总按后端分别给比例():
    stats = [
        FileStat("甲.jpg", "vlm", image_entries=2, with_details=1),
        FileStat("乙.jpg", "vlm", image_entries=2, with_entry_text=1, with_details=1),
        FileStat("甲.jpg", "pipeline", image_entries=2),
    ]

    text = report(stats)

    assert "| vlm | 4 | 1 | **25.0%** |" in text
    assert "| pipeline | 2 | 2 | **100.0%** |" in text
    assert "甲.jpg" in text and "乙.jpg" in text


def test_解析失败的那一张照常进表_但不进分母():
    """一张图挂了不该让整轮白跑，但它也不该被当成「有图无字」混进比例里。"""
    stats = [
        FileStat("甲.jpg", "vlm", error="MinerU 解析失败（文件损坏）"),
        FileStat("乙.jpg", "vlm", image_entries=2),
    ]

    text = report(stats)

    assert "文件损坏" in text
    assert "另有 1 张解析失败" in text
    assert "| vlm | 2 | 2 | **100.0%**" in text


def test_一个图片条目都没有时不给比例():
    """分母是零：这时候给个 0% 会被读成「不用二次 OCR」，恰好是反的。"""
    text = report([FileStat("甲.jpg", "vlm")])

    assert "—（一个图片条目都没有）" in text


def test_只挑图片扩展名_顺序稳定(tmp_path: Path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.JPG").write_bytes(b"x")
    (tmp_path / "攻略.md").write_text("正文", encoding="utf-8")
    (tmp_path / "笔记.txt").write_text("正文", encoding="utf-8")

    assert [path.name for path in images_in(tmp_path)] == ["a.JPG", "b.png"]


def test_目录不存在时当场报错(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        images_in(tmp_path / "没有这个目录")
