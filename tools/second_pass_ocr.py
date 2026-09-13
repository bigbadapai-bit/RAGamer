"""T12 的实验：二次 OCR 到底能把多少图内文字拿回来。

    uv run python tools/second_pass_ocr.py <截图目录> [--out docs/experiments/second-pass-ocr.md]

T10 量出的那一半是「**必须**依赖二次 OCR 的比例 = 100%」（56/56 个图片条目，
`docs/experiments/mineru-ocr.md`），另一半留给了本票：**二次 OCR 出来多少能用**——
那也是那张票「限制」一节里点名要在这里看的。这个数只有真跑一遍才知道：识别质量取决于
引擎与素材，查文档问不出来。

跑的是 `ragamer.ocr` 那个引擎本身（可选的 `ocr` 组，先 `uv sync --extra ocr`）。
**不是另写一个识别**：实验的口径必须与生产那条路完全一致，否则量出来的数与线上无关。
要跑哪个引擎由 `ragamer.ocr` 决定，这里不另开参数。

素材用 T10 那同一批截图——两次实验要能对着看，换一批就比不上了。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mineru_ocr_experiment import images_in

from ragamer.logging import get_logger, setup_logging
from ragamer.ocr import OcrEngine, OcrError, RapidOcrEngine

logger = get_logger(__name__)


@dataclass(frozen=True)
class ImageStat:
    """一张截图的二次 OCR 结果。失败也是结果——一张图挂了不该让整轮白跑。"""

    name: str
    chars: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Summary:
    """一轮的总账。**没认出字的单独数出来**：它和「整批都很好」是两回事，
    混进平均值里就看不出来了。"""

    total: int
    empty: int
    failed: int
    chars: tuple[int, ...]
    seconds: float

    @property
    def median_chars(self) -> int:
        return int(statistics.median(self.chars)) if self.chars else 0

    @property
    def empty_ratio(self) -> str:
        done = self.total - self.failed
        return "—（一张都没跑成）" if done == 0 else f"{self.empty / done:.1%}"


def summarize(stats: Sequence[ImageStat]) -> Summary:
    done = [stat for stat in stats if stat.ok]
    return Summary(
        total=len(stats),
        empty=sum(1 for stat in done if stat.chars == 0),
        failed=sum(1 for stat in stats if not stat.ok),
        chars=tuple(stat.chars for stat in done),
        seconds=sum(stat.seconds for stat in stats),
    )


def run(directory: Path, engine: OcrEngine) -> list[ImageStat]:
    """目录里的截图逐张识别。**一张都不跳过**：范围就是每一个图片条目（T10 的结论）。"""
    files = images_in(directory)
    logger.info("截图 %d 张，逐张做二次 OCR", len(files))
    return [_one(engine, path) for path in files]


def _one(engine: OcrEngine, path: Path) -> ImageStat:
    start = time.monotonic()
    try:
        text = engine.read(path.read_bytes())
    except OcrError as exc:
        logger.error("%s：%s", path.name, exc)
        return ImageStat(name=path.name, error=str(exc))
    seconds = time.monotonic() - start
    logger.info("%s：%d 字，%.1f 秒", path.name, len(text), seconds)
    return ImageStat(name=path.name, chars=len(text), seconds=seconds)


def report(stats: Sequence[ImageStat]) -> str:
    """逐张明细 + 一张总账。生成的部分会被人工写的结论覆盖前，先整份重写。"""
    lines = ["| 截图 | 字数 | 耗时（秒） |", "| --- | --- | --- |"]
    for stat in stats:
        if not stat.ok:
            lines.append(f"| {stat.name} | 识别失败 | {_cell(stat.error)} |")
            continue
        lines.append(f"| {stat.name} | {stat.chars} | {stat.seconds:.1f} |")

    summary = summarize(stats)
    lines += [
        "",
        "## 汇总",
        "",
        "| 截图数 | 识别失败 | 一个字都没认出来 | 字数中位数 | 总耗时（秒） |",
        "| --- | --- | --- | --- | --- |",
        f"| {summary.total} | {summary.failed} | {summary.empty}（{summary.empty_ratio}） "
        f"| {summary.median_chars} | {summary.seconds:.1f} |",
    ]
    return "\n".join(lines) + "\n"


def _cell(value: str | None) -> str:
    """表格单元格里的东西：换行会把整张表冲散。"""
    return " ".join((value or "").split()).replace("|", "\\|")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="二次 OCR 出字量实测（T12）")
    parser.add_argument("directory", type=Path, help="放截图的目录（用 T10 那同一批）")
    parser.add_argument("--out", type=Path, default=None, help="把报告写成 markdown 存到这个路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("INFO")  # 一张图几秒到十几秒，进度得看得见
    try:
        stats = run(args.directory, RapidOcrEngine())
    except NotADirectoryError as exc:
        logger.error("%s", exc)
        return 1
    text = report(stats)
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        logger.info("报告写到 %s（重跑会整份覆盖，结论那几节要人工补回）", args.out)
    # 一张都没成，说明这条链路根本没通，别让人拿着空表当结论
    return 0 if any(stat.ok for stat in stats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
