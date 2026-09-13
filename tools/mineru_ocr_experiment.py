"""T10 的实验：真实攻略截图分别跑 pipeline 与 vlm，量出「必须依赖二次 OCR 的比例」。

    uv run python tools/mineru_ocr_experiment.py <截图目录> [--out docs/experiments/mineru-ocr.md]

**这个数字必须实测，不能凭文档推测**——它直接决定 T12（图片补全）的范围。
MinerU 不把图片区域里的文字 OCR 成正文（它的产品决策），但 vlm 后端带图内文字分析，
结果落在正文的 `<details>` 折叠块里；云端跑 vlm 时是否真的开着，属于未确证项
（docs/ARCHITECTURE.md §1.2）。两种后端差多少，只有真跑一遍才知道。

截图目录里放 20 张真实攻略截图：长图、表格截图、wiki 截图各若干。

**判定口径**（一个图片条目算「有文字」的两种情形）：

1. 条目自身带 `text` / `content` 字段——pipeline 后端若能直接抽到图内文字就是这种；
2. 正文里这张图的引用之后紧跟一个标着 `text_image` 的 `<details>` 折叠块——
   vlm 后端走的是这种（§1.2 那个例子的原样）。

两种都没有，才算「有图但没提取出文字」，也就是必须靠二次 OCR 补的那种。
第 2 条认的是标记而不只是 `<details>`：正文里别的折叠块（表格、公式）不算这张图的文字。

凭据从 `.env` 读（`RAGAMER_MINERU_API_KEY`），不从命令行传：命令行参数会进 shell 历史。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ragamer.config import ConfigError, Settings, get_settings
from ragamer.logging import get_logger, setup_logging
from ragamer.mineru import IMAGE_SUFFIXES, MineruError, MineruParser
from ragamer.sources import NormalizedDoc, SourceDocument

logger = get_logger(__name__)

#: 两个后端各跑一遍。只有 vlm 默认带图内文字分析。
BACKENDS = ("pipeline", "vlm")

#: 判定「这张图后面跟了提取出来的文字」的窗口（字符）。
#: 折叠块紧跟图片引用（§1.2 那个例子），取这么长够到它，又不会把后面正文里的算进来。
DETAILS_WINDOW = 500

#: 图内文字那个折叠块的标记。认它而不只是认 `<details>`：正文里别的折叠块
#: （表格、公式）不是这张图的文字，算进来会把比例压低。
DETAILS_MARK = "<summary>text_image</summary>"


@dataclass(frozen=True)
class FileStat:
    """一张截图在一个后端下的结果。失败也是结果——一张图挂了不该让整轮白跑。"""

    name: str
    backend: str
    image_entries: int = 0
    with_entry_text: int = 0
    with_details: int = 0
    error: str | None = None

    @property
    def missing(self) -> int:
        """有图但两处都没有文字的条数——**必须依赖二次 OCR 的就是它**。"""
        return self.image_entries - self.with_entry_text - self.with_details

    @property
    def ok(self) -> bool:
        return self.error is None


def measure(name: str, backend: str, doc: NormalizedDoc) -> FileStat:
    """一份解析产物里，图片条目各有多少带上了文字。"""
    entries = [entry for entry in doc.content_list if entry.get("type") == "image"]
    entry_text = details = 0
    for entry in entries:
        if str(entry.get("text") or entry.get("content") or "").strip():
            entry_text += 1
        elif _details_after(doc.markdown, str(entry.get("img_path") or "")):
            details += 1
    return FileStat(
        name=name,
        backend=backend,
        image_entries=len(entries),
        with_entry_text=entry_text,
        with_details=details,
    )


def _details_after(markdown: str, ref: str) -> bool:
    """这张图的引用后面跟了标着 `text_image` 的折叠块吗。

    只在引用之后的那一段窗口里找，而且认标记：窗口里碰巧有个表格折叠块，
    不能算成这张图提取到了文字——那正是把这个比例算歪的方式。
    """
    if not ref:
        return False
    start = markdown.find(ref)
    if start < 0:
        return False
    return DETAILS_MARK in markdown[start : start + len(ref) + DETAILS_WINDOW]


def run(
    directory: Path, settings: Settings, *, backends: Sequence[str] = BACKENDS
) -> list[FileStat]:
    """目录里的截图，每个后端各跑一遍。"""
    files = images_in(directory)
    logger.info(
        "截图 %d 张，后端 %s，一共 %d 次解析",
        len(files),
        "、".join(backends),
        len(files) * len(backends),
    )
    stats: list[FileStat] = []
    for backend in backends:
        parser = MineruParser(settings.mineru.model_copy(update={"model_version": backend}))
        for path in files:
            stats.append(_one(parser, path, backend))
    return stats


def images_in(directory: Path) -> list[Path]:
    """目录里的图片，按名字排序——两次运行的结果要能对得上。"""
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} 不是一个目录")
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _one(parser: MineruParser, path: Path, backend: str) -> FileStat:
    source = SourceDocument(filename=path.name, data=path.read_bytes())
    try:
        stat = measure(path.name, backend, parser.parse(source))
    except MineruError as exc:
        logger.error("%s · %s：%s", path.name, backend, exc)
        return FileStat(name=path.name, backend=backend, error=str(exc))
    logger.info(
        "%s · %s：图片条目 %d，有图无字 %d",
        path.name,
        backend,
        stat.image_entries,
        stat.missing,
    )
    return stat


def report(stats: Sequence[FileStat]) -> str:
    """一张表 + 每个后端一行的比例。比例是这张票要的那个数字。"""
    lines = [
        "| 截图 | 后端 | 图片条目 | 条目自带文字 | 折叠块里有文字 | **有图无字** |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for stat in stats:
        if not stat.ok:
            lines.append(f"| {stat.name} | {stat.backend} | 解析失败 | | | {_cell(stat.error)} |")
            continue
        lines.append(
            f"| {stat.name} | {stat.backend} | {stat.image_entries} | {stat.with_entry_text} "
            f"| {stat.with_details} | **{stat.missing}** |"
        )
    lines += [
        "",
        "## 汇总",
        "",
        "| 后端 | 图片条目 | 有图无字 | 必须依赖二次 OCR 的比例 |",
        "| --- | --- | --- | --- |",
    ]
    for backend in dict.fromkeys(stat.backend for stat in stats):
        rows = [stat for stat in stats if stat.backend == backend and stat.ok]
        entries = sum(stat.image_entries for stat in rows)
        missing = sum(stat.missing for stat in rows)
        failed = [stat for stat in stats if stat.backend == backend and not stat.ok]
        ratio = "—（一个图片条目都没有）" if entries == 0 else f"{missing / entries:.1%}"
        note = f"（另有 {len(failed)} 张解析失败）" if failed else ""
        lines.append(f"| {backend} | {entries} | {missing} | **{ratio}**{note} |")
    return "\n".join(lines) + "\n"


def _cell(value: str | None) -> str:
    """表格单元格里的东西：换行会把整张表冲散。"""
    return " ".join((value or "").split()).replace("|", "\\|")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MinerU 图内文字提取对比实验（T10）")
    parser.add_argument("directory", type=Path, help="放截图的目录")
    parser.add_argument("--out", type=Path, default=None, help="把报告写成 markdown 存到这个路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("INFO")  # 一轮要跑几十次云端解析，进度得看得见
    try:
        settings = get_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2  # 与 `ragamer` 那条命令同一个含义：配置有问题
    try:
        stats = run(args.directory, settings)
    except NotADirectoryError as exc:
        logger.error("%s", exc)
        return 1
    text = report(stats)
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        logger.info("报告写到 %s", args.out)
    # 一张都没成，说明这条链路根本没通，别让人拿着空表当结论
    return 0 if any(stat.ok for stat in stats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
