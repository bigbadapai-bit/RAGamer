"""导入任务：提交就返回、后台跑、快照能看见每一条走到哪了。

这里验的是「跑起来之后界面看得到什么」，不是切分与打标本身——那一层各有各的测试。
时间上不赌运气：要卡住的地方用 Event 卡住，要放行的地方由测试放行。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import pytest

from ragamer.chunking import ChunkRules
from ragamer.importing import ImportStage, SourceKind
from ragamer.jobs import ImportJobs, JobSnapshot
from ragamer.sources import SourceDocument
from ragamer.stores.base import UNVERSIONED
from ragamer.stores.memory import InMemoryChunkStore
from ragamer.tagging import SubjectType, TagVocabulary
from ragamer.vectors.fake import FakeEmbedder

from .conftest import FakeCrawler
from .test_importing import PAGE_URL, SECOND_ARTICLE, WIKI_ARTICLE

RULES = ChunkRules(max_chars=200, min_chars=40, heading_density=0.02)
GAME = "black_myth"
VOCAB = TagVocabulary(
    subject_types=tuple(SubjectType), term_mapping={"妖王": SubjectType.CHARACTER}
)

#: 一次导入最多等这么久。够宽松，卡住时是「测试失败」而不是「测试永远不返回」。
TIMEOUT = 10.0


class Gate:
    """卡在向量化那一步，等测试放行。

    真的按时间睡是不行的：快一点慢一点的机器上结论会不一样，而这里要断言的是
    「跑着的时候快照里看得到它走到哪了」。
    """

    def __init__(self) -> None:
        self.opened = threading.Event()
        self.entered = threading.Event()

    def embed(self, texts: Sequence[str]):
        self.entered.set()
        assert self.opened.wait(timeout=TIMEOUT), "测试没放行，这一步卡住了"
        return FakeEmbedder().embed(texts)


def doc(name: str, text: str = WIKI_ARTICLE) -> SourceDocument:
    return SourceDocument(filename=name, data=text.encode("utf-8"))


def make_jobs(chunks=None, *, embedder=None, crawler=None, **kwargs) -> ImportJobs:
    from ragamer.importing import Importer

    return ImportJobs(
        Importer(
            chunks=chunks if chunks is not None else InMemoryChunkStore(),
            embedder=embedder if embedder is not None else FakeEmbedder(),
            crawler=crawler,
            rules=RULES,
            # 任务自己会给每一批挂回调，这里显式关掉默认的日志回调
            on_progress=None,
        ),
        **kwargs,
    )


def wait(jobs: ImportJobs, job_id: str, *, until=None) -> JobSnapshot:
    """等到快照满足条件为止。等不到就把最后那一眼抛出去，别让测试挂死。"""
    deadline = time.monotonic() + TIMEOUT
    seen = jobs.snapshot(job_id)
    while time.monotonic() < deadline:
        seen = jobs.snapshot(job_id)
        assert seen is not None
        if until is None:
            if seen.finished:
                return seen
        elif until(seen):
            return seen
        time.sleep(0.02)
    raise AssertionError(f"等超时了，最后看到的是 {seen}")


# --- 提交就返回 ---


def test_提交之后立刻能拿到快照_每条都列着还没跑():
    """页面上一提交就该看到这一批有哪些条目，而不是空白等整批跑完。"""
    jobs = make_jobs()

    job_id = jobs.submit([doc("甲.md"), doc("乙.md", SECOND_ARTICLE)], game_id=GAME, version="")

    snapshot = jobs.snapshot(job_id)
    assert snapshot is not None
    assert [item.source for item in snapshot.items] == ["甲.md", "乙.md"]
    assert [item.kind for item in snapshot.items] == [SourceKind.FILE, SourceKind.FILE]
    assert all(item.result is None for item in snapshot.items)
    wait(jobs, job_id)


def test_没这个任务号时给_None():
    """服务重启过、或者任务早被丢掉：页面要能说一句人话，而不是 500。"""
    assert make_jobs().snapshot("没有这个号") is None


def test_网址与文件混在一批里_种类分得清():
    jobs = make_jobs(crawler=FakeCrawler(**{PAGE_URL: WIKI_ARTICLE}))

    job_id = jobs.submit([doc("甲.md")], urls=[PAGE_URL], game_id=GAME, version="")

    snapshot = jobs.snapshot(job_id)
    assert snapshot is not None
    assert [(item.source, item.kind) for item in snapshot.items] == [
        ("甲.md", SourceKind.FILE),
        (PAGE_URL, SourceKind.URL),
    ]
    wait(jobs, job_id)


# --- 跑着的时候看得见进度 ---


def test_跑着的时候快照里看得到这一条走到了哪一步():
    gate = Gate()
    jobs = make_jobs(embedder=gate)

    job_id = jobs.submit([doc("甲.md")], game_id=GAME, version="")

    assert gate.entered.wait(timeout=TIMEOUT), "这一批没跑到向量化"
    running = jobs.snapshot(job_id)
    assert running is not None
    assert running.running and not running.finished
    assert running.items[0].stage is ImportStage.EMBED
    assert running.items[0].result is None

    gate.opened.set()
    done = wait(jobs, job_id)
    assert done.finished
    assert done.items[0].result is not None and done.items[0].result.ok


def test_两条的进度各算各的():
    """第一条卡在向量化时，第二条还停在归一化之前——快照按条记，不是整批一个状态。"""
    gate = Gate()
    jobs = make_jobs(embedder=gate)

    job_id = jobs.submit([doc("甲.md"), doc("乙.md", SECOND_ARTICLE)], game_id=GAME, version="")

    assert gate.entered.wait(timeout=TIMEOUT)
    running = jobs.snapshot(job_id)
    assert running is not None
    first, second = running.items
    assert first.stage is ImportStage.EMBED
    assert second.stage is None  # 还没轮到它
    gate.opened.set()
    wait(jobs, job_id)


def test_跑完的快照里每条都有结果():
    jobs = make_jobs()

    job_id = jobs.submit([doc("甲.md"), doc("乙.md", SECOND_ARTICLE)], game_id=GAME, version="")
    done = wait(jobs, job_id)

    assert done.finished and not done.running
    assert [item.result.ok for item in done.items] == [True, True]
    assert len(done.results) == 2


def test_一条失败不牵连这一批的其余():
    jobs = make_jobs()

    job_id = jobs.submit(
        [doc("甲.md"), SourceDocument(filename="攻略.pdf", data=b"%PDF-1.7")],
        game_id=GAME,
        version="",
    )
    done = wait(jobs, job_id)

    assert [item.result.ok for item in done.items] == [True, False]
    assert done.items[1].result.stage is ImportStage.NORMALIZE


# --- 排队与收拾 ---


def test_同时在跑的还是只有一批_后提交的排队():
    """并发跑只会互相拖慢，而且跨批次写同一份文档没人拦。所以排队，但都收下。"""
    gate = Gate()
    jobs = make_jobs(embedder=gate)

    first = jobs.submit([doc("甲.md")], game_id=GAME, version="")
    assert gate.entered.wait(timeout=TIMEOUT)
    second = jobs.submit([doc("乙.md", SECOND_ARTICLE)], game_id=GAME, version="")

    queued = jobs.snapshot(second)
    assert queued is not None and not queued.finished
    assert all(item.result is None for item in queued.items)

    gate.opened.set()
    wait(jobs, first)
    wait(jobs, second)


def test_跑完的任务留着供翻看_超出上限从旧的开始丢():
    jobs = make_jobs(history=2)

    ids = [jobs.submit([doc(f"{index}.md")], game_id=GAME, version="") for index in range(4)]

    wait(jobs, ids[-1])
    assert jobs.snapshot(ids[-1]) is not None
    assert jobs.snapshot(ids[0]) is None  # 最早那条被丢掉了


def test_在跑的任务不会被丢掉():
    """丢掉跑着的任务，那一批就没人在看它了——留着，宁可多占一点内存。"""
    gate = Gate()
    jobs = make_jobs(embedder=gate, history=1)

    first = jobs.submit([doc("甲.md")], game_id=GAME, version="")
    assert gate.entered.wait(timeout=TIMEOUT)
    for index in range(3):
        jobs.submit([doc(f"{index}.md")], game_id=GAME, version="")

    assert jobs.snapshot(first) is not None
    gate.opened.set()


# --- 换个页回来还认得出那一批 ---


def test_都跑完了就认领最后提交的那条():
    jobs = make_jobs()

    first = jobs.submit([doc("甲.md")], game_id=GAME, version="")
    wait(jobs, first)
    second = jobs.submit([doc("乙.md", SECOND_ARTICLE)], game_id=GAME, version="")
    wait(jobs, second)

    claimed = jobs.latest(GAME)
    assert claimed is not None and claimed.job_id == second


def test_认领时取正在跑的那条_不是排在后面的那条():
    """工作线程一次只跑一批：同时提交两批时显示排队那条，等于报一个假的进度。"""
    gate = Gate()
    jobs = make_jobs(embedder=gate)

    first = jobs.submit([doc("甲.md")], game_id=GAME, version="")
    assert gate.entered.wait(timeout=TIMEOUT)
    jobs.submit([doc("乙.md", SECOND_ARTICLE)], game_id=GAME, version="")

    claimed = jobs.latest(GAME)
    assert claimed is not None and claimed.job_id == first
    gate.opened.set()


def test_认领不越库_留空才不限库():
    """留空那一支是给没带库的导入页用的：总得让人看见还在跑的那批。"""
    jobs = make_jobs()

    job_id = jobs.submit([doc("甲.md")], game_id=GAME, version="")
    wait(jobs, job_id)

    assert jobs.latest("别的库") is None
    assert jobs.latest() is not None


def test_一条任务都没有时认领给_None():
    assert make_jobs().latest(GAME) is None


def test_整批没跑起来时任务带着原因收场():
    """给了网址却没接抓取器是接线错，不是某一条资料的错——但界面也得有个交代，
    不能让那一页永远转圈。"""
    jobs = make_jobs(crawler=None)

    job_id = jobs.submit(urls=[PAGE_URL], game_id=GAME, version="")
    done = wait(jobs, job_id)

    assert done.finished
    assert "抓取器" in done.error
    assert not done.running


def test_一批炸了工作线程还活着():
    """线程死了队列就永远堵着——后面的批次一条都跑不了，而且不报错。"""
    jobs = make_jobs(crawler=None)

    broken = jobs.submit(urls=[PAGE_URL], game_id=GAME, version="")
    good = jobs.submit([doc("甲.md")], game_id=GAME, version="")

    assert wait(jobs, broken).finished
    assert [item.result.ok for item in wait(jobs, good).items] == [True]


def test_进度照旧落日志(caplog):
    """快照是给页面看的，日志是这条路出问题时唯一留下的现场——两份都要有。

    只把 on_progress 换成快照那条的话，一批几分钟的导入在日志里就只剩「收了」与「跑完」，
    中途哪一条卡在哪一步回头看什么都没有。这里刻意不覆盖 `on_progress`，用的就是
    Importer 自己的默认实现（落日志）。
    """
    from ragamer.importing import Importer

    jobs = ImportJobs(Importer(chunks=InMemoryChunkStore(), embedder=FakeEmbedder(), rules=RULES))

    with caplog.at_level(logging.INFO):
        job_id = jobs.submit([doc("甲.md")], game_id=GAME, version="")
        wait(jobs, job_id)

    assert "导入 甲.md：[1/1] 归一化" in caplog.text


def test_跑完就把上传的字节丢掉():
    """一批几十上百 MB 的资料留在内存里等被清掉是白占——结果已经在每一条里了。

    字节有意不进快照（页面上用不到），所以要验只能直接看任务对象。这个测试盯的就是
    那份内部记账，不是外部行为。
    """
    jobs = make_jobs()

    job_id = jobs.submit([doc("甲.md")], game_id=GAME, version="")
    wait(jobs, job_id)

    assert jobs.snapshot(job_id).items[0].result is not None
    assert jobs._jobs[job_id].sources == ()
    assert jobs._jobs[job_id].urls == ()


@pytest.mark.parametrize("version", [UNVERSIONED, "2.0"])
def test_版本原样带给这一批(version):
    chunks = InMemoryChunkStore()
    jobs = make_jobs(chunks)

    job_id = jobs.submit([doc("甲.md")], game_id=GAME, version=version)
    wait(jobs, job_id)

    assert {chunk.version for chunk in chunks.fetch_document(GAME, "二郎神", version=version)} == {
        version
    }
