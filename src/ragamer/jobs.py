"""导入任务：一批资料丢进后台跑，界面边跑边看进度。

导入是**分钟级**的一段（PDF 与图片要走 MinerU 的云端解析、图片还要本地二次 OCR、
网页要出网抓），压在请求里跑会有两个后果：请求那头一直转圈、什么都看不见；
而这一层是 `async def`，占着事件循环会把整个应用一起卡住——别的页面、别的端点
全都得排队等它。所以把这一段挪到后台线程，界面改成轮询一份**快照**。

三个取舍：

- **一次跑一批**。同时提交的几批在队列里排队，各自显示「排队中」。补图与向量化是
  CPU 密集的活，并发跑只会互相拖慢，而且两个批次同时写同一份文档时，本批之内的
  认领表拦不住跨批次的覆盖（见 `ragamer.importing` 的文档标识那条）。
- **状态在内存里**。服务重启就没了——但导入本身也活不过重启，落库只会留下一份
  写了一半的进度，不如干脆不留。跑完的任务留着供翻看，超过 `history` 条从旧的开始丢。
- **快照不可变**。工作线程持锁改状态，读的人拿到的是复制出来的一份，
  页面渲染期间状态怎么变都不会改到它手上这一份。
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from queue import SimpleQueue

from ragamer.importing import Importer, ImportResult, ImportStage, ProgressEvent, SourceKind
from ragamer.logging import get_logger
from ragamer.sources import SourceDocument
from ragamer.tagging import TagVocabulary

logger = get_logger(__name__)

#: 跑完的任务留几条供翻看。再多也只是占内存。
DEFAULT_HISTORY = 8


@dataclass(frozen=True)
class JobItem:
    """任务里的一条资料：它从哪来，走到哪了，结果是什么。

    `result` 为空表示还没轮到它、或者正在跑——这时看 `stage`。
    """

    source: str
    kind: SourceKind
    #: 已经进过的阶段，按先后。界面上的「走到：归一化 › 补图 › …」来自它。
    stages: tuple[ImportStage, ...] = ()
    result: ImportResult | None = None

    @property
    def stage(self) -> ImportStage | None:
        """正在跑的那一步。还没轮到、或者已经跑完都是 `None`。"""
        return self.stages[-1] if self.result is None and self.stages else None


@dataclass(frozen=True)
class JobSnapshot:
    """某个任务此刻的样子。**渲染用的就是它**，不再回头读任务对象。"""

    job_id: str
    game_id: str
    version: str
    items: tuple[JobItem, ...]
    #: 整批都跑完了（不管是全部成功还是各有各的失败）。
    finished: bool = False
    #: 整批压根没跑起来的原因（例如给了网址却没接抓取器）。它与单条的失败不是一回事。
    error: str = ""

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def results(self) -> tuple[ImportResult, ...]:
        """已经出结果的那些，按提交顺序。还没跑到的跳过。"""
        return tuple(item.result for item in self.items if item.result is not None)

    @property
    def running(self) -> bool:
        return not self.finished and not self.error


class _Job:
    """一个任务的内部状态。**只有工作线程写、别的线程读快照**，所以配一把锁。"""

    def __init__(
        self,
        job_id: str,
        items: Sequence[tuple[str, SourceKind]],
        *,
        sources: Sequence[SourceDocument],
        urls: Sequence[str],
        game_id: str,
        version: str,
        vocabulary: TagVocabulary | None,
    ) -> None:
        self.id = job_id
        self.sources = tuple(sources)
        self.urls = tuple(urls)
        self.game_id = game_id
        self.version = version
        self.vocabulary = vocabulary
        self._items = [(source, kind, (), None) for source, kind in items]
        self._finished = False
        self._error = ""
        self._lock = threading.Lock()

    def note(self, event: ProgressEvent) -> None:
        """进度回调。`file_number` 就是这一条在批次里的位置，从 1 起。"""
        with self._lock:
            index = event.file_number - 1
            if 0 <= index < len(self._items):
                source, kind, stages, result = self._items[index]
                self._items[index] = (source, kind, (*stages, event.stage), result)

    def finish(self, results: Sequence[ImportResult]) -> None:
        """一批跑完了。结果按提交顺序对上每一条。"""
        with self._lock:
            for index, result in enumerate(results):
                if index < len(self._items):
                    source, kind, stages, _ = self._items[index]
                    self._items[index] = (source, kind, stages, result)
            self._finished = True

    def fail(self, error: str) -> None:
        """整批没跑起来。这一条也要算「跑完了」，否则界面会一直转圈。"""
        with self._lock:
            self._error = error
            self._finished = True

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                job_id=self.id,
                game_id=self.game_id,
                version=self.version,
                items=tuple(JobItem(*item) for item in self._items),
                finished=self._finished,
                error=self._error,
            )


@dataclass
class ImportJobs:
    """导入任务的登记处：提交、排队、跑、按 id 查快照。

    做成对象而不是一组模块级的函数，是为了没有模块级单例（见 `tests/test_conventions.py`）：
    由组合根造一个、注入给页面，测试里换得掉。
    """

    importer: Importer
    #: 跑完的任务留几条。
    history: int = DEFAULT_HISTORY
    _jobs: OrderedDict[str, _Job] = field(default_factory=OrderedDict, init=False)
    _queued: SimpleQueue[_Job] = field(default_factory=SimpleQueue, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _worker: threading.Thread | None = field(default=None, init=False)

    def submit(
        self,
        sources: Sequence[SourceDocument] = (),
        *,
        urls: Sequence[str] = (),
        game_id: str,
        version: str,
        vocabulary: TagVocabulary | None = None,
    ) -> str:
        """收下一批，立刻返回任务号。**读上传字节已经在调用方做完了**（那是请求的一部分）。"""
        items = [(source.filename, SourceKind.FILE) for source in sources]
        items += [(url, SourceKind.URL) for url in urls]
        job = _Job(
            uuid.uuid4().hex[:12],
            items,
            sources=sources,
            urls=urls,
            game_id=game_id,
            version=version,
            vocabulary=vocabulary,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._forget_old()
        self._start_worker()
        self._queued.put(job)
        logger.info("导入任务 %s 收了 %d 条（知识库 %s）", job.id, len(items), game_id)
        return job.id

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        """这个任务现在什么样。没这个号（服务重启过、或者早就被丢掉了）返回 `None`。"""
        with self._lock:
            job = self._jobs.get(job_id)
        return None if job is None else job.snapshot()

    def _forget_old(self) -> None:
        """超出 history 就从最早的开始丢，**只丢跑完的**——在跑的丢掉就没人认得出它了。"""
        for job_id, job in list(self._jobs.items()):
            if len(self._jobs) <= self.history:
                return
            if job.finished:
                del self._jobs[job_id]

    def _start_worker(self) -> None:
        """第一次提交时才起线程：没导入过的进程不必挂一个空转的线程。"""
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._work, name="ragamer-import", daemon=True)
        self._worker.start()

    def _work(self) -> None:
        """工作线程：一批接一批地跑，**一批炸了不能让线程死掉**——死了队列就永远堵着。"""
        while True:
            job = self._queued.get()
            try:
                self._run(job)
            except Exception as exc:  # 兜住是这一层的职责，理由同上
                logger.error("导入任务 %s 挂了：%s", job.id, exc, exc_info=True)
                job.fail(str(exc))
            finally:
                # 跑完就收拾一次：攒着等下一次提交才清的话，一个长期只在收尾的进程
                # 会把跑完的任务一直堆在内存里
                with self._lock:
                    self._forget_old()

    def _run(self, job: _Job) -> None:
        # 只给这一批挂进度回调：Importer 是 frozen 的，replace 出来一份改回调
        importer = replace(self.importer, on_progress=job.note)
        results = importer.batch(
            job.sources,
            urls=job.urls,
            game_id=job.game_id,
            version=job.version,
            vocabulary=job.vocabulary,
        )
        job.finish(results)
        logger.info("导入任务 %s 跑完：%s", job.id, _summary(results))


def _summary(results: Sequence[ImportResult]) -> str:
    ok = sum(1 for result in results if result.ok)
    return f"{len(results)} 条里成功 {ok} 条"
