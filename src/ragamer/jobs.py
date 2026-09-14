"""导入任务：一批资料丢进后台跑，界面边跑边看进度。

导入是**分钟级**的一段（PDF 与图片要走 MinerU 的云端解析、图片还要本地二次 OCR、
网页要出网抓），压在请求里跑会有两个后果：请求那头一直转圈、什么都看不见；
而这一层是 `async def`，占着事件循环会把整个应用一起卡住——别的页面、别的端点
全都得排队等它。所以把这一段挪到后台线程，界面改成轮询一份**快照**。

三个取舍：

- **页面那几批一次跑一批**。同时提交的几批在队列里排队，各自显示「排队中」——补图与
  向量化是 CPU 密集的活，并发跑只会互相拖慢。**这条只管页面这条路**：JSON 端点那条是
  同步调用方自己等结果，它跑在自己的线程池里，与页面上的批次可能同时进行。跨批次同时
  写同一份文档也没人拦（认领表只在一次提交之内，见 `ragamer.importing` 的文档标识那条），
  单机自用可以接受，多用户部署要先解决这个。
- **状态在内存里**。服务重启就没了——但导入本身也活不过重启，落库只会留下一份
  写了一半的进度，不如干脆不留。跑完的任务留最近 `history` 条供翻看（**上传的字节
  跑完就丢**，留着的只有结果），再早的连同它的任务号一起消失。
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

from ragamer.importing import (
    Importer,
    ImportResult,
    ImportStage,
    ProgressCallback,
    ProgressEvent,
    SourceKind,
)
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
    #: 还没轮到它：工作线程一次跑一批，它排在别人后面。界面据此把它标成「排队中」，
    #: 而不是与刚被取走的那批混为一谈（两者的条目上都没有任何进度事件）。
    queued: bool = False
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
        self._started = False
        self._error = ""
        self._lock = threading.Lock()

    def note(self, event: ProgressEvent) -> None:
        """进度回调。`file_number` 就是这一条在批次里的位置，从 1 起。"""
        with self._lock:
            index = event.file_number - 1
            if 0 <= index < len(self._items):
                source, kind, stages, result = self._items[index]
                self._items[index] = (source, kind, (*stages, event.stage), result)

    def start(self) -> None:
        """工作线程把它取走了。**「排队中」与「刚开始跑」在快照上分得开**靠的就是这一笔。

        两者都没有任何进度事件，光看条目分不出：刚被取走的那批界面上还没走到第一步，
        说它「排队中」是错的（它前面已经没有别人了）。
        """
        with self._lock:
            self._started = True

    def finish(self, results: Sequence[ImportResult]) -> None:
        """一批跑完了。结果按提交顺序对上每一条。"""
        with self._lock:
            for index, result in enumerate(results):
                if index < len(self._items):
                    source, kind, stages, _ = self._items[index]
                    self._items[index] = (source, kind, stages, result)
            self._finished = True
            self._drop_payload()

    def fail(self, error: str) -> None:
        """整批没跑起来。这一条也要算「跑完了」，否则界面会一直转圈。"""
        with self._lock:
            self._error = error
            self._finished = True
            self._drop_payload()

    def _drop_payload(self) -> None:
        """跑完就把上传的字节丢掉。**只有调用方持锁时才调**。

        结果已经在每一条的 `result` 里了，字节没人再看；留着一批几十上百 MB 的资料
        在内存里等被清掉，是白占——界面要重试时本来也得让人重新选文件。
        """
        self.sources = ()
        self.urls = ()

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def queued(self) -> bool:
        """还没轮到它：工作线程没取走、也没跑完。"""
        with self._lock:
            return not self._started and not self._finished

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                job_id=self.id,
                game_id=self.game_id,
                version=self.version,
                items=tuple(JobItem(*item) for item in self._items),
                finished=self._finished,
                queued=not self._started and not self._finished,
                error=self._error,
            )


@dataclass
class ImportJobs:
    """导入任务的登记处：提交、排队、跑、按 id 或按库查快照。

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
        """收下一批，立刻返回任务号。**读上传字节已经在调用方做完了**（那是请求的一部分）。

        条目顺序**必须与 `Importer.batch` 的编号一致**（文件在前、网址在后，各自按提交
        顺序）：进度事件里的 `file_number` 是从 1 起的位置，按它回填到第几条。
        """
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
            # 起线程也放在锁里：check-then-set 漏在外面的话，两个并发的提交会各起一个，
            # 「一次跑一批」当场就没了
            self._start_worker()
        self._queued.put(job)
        logger.info("导入任务 %s 收了 %d 条（知识库 %s）", job.id, len(items), game_id)
        return job.id

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        """这个任务现在什么样。没这个号（服务重启过、或者早就被丢掉了）返回 `None`。"""
        with self._lock:
            job = self._jobs.get(job_id)
        return None if job is None else job.snapshot()

    def latest(self, game_id: str = "") -> JobSnapshot | None:
        """这个库最近该看的那条任务，没有就是 `None`。

        **在跑的那条优先**：换个页回来时人想看的是「我那一批跑到哪了」，不是上一批的旧
        结果。同时在跑的取**最早**那条——工作线程一次只跑一批，最早未完成的正是它手上
        那条，界面显示它才对得上实际的进度。

        `game_id` 留空即不限库。导入页没带库时靠它兜住「总得让人看见还在跑的那批」：
        那一页会把下拉默认选到字母序第一个库，与实际在跑的那个往往不是同一个。

        任务早被丢掉时（服务重启过、或者超出 `history`）返回 `None`，与 `snapshot` 一致。
        """
        with self._lock:
            candidates = [
                job for job in self._jobs.values() if not game_id or job.game_id == game_id
            ]
        if not candidates:
            return None
        running = [job for job in candidates if not job.finished]
        return (running[0] if running else candidates[-1]).snapshot()

    def queued_behind(self, job_id: str) -> tuple[JobSnapshot, ...]:
        """这条任务后面还**排着**的批次，按提交顺序（只算还没被取走的）。

        工作线程一次跑一批，所以「我刚提交的那批」在轮到自己之前一条进度都没有。界面上
        把它们的回执一并摆出来（见 `ragamer.web.pages._job_view`）：提交完就能看见
        「我交的那批在队里、共几条」，而不是只有正在跑的那批。

        认不出这个任务号时返回空元组：那一页会说「这个任务不在了」，不必再多一句。
        """
        with self._lock:
            jobs = list(self._jobs.items())
        behind = False
        waiting: list[JobSnapshot] = []
        for other_id, job in jobs:
            if other_id == job_id:
                behind = True
                continue
            if behind and job.queued:
                waiting.append(job.snapshot())
        return tuple(waiting)

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
            job.start()
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
        importer = replace(self.importer, on_progress=self._progress(job))
        results = importer.batch(
            job.sources,
            urls=job.urls,
            game_id=job.game_id,
            version=job.version,
            vocabulary=job.vocabulary,
        )
        job.finish(results)
        logger.info("导入任务 %s 跑完：%s", job.id, _summary(results))

    def _progress(self, job: _Job) -> ProgressCallback:
        """这一批的进度回调：**组合根配的那个照跑**（默认是落日志），再往快照里记一份。

        直接把 `on_progress` 换成快照那条是不行的：一批几分钟的导入在日志里就只剩
        「收了」与「跑完」两行，中途出问题回头什么都看不到——而日志是这条路出问题时
        唯一留下的现场。
        """
        wired = self.importer.on_progress

        def report(event: ProgressEvent) -> None:
            if wired is not None:
                wired(event)
            job.note(event)

        return report


def _summary(results: Sequence[ImportResult]) -> str:
    ok = sum(1 for result in results if result.ok)
    return f"{len(results)} 条里成功 {ok} 条"
