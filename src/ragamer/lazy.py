"""懒加载句柄：模型与引擎只在第一次真的要用时才加载。

组合根构造适配器时不该等几个 G 的权重——`ragamer` 那条命令是用来快速判断
「配置对不对」的，卡在权重下载上就没法当这个用了。所以真实模型的加载一律推后到
第一次调用，之后整个进程复用同一个实例。

加锁是冲着**并发首次调用**去的：界面后端会把同步端点丢进线程池，两个线程同时看见
「还没加载」，各自加载一遍，白白多占一份显存（或一份内存）。锁只护加载，不护推理。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ragamer.logging import get_logger

logger = get_logger(__name__)


class LazyModel[ModelT]:
    """一个只加载一次的句柄。第一次 `get` 时才构造，之后直接给同一个实例。

    `describe` 是进日志的名字（「向量化模型」这类），加载要花掉几秒到几分钟，
    那行日志是这期间唯一看得见的东西。
    """

    def __init__(self, describe: str, loader: Callable[[], ModelT]) -> None:
        self._describe = describe
        self._loader = loader
        self._lock = threading.Lock()
        self._model: ModelT | None = None

    def get(self) -> ModelT:
        with self._lock:
            if self._model is None:
                logger.info("加载%s", self._describe)
                self._model = self._loader()
            return self._model
