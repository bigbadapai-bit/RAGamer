"""组合根：全部外部依赖在这里构造一次，然后注入。

三个客户端一个都不做成模块级单例——单例在测试里换不掉，缝也就没了。
`build_container` 每次都新建一套：调用两次得到两套互不相干的客户端。

启动自检也在这里：`Container.check` 一次报出全部不通的服务。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ragamer.config import Settings
from ragamer.logging import get_logger
from ragamer.stores.base import (
    ChunkStore,
    DocStore,
    ObjectStore,
    Store,
    StoreCheckError,
    StoreUnavailableError,
)
from ragamer.stores.chunks import MilvusChunkStore
from ragamer.stores.documents import MongoDocStore
from ragamer.stores.objects import MinioObjectStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class Container:
    """一套存储客户端。测试里换成内存假件即可整条链路照跑。"""

    chunks: ChunkStore
    docs: DocStore
    objects: ObjectStore

    def stores(self) -> tuple[Store, ...]:
        """三个服务，自检按这个顺序走。"""
        return (self.chunks, self.docs, self.objects)

    def check(self) -> None:
        """启动自检：三个服务都要连得上，每个服务各自确保自己的命名空间。

        Milvus 的 database 与 MinIO 的桶都在这一步落下——命名空间得在任何一条数据路径
        之前定下来，否则哪条路径漏了，数据会静默落到与原项目共用的那个命名空间里。
        桶还牵着一件部署侧的事：容器以非 root 身份跑时卷属主不对就建不了桶，
        这属于启动阶段就该暴露的问题，不该等到第一次导入原图。

        有不通的把不通的都列出来一起报——启动时一次看清全部问题，而不是修一个重启一次。

        :raises StoreCheckError: 有服务不通。
        """
        failures: list[StoreUnavailableError] = []
        for store in self.stores():
            _collect(failures, store.check)
        if failures:
            raise StoreCheckError(failures)
        logger.info("存储自检通过：%s", "、".join(store.name for store in self.stores()))


def _collect(failures: list[StoreUnavailableError], probe: Callable[[], None]) -> None:
    """自检里的每一步都记下来，不因为上一步失败就跳过后面。"""
    try:
        probe()
    except StoreUnavailableError as exc:
        failures.append(exc)


def build_container(settings: Settings) -> Container:
    """按配置构造三个客户端。构造只发生在这里。"""
    timeout = settings.store_timeout_seconds
    return Container(
        chunks=MilvusChunkStore(settings.milvus, timeout=timeout),
        docs=MongoDocStore(settings.mongo, timeout=timeout),
        objects=MinioObjectStore(settings.minio, timeout=timeout),
    )
