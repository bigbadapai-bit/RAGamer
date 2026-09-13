"""组合根：全部外部依赖在这里构造一次，然后注入。

三个客户端一个都不做成模块级单例——单例在测试里换不掉，缝也就没了。
`build_container` 每次都新建一套：调用两次得到两套互不相干的客户端。

启动自检也在这里：`Container.check` 一次报出全部不通的服务。
**三个模型都不进自检**——两个本地模型的权重是几个 G，要等第一次真的用到时才加载；
语言模型则要发一次网络请求，那是运行期的事。自检卡在这些上面，
`ragamer` 这条命令就没法当"配置对不对"的快速检查用了。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ragamer.config import Settings
from ragamer.llm import LlmClient, OpenAiLlm
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
from ragamer.vectors.base import Embedder, Reranker
from ragamer.vectors.bge import BgeM3Embedder, BgeReranker
from ragamer.websearch import BochaWebSearch, WebSearch

logger = get_logger(__name__)


@dataclass(frozen=True)
class Container:
    """一套外部依赖。测试里换成内存假件即可整条链路照跑。"""

    chunks: ChunkStore
    docs: DocStore
    objects: ObjectStore
    embedder: Embedder
    reranker: Reranker
    #: 语言模型。打标兜底、查询路由、多查询改写、HyDE、生成都走它。
    llm: LlmClient
    #: 联网兜底那一路的外部检索。**没配就是 `None`**——整组配置可以不填，这一路随之
    #: 跳过，其余几路照常作答（见 `ragamer.websearch`）。
    search: WebSearch | None = None

    def stores(self) -> tuple[Store, ...]:
        """三个存储服务，自检按这个顺序走。"""
        return (self.chunks, self.docs, self.objects)

    def check(self) -> None:
        """启动自检：三个服务都要连得上，每个服务各自确保自己的命名空间。

        Milvus 的 database 与 MinIO 的桶都在这一步落下——命名空间得在任何一条数据路径
        之前定下来，否则哪条路径漏了，数据会静默落到与原项目共用的那个命名空间里。

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
    """按配置构造全部外部依赖。构造只发生在这里。

    三个模型在这里只是被造出来：两个本地模型的权重等第一次真的要用时才加载
    （见 `ragamer.vectors.bge`），语言模型则要等第一次调用才连。
    """
    timeout = settings.store_timeout_seconds
    return Container(
        chunks=MilvusChunkStore(settings.milvus, timeout=timeout),
        docs=MongoDocStore(settings.mongo, timeout=timeout),
        objects=MinioObjectStore(settings.minio, timeout=timeout),
        embedder=BgeM3Embedder(settings.embed, settings.models),
        reranker=BgeReranker(settings.rerank, settings.models),
        llm=OpenAiLlm(settings.llm),
        # 没配密钥就是没这一路。**不构造一个「搜不到东西」的实现**：那会让「没配」
        # 与「搜了但没有结果」在日志与答案里长得一模一样，查无可查
        search=BochaWebSearch(settings.search) if settings.search.api_key else None,
    )
