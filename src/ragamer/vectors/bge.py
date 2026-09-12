"""真实模型适配器：BGE-M3（稠密 + 稀疏）与长上下文 reranker。

四件事必须一次做对：

- **一次调用出两路**：`BGEM3FlagModel.encode(return_dense=True, return_sparse=True)`
  在同一次前向里取稠密与稀疏，不为了稀疏再跑一遍模型。
- **长上下文**：`max_length` 从配置来，默认 8192。**两个库的默认值都是 512** ——
  那正好是原项目「超长就 LLM 摘要压缩再重试 4 次」那条路径的来源，照默认值用等于把
  坑原样搬回来。适配器自己不做任何截断与摘要，文本整段进模型（docs §3.3）。
- **加载一次**：权重在第一次真的要用时才加载（`ragamer` 启动自检不该等几个 G 的
  权重），之后整个进程复用同一个实例。
- **值不许悄悄变形**：稀疏向量的键在 FlagEmbedding 里是**字符串** token id，而
  Milvus 要 int；返回条数与输入对不上时当场炸，不按短的那边截齐。

FlagEmbedding 拉进 torch 与 transformers，所以它放在可选的 `models` 组里
（`uv sync --extra models`）。本模块因此对它做懒导入：没装也能 import，
只在真的要用模型时报错，并说清怎么装。测试注入 `loader` 就能绕开真实模型，
于是「只加载一次」「不截断」「两路同源」这些断言不用装 torch 也能跑。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from ragamer.config import EmbedSettings, ModelSettings, RerankSettings
from ragamer.logging import get_logger
from ragamer.vectors.base import (
    Embedding,
    ModelOutputError,
    ModelUnavailableError,
)

logger = get_logger(__name__)

#: FlagEmbedding 没装时的提示。
_INSTALL_HINT = "真实模型在可选的 models 组里，装法：uv sync --extra models"


class _M3Model(Protocol):
    """`BGEM3FlagModel` 里我们用到的那一部分。"""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        """返回 `dense_vecs`（每条一个向量）与 `lexical_weights`（每条一个词权重字典）。"""
        ...


class _RerankerModel(Protocol):
    """`FlagReranker` 里我们用到的那一部分。"""

    def compute_score(self, sentence_pairs: Sequence[tuple[str, str]], **kwargs: Any) -> Any:
        """返回每个候选一个分数，顺序与传入的句子对一致。"""
        ...


#: 模型加载器。默认是真实加载，测试换掉它就不必装 torch。
M3Loader = Callable[[EmbedSettings, ModelSettings], _M3Model]
RerankerLoader = Callable[[RerankSettings, ModelSettings], _RerankerModel]


class BgeM3Embedder:
    """BGE-M3：同一次调用产出稠密与稀疏两路向量。

    稠密向量开 `normalize_embeddings=True`（坑 #2：归一化后配 IP 度量等价余弦），
    这是接口的后置条件，不留给调用方自己做。
    """

    def __init__(
        self,
        config: EmbedSettings,
        shared: ModelSettings,
        *,
        loader: M3Loader | None = None,
    ) -> None:
        self._config = config
        self._shared = shared
        self._loader = loader if loader is not None else load_bge_m3
        self._lock = threading.Lock()
        self._model: _M3Model | None = None

    def embed(self, texts: Sequence[str]) -> Embedding:
        batch = list(texts)
        if not batch:
            # 空批次不惊动模型：这一趟没有意义，还会白加载一次权重
            return Embedding(dense=(), sparse=())
        result = self._model_for_use().encode(
            batch,
            batch_size=self._config.batch_size,
            max_length=self._config.max_length,
            return_dense=True,
            return_sparse=True,
        )
        return _to_embedding(result, len(batch))

    def _model_for_use(self) -> _M3Model:
        """第一次用到时才加载，之后复用同一个实例。

        加载过程放在锁里：并发首次调用（界面后端会把同步端点丢进线程池）会同时看见
        `None`，各自加载一遍，白白多占一份显存。锁只护加载，不护推理。
        """
        with self._lock:
            if self._model is None:
                logger.info(
                    "加载向量化模型 %s（device=%s，fp16=%s）",
                    self._config.model,
                    self._shared.device,
                    self._shared.fp16,
                )
                self._model = self._loader(self._config, self._shared)
            return self._model


class BgeReranker:
    """长上下文精排（bge-reranker-v2-m3，上限 8192 token）。

    **不截断、不摘要**：候选整段进模型，长度只由 `max_length` 决定。原项目那条
    「超长就摘要压缩再重试」的路径在这里没有对应物——不是被优化掉了，是不需要。

    分数按 sigmoid 归一化到 [0, 1]，与下游断崖截断的 0.3 / 0.5 同一量纲。
    """

    def __init__(
        self,
        config: RerankSettings,
        shared: ModelSettings,
        *,
        loader: RerankerLoader | None = None,
    ) -> None:
        self._config = config
        self._shared = shared
        self._loader = loader if loader is not None else load_bge_reranker
        self._lock = threading.Lock()
        self._model: _RerankerModel | None = None

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        candidates = list(docs)
        if not candidates:
            return []
        scores = self._model_for_use().compute_score(
            [(query, candidate) for candidate in candidates],
            batch_size=self._config.batch_size,
            max_length=self._config.max_length,
            normalize=True,
        )
        return _scores(scores, len(candidates))

    def _model_for_use(self) -> _RerankerModel:
        """与 :meth:`BgeM3Embedder._model_for_use` 同一套：懒加载、加锁、只加载一次。"""
        with self._lock:
            if self._model is None:
                logger.info(
                    "加载精排模型 %s（device=%s，fp16=%s）",
                    self._config.model,
                    self._shared.device,
                    self._shared.fp16,
                )
                self._model = self._loader(self._config, self._shared)
            return self._model


def load_bge_m3(config: EmbedSettings, shared: ModelSettings) -> _M3Model:
    """真实加载 BGE-M3。"""
    model_class = _flag_embedding("BGEM3FlagModel")
    return model_class(
        config.model,
        # 归一化是接口的后置条件（坑 #2），不靠库的默认值
        normalize_embeddings=True,
        use_fp16=shared.fp16,
        devices=shared.device,
    )


def load_bge_reranker(config: RerankSettings, shared: ModelSettings) -> _RerankerModel:
    """真实加载长上下文 reranker。"""
    model_class = _flag_embedding("FlagReranker")
    return model_class(
        config.model,
        use_fp16=shared.fp16,
        devices=shared.device,
    )


def _flag_embedding(name: str) -> Any:
    """懒导入 FlagEmbedding，取其中一个类。

    缺依赖时说清怎么装，而不是抛一个 ImportError 让人自己猜。
    """
    try:
        import FlagEmbedding
    except ImportError as exc:
        raise ModelUnavailableError(
            f"没有装 FlagEmbedding（{name} 需要它）。{_INSTALL_HINT}"
        ) from exc
    return getattr(FlagEmbedding, name)


def _to_embedding(result: Any, expected: int) -> Embedding:
    """把 FlagEmbedding 的返回翻成 :class:`Embedding`。"""
    if not isinstance(result, Mapping):
        raise ModelOutputError(f"向量化模型返回的不是字典：{type(result).__name__}")
    return Embedding(
        dense=_dense_rows(result.get("dense_vecs"), expected),
        sparse=_sparse_rows(result.get("lexical_weights"), expected),
    )


def _dense_rows(value: Any, expected: int) -> tuple[tuple[float, ...], ...]:
    rows = _as_list(value, "dense_vecs")
    if len(rows) != expected:
        raise ModelOutputError(f"稠密向量有 {len(rows)} 条，送进去的文本却有 {expected} 条")
    return tuple(_vector(row) for row in rows)


def _vector(row: Any) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in row)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"稠密向量里有不是数值的元素：{row!r}") from exc


def _sparse_rows(value: Any, expected: int) -> tuple[Mapping[int, float], ...]:
    items = _as_list(value, "lexical_weights")
    if len(items) != expected:
        raise ModelOutputError(f"稀疏向量有 {len(items)} 条，送进去的文本却有 {expected} 条")
    return tuple(_term_weights(item) for item in items)


def _term_weights(item: Any) -> Mapping[int, float]:
    """一条稀疏向量：`{token_id: 权重}`，键转成 int。

    FlagEmbedding 内部把 token id 写成 `str(idx)` 才当字典键，所以这里必须转一次。
    让字符串键漏下去，写 Milvus 时要么报错、要么更糟——被当成另一套词表静默错位。
    """
    if not isinstance(item, Mapping):
        raise ModelOutputError(f"一条稀疏向量不是字典：{type(item).__name__}")
    weights: dict[int, float] = {}
    for term, weight in item.items():
        try:
            weights[int(term)] = float(weight)
        except (TypeError, ValueError) as exc:
            raise ModelOutputError(f"稀疏向量的词权重不是数值：{term!r} → {weight!r}") from exc
    return weights


def _scores(value: Any, expected: int) -> list[float]:
    """分数与候选一一对应，条数对不上就是接线错了。"""
    try:
        scores = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"精排返回的分数不是数值：{value!r}") from exc
    if len(scores) != expected:
        raise ModelOutputError(f"精排返回 {len(scores)} 个分数，候选却有 {expected} 个")
    return scores


def _as_list(value: Any, field: str) -> list[Any]:
    if value is None:
        raise ModelOutputError(f"向量化模型的返回里没有 {field}")
    try:
        return list(value)
    except TypeError as exc:
        raise ModelOutputError(f"{field} 不是可遍历的：{type(value).__name__}") from exc
