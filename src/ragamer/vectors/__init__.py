"""向量化与精排适配器：两个协议、两个真实实现、两个确定性假件。

业务层 `from ragamer.vectors import Embedder` 拿协议，实现由组合根注入。
真实模型（BGE-M3 与长上下文 reranker）需要可选的 `models` 组，见 `ragamer.vectors.bge`。
"""

from __future__ import annotations

from ragamer.vectors.base import (
    DENSE_DIM,
    Embedder,
    Embedding,
    ModelError,
    ModelOutputError,
    ModelUnavailableError,
    Reranker,
    normalize_dense,
)
from ragamer.vectors.bge import BgeM3Embedder, BgeReranker
from ragamer.vectors.fake import FakeEmbedder, FakeReranker

__all__ = [
    "DENSE_DIM",
    "BgeM3Embedder",
    "BgeReranker",
    "Embedder",
    "Embedding",
    "FakeEmbedder",
    "FakeReranker",
    "ModelError",
    "ModelOutputError",
    "ModelUnavailableError",
    "Reranker",
    "normalize_dense",
]
