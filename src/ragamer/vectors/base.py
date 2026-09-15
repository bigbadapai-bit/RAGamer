"""向量化与精排的共同约定：协议、共享类型与错误。

两个适配器共用一套词汇：文本进，向量／分数出。业务层只认这里的协议，
真实模型（`bge`）与确定性假件（`fake`）实现的是同一组协议，
所以在组合根换掉实现，整条链路照跑——测试缝能立起来靠的就是这一层。

三条贯穿全局的约定：

- **一个模型同时产出稠密与稀疏两路**（混合检索的基础，见 docs/ARCHITECTURE.md §3.2）。
  :meth:`Embedder.embed` 一次调用交出两路，而不是让调用方各调一次——两次调用意味着
  同一批文本被模型看两遍，还意味着两路可能对不上号。
- **稠密向量一律归一化**，与 Milvus 的 IP 度量配合等价于余弦相似度（坑 #2）。
  归一化是接口的后置条件，不是某个实现的内部细节。
- **精排不截断、不摘要**：文本原样交给模型，长度只由模型自己的上下文上限决定。
  原项目被 reranker 的 512 token 上限逼出了一条「超长就 LLM 摘要压缩再重试 4 次」的
  路径，换成长上下文模型之后那条路径不需要存在（docs/ARCHITECTURE.md §3.3）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: 稠密向量的维度（BGE-M3）。collection 的 `FLOAT_VECTOR` 与它对齐。
DENSE_DIM = 1024

#: 归一化向量的模长容差。float32 归一化完与 1 的偏差在 1e-7 量级，这里留宽几百倍。
NORM_TOLERANCE = 1e-3


class ModelError(Exception):
    """模型相关的失败。"""


class ModelUnavailableError(ModelError):
    """模型用不了：依赖没装、权重取不下来、设备不可用。"""


class ModelOutputError(ModelError):
    """模型返回的东西不认识：结构不对，或条数与输入对不上。

    宁可当场炸。按短的那边截齐、或者把认不出的结构当成空结果接着往下走，
    都会变成一条检索不到的切片或一个静默错位的排序——查不出、也不报错。
    """


@dataclass(frozen=True, slots=True)
class Embedding:
    """一批文本的两路向量。

    `dense[i]` 与 `sparse[i]` 说的是同一段文本——两路同源是混合检索的前提，
    所以两者在一次调用里一起产出，也在这里一起被校验。
    """

    dense: tuple[tuple[float, ...], ...]
    sparse: tuple[Mapping[int, float], ...]

    def __post_init__(self) -> None:
        if len(self.dense) != len(self.sparse):
            raise ModelOutputError(
                f"稠密与稀疏的条数对不上：{len(self.dense)} 条稠密、{len(self.sparse)} 条稀疏"
            )
        for index, vector in enumerate(self.dense):
            if len(vector) != DENSE_DIM:
                raise ModelOutputError(
                    f"第 {index} 条稠密向量是 {len(vector)} 维，"
                    f"collection schema 要的是 {DENSE_DIM} 维"
                )
            # 归一化在这里兜住，而不是只写在文档里：没归一化的向量照样写进 Milvus、
            # 照样检索得回来，只是 IP 从此不再等价于余弦——分数悄悄变了意思，不报错。
            length = math.sqrt(sum(value * value for value in vector))
            if abs(length - 1.0) > NORM_TOLERANCE:
                raise ModelOutputError(
                    f"第 {index} 条稠密向量没有归一化（模长 {length:g}）："
                    "配 IP 度量才算得了余弦相似度"
                )

    def __len__(self) -> int:
        return len(self.dense)


@runtime_checkable
class Embedder(Protocol):
    """把文本变成向量。换实现只换这里。"""

    def embed(self, texts: Sequence[str]) -> Embedding:
        """一批文本 → 稠密与稀疏两路，一次调用产出，顺序与 `texts` 一一对应。"""
        ...

    def warm(self) -> None:
        """把权重提前读进来。不产生任何输出，也不改变之后任何一次调用的结果。

        **只为把冷启动那份等待从提问者头上挪到启动时**：真实适配器是懒加载的
        （`ragamer.lazy`），不预热的话服务起来之后的第一条提问要先把几个 G 的权重
        从盘上读进来——那不是「检索慢」，而它每次都落在第一个提问的人身上。

        实现必须幂等：预热过再调一次什么也不做（`LazyModel.get` 本身就保证这一条）。
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """把候选按与问题的相关度重新打分。换实现只换这里。"""

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        """逐个候选打分，顺序与 `docs` 一一对应；分数越大越相关。

        分数与候选的条数必须一样多。候选可以是整篇长文，适配器不做截断。
        """
        ...

    def warm(self) -> None:
        """把权重提前读进来。约定与 :meth:`Embedder.warm` 同一条。"""
        ...


def normalize_dense(vector: Sequence[float]) -> tuple[float, ...]:
    """把稠密向量缩到单位长度，与 IP 度量配合等价于余弦相似度（坑 #2）。

    零向量原样返回：它的方向没有定义，除法只会得到 NaN，而 NaN 进了 Milvus
    既排不上序也不报错。留着这个明显不正常的零向量，下一步构造 :class:`Embedding`
    时就会被拦下——那是期望的失败方式。
    """
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0.0:
        return tuple(vector)
    return tuple(value / length for value in vector)
