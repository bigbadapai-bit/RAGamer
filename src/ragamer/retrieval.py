"""主检索路：取候选、精排、按分数落差截断。

这是 `docs/ARCHITECTURE.md` §3 那条链路里的 P1，三段依次是：混合检索召回一批候选、
精排重新打分、断崖截断决定交多少条给生成。其余五路召回、RRF 融合与查询路由在后面
几张票里接，那时复用的是后两段，所以这里把它们各自切开。

四处容易做错、做错了又不报错的：

- **两路必须同一次产出**。稠密与稀疏出自同一个模型、同一批文本，所以向量化只调一次
  （`:meth:`~ragamer.vectors.base.Embedder.embed``），融合交给存储适配器做——那是
  「同一批数据上融合两路」（§3.2 的混合检索），不是多路召回。分两次调向量化，
  两批文本对不上号既不报错也查不出来。
- **精排吃正文，不吃 `content_meta`**。后者按设计不参与向量化（§2.2），打分同理：
  表格里整列降级进去的长文本会把分数带偏。
- **截断按分数落差，不取固定前 K**（§3.3、坑 #14）。凑数凑进来的那几条会把上下文
  稀释掉——答案悄悄变差，而且看不出是哪一步的问题。
- **候选池要明显大于截断上界**。池子只有上界那么大时，上界会把每一条候选都保下来，
  断崖等于没生效，而两处的参数看上去都还写在那里。

分数一律是精排分，落在 [0, 1]（真实精排 sigmoid 之后，见 `ragamer.vectors.bge`）。
"""

from __future__ import annotations

from collections.abc import Sequence

from ragamer.stores.base import ChunkFilter, ChunkHit, ChunkStore
from ragamer.vectors.base import Embedder, ModelOutputError, Reranker

#: 交出去的内容条数上界。**上界是「最多」，不是「取前 K 条」**——实际给几条由
#: 断崖说了算（§3.3、坑 #14 的 `min(10, 候选数)`）。
MAX_PASSAGES = 10

#: 断崖的两个口径：绝对落差、相对落差（相邻分差 ÷ 前一条分数）。两者都是「大于」才
#: 算断崖，取自原项目标定过的值（§3.3）。
CLIFF_ABSOLUTE = 0.3
CLIFF_RELATIVE = 0.5

#: 精排前的候选池：截断上界的若干倍。
#:
#: 倍数本身与全体阈值一样是**没有评测集时的占位**（§11：任何调参都应先有评测集）。
#: 这里能讲清的只有它必须大于 1——池子不大于上界，上界就把候选全保下来了。
#: 乘出来的池子大小还要够存储一次取回，所以按上界的倍数给，不写死一个数。
CANDIDATE_FACTOR = 3

#: 一次检索取回多少条候选。数值由 `MAX_PASSAGES` 与 `CANDIDATE_FACTOR` 推出来。
CANDIDATE_LIMIT = MAX_PASSAGES * CANDIDATE_FACTOR


def retrieve(
    query: str,
    *,
    game_id: str,
    chunks: ChunkStore,
    embedder: Embedder,
    reranker: Reranker,
    where: ChunkFilter | None = None,
    candidates: int = CANDIDATE_LIMIT,
) -> tuple[ChunkHit, ...]:
    """主检索路：一批候选进，截断后的一批进生成。

    返回的分数是**精排分**，顺序即交出去的顺序。一条都没检索到时返回空元组——
    由调用方决定没有资料时怎么办，这一层不编造内容。

    :param query: 用来向量化与精排的文本。改写（`ragamer.query`）在外面做完再进来。
    :param where: 结构化过滤条件，版本那一条由 `ragamer.query.version_filter` 给出。
    :param candidates: 精排前的候选池大小。
    :raises ModelOutputError: 向量化或精排的条数与候选对不上。宁可当场炸：
        按短的一边截齐会得到一个静默错位的排序，查不出、也不报错。
    """
    embedding = embedder.embed([query])
    if len(embedding) != 1:
        raise ModelOutputError(f"向量化返回了 {len(embedding)} 条，喂进去的是一个问题")
    found = chunks.search(
        game_id,
        dense=embedding.dense[0],
        sparse=embedding.sparse[0],
        where=where,
        limit=candidates,
    )
    if not found:
        return ()  # 空候选上白调一次精排
    return cliff_cut(_reranked(query, found, reranker))


def _reranked(query: str, found: Sequence[ChunkHit], reranker: Reranker) -> list[ChunkHit]:
    """重新打分并按分数降序。

    同分时按切片序号定序：候选集合相同就必须排出同一个顺序，否则截断位置会在两条
    同分候选之间挪来挪去，同一个问题两次问出不同的答案。
    """
    scores = reranker.rerank(query, [hit.chunk.content for hit in found])
    if len(scores) != len(found):
        raise ModelOutputError(f"精排返回了 {len(scores)} 个分数，候选是 {len(found)} 条")
    ranked = [
        ChunkHit(chunk=hit.chunk, score=score) for hit, score in zip(found, scores, strict=True)
    ]
    ranked.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
    return ranked


def cliff_cut(hits: Sequence[ChunkHit]) -> tuple[ChunkHit, ...]:
    """断崖截断：按相邻候选的分数落差决定切在哪里（§3.3）。

    从最相关的一条起往下走，遇到断崖就停——**落差本身就是「后面那些是另一档」的
    信号**。取固定前 K 条会把另一档的内容一起塞进上下文，稀释掉真正相关的那几条。

    两个口径任一命中即算断崖，两者都是「大于」：

    - 绝对：相邻分差 > 0.3
    - 相对：相邻分差 ÷ 前一条分数 > 0.5

    相对那一条要除以「前一条的分数」。前一条不超过 0 时它定义不了（除零，或者把负数
    除法算成一个假的断崖），这时只按绝对落差判——真实精排的分数落在 [0, 1]，
    走到这一支说明分数已经贴底，本来也没什么可再切的。

    上界是 `min(MAX_PASSAGES, 候选数)`：**候选本来就少时不凑数**。传进来的应当是
    已经按分数降序排好的候选。
    """
    selected: list[ChunkHit] = []
    for hit in hits[:MAX_PASSAGES]:
        if selected and _is_cliff(selected[-1].score, hit.score):
            break
        selected.append(hit)
    return tuple(selected)


def _is_cliff(previous: float, current: float) -> bool:
    drop = previous - current
    if drop > CLIFF_ABSOLUTE:
        return True
    return previous > 0 and drop / previous > CLIFF_RELATIVE
