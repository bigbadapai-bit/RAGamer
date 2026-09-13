"""主检索路：取候选、精排、按分数落差截断，再按文档聚合父块。

这是 `docs/ARCHITECTURE.md` §3 那条链路里的 P1，四段依次是：混合检索召回一批候选、
精排重新打分、断崖截断决定交多少条给生成、按文档聚合父块决定每条交出去多少内容。
其余五路召回、RRF 融合与查询路由在后面几张票里接，那时复用的是中间两段，
所以这里把它们各自切开。

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
- **父块是查出来的，不单独入库**（§2.5）。回查永远与子块同源，重导一份文档不需要
  额外的同步步骤，删库也不会留下孤儿父块——有第二份存储就总有漏同步的那一天。

分数一律是精排分，落在 [0, 1]（真实精排 sigmoid 之后，见 `ragamer.vectors.bge`）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragamer.logging import get_logger
from ragamer.stores.base import Chunk, ChunkFilter, ChunkHit, ChunkStore
from ragamer.vectors.base import Embedder, ModelOutputError, Reranker

logger = get_logger(__name__)

#: 交出去的切片条数上界。**上界是「最多」，不是「取前 K 条」**——实际给几条由
#: 断崖说了算（§3.3、坑 #14 的 `min(10, 候选数)`）。
MAX_CHUNKS = 10

#: 断崖的两个口径：绝对落差、相对落差（相邻分差 ÷ 前一条分数）。两者都是「大于」才
#: 算断崖，取自原项目标定过的值（§3.3），量纲是精排分的 [0, 1]。
CLIFF_ABSOLUTE = 0.3
CLIFF_RELATIVE = 0.5

#: 精排前的候选池：截断上界的若干倍。
#:
#: 倍数本身与全体阈值一样是**没有评测集时的占位**（§11：任何调参都应先有评测集）。
#: 这里能讲清的只有它必须大于 1——池子不大于上界，上界就把候选全保下来了。
#: 乘出来的池子大小还要够存储一次取回，所以按上界的倍数给，不写死一个数。
CANDIDATE_FACTOR = 3

#: 一次检索取回多少条候选。数值由 `MAX_CHUNKS` 与 `CANDIDATE_FACTOR` 推出来。
CANDIDATE_LIMIT = MAX_CHUNKS * CANDIDATE_FACTOR

#: 父块的长度上限（字符数）。整篇文档超过它就按命中切片所在的小节收敛（§2.5）。
#:
#: 量级由截断上界与单片上界乘出来：截断最多交 :data:`MAX_CHUNKS` 条，切分器单片正文
#: 上限 800 字符（`ragamer.chunking.ChunkRules.max_chars`）。**数值本身仍是没有评测集时的
#: 占位**（§11）——模板块不受单片上限约束，这条乘法只是"一页"的大致边界，
#: 真章是超长文档不该整页喂进去。
MAX_PARENT_CHARS = MAX_CHUNKS * 800


def retrieve(
    query: str,
    *,
    game_id: str,
    chunks: ChunkStore,
    embedder: Embedder,
    reranker: Reranker,
    where: ChunkFilter | None = None,
) -> tuple[ChunkHit, ...]:
    """主检索路：一批候选进，截断后的一批进生成。

    返回的分数是**精排分**，顺序即交出去的顺序。一条都没检索到时返回空元组——
    由调用方决定没有内容时怎么办，这一层不编造内容。

    候选池固定取 :data:`CANDIDATE_LIMIT`，不做成入参：池子一旦可以被调小到上界以内，
    上界就会把候选全保下来，断崖等于没生效，而两个参数看上去都还写在那里。

    :param query: 用来向量化与精排的文本。改写（`ragamer.query`）在外面做完再进来。
    :param where: 结构化过滤条件，版本那一条由 `ragamer.query.version_filter` 给出。
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
        limit=CANDIDATE_LIMIT,
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

    上界是 `min(MAX_CHUNKS, 候选数)`：**候选本来就少时不凑数**。传进来的应当是
    已经按分数降序排好的候选。
    """
    _warn_on_scale(hits)
    selected: list[ChunkHit] = []
    for hit in hits[:MAX_CHUNKS]:
        if selected and _is_cliff(selected[-1].score, hit.score):
            break
        selected.append(hit)
    return tuple(selected)


@dataclass(frozen=True)
class ParentBlock:
    """命中并截断之后，按文档聚合出来的父块（§2.5 的父子块）。

    `ancestor_path` 是空串表示**整篇**——文档不长，整页交给生成；非空表示这份文档
    超长、已收敛到命中的那个小节。两种情况都随 `chunks` 给出原始切片：父块不是另一份
    内容，只是同一批切片的一个视图，所以它与子块永远同源。
    """

    doc_title: str
    ancestor_path: str
    chunks: tuple[Chunk, ...]


def aggregate_parents(
    hits: Sequence[ChunkHit],
    *,
    game_id: str,
    chunks: ChunkStore,
    where: ChunkFilter | None = None,
) -> tuple[ParentBlock, ...]:
    """命中并截断之后，按文档回查兄弟切片、拼成交给生成的父块。

    用户问"二郎神怎么打"时模型拿到整页——包括"掉落"，所以追问"掉什么"不必重新检索。
    **父块是查出来的，不单独入库**（§2.5）：一个文档对应一个主体，所以"该主体在该文档
    的聚合父块"就是该文档的全部切片，按 `doc_title` 回查、按 `chunk_index` 升序拼起来
    即可。另存一份父块就得在每次重导后记得同步，漏掉是静默失效。

    传进来的应当是**已经过断崖截断**的那批命中：先截断再聚合，否则会为即将被丢弃的
    切片白拼一遍父块。反过来，被截断掉的切片仍可能经由同文档的兄弟关系进到父块里——
    那正是父块的意义，问"怎么打"时把同页的"掉落"一并带上。

    三条口径：

    - **版本过滤与检索同一个**。回查用 `where.version`，与 `search` 走的是同一套
      「该版本 **或** 未标注版本」；两个版本各留一份切片时，父块里不会混版本。
    - **超长文档收敛到小节**。整页超过 :data:`MAX_PARENT_CHARS` 就只留与命中切片同一
      条 `ancestor_path` 的切片，不把整页喂进去。同一文档命中多处小节时各成一个父块。
    - **顺序**：父块按命中的先后（即精排分从高到低），块内按 `chunk_index` 升序。

    同文档只回查一次；同一个 (文档, 小节) 只出一个父块。命中切片按 `doc_title` 回查
    一定查得到（它就是照这个条件检出来的），查不到说明索引与数据对不上——跳过并留痕，
    不产出内容为空的引用。

    :param where: 与 `retrieve` 收到的是同一个。只取其中的版本口径。
    """
    version = None if where is None else where.version
    blocks: list[ParentBlock] = []
    seen: set[tuple[str, str]] = set()
    siblings: dict[str, tuple[Chunk, ...]] = {}
    for hit in hits:
        title = hit.chunk.doc_title
        if title not in siblings:
            siblings[title] = tuple(chunks.fetch_document(game_id, title, version=version))
        found = siblings[title]
        if not found:
            logger.warning(
                "切片 %d 属于文档 %r，按文档回查却一条都没有：索引与数据对不上，跳过它",
                hit.chunk.chunk_id,
                title,
            )
            continue
        section = hit.chunk.ancestor_path if _too_long(found) else ""
        if (title, section) in seen:
            continue
        seen.add((title, section))
        blocks.append(
            ParentBlock(
                doc_title=title,
                ancestor_path=section,
                chunks=tuple(
                    chunk for chunk in found if not section or chunk.ancestor_path == section
                ),
            )
        )
    return tuple(blocks)


def _too_long(chunks: Sequence[Chunk]) -> bool:
    """整页交给生成是不是太长了。

    `content_meta` 也算进来：它随结果一起交给模型（§2.2），只按正文算会低估实际喂进去的量。
    """
    width = sum(len(chunk.content) + len(chunk.content_meta) for chunk in chunks)
    return width > MAX_PARENT_CHARS


def _is_cliff(previous: float, current: float) -> bool:
    drop = previous - current
    if drop > CLIFF_ABSOLUTE:
        return True
    return previous > 0 and drop / previous > CLIFF_RELATIVE


def _warn_on_scale(hits: Sequence[ChunkHit]) -> None:
    """分数不在 [0, 1] 里时提醒一声：断崖的两个阈值是按这个量纲标定的。

    真实精排是 sigmoid 之后的取值（`ragamer.vectors.bge`），假件也照同一量纲造。
    换上一个返回 logits 的精排，0.3 与 0.5 就不再是原来那个意思——绝对落差几乎必然
    命中、一上来就切，相对落差则被除以「前一条分数」那一步整个关掉。两条都不报错，
    截断位置于是悄悄换了个依据，所以在这里留一条痕。
    """
    outside = next((hit.score for hit in hits if not 0.0 <= hit.score <= 1.0), None)
    if outside is not None:
        logger.warning(
            "精排分不在 [0, 1] 里（如 %g）：断崖的 %g / %g 是按这个量纲标定的，"
            "这次的截断位置不能按那两个口径读",
            outside,
            CLIFF_ABSOLUTE,
            CLIFF_RELATIVE,
        )
