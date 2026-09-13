"""生成：把检索到的资料交给模型，产出一段**能核对**的答案。

读取侧到这一步为止：问题进，答案与引用出来。中间是主检索路（`ragamer.retrieval`）
——取候选、精排、断崖截断、按文档聚合父块。

四件事在这里定死：

- **一条资料是一个父块，不是一个切片**。命中并截断之后按文档回查兄弟切片
  （`aggregate_parents`），于是问"二郎神怎么打"时模型拿到的是整页——包括"掉落"，
  追问"掉什么"不必重新检索。进父块的是被截断那批切片**所属的文档**，
  不是它们自己那几句：引用因此指向文档（超长文档里则指向那一小节），不是某一句话。
- **答案必须带引用来源**：交给模型的每一个父块都编了号，编号连同「文档标题 + 祖先标题
  路径」一起进提示词，也一起随答案交回。没有引用的答案是一段无从核对的话——
  用户没法知道它是切片里写的还是模型编的。编号对不对得上由「引用与内容同批同序」
  保证：两者绑在同一个 :class:`_Source` 里，不是两个各排一遍的序列。
- **答案还要带图片地址**。图片留在正文里（§1.3），从交给模型的那批内容里取出来随答案
  交回，用户才不必跳出去找原图。它跟着引用走，不是另一次检索。
- **检索不到就直说**。候选一条都没有时**不调模型**：没有内容可依据，让它自由发挥
  只会得到一段编造的游戏攻略，而且看起来和真答案一样。回复是这里的常量。
- **生成失败照抛**：没有答案就是没有答案。降级成一段「抱歉我答不上来」会把故障
  伪装成结果，比报错难查得多（与 `ragamer.query` 那一步的降级不同——那一步降级之后
  整条链路还能继续，这一步降级之后没有东西可以继续）。

引用里的编号是**人看的编号，从 1 起**，与提示词里的编号一致；列表下标那套不出现
在答案里。模型有没有真的在正文里引用某一条，v1 不做校验——那属于忠实度评测要做的事
（架构文档 §8）。能做的机械检查只有一条：正文里出现了范围之外的编号就留痕，
那种答案指向不存在的来源。

提示词里管这批切片叫「资料」是给模型看的说法，代码里不引这个词——
`ragamer.importing` 那边「资料」指的是一份源文件，两个意思撞在一起会读岔。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ragamer.chunking import PATH_SEPARATOR
from ragamer.llm import LlmClient, LlmRequest, Message
from ragamer.logging import get_logger
from ragamer.query import version_filter
from ragamer.retrieval import ParentBlock, aggregate_parents, retrieve
from ragamer.stores.base import Chunk, ChunkStore
from ragamer.vectors.base import Embedder, Reranker

logger = get_logger(__name__)

#: 生成用的温度。钉死 0：答案要照着资料写、不追求多样性，同一个问题两次问出同一个
#: 答案才谈得上核对，缓存层（架构文档 §4）也才好命中。
TEMPERATURE = 0.0

#: 检索不到内容时的回复。**这不是模型写的**，也没有调模型——理由见模块说明。
NOT_FOUND = (
    "知识库里没有找到与这个问题相关的资料。换个问法试试，或者先把这个游戏的相关资料导入知识库。"
)

#: 生成用的系统提示。三条约束各对应一种失败：编造资料里没有的、结论与来源对不上、
#: 以及把提示词本身复述成答案。
_INSTRUCTION = (
    "你是游戏攻略助手。只依据下面列出的资料回答用户的问题。\n"
    "资料里没写到的就说资料里没有，不要凭印象补充数值、打法或结论。\n"
    "每条结论后面用 [编号] 标出它出自哪条资料，编号就是资料前面的那个。\n"
    "用中文直接回答，不要复述这几条要求，也不要使用资料之外的编号。\n"
    "资料：\n"
)

#: 正文里的引用编号。答案里出现范围之外的编号，指向的是一条不存在的来源。
_MARKER = re.compile(r"\[(\d+)\]")

#: 正文里的图片引用。图片地址是留在正文里的（§1.3），答案要把原图带回去就得从这儿取。
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")


@dataclass(frozen=True)
class Citation:
    """答案的一个来源。

    `index` 是**正文里 [n] 指的那个编号**，从 1 起；引用在元组里的顺序就是交给模型的
    顺序，两者一一对应。文档标题与祖先标题路径都要给出来——只给标题的话，一份长词条
    里是「打法」那一段还是「获取方式」那一段，读的人仍然对不上。
    """

    index: int
    doc_title: str
    ancestor_path: str

    @property
    def label(self) -> str:
        """给人和模型看的一行来源。

        祖先标题路径通常以文档标题开头（一级标题就是文档标题，它是标题树的第一层），
        所以对得上时不再重复念一遍。比对前缀时要落在分隔符上：文档「二郎神」与文档
        「二郎神外传」是两份文档，只按字面前缀比会把后者误当成前者的一部分。
        """
        if not self.ancestor_path:
            return self.doc_title
        if self.ancestor_path == self.doc_title or self.ancestor_path.startswith(
            self.doc_title + PATH_SEPARATOR
        ):
            return self.ancestor_path
        return f"{self.doc_title}{PATH_SEPARATOR}{self.ancestor_path}"


@dataclass(frozen=True)
class _Source:
    """编号好的一个父块：引用 + 它的内容。

    两者绑在一个类型里而不是两个平行序列：编号与内容本来就是同一件事的两面，
    分开放就得靠调用方保证两边同长同序，对不上时是静默的（编号指向另一条内容）。
    """

    citation: Citation
    block: ParentBlock


@dataclass(frozen=True)
class Answer:
    """一次提问的结果。

    `citations` 为空即「没有检索到内容」，这时 `text` 是 `NOT_FOUND` 那段常量。
    调用方不必另外判断有没有答案——空引用就是那个信号。

    `images` 是交给生成的那批内容里出现过的图片地址：答案里要能直接展示原图，
    用户不必跳出去找（用户故事 52）。它是**跟着引用走**的，不是另一次检索的结果。
    """

    text: str
    citations: tuple[Citation, ...]
    images: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamingAnswer:
    """一次作答的流式形态：正文一片片出来，引用与图片在第一个字之前就已确定。

    引用与图片之所以能提前给出，是因为检索在生成之前——不必等答案写完才知道它出自哪里。
    两者与正文放在同一层返回，而不是各成一路：分开成两个平行序列就得靠调用方保证
    两边对得上，对不上时是静默的（引用指向另一段正文）。

    `text` 是一个生成器：**消费它才会真的去走模型**（未命中缓存时），也可能一次都不走
    （命中缓存后重放，或压根没检索到内容）。
    """

    citations: tuple[Citation, ...]
    images: tuple[str, ...]
    text: Iterator[str]


def _prompt_text(block: ParentBlock) -> str:
    """一个父块交给模型时的全文：块内每条切片按源文档顺序拼起来。

    顺序就是 `chunk_index` 升序（`fetch_document` 已排好），所以拼出来的读法与源文档一致。
    """
    return "\n".join(_chunk_text(chunk) for chunk in block.chunks)


def _chunk_text(chunk: Chunk) -> str:
    """块内的一条切片：正文 + 不参与向量化的附加文本。

    `content_meta` **必须带上**：表格里那些长文本列整列降级在那里（§2.5），
    只给正文等于把整列说明丢掉——它本来就是「随结果返回但不打分」的那部分（§2.2）。
    """
    meta = chunk.content_meta.strip()
    return f"{chunk.content}\n{meta}" if meta else chunk.content


def _citations(sources: Sequence[_Source]) -> tuple[Citation, ...]:
    """编号好的这批来源，顺序即交给模型的顺序。"""
    return tuple(source.citation for source in sources)


def image_urls(sources: Sequence[_Source]) -> tuple[str, ...]:
    """交给生成的这批内容里出现过的图片地址，按首次出现的顺序去重。

    图片地址本来就留在正文里（§1.3：原图保留，供答案展示），所以这里是从**已经要
    交给模型的那批内容**里取，不是另查一次——另查一次就会与引用对不上。
    `content_meta` 也算：表格的长文本列整列降级在那里（§2.5），里面同样可以有图。

    只认 Markdown 的 `![alt](地址)`：管道里的一切先归一成 md（ADR-0006）。
    """
    return tuple(
        dict.fromkeys(
            match.group(1)
            for source in sources
            for chunk in source.block.chunks
            for text in (chunk.content, chunk.content_meta)
            for match in _MD_IMAGE.finditer(text)
        )
    )


@dataclass(frozen=True)
class Answerer:
    """读取侧的唯一入口。外部依赖由组合根注入（见 `ragamer.container`）。

    与写入侧的 `ragamer.importing.Importer` 是同一个打法：做成不可变对象，一次接线
    反复使用——每问一次都要重传一遍五个依赖，线就总有接错的机会。
    """

    chunks: ChunkStore
    embedder: Embedder
    reranker: Reranker
    llm: LlmClient

    def answer(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
    ) -> Answer:
        """读一个问题，给出答案与它的来源。**只走主检索路**。

        截断之后按文档聚合父块，交给生成的是整页而不是命中那几句（§2.5）。
        没有检索到内容时返回 `NOT_FOUND` 与空引用，不调模型。

        :param game_id: 进哪个游戏知识库检索。
        :param version: 这次按哪个版本检索。空串表示没点名（回落 `current_version`）。
        :param current_version: 知识库标着的现行版本。两个都是空串时不做版本过滤，
            并留一条 warning——见 `ragamer.query.version_filter`。
        :raises ValueError: 问题为空。空问题会让检索查出任意一批切片。
        :raises ragamer.llm.LlmError: 生成失败。没有答案就是没有答案，不降级。
        """
        sources = self._prepare(
            question, game_id=game_id, version=version, current_version=current_version
        )
        if not sources:
            return Answer(NOT_FOUND, ())
        text = self.llm.complete(_request(question, sources))
        _warn_on_unknown_citations(text, len(sources))
        return Answer(text, _citations(sources), image_urls(sources))

    def stream_answer(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
    ) -> StreamingAnswer:
        """与 :meth:`answer` 同一条路，只是正文逐字流出来。

        两条路的取内容与编号完全一致（共用 :meth:`_prepare`）：同一个问题走不走流式，
        拿到的引用与图片必须是同一批，否则缓存命中与否会给出两种结果。

        **只在真的有内容可依据时才调模型**：没有检索到时吐出的就是 `NOT_FOUND`
        那一段常量（与 :meth:`answer` 同一个），不编一个答案。

        :raises ValueError: 问题为空。
        :raises ragamer.llm.LlmError: 生成失败。流到一半失败也照抛——半个答案比报错难查。
        """
        sources = self._prepare(
            question, game_id=game_id, version=version, current_version=current_version
        )
        if not sources:
            return StreamingAnswer((), (), iter((NOT_FOUND,)))
        return StreamingAnswer(
            _citations(sources), image_urls(sources), self._stream(question, sources)
        )

    def _prepare(
        self,
        question: str,
        *,
        game_id: str,
        version: str,
        current_version: str,
    ) -> tuple[_Source, ...] | None:
        """检索、聚合、编号。没有内容可依据时返回 `None`。

        返回 `None` 的两条路各自留痕：候选一条都没检到，以及命中了却按文档回查不出父块
        （索引与数据对不上）。两种情况下手里都没有内容，不调模型。
        """
        if not question.strip():
            raise ValueError("问题不能为空：空问题会让检索查出任意一批切片，答案也就是编的")
        where = version_filter(version, current_version=current_version)
        found = retrieve(
            question,
            game_id=game_id,
            chunks=self.chunks,
            embedder=self.embedder,
            reranker=self.reranker,
            where=where,
        )
        if not found:
            logger.info("提问 %r 没检索到内容，回明确回复，不调模型", question)
            return None
        # 聚合在截断之后：先由断崖定下哪些文档进得来，再按文档把兄弟切片一次查齐
        blocks = aggregate_parents(found, game_id=game_id, chunks=self.chunks, where=where)
        if not blocks:
            logger.warning(
                "提问 %r 命中 %d 条切片却聚合不出父块，按检索不到处理", question, len(found)
            )
            return None
        sources = tuple(
            _Source(Citation(index, block.doc_title, block.ancestor_path), block)
            for index, block in enumerate(blocks, start=1)
        )
        logger.info(
            "提问 %r 命中 %d 条切片、聚成 %d 个父块，交给生成", question, len(found), len(sources)
        )
        return sources

    def _stream(self, question: str, sources: Sequence[_Source]) -> Iterator[str]:
        """模型的流：一边吐一边攒，吐完才检查正文里有没有越界的编号。

        越界编号只有拿到整段答案才判得出来，而流式恰恰不留全文——所以在这里攒一份。
        调用方中途断开时这个生成器不会被走完，那次检查也就不做：没有答案可检查。
        """
        pieces: list[str] = []
        for piece in self.llm.stream(_request(question, sources)):
            pieces.append(piece)
            yield piece
        _warn_on_unknown_citations("".join(pieces), len(sources))


def _request(question: str, sources: Sequence[_Source]) -> LlmRequest:
    """一次生成调用：来源在系统提示里，问题在用户消息里。"""
    return LlmRequest(
        messages=[
            Message("system", _sources(sources)),
            Message("user", question),
        ],
        temperature=TEMPERATURE,
    )


def _sources(sources: Sequence[_Source]) -> str:
    """系统提示 = 约束 + 编号好的切片。

    编号取自 `source.citation.index`，与随答案交回去的那批是同一个值——不是在这里
    重新数一遍。
    """
    parts = [
        f"[{source.citation.index}] {source.citation.label}\n{_prompt_text(source.block)}"
        for source in sources
    ]
    return _INSTRUCTION + "\n\n".join(parts)


def _warn_on_unknown_citations(text: str, given: int) -> None:
    """答案里引了没给出的编号：那条来源不存在，这段答案无从核对。

    不改成错误——模型偶尔会多写一个编号，答案本身通常还有用；但一定要留痕，
    否则「引用了一个不存在的来源」这件事在界面上与正常答案长得一模一样。
    """
    cited = {int(found) for found in _MARKER.findall(text)}
    unknown = sorted(number for number in cited if not 1 <= number <= given)
    if unknown:
        logger.warning(
            "答案里引用了没有给出的资料编号 %s（这次给的是 1-%d 号）",
            "、".join(f"[{index}]" for index in unknown),
            given,
        )
