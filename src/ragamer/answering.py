"""生成：把检索到的资料交给模型，产出一段**能核对**的答案。

读取侧到这一步为止：问题进，答案与引用出来。中间是主检索路（`ragamer.retrieval`）
——取候选、精排、断崖截断。

三件事在这里定死：

- **答案必须带引用来源**：交给模型的每一条资料都编了号，编号连同「文档标题 + 祖先标题
  路径」一起进提示词，也一起随答案交回。没有引用的答案是一段无从核对的话——
  用户没法知道它是资料里写的还是模型编的。编号对不对得上由「引用与资料同批同序」
  保证：两边是从同一个序列里出来的，不是各算一遍。
- **检索不到就直说**。候选一条都没有时**不调模型**：没有资料可依据，让它自由发挥
  只会得到一段编造的游戏攻略，而且看起来和真答案一样。回复是这里的常量。
- **生成失败照抛**：没有答案就是没有答案。降级成一段「抱歉我答不上来」会把故障
  伪装成结果，比报错难查得多（与 `ragamer.query` 那一步的降级不同——那一步降级之后
  整条链路还能继续，这一步降级之后没有东西可以继续）。

引用里的编号是**人看的编号，从 1 起**，与提示词里资料的编号一致；列表下标那套不出现
在答案里。模型有没有真的在正文里引用某一条，v1 不做校验——那属于忠实度评测要做的事
（架构文档 §8）。能做的机械检查只有一条：正文里出现了范围之外的编号就留痕，
那种答案指向不存在的来源。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ragamer.chunking import PATH_SEPARATOR
from ragamer.llm import LlmClient, LlmRequest, Message
from ragamer.logging import get_logger
from ragamer.query import version_filter
from ragamer.retrieval import retrieve
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


@dataclass(frozen=True)
class Citation:
    """答案的一个来源。

    `index` 是**正文里 [n] 指的那个编号**，从 1 起；引用在元组里的顺序就是资料交给
    模型的顺序，两者一一对应。文档标题与祖先标题路径都要给出来——只给标题的话，
    一份长词条里是「打法」那一段还是「获取方式」那一段，读的人仍然对不上。
    """

    index: int
    doc_title: str
    ancestor_path: str

    @property
    def label(self) -> str:
        """给人和模型看的一行来源。

        祖先标题路径通常以文档标题开头（一级标题就是文档标题，它是标题树的第一层），
        所以对得上时不再重复念一遍。《》是给路径里可能出现的分隔符留的边界。
        """
        if not self.ancestor_path:
            return self.doc_title
        if self.ancestor_path == self.doc_title or self.ancestor_path.startswith(
            self.doc_title + PATH_SEPARATOR
        ):
            return self.ancestor_path
        return f"{self.doc_title}{PATH_SEPARATOR}{self.ancestor_path}"


@dataclass(frozen=True)
class Answer:
    """一次提问的结果。

    `citations` 为空即「没有检索到内容」，这时 `text` 是 `NOT_FOUND` 那段常量。
    调用方不必另外判断有没有答案——空引用就是那个信号。
    """

    text: str
    citations: tuple[Citation, ...]


def passage_text(chunk: Chunk) -> str:
    """一条资料交给模型的全文：正文 + 不参与向量化的附加文本。

    `content_meta` **必须带上**：表格里那些长文本列整列降级在那里（§2.5），
    只给正文等于把整列说明丢掉——它本来就是「随结果返回但不打分」的那部分（§2.2）。
    """
    meta = chunk.content_meta.strip()
    return f"{chunk.content}\n{meta}" if meta else chunk.content


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

        没有检索到内容时返回 `NOT_FOUND` 与空引用，不调模型。

        :param game_id: 进哪个游戏知识库检索。
        :param version: 这次按哪个版本检索。空串表示没点名（回落 `current_version`）。
        :param current_version: 知识库标着的现行版本。两个都是空串时不做版本过滤，
            并留一条 warning——见 `ragamer.query.version_filter`。
        :raises ValueError: 问题为空。空问题会让检索查出任意一批切片。
        :raises ragamer.llm.LlmError: 生成失败。没有答案就是没有答案，不降级。
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
            return Answer(NOT_FOUND, ())
        citations = tuple(
            Citation(index, hit.chunk.doc_title, hit.chunk.ancestor_path)
            for index, hit in enumerate(found, start=1)
        )
        logger.info("提问 %r 检索到 %d 条，交给生成", question, len(found))
        text = self.llm.complete(_request(question, [hit.chunk for hit in found], citations))
        _warn_on_unknown_citations(text, len(citations))
        return Answer(text, citations)


def _request(question: str, chunks: Sequence[Chunk], citations: Sequence[Citation]) -> LlmRequest:
    """一次生成调用：资料在系统提示里，问题在用户消息里。"""
    return LlmRequest(
        messages=[
            Message("system", _sources(chunks, citations)),
            Message("user", question),
        ],
        temperature=TEMPERATURE,
    )


def _sources(chunks: Sequence[Chunk], citations: Sequence[Citation]) -> str:
    """系统提示 = 约束 + 编号好的资料。

    资料与引用是两个序列但同样长、同次序，所以编号是数出来的而不是各写一遍
    ——两边各数一次，早晚有一边会数错。
    """
    blocks = [
        f"[{citation.index}] {citation.label}\n{passage_text(chunk)}"
        for chunk, citation in zip(chunks, citations, strict=True)
    ]
    return _INSTRUCTION + "\n\n".join(blocks)


def _warn_on_unknown_citations(text: str, given: int) -> None:
    """答案里引了没给出的编号：那条来源不存在，这段答案无从核对。

    不改成错误——模型偶尔会多写一个编号，答案本身通常还有用；但一定要留痕，
    否则「引用了一个不存在的来源」这件事在界面上与正常答案长得一模一样。
    """
    unknown = sorted(
        {int(found) for found in _MARKER.findall(text) if not 1 <= int(found) <= given}
    )
    if unknown:
        logger.warning(
            "答案里引用了没有给出的资料编号 %s（这次给的是 1-%d 号）",
            "、".join(f"[{index}]" for index in unknown),
            given,
        )
