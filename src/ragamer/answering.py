"""生成：把检索到的资料交给模型，产出一段**能核对**的答案。

读取侧到这一步为止：问题进，答案与引用出来。中间是检索（`ragamer.retrieval`）
——按路由走选中的那几路取候选、融合、精排、断崖截断、按文档聚合父块，外加联网兜底
搜回来的那几条。

**两种给法**：:meth:`Answerer.answer` 一次给全，:meth:`Answerer.stream` 逐字给。
两者共用同一段检索与同一份提示词（:meth:`Answerer._sources`），差别只在正文怎么出来；
引用在流式这一路是**先**出来的，因为它在检索那一步就定下来了。

五件事在这里定死：

- **一条资料是一个父块，不是一个切片**。命中并截断之后按文档回查兄弟切片
  （`aggregate_parents`），于是问"二郎神怎么打"时模型拿到的是整页——包括"掉落"，
  追问"掉什么"不必重新检索。进父块的是被截断那批切片**所属的文档**，
  不是它们自己那几句：引用因此指向文档（超长文档里则指向那一小节），不是某一句话。
- **答案带引用来源，正文里不带编号**：交给模型的每一个父块都编了号，编号连同「文档标题
  + 祖先标题路径」一起进提示词，也一起随答案交回，界面把它们列成一份来源清单。
  **编号不进正文**：它是给人核对的，而读的人手上是一份并排的来源列表，句末挂一个
  数字既指不出是哪条、又铺满整段（一段答案全出自同一个父块时，每一行都带 `[1]`）。
  所以提示词里明说不标。引用与内容同批同序由两者绑在同一个 :class:`_Source` 里保证：
  不是两个各排一遍的序列。
- **答案还要带图片地址**。图片留在正文里（§1.3），从交给模型的那批内容里取出来随答案
  交回，用户才不必跳出去找原图。它跟着引用走，不是另一次检索。
- **检索不到就直说**。候选一条都没有时**不调模型**：没有内容可依据，让它自由发挥
  只会得到一段编造的游戏攻略，而且看起来和真答案一样。回复是这里的常量。
- **网络来源与语料分开标**。两者混在一份答案里而不标出来，等于把「这是我们语料里
  写的」与「这是网上说的」说成同一件事——而网络内容可能过时、也可能与知识库冲突。
  标注落在两处：资料清单里那一行的前缀（:data:`WEB_PREFIX`，给模型看），
  以及引用上的 `url`／`origin`（给人看、也给界面判）。
- **生成失败照抛**：没有答案就是没有答案。降级成一段「抱歉我答不上来」会把故障
  伪装成结果，比报错难查得多（与 `ragamer.query` 那一步的降级不同——那一步降级之后
  整条链路还能继续，这一步降级之后没有东西可以继续）。

资料清单里的编号是**人看的编号，从 1 起**，也是来源清单的顺序；列表下标那套不出现
在答案里。**正文里不要求出现编号**（见上），所以「模型有没有引用某一条」v1 不校验
——那属于忠实度评测要做的事（架构文档 §8）。留着的机械检查只有一条：正文里出现了
范围之外的编号就留痕，那种答案指向一条不存在的来源。提示词已经明说不标了它还标出来，
多半是惯性，更该看见。

提示词里管这批切片叫「资料」是给模型看的说法，代码里不引这个词——
`ragamer.importing` 那边「资料」指的是一份源文件，两个意思撞在一起会读岔。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ragamer.chunking import PATH_SEPARATOR
from ragamer.live import TurnCancelled
from ragamer.llm import LlmClient, LlmRequest, Message
from ragamer.logging import get_logger
from ragamer.query import version_filter
from ragamer.retrieval import ParentBlock, aggregate_parents, retrieve
from ragamer.routing import Route
from ragamer.stores.base import Chunk, ChunkStore, is_image_key
from ragamer.vectors.base import Embedder, Reranker
from ragamer.websearch import WebResult, WebSearch

logger = get_logger(__name__)

#: 生成用的温度。钉死 0：答案要照着资料写、不追求多样性，同一个问题两次问出同一个
#: 答案才谈得上核对，缓存层（架构文档 §4）也才好命中。
TEMPERATURE = 0.0

#: 检索不到内容时的回复。**这不是模型写的**，也没有调模型——理由见模块说明。
NOT_FOUND = (
    "知识库里没有找到与这个问题相关的资料。换个问法试试，或者先把这个游戏的相关资料导入知识库。"
)

#: 网络来源在资料清单里的前缀。**必须与语料里的资料分开标**：网络内容可能过时、
#: 也可能与知识库里的说法冲突，两者混在一份答案里而读的人看不出来，等于把
#: 「这是我们语料里写的」与「这是网上说的」说成同一件事。
WEB_PREFIX = "【网络】"

#: 生成用的系统提示。这几条约束各对应一种失败：编造资料里没有的、把提示词本身复述成
#: 答案、正文里挂满引用编号，以及把网上的说法当成知识库里的说法。
_INSTRUCTION = (
    "你是游戏攻略助手。只依据下面列出的资料回答用户的问题。\n"
    "资料里没写到的就说资料里没有，不要凭印象补充数值、打法或结论。\n"
    f"标着{WEB_PREFIX}的那几条来自网络检索，可能滞后或与知识库不一致；"
    "用到它们时要在答案里说明这是网络上的说法。\n"
    "用中文直接回答，不要复述这几条要求。正文里不要写引用编号或别的出处标记"
    "——资料前面的编号是给系统对号用的，来源会由界面单独列出来。\n"
    "资料：\n"
)

#: 正文里的引用编号。**提示词已经不要求标了**（`_INSTRUCTION`），留着这一条是兜底：
#: 模型照旧标的时候，范围之外的编号指向的是一条不存在的来源，那种答案无从核对。
_MARKER = re.compile(r"\[(\d+)\]")

#: 答案里最多带几张原图。
#:
#: 聚合的是父块而不是命中的那几句（§2.5），一个词条页整页进来时能带十几张，全铺在
#: 答案下面会把正文淹掉——而图是**用来看出处的**，不是答案本身。按首次出现的顺序截
#: 前几张：顺序跟着引用走，前几张就是最相关那几个父块里的。
#: 与别的经验值一样，要调先有评测集（§11）。
MAX_IMAGES = 6


@dataclass(frozen=True)
class Citation:
    """答案的一个来源。

    `index` 是**正文里 [n] 指的那个编号**，从 1 起；引用在元组里的顺序就是交给模型的
    顺序，两者一一对应。文档标题与祖先标题路径都要给出来——只给标题的话，一份长词条
    里是「打法」那一段还是「获取方式」那一段，读的人仍然对不上。

    **`url` 非空即网络来源**（`origin` 那个属性就是照它判的）。分成两个字段而不是一个
    `origin` 枚举：对网络来源来说地址本来就要显示出来，而语料里的切片没有地址可给——
    一个空串与一个有值的串，比「枚举 + 可能为空的地址」少一种对不上的组合。

    这个判据的边界要记住：`url` 说的是**检索期从外面搜回来的那一条**。另一条堆叠线上
    的 `Chunk.source_url`（导入的网页）是语料自己的出处，不是网络来源——哪天要把它也
    显示出来，这个判据就得换成显式的来源标记，否则每一份导入的网页都会变成「网络来源」，
    而那是静默的。
    """

    index: int
    doc_title: str
    ancestor_path: str
    #: 网络来源的地址。语料里查到的切片是空串。
    url: str = ""
    #: 网络来源的发布时间，服务给什么就是什么。语料里的是空串。
    published_at: str = ""

    @property
    def origin(self) -> Literal["local", "web"]:
        """这条来源是知识库里查到的，还是网上搜来的。

        读答案的人要分得清——网络内容可能过时、也可能与知识库里的说法冲突，两种来源
        混在一份答案里而不标出来，等于把「这是我们语料里写的」与「这是网上说的」
        说成同一件事。
        """
        return "web" if self.url else "local"

    @property
    def label(self) -> str:
        """给人和模型看的一行来源。

        祖先标题路径通常以文档标题开头（一级标题就是文档标题，它是标题树的第一层），
        所以对得上时不再重复念一遍。比对前缀时要落在分隔符上：文档「二郎神」与文档
        「二郎神外传」是两份文档，只按字面前缀比会把后者误当成前者的一部分。

        网络来源给的是地址（必要时带发布时间）：它没有祖先标题路径可言，而读的人要能
        自己去看一眼原文。
        """
        if self.url:
            return (
                f"{self.doc_title}（{self.url}）"
                if not self.published_at
                else (f"{self.doc_title}（{self.url}，{self.published_at}）")
            )
        if not self.ancestor_path:
            return self.doc_title
        if self.ancestor_path == self.doc_title or self.ancestor_path.startswith(
            self.doc_title + PATH_SEPARATOR
        ):
            return self.ancestor_path
        return f"{self.doc_title}{PATH_SEPARATOR}{self.ancestor_path}"


@dataclass(frozen=True)
class _Source:
    """编号好的一个来源：引用 + 交给生成的那段文字。

    两者绑在一个类型里而不是两个平行序列：编号与内容本来就是同一件事的两面，
    分开放就得靠调用方保证两边同长同序，对不上时是静默的（编号指向另一条内容）。

    语料里来的那段是父块拼出来的整页（:func:`_prompt_text`），网络来的是搜索服务给的
    摘要——两者的取法不同，但到了这一步都只是「一段要编号的文字」。
    """

    citation: Citation
    text: str
    #: 这条来源带出来的图片地址。语料里的取自切片自己的字段（:func:`_block_images`），
    #: 网络来源没有——搜索服务给的是一段摘要，没有图片可言。**地址不进 `text`**：
    #: 它唯一的去处是答案下面那排图，不该占着模型的上下文。
    images: tuple[str, ...] = ()


def require_question(question: str) -> None:
    """问题不能是空的。读取侧的入口都从这里过一遍（`Answerer.answer`、`Answerer.stream`、
    `ragamer.clarifying.Clarifier.decide`、`ragamer.conversations.Chat.ask`）——空问题会让
    检索查出任意一批切片，答案也就是编的，而这几种失败都不会报错。

    提到一个函数里是因为几个入口各自守一遍时，那句话会被抄成几份——文案一旦分岔，
    同一个毛病在两处就说成两件事了（与 `ragamer.stores.base.require_vectors` 同一个打法）。
    会话那一步尤其要在调模型**之前**先拦一次：晚一步就白花一次提问理解的调用。
    """
    if not question.strip():
        raise ValueError("问题不能为空：空问题会让检索查出任意一批切片，答案也就是编的")


@dataclass(frozen=True)
class Answer:
    """一次提问的结果。

    `citations` 为空即「没有检索到内容」，这时 `text` 是 `NOT_FOUND` 那段常量。
    调用方不必另外判断有没有答案——空引用就是那个信号。

    `images` 是交给生成的那批内容里出现过的图片地址：答案里要能直接展示原图，
    用户不必跳出去找（用户故事 52）。它是**跟着引用走**的，不是另一次检索的结果——
    地址跟着切片走（`Chunk.image_urls`），切片跟着父块走，父块跟着引用走。至多
    :data:`MAX_IMAGES` 张。正文里没有它们：地址在切分时就摘走了（`ragamer.chunking`）。
    """

    text: str
    citations: tuple[Citation, ...]
    images: tuple[str, ...] = ()


@runtime_checkable
class ReadSide(Protocol):
    """读取侧对外的那两个动作：一次问全、逐字问。

    :class:`Answerer` 是直接检索生成的那一份，`ragamer.caching.CachedAnswerer` 在它前面
    挡了一层缓存。**两者实现同一组签名**，所以对话那一层（`ragamer.conversations`）
    只认这一个协议——接不接缓存是组合根的事，多轮对话那一层不必知道。
    """

    def answer(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
        route: Route | None = None,
    ) -> Answer:
        """读一个问题，给出答案与它的来源。"""
        ...

    def stream(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
        route: Route | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AnswerStream:
        """与 :meth:`answer` 同一套检索与提示，只是正文逐字产出。"""
        ...


@dataclass(frozen=True)
class AnswerStream:
    """一次提问的流式形态：**引用与图片先定下来，正文逐字来**。

    检索、精排、聚合父块在 :meth:`Answerer.stream` 返回之前就跑完了，所以 `citations`
    与 `images` 到这一刻已经是最终的那一批——界面可以先把来源与图片渲染出来，
    不必等正文吐完。两者与正文放在同一层返回，而不是各成一路：分开成两个平行序列
    就得靠调用方保证两边对得上，对不上时是静默的（引用指向另一段正文）。

    `deltas` 是**惰性**的：模型那一段要等调用方真的开始迭代才发出去。好处不只是省一次
    网络往返——调用方**随时可以把它丢掉**（客户端断开、页面关掉），丢掉之后这边不留
    任何痕迹，因为这一层从头到尾没有累积过整段正文（唯一那份累积在
    `Answerer._streamed` 里，它随生成器一起被关掉）。会话历史要不要记下这一轮，
    因此完全是调用方的事，见 `ragamer.conversations`。

    `citations` 为空即「没有检索到内容」，这时 `deltas` 只有 `NOT_FOUND` 那一片，
    与 :class:`Answer` 是同一条口径。
    """

    citations: tuple[Citation, ...]
    #: 交给生成的那批内容里出现过的图片地址，见 :class:`Answer`。
    images: tuple[str, ...]
    deltas: Iterator[str]


def _numbered(blocks: Sequence[ParentBlock], web: Sequence[WebResult]) -> tuple[_Source, ...]:
    """语料的父块与网络来源合成一份**连续编号**的资料清单。

    语料在前、网络在后：编号是正文里 [n] 指的那个号，读的人顺着往下看时先看到的是
    知识库里查到的，网络那几条排在末尾——它是兜底，本该如此。

    编号在**两批之间连续**，不是各编各的：模型看到的是一份资料清单，断号会让它以为
    中间还有没给它的东西。
    """
    sources = [
        _Source(
            Citation(index, block.doc_title, block.ancestor_path),
            _prompt_text(block),
            _block_images(block),
        )
        for index, block in enumerate(blocks, start=1)
    ]
    sources += [
        _Source(
            Citation(
                len(sources) + offset,
                result.title,
                "",
                url=result.url,
                published_at=result.published_at,
            ),
            result.text,
        )
        for offset, result in enumerate(web, start=1)
    ]
    return tuple(sources)


def _citations(sources: Sequence[_Source]) -> tuple[Citation, ...]:
    """编号好的来源 → 交回给调用方的引用。顺序就是提示词里的顺序，一一对应。"""
    return tuple(source.citation for source in sources)


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


def _block_images(block: ParentBlock) -> tuple[str, ...]:
    """一个父块带出来的图片地址：块内每条切片自己的，按源文档顺序去重。

    切片里的地址是**切分时从正文摘下来的**（`ragamer.chunking`），不再拿正则扫一遍
    正文——正文里已经没有地址了。所以这里是读字段，不是从交给模型的那段文字里挑。

    **再按对象 key 过一道**（`is_image_key`）：切分那一层已经只留取得到原图的
    （`_servable`），这里是读侧的兜底——回显是拿地址去对象存储取的，库里要是留着一条
    不是 key 的旧地址（这一层加上之前导进去的），答案里就多一条取不到的死图。
    """
    return tuple(
        dict.fromkeys(
            url for chunk in block.chunks for url in chunk.image_urls if is_image_key(url)
        )
    )


def _image_urls(sources: Sequence[_Source]) -> tuple[str, ...]:
    """交给生成的这批内容里出现过的图片地址，按首次出现的顺序去重，至多 :data:`MAX_IMAGES` 张。

    **跟着引用走**：哪几条内容进得了提示词，它们带的图就在答案里显示。各来源自己的
    图片在组装来源清单时就定下了（`_numbered`），这里只做去重、排序与截断——同一个
    地址经两个父块交回来是常事（同文档的两个小节各带一次）。
    """
    addresses = dict.fromkeys(url for source in sources for url in source.images)
    return tuple(list(addresses)[:MAX_IMAGES])


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
    #: 联网兜底那一路要用的外部检索。**不配就是 `None`**：这一路跳过，其余照跑
    #: （见 `ragamer.websearch`）——它不是「答不出来」的理由。
    search: WebSearch | None = None

    def answer(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
        route: Route | None = None,
    ) -> Answer:
        """读一个问题，给出答案与它的来源。

        截断之后按文档聚合父块，交给生成的是整页而不是命中那几句（§2.5）。
        没有检索到内容时返回 `NOT_FOUND` 与空引用，不调模型。

        :param game_id: 进哪个游戏知识库检索。
        :param version: 这次按哪个版本检索。空串表示没点名（回落 `current_version`）。
        :param current_version: 知识库标着的现行版本。两个都是空串时不做版本过滤，
            并留一条 warning——见 `ragamer.query.version_filter`。
        :param route: 这次走哪几路召回。**不给就只走主检索路**——选路要的是查询类型，
            而那是 `ragamer.query.understand` 的产物，这一层拿不到也不该假装拿得到
            （`ragamer.conversations` 那边判出来再传进来）。
        :raises ValueError: 问题为空（由 :func:`require_question` 报出来）。
        :raises ragamer.llm.LlmError: 生成失败。没有答案就是没有答案，不降级。
        """
        require_question(question)
        sources = self._sources(
            question,
            game_id=game_id,
            version=version,
            current_version=current_version,
            route=route,
        )
        if not sources:
            return Answer(NOT_FOUND, ())
        text = self.llm.complete(_request(question, sources))
        _warn_on_unknown_citations(text, len(sources))
        return Answer(text, _citations(sources), _image_urls(sources))

    def stream(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
        route: Route | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AnswerStream:
        """与 :meth:`answer` 同一套检索与提示，只是正文逐字产出。

        引用与正文分两步给：检索那一段**在返回之前**就跑完了（它决定引用与图片，也决定
        要不要调模型），模型那一段则等调用方开始时才发出去。没有检索到内容时同样不调模型，
        `NOT_FOUND` 那段常量作为流的第一片也是唯一一片交出去——界面上两种情况的呈现
        一样，只是这一种不会有引用。

        这一步与 `answer` 一样**不落任何盘**：调用方可以在正文吐到一半时把它丢掉，
        不留痕迹。会话历史该不该记下这一轮，由调用方在收完之后决定（`ragamer.conversations`）。

        :param game_id: 进哪个游戏知识库检索。
        :param version: 这次按哪个版本检索。空串表示没点名（回落 `current_version`）。
        :param current_version: 知识库标着的现行版本。
        :param route: 这次走哪几路召回。理由同 :meth:`answer`。
        :raises ValueError: 问题为空。空问题会让检索查出任意一批切片。
        :raises ragamer.llm.LlmError: 生成失败。`deltas` 迭代到一半才炸是常事——
            这时已经吐出去的内容是收不回的，调用方应当把整轮丢掉而不是记半句。
        """
        require_question(question)
        sources = self._sources(
            question,
            game_id=game_id,
            version=version,
            current_version=current_version,
            route=route,
            cancelled=cancelled,
        )
        if not sources:
            return AnswerStream((), (), iter((NOT_FOUND,)))
        return AnswerStream(
            _citations(sources),
            _image_urls(sources),
            self._streamed(question, sources, cancelled=cancelled),
        )

    def _sources(
        self,
        question: str,
        *,
        game_id: str,
        version: str,
        current_version: str,
        route: Route | None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[_Source, ...]:
        """检索、聚合父块、编号。**一条内容都没有时返回空元组**，不编造内容。

        `answer` 与 `stream` 共用这一段：两条路给出去的引用与图片必须是同一批、同一个顺序，
        各写一遍迟早会分岔——而引用对不上内容这件事，从答案本身看不出来。

        `cancelled` 只走 `stream` 那条路：`answer` 是同步一次给全的，调用方没有中途
        收手的时机。
        """
        where = version_filter(version, current_version=current_version)
        found = retrieve(
            question,
            game_id=game_id,
            chunks=self.chunks,
            embedder=self.embedder,
            reranker=self.reranker,
            where=where,
            route=route,
            llm=self.llm,
            search=self.search,
            cancelled=cancelled,
        )
        # 聚合在截断之后：先由断崖定下哪些文档进得来，再按文档把兄弟切片一次查齐
        blocks = (
            aggregate_parents(found.hits, game_id=game_id, chunks=self.chunks, where=where)
            if found.hits
            else ()
        )
        if found.hits and not blocks:
            # 命中了却一条都回查不出来：索引与数据对不上。这种时候不该
            # 悄悄换成联网那批顶上——那是两种完全不同的故障，混在一起就查不出了
            logger.warning(
                "提问 %r 命中 %d 条切片却聚合不出父块，按检索不到处理", question, len(found.hits)
            )
        sources = _numbered(blocks, found.web)
        if not sources:
            logger.info("提问 %r 没检索到内容，回明确回复，不调模型", question)
            return ()
        logger.info(
            "提问 %r 命中 %d 条切片、聚成 %d 个父块、另有 %d 条网络来源，交给生成",
            question,
            len(found.hits),
            len(blocks),
            len(found.web),
        )
        return sources

    def _streamed(
        self,
        question: str,
        sources: Sequence[_Source],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """逐字转出去，**吐完之后**才检查引用编号。

        编号检查要整段正文才做得成，而流式这一路没有累积——所以在这里攒一份。
        调用方在正文收完之前就把流丢掉时，这个生成器会被关掉，检查也就不做了：
        那一轮本来就不该留下任何东西，没有正文可核对。

        **取消那一查放在这一层，而不是调用方的循环里**：模型那段等待发生在两次
        `yield` 之间，只有在这里才看得见它。抛出去时这一帧跟着销毁，上游那个模型流
        被关闭（`GeneratorExit` 传进去），那边的 HTTP 请求也就断了——不必等它吐完。
        """
        produced: list[str] = []
        for piece in self.llm.stream(_request(question, sources)):
            if cancelled is not None and cancelled():
                raise TurnCancelled
            produced.append(piece)
            yield piece
        _warn_on_unknown_citations("".join(produced), len(sources))


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
    """系统提示 = 约束 + 编号好的资料。

    编号取自 `source.citation.index`，与随答案交回去的那批是同一个值——不是在这里
    重新数一遍。**这个编号只在清单里用**：正文里不写它（`:data:`_INSTRUCTION` 里
    明说了），它是给人核对清单用的。网络来源那一行带 :data:`WEB_PREFIX`：模型据此
    在答案里交代出处。
    """
    parts = [
        f"[{source.citation.index}] {_mark(source.citation)}{source.citation.label}\n{source.text}"
        for source in sources
    ]
    return _INSTRUCTION + "\n\n".join(parts)


def _mark(citation: Citation) -> str:
    """资料清单里那一行的来路前缀。语料里查到的不标——标的是少数那一类。

    判据取 `citation.origin` 而不是「url 是不是空串」：两者今天等价，但来源的判法
    只该有一处（`Citation.origin`），在这里重写一遍就等着哪天两处对不上。
    """
    return WEB_PREFIX if citation.origin == "web" else ""


def _warn_on_unknown_citations(text: str, given: int) -> None:
    """答案里引了没给出的编号：那条来源不存在，这段答案无从核对。

    提示词已经明说不要标编号了，所以这一条现在是**兜底**：模型照旧标（它见到的
    资料清单上就带着编号，惯性很难免），而范围之外的编号尤其危险——它指的那条来源
    根本不在这次给出的资料里。

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
