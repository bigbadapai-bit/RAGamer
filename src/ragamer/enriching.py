"""补图：把图片里的信息变成可检索的文字（docs/ARCHITECTURE.md §1.3）。

MinerU 把版面分成「文字区域」与「图片区域」两块，**图片区域里的文字一个字都不给**
（§1.2，T10 实测 56 / 56 个图片条目，两个后端逐张一致）。独立上传的攻略截图是最坏的
情况——它没有周边正文，图内文字就是全部信息。所以这一层做三件事，缺一不可：

1. **展开 HTML 折叠块**：MinerU 若哪天把图内文字的分析结果塞进
   `<details><summary>text_image</summary>…</details>`，标记去掉、文字留下。
   切分器也认这几个标记，但它只去标记（`ragamer.chunking`）——`text_image` 这种
   摘要标记会跟着进正文，而折叠块整块落在表格或围栏里时它也不管。
2. **二次 OCR 回填**：对 `content_list` 里每一个 `type == "image"` 的条目取原图做
   OCR，文字插回正文中该图引用的**之后**。原图本身留着，答案里要能展示它。
3. **视觉摘要写进 alt**：给替代文本空着的图补一段可检索的描述。图里没有文字的那种
   （立绘、纯色块、示意图）不走这条路就一点可检索的文本都没有。

**范围按 T10 的实验定，不凭猜测**：每个 `type == "image"` 的条目都做，不设阈值、
不挑大小图——比例是 100%，跳过任何一张都是静默丢信息；而且要能对付「整页一张大图」
（22 张里 8 张如此），对这种图二次 OCR 等于把整页重新识别一遍。实验没有定下任何
识别阈值，所以这里也不留阈值配置（§11）。

**三条来源一起管。** 早先只有 MinerU 那条路进来（它带 `content_list`），网页与 md
的图是外链、按对象 key 取不到，一张都补不上。现在归一化那一层把外链也收进了对象存储
（`ragamer.sources.fetch_images`），所以这一层的输入又只剩一种形态：**正文里的图片引用
是对象 key**，按 key 取原图即可，不必知道这份资料从哪来。

只有一件事仍按来源分：**展开折叠块**只对 MinerU 那类产物做——md 与网页里的
`<details>` 可能是作者真的在讲这个元素，拆掉是在改用户写的东西。

**一张图补不上不让整份资料失败**（与 `ragamer.sources._unmapped` 同一条规矩）：
取不到原图、识别不出来、摘要调用失败，各留一条日志接着走——但**引擎级**的失败
（`OcrUnavailable`：依赖没装）不在此列，那会让每一张图都补不上，正是要防的静默丢失。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ragamer.llm import ImagePart, LlmClient, LlmError, LlmRequest, Message
from ragamer.logging import get_logger
from ragamer.ocr import OcrEngine, OcrError, OcrUnavailable
from ragamer.sources import (
    ImageRef,
    NormalizedDoc,
    image_refs,
    image_refs_in,
    set_image_alt,
    strip_image_refs,
)
from ragamer.stores.base import ObjectStore, StoreError, is_image_key

logger = get_logger(__name__)

#: MinerU 折叠块的摘要标记。它说明的是「这一块是图内文字」，不是内容本身，
#: 留着会变成一个进正文的假词。
_FOLD_SUMMARY = "text_image"

#: 整个 `<summary>…</summary>` 元素。摘要要按内容判断留不留，所以先整段取出来。
_SUMMARY = re.compile(r"<summary\b[^>]*>(.*?)</summary>", re.IGNORECASE | re.DOTALL)

#: 剩下的折叠块标记。只去标记，块内的文字留着。
_FOLD_TAG = re.compile(r"</?(?:details|summary)(?:\s[^>]*)?>", re.IGNORECASE)

#: 摘要的提示。要的是「不打开图也能搜到它」，所以问的是画了什么、有什么信息，
#: 不是问它好看不好看。**「一句话」由这句话本身约束**，不靠 `max_tokens`——理由见下面。
#:
#: 「只描述图上真有的字与物、不交代出处」这一句是必须的：模型很爱开头就报出这是哪款
#: 游戏的截图，而它**是靠猜的**——同一张燕云十六声的截图，两次分别猜成《永劫无间》与
#: 《逆水寒手游》。知识库本来就一个游戏一个库，游戏名在这里对检索毫无用处，猜错了却会
#: 把别的游戏的名字写进语料里被搜出来。**顺带还让摘要更具体**：不提出处之后它转而引
#: 图上真实存在的词（招式名、数值、界面文字），那正是检索用得上的东西。实测同一张图，
#: 旧提示词给出「NGA论坛《逆水寒手游》求助帖截图」这类开场，新提示词直接引原文。
_SUMMARY_PROMPT = (
    "这是从一份游戏攻略资料里切出来的一张图。用一句中文说清画面里看得见的内容："
    "界面元素、数值、专有名词、人物动作。只描述图上真有的字与物，不要交代它出自哪里"
    "——不写游戏名、作品名、网站名：那些对检索没用，猜错了还会把错名字混进语料。"
    "只输出这句话，不要前后缀。"
)

#: 摘要的 token 上限。**它是上限不是开销**：简单图照样只花几百个，抬高只是让复杂的图
#: 能跑完，不会让简单的图变贵。
#:
#: 定这么大是因为推理模型：`deepseek-flash` 这类模型会先烧掉一大段推理 token 才开口，
#: 而推理量随图的复杂度涨。实测 29 张真实截图：200 只够 25 张，4000 还差一张 95KB 的
#: 长帖截图。给不够时抛 `LlmTruncated`——那句话一个字都拿不到、alt 留空，图里没字的
#: 那几张因此少掉全部可检索文本，而界面上看不出来。
#:
#: **留的余量比实测值大一倍，因为推理长度会波动**：同一张 95KB 的图、同一提示词，
#: 8000 下有一次截断、另一次又通过。卡在实测边界上等于让它偶发失败，而失败是静默的。
_SUMMARY_MAX_TOKENS = 16000

#: 上下文跟着提示词走的那一段。
#:
#: **「只写图上才有、前后文字里没说过的」这句是必须的**：不这么写，模型会把周围正文
#: 复述一遍，那句摘要进了切片只是把同一段话算两遍分，榨不出新东西。
#:
#: **试过也否掉了一版**：曾经加过一句「说得出它属于谁、是哪一件就点出来」，想让摘要
#: 自报家门。实测同一段上下文里并列着九转金丹、太乙紫金丹、碧藕金丹三个名字，而图是
#: 碧藕金丹，模型报了「九转金丹」——**它有候选可挑，就一定会挑一个**。这与提示词上面
#: 那条「不要交代出处」是同一类事：错名字一旦进了语料就会被搜出来，而检索上不赚什么
#: （正文本就在同一片切片里，那个名字本来就检索得到）。所以只让它**认**图，不让它**报**名。
_SUMMARY_CONTEXT_PROMPT = (
    "\n\n这张图前后的原文如下，供你认出图上的是什么。"
    "只写图上才有、前后文字里没说过的信息，不要把前后文字复述一遍"
    "——点名它是谁、是哪一件也属于复述，不要写。\n"
)

#: 给视觉模型的上下文取多宽：图片前后各这么多字符。
#:
#: 量级按切片上限（`ragamer.chunking.DEFAULT_MAX_CHARS` = 800）的四分之一取：
#: 够认出「这是谁、这是哪一处」，又不至于把整节正文喂进去。与别的阈值一样，
#: **要调先有评测集**（§11）。
CONTEXT_CHARS = 100


def _context(stripped: str, item: ImageRef) -> str:
    """图片前后的一段原文，给视觉模型认人用。入参是**摘完地址**的正文。

    为什么要用摘完的那一份：窗口是硬切的 100 字符，边界会从中间切断邻居的地址，
    而半截地址 `strip_image_refs` 认不出来，就留在上下文里了——留给模型的是
    `thumb/b/b1/77u19lle…png/18px-%E5%9B%BE%E6%A0%87.png` 这种半截 URL 加一串
    百分号转义（真跑出来的样子）。那正是「地址混进正文」这件事，从提示词这条路
    又回来了。在摘完的正文上取窗口，就不存在被切断的地址。

    压平空白、修掉边上的碎片：取到的常常是表格行的尾巴（`| 说明 |`）。
    """
    start = max(0, item.start - CONTEXT_CHARS)
    end = min(len(stripped), item.end + CONTEXT_CHARS)
    return " ".join(stripped[start:end].split()).strip("|·-— ")


def _summary_message(context: str) -> str:
    return f"{_SUMMARY_PROMPT}{_SUMMARY_CONTEXT_PROMPT}{context}" if context else _SUMMARY_PROMPT


@dataclass(frozen=True)
class ImageEnricher:
    """补图。进出一份 :class:`NormalizedDoc`，位置在归一化与切分之间。

    `objects` 里存的是 `publish_assets` 发出去的原图——**进来之前那一步已经做过了**，
    所以 `content_list` 的 `img_path` 与正文里的引用这时都是对象 key，按 key 取图。
    """

    objects: ObjectStore
    ocr: OcrEngine
    #: 视觉模型。不接就只做二次 OCR——图里没有文字的那几张会少掉可检索的文本，
    #: 其余照旧。配不配由组合根决定（见 `ragamer.config.VisionSettings`）。
    vision: LlmClient | None = None

    def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
        """展开折叠块、补二次 OCR 的文字、给空 alt 写摘要。

        三条来源都过这里。**折叠块只对 MinerU 那类产物展开**：那对标记在 md 与网页里
        可能是作者真的在讲这个元素（理由见模块文档）。

        二次 OCR 也仍只对 MinerU 的条目做：网页与 md 的图，文字大多就在页面正文里；
        而识别是本地 CPU，实测 4.4–11.7 秒一张，铺开去等于把一次导入从几分钟拖成
        几十分钟（docs/experiments/second-pass-ocr.md）。
        """
        markdown = unfold(doc.markdown) if doc.content_list else doc.markdown
        entries = _image_entries(doc.content_list)
        assets = self._assets(markdown, entries)
        if doc.content_list:
            markdown = _backfill(markdown, self._texts(entries, assets))
        markdown = set_image_alt(markdown, self._alt_texts(markdown, assets))
        return replace(doc, markdown=markdown, images=image_refs(markdown))

    def _assets(self, markdown: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
        """要处理的那些图对应的原图字节，按对象 key 取。取不到的留一条痕就跳过。

        两个来源取并集：**正文里引用到的**，加上**条目级结构里点名的**。后者可能没在
        正文里留下引用（MinerU 偶尔会切出这样的条目），它的文字照样不该丢——
        `_backfill` 会把找不到落点的那段接到文末。

        **只认对象 key**（`is_image_key`）。归一化之后正文里的引用本该全是它
        （`ragamer.sources`）；剩下的不是故意没收的行内图标，就是取不到的相对路径。
        按 key 去取它们只会得到一串「原图取不到」的假警报，而那些图本来就没有原图
        在这一层。

        一张图缺了不该让整份资料进不了库，但静默跳过也不行——这一层存在的理由
        就是「图里的内容别再悄悄丢掉」。
        """
        wanted = dict.fromkeys(
            [
                *(item.ref for item in image_refs_in(markdown)),
                *(ref for ref in map(_img_path, entries) if ref),
            ]
        )
        assets: dict[str, bytes] = {}
        for ref in wanted:
            if not is_image_key(ref):
                continue
            try:
                assets[ref] = self.objects.get(ref)
            except StoreError as exc:
                logger.warning("%s：这张原图取不到（%s），它的文字这一趟补不上", ref, exc)
        return assets

    def _texts(
        self, entries: Sequence[Mapping[str, Any]], assets: Mapping[str, bytes]
    ) -> dict[str, str]:
        """每个图片条目一段可检索的文字。

        **条目自带了文字就不做 OCR**：`text` / `content` 是 MinerU 自己给的图内文字
        （今天的实测里 `content` 恒为空串，见 experiments/mineru-ocr.md），有它就没必要
        再识别一遍——顺便也让这条路径在 MinerU 哪天真的填上它时自动省下一次识别。
        """
        texts: dict[str, str] = {}
        for entry in entries:
            ref = _img_path(entry)
            if not ref or ref in texts:
                continue
            given = _given_text(entry)
            if given:
                texts[ref] = given
                continue
            data = assets.get(ref)
            if data is None:
                continue
            try:
                text = self.ocr.read(data)
            except OcrUnavailable:
                # 引擎用不了是整份资料的事，不是这一张图的事：抛出去让编排器
                # 按文件兜住，而不是在这里静默地一张一张丢掉
                raise
            except OcrError as exc:
                logger.warning("%s：这张图二次 OCR 失败（%s），它的文字这一趟补不上", ref, exc)
                continue
            if text.strip():
                texts[ref] = text
        return texts

    def _alt_texts(self, markdown: str, assets: Mapping[str, bytes]) -> dict[str, str]:
        """给替代文本空着的图配一段摘要。没接视觉模型时一个都不做。

        Markdown 与 HTML 两种引用都做：表格内嵌的图是 HTML 那种，而它恰恰是
        「图里没有文字」概率最高的一类（图标、示意图），不补就一点可检索的文本都没有。

        **遍历的是 `strip_image_refs` 给出的那批引用**，不是 `image_refs_in` 的：
        上下文要在摘完地址的正文上取，而只有它给出的落点是在那份正文里的（见 `_context`）。
        两者认的是同一批引用，`ImageRef` 上该有的都在。
        """
        vision = self.vision
        if vision is None:
            return {}
        stripped, refs = strip_image_refs(markdown)
        alt_by_ref: dict[str, str] = {}
        for item in refs:
            if item.alt.strip() or item.ref in alt_by_ref:
                continue
            data = assets.get(item.ref)
            if data is None:
                continue
            try:
                summary = self._summarize(vision, data, _context(stripped, item))
            except LlmError as exc:
                logger.warning("%s：这张图的摘要没取到（%s），它的 alt 留空", item.ref, exc)
                continue
            if summary:
                alt_by_ref[item.ref] = summary
        return alt_by_ref

    def _summarize(self, vision: LlmClient, data: bytes, context: str) -> str:
        """一次带图的调用换一句话。返回值已压成一行、去掉方括号（见 `_alt_text`）。"""
        answer = vision.complete(
            LlmRequest(
                messages=[
                    Message(
                        "user",
                        _summary_message(context),
                        images=(ImagePart(data=data, content_type=_media_type(data)),),
                    )
                ],
                max_tokens=_SUMMARY_MAX_TOKENS,
            )
        )
        return _alt_text(answer)


def unfold(markdown: str) -> str:
    """展开 MinerU 的 `<details>` 折叠块：标记去掉、块里的文字留下。

    **只在带条目级结构的产物上调用**（`ImageEnricher.enrich` 已经把门关好了）：
    md 与网页里那对标记可能是真的在讲这个元素，把它拆掉是在改用户写的东西。

    摘要那一段整段丢掉：MinerU 的折叠块里它是一句标签（`text_image`，
    说明「这一块是图内文字」），留着会变成一个进正文的假词。**别的摘要文本留着**——
    那可能是真的正文（`<summary>安装步骤</summary>`），丢了就找不回来了。
    """
    return _FOLD_TAG.sub("", _SUMMARY.sub(_unsummarize, markdown))


def _unsummarize(match: re.Match[str]) -> str:
    return "" if match.group(1).strip() == _FOLD_SUMMARY else match.group(1)


def _image_entries(content_list: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """条目级结构里的图片条目。实验的口径就是它：`type == "image"`，不设别的条件。"""
    return tuple(entry for entry in content_list if entry.get("type") == "image")


def _img_path(entry: Mapping[str, Any]) -> str:
    ref = entry.get("img_path")
    return ref if isinstance(ref, str) else ""


def _given_text(entry: Mapping[str, Any]) -> str:
    """条目自带的图内文字。`text` 与 `content` 两个键名都认——后端不同键名不同。"""
    for key in ("text", "content"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _backfill(markdown: str, texts: Mapping[str, str]) -> str:
    """把二次 OCR 的文字插到对应图片引用的**之后**。

    插在引用**所在那一行的末尾**，不是紧贴引用的右括号：引用常常在一行字当中
    （表格内嵌的图就是这样），贴着插会把那句话拦腰截断。原图引用本身不动。

    正文里找不到引用的那些地址接到文末并留一条痕。**不丢**：走到这一步的每个字
    都是 MinerU 少给的那部分，丢掉就又是这一层要防的那件事——而「接在文末」至少
    还检索得到，比只留在日志里强。
    """
    if not texts:
        return markdown
    cuts: list[tuple[int, str]] = []
    placed: set[str] = set()
    for item in image_refs_in(markdown):
        text = texts.get(item.ref)
        # 同一张图在正文里被引用两次时只插一次：插两遍会让同一段文字进两个切片
        if text is None or item.ref in placed:
            continue
        placed.add(item.ref)
        cuts.append((_line_end(markdown, item.end), text))
    orphan = {ref: text for ref, text in texts.items() if ref not in placed}
    if orphan:
        logger.warning(
            "%d 个图片条目的文字在正文里找不到可落的引用，接到文末（%s）",
            len(orphan),
            "、".join(orphan),
        )
        cuts.append((len(markdown), "\n\n".join(orphan.values())))
    for position, text in reversed(cuts):
        markdown = f"{markdown[:position]}\n\n{text}{markdown[position:]}"
    return markdown


def _line_end(markdown: str, position: int) -> int:
    """这一处所在那一行的行尾（换行符的位置，没有换行就是文末）。"""
    found = markdown.find("\n", position)
    return len(markdown) if found < 0 else found


def _alt_text(answer: str) -> str:
    """把模型给的一句话压成一行，并去掉会撑破容器的字符。

    替代文本有两个落点：Markdown 的 `![…]` 里出现方括号会把这一段引用拆掉，
    HTML 的 `alt="…"` 里出现引号同理——两处都得清，清多了只是少两个字。
    """
    return re.sub(r"[\[\]\"]", "", " ".join(answer.split())).strip()


def _media_type(data: bytes) -> str:
    """图片的形态，写进 data URI 给模型看。

    按魔数认而不是按对象名的后缀：**类型报错在有些服务端上不是报错，是把图当成坏的
    直接忽略**，而对象 key 的后缀来自解析产物的文件名，不由这一层保证。
    认不出的一律按 JPEG 报——截图与长图绝大多数是它。
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"
