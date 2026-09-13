"""读取侧入口：缓存挡在主检索路前面（docs/ARCHITECTURE.md §4）。

热门问题秒回。游戏攻略的查询分布极度倾斜——「二郎神怎么打」会被成千上万个玩家问到，
是同一句话；而改写那一步（`ragamer.query.understand`）已经把口语问法归一了，
所以精确匹配就有很高的命中率。

四条口径在这里定死：

- **命中就整份交回**：答案、引用、图片一个不少（`CachedAnswer`）。只缓存文本的话，
  命中之后引用与图片就没了——那条路径会静默地比未命中时少东西，界面上看不出区别。
- **命中也要逐字流式**。缓存里的答案是一整段，直接一次吐出去就是「流式」在缓存路径上
  静默失效：未命中时一个字一个字出来，命中时整段砸下来，而命中恰恰是最常走的那条路。
  `replay` 按同一个粒度把它重放出去，两条路的观感因此一致。
- **没检索到的结果不入缓存**：它随时会因为新资料而改变，存下来等于把一次「暂时没有」
  钉成 7 天的「没有」。
- **缓存不可用一律降级**：读、写、计数任何一步失败都只留一条日志，这次按没缓存走，
  **作答照常**。缓存是加速器，不是链路的一环。

改写那一步不在这里，也不因为缓存命中而省掉：缓存键就建立在它的输出上（§4）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ragamer.answering import Answer, Answerer, AnswerStream
from ragamer.caching.base import (
    TOP_QUESTIONS,
    TTL_SECONDS,
    AnswerCache,
    CachedAnswer,
    CacheError,
    cache_key,
)
from ragamer.logging import get_logger
from ragamer.query import effective_version, normalize_query

logger = get_logger(__name__)


def replay(text: str) -> Iterator[str]:
    """把一条缓存好的答案**逐字**吐出来。命中缓存时用它伪装成流式（§4）。

    粒度与未命中那条路取齐：那边吐的是模型流下来的 delta，一次也就一两个字，验收口径
    写的是「逐字流式输出」。缓存这边若按短句成片地给（40 字一跳），用户看到的是两种
    观感——而未命中时的「等」被摊平在几百个 delta 上，缓存命中反而是这里最常走的一条路，
    差异最容易被看见。

    逐字之间不带等待：节奏是传输层的事（假模型逐字吐也是这个口径），这一层只负责
    「不要退化成一次性返回」。
    """
    yield from text


@dataclass(frozen=True)
class CachedAnswerer:
    """读取侧的唯一入口。缓存挡在 `ragamer.answering.Answerer` 前面。

    与写入侧的 `ragamer.importing.Importer` 是同一个打法：做成不可变对象，一次接线
    反复使用。`answer` 与 `stream` 是同一件事的两种吐法，两者**共用同一套键与同一份
    缓存值**——一个问题走不走流式，命中的必须是同一条。
    """

    answers: Answerer
    cache: AnswerCache
    #: 缓存存活时间。默认 7 天：长过期是兜底，主要失效手段是导入时按游戏前缀批量删。
    ttl: int = TTL_SECONDS

    def answer(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
        rewritten_query: str = "",
    ) -> Answer:
        """命中缓存就拼出上次那份答案，未命中就走主检索路并把结果写回。

        :param game_id: 进哪个游戏知识库检索，同时也是缓存键里的游戏那一段。
        :param version: 这次按哪个版本检索。空串表示没点名（回落 `current_version`）。
        :param current_version: 知识库标着的现行版本。
        :param rewritten_query: 改写后的问法（`ragamer.query.understand`）。空串表示那一步
            降级了，这时按原问题算键——**绝不落到空串上**，那会让所有降级提问共用一个键。
        :raises ValueError: 问题为空。这时不查也不写缓存，直接交给下层当场报错。
        :raises ragamer.llm.LlmError: 生成失败。缓存只在生成成功之后才写。
        """
        key = self._key(
            question,
            game_id=game_id,
            version=version,
            current_version=current_version,
            rewritten_query=rewritten_query,
        )
        cached = None if key is None else self._read(key, question)
        if cached is not None:
            return Answer(cached.text, cached.citations, cached.images)
        answer = self.answers.answer(
            question, game_id=game_id, version=version, current_version=current_version
        )
        if key is not None:
            self._write(key, CachedAnswer.of(answer), question)
        return answer

    def stream(
        self,
        question: str,
        *,
        game_id: str,
        version: str = "",
        current_version: str = "",
        rewritten_query: str = "",
    ) -> AnswerStream:
        """:meth:`answer` 的流式形态：命中就重放缓存，未命中边走边吐。

        两条路吐出来的东西一样多——**命中时逐字流式不是可选项**，它是「体验与未命中
        一致」的全部内容。

        :raises ValueError: 问题为空。
        :raises ragamer.llm.LlmError: 生成失败。流到一半失败也照抛。
        """
        key = self._key(
            question,
            game_id=game_id,
            version=version,
            current_version=current_version,
            rewritten_query=rewritten_query,
        )
        cached = None if key is None else self._read(key, question)
        if cached is not None:
            return AnswerStream(cached.citations, cached.images, replay(cached.text))
        streamed = self.answers.stream(
            question, game_id=game_id, version=version, current_version=current_version
        )
        return AnswerStream(
            streamed.citations, streamed.images, self._written_back(key, streamed, question)
        )

    def top_questions(
        self, game_id: str, limit: int = TOP_QUESTIONS
    ) -> tuple[tuple[str, int], ...]:
        """这个游戏被问得最多的问法。缓存不可用时返回空——热门问题列表不值得让它报错。"""
        try:
            return self.cache.top_questions(game_id, limit)
        except CacheError as exc:
            logger.warning("热门问题读不出来：%s", exc)
            return ()

    def _key(
        self,
        question: str,
        *,
        game_id: str,
        version: str,
        current_version: str,
        rewritten_query: str,
    ) -> str | None:
        """算这次提问的键，顺带给它计一次数。空问题返回 `None`。

        计数放在这里而不是写入成功那里：**命中与否都要记**。热门问题问的是
        「大家在问什么」，不是「什么被缓存了」——只记未命中的话，最热的那批问题
        一次都不会被数到。
        """
        asked = normalize_query(rewritten_query) or normalize_query(question)
        if not asked:
            return None
        self._count(game_id, asked)
        chosen = effective_version(version, current_version=current_version)
        return cache_key(game_id, chosen, asked)

    def _read(self, key: str, question: str) -> CachedAnswer | None:
        try:
            cached = self.cache.get(key)
        except CacheError as exc:
            logger.warning("缓存读不了（%s），这次按没命中走：%s", key, exc)
            return None
        if cached is not None:
            logger.info("提问 %r 命中缓存 %s：不检索、不生成", question, key)
        return cached

    def _write(self, key: str, answer: CachedAnswer, question: str) -> None:
        """写回缓存。没有引用的结果不写（见模块说明）。"""
        if answer.not_found:
            logger.info("提问 %r 没检索到内容，不入缓存", question)
            return
        try:
            self.cache.set(key, answer, ttl=self.ttl)
        except CacheError as exc:
            logger.warning("缓存写不进去（%s）：这次照常作答，只是下次仍要重算：%s", key, exc)

    def _written_back(
        self, key: str | None, streamed: AnswerStream, question: str
    ) -> Iterator[str]:
        """边吐边攒，吐完写回缓存。**没吐完不写**：半个答案进缓存比不缓存更糟——
        下一次命中它，用户拿到的是一段断掉的话，而且看不出这是缓存给的。
        """
        pieces: list[str] = []
        for piece in streamed.deltas:
            pieces.append(piece)
            yield piece
        if key is None:
            return
        self._write(
            key,
            CachedAnswer("".join(pieces), streamed.citations, streamed.images),
            question,
        )

    def _count(self, game_id: str, asked: str) -> None:
        try:
            self.cache.record_question(game_id, asked)
        except CacheError as exc:
            logger.warning("提问计数写不进去：%s", exc)
