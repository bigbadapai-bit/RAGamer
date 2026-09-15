"""起服务那一段里的模型预热。

预热**不改变任何一次调用的结果**，它省下的只是「服务起来之后第一位提问者干等权重
加载」——实测向量化那一段冷启 23.7 秒、热了 0.4 秒，精排冷启 53.3 秒、热了 43.9 秒。
正因为它的收益与结果无关，它的失败也不该有后果：权重读不出来该表现为「这一条提问
报错」，而不是「服务起不来」——后者会让人去查配置，而问题不在那里。
"""

from __future__ import annotations

import logging
import time

from ragamer.app import _warm_models

from .conftest import make_container


class _Model:
    """记下自己被预热过几次。`failing` 时按真实加载器失败的样子炸掉。"""

    def __init__(self, *, failing: bool = False, seconds: float = 0.0) -> None:
        self.failing = failing
        self.seconds = seconds
        self.warms = 0

    def warm(self) -> None:
        self.warms += 1
        time.sleep(self.seconds)
        if self.failing:
            raise RuntimeError("权重目录不在")


def test_两个模型都被预热():
    embedder, reranker = _Model(), _Model()

    _warm_models(make_container(embedder=embedder, reranker=reranker)).join(timeout=10)

    assert (embedder.warms, reranker.warms) == (1, 1)


def test_预热失败不拦启动(caplog):
    """一个模型没预热上，另一个照预热点——两个各试各的，不互相牵连。"""
    reranker = _Model()
    container = make_container(embedder=_Model(failing=True), reranker=reranker)

    with caplog.at_level(logging.ERROR, logger="ragamer.app"):
        thread = _warm_models(container)
    thread.join(timeout=10)

    assert not thread.is_alive()  # 异常穿过线程就没人接得住了
    assert reranker.warms == 1
    assert any("没预热上" in record.getMessage() for record in caplog.records)


def test_预热不挡住起服务这一条路():
    """就地加载要几十秒，那会把「服务起没起来」也一起拖住——而这段时间里页面本来开得了
    （列知识库、翻历史走的是 Mongo／MinIO，用不到这两个模型）。
    """
    slow = _Model(seconds=0.4)

    started = time.perf_counter()
    thread = _warm_models(make_container(embedder=slow, reranker=_Model()))
    elapsed = time.perf_counter() - started
    thread.join(timeout=10)

    assert elapsed < 0.2
    assert slow.warms == 1
