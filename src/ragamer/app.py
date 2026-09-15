"""可跑的应用：把 JSON 端点与页面装到一起，并提供起服务的入口。

`ragamer.api` 造的是**只有 JSON 端点**的应用——那是写入侧对外的契约，测试直接打它。
给人用的那个在这里：同样的端点，外加 `ragamer.web` 的页面。两个模块各自不认识对方，
装配只发生在这一处。

起服务前先跑一遍存储自检：界面上每个动作都要读写它们，配置不对或服务不通时就该在这里
一次报清楚，而不是等人点进页面一个个撞。退出码与 `ragamer` 命令一致，脚本里能同样处理。
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI

from ragamer.__main__ import EXIT_CONFIG, EXIT_OK, EXIT_STORE
from ragamer.api import create_app as create_api_app
from ragamer.config import ConfigError, get_settings
from ragamer.container import Container, build_container
from ragamer.conversations import CONVERSATIONS, SESSION_INDEX, build_chat
from ragamer.logging import get_logger, setup_logging
from ragamer.stores.base import StoreError
from ragamer.web import create_router

logger = get_logger(__name__)

#: 默认只监听本机：这一版没有登录，不该顺手暴露到局域网上。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def create_app(container: Container) -> FastAPI:
    """完整应用：JSON 端点 + 页面。

    **读取侧只接一套**，两组路由共用：它们打的是同一批会话、同一份缓存，
    各接一遍就可能一边挡了缓存、一边没挡，而那种差别在界面上看不出来。
    """
    stack = build_chat(container)
    app = create_api_app(container, chat=stack.chat, turns=stack.turns)
    app.include_router(create_router(container, stack))
    return app


def _ensure_session_index(container: Container) -> None:
    """把会话列表要用的复合索引落下来。**建不上不拦启动**。

    没有它，每次列会话都是「全表扫 + 内存排序」——一页一次，而翻页把这份成本乘以页数。
    但索引只影响快慢：建不上（没权限、库刚被别人删了）时列表照常能用，只是慢。
    所以这里记一条 ERROR 继续走，不学存储自检那样把进程拦住。

    放在这里而不是 `check()` 里：适配器不知道业务层查哪些集合，而自检那条路是刻意只读的
    （它连库都不会平白建一个）。这里是唯一知道「哪个集合要哪个索引」的地方。
    """
    try:
        container.docs.ensure_indexes(CONVERSATIONS, SESSION_INDEX)
    except Exception as exc:  # 适配器只把「连不上」包成 StoreError，这里要的是「什么都不拦」
        logger.error("会话列表的索引没建上，列表会退化成全表扫：%s", exc)


def _warm_models(container: Container) -> threading.Thread:
    """把两个本地模型的权重提前读进来。**丢在后台线程里，失败不拦启动。**

    两个模型都是懒加载的（`ragamer.lazy`），不预热的话，服务起来之后的**第一条提问**
    要先把几个 G 的权重从盘上读进来。实测这笔账是：向量化那一段冷启 23.7 秒、热了
    0.4 秒；精排冷启 53.3 秒、热了 43.9 秒——**三十几秒全落在第一个提问的人头上**，
    而它跟那次提问问的是什么毫无关系。

    放后台线程而不是就地加载：加载要几十秒，就地做会把「服务起没起来」也一起拖住，
    而这段时间里页面本来是可以开的（列知识库、翻历史走的是 Mongo／MinIO）。

    失败只记一条 ERROR、不拦启动：存储已经自检过了，模型加载不出来该表现为
    「这一条提问报错」，而不是「服务起不来」——后者会让人去查配置，而问题不在那里。
    这里**宽泛地接 `Exception`**（与 `_ensure_session_index` 同一个理由）：加载器
    自己承诺的是 `ModelError`，但 transformers 那条路会抛别的东西，而这条路径的
    约定是「什么都不拦」。

    返回那个线程**只是为了测试能 `join` 它**——调用方不必管它。
    """

    def load() -> None:
        for name, model in (
            ("向量化模型", container.embedder),
            ("精排模型", container.reranker),
        ):
            try:
                model.warm()
            except Exception as exc:  # 见 docstring：这条路径不拦任何失败
                logger.error("%s没预热上，第一次用到它时会再加载一次：%s", name, exc)

    thread = threading.Thread(target=load, name="warm-models", daemon=True)
    thread.start()
    return thread


def main(argv: Sequence[str] | None = None) -> int:
    """起界面与后端的本地服务。0 = 正常退出，2 = 配置有问题，3 = 服务不可达。"""
    parser = argparse.ArgumentParser(
        prog="ragamer-web", description="起 RAGamer 的界面与后端（本地跑，数据库在云端）"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"默认 {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"默认 {DEFAULT_PORT}")
    args = parser.parse_args(argv)

    setup_logging()  # 先按默认等级装上，配置装载失败时才有地方报错
    try:
        settings = get_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_CONFIG

    setup_logging(settings.log_level)
    container = build_container(settings)
    try:
        container.check()
    except StoreError as exc:
        logger.error("%s", exc)
        return EXIT_STORE

    _ensure_session_index(container)
    _warm_models(container)
    logger.info("界面在 http://%s:%d（Ctrl-C 停）", args.host, args.port)
    uvicorn.run(create_app(container), host=args.host, port=args.port)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
