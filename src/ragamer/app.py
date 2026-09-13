"""可跑的应用：把 JSON 端点与页面装到一起，并提供起服务的入口。

`ragamer.api` 造的是**只有 JSON 端点**的应用——那是写入侧对外的契约，测试直接打它。
给人用的那个在这里：同样的端点，外加 `ragamer.web` 的页面。两个模块各自不认识对方，
装配只发生在这一处。

起服务前先跑一遍存储自检：界面上每个动作都要读写它们，配置不对或服务不通时就该在这里
一次报清楚，而不是等人点进页面一个个撞。退出码与 `ragamer` 命令一致，脚本里能同样处理。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI

from ragamer.__main__ import EXIT_CONFIG, EXIT_OK, EXIT_STORE
from ragamer.api import create_app as create_api_app
from ragamer.config import ConfigError, get_settings
from ragamer.container import Container, build_container
from ragamer.logging import get_logger, setup_logging
from ragamer.stores.base import StoreError
from ragamer.web import create_router

logger = get_logger(__name__)

#: 默认只监听本机：这一版没有登录，不该顺手暴露到局域网上。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def create_app(container: Container) -> FastAPI:
    """完整应用：JSON 端点 + 页面。"""
    app = create_api_app(container)
    app.include_router(create_router(container))
    return app


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

    logger.info("界面在 http://%s:%d（Ctrl-C 停）", args.host, args.port)
    uvicorn.run(create_app(container), host=args.host, port=args.port)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
