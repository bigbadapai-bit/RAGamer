"""日志的唯一入口。

业务模块用 `get_logger(__name__)` 取 logger，启动时调一次 `setup_logging`。
源码里不出现 `print`——由 `tests/test_conventions.py` 机械拦截。
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

#: 本项目所有 logger 的根名字。
LOGGER_NAME = "ragamer"

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """取 logger。传 `__name__` 得到 `ragamer.<模块>`，不传得到项目根 logger。"""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """装配项目根 logger。启动时调用；重复调用不会叠加 handler。

    `stream` 默认取调用时的 `sys.stderr`，方便测试捕获。
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    # 本项目日志只走这一个 handler：不往上冒泡，root 上再挂 handler 也不会打印两遍
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    logger.addHandler(handler)
