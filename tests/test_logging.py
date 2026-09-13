"""日志：统一入口、可重复初始化、按等级过滤、不重复输出。"""

from __future__ import annotations

import io
import logging

from ragamer.logging import LOGGER_NAME, get_logger, setup_logging


def test_取到的是本项目的子_logger():
    assert get_logger().name == LOGGER_NAME
    assert get_logger("ragamer.pipeline").name == "ragamer.pipeline"
    assert get_logger("pipeline").name == "ragamer.pipeline"
    assert get_logger("tests.test_logging").name == "ragamer.tests.test_logging"


def test_重复初始化不叠加处理器():
    setup_logging()
    count = len(logging.getLogger(LOGGER_NAME).handlers)

    setup_logging()
    setup_logging("DEBUG")

    assert count >= 1
    assert len(logging.getLogger(LOGGER_NAME).handlers) == count


def test_初始化后按等级过滤(capsys):
    setup_logging("WARNING")
    logger = get_logger("tests")

    logger.info("不该出现")
    logger.warning("该出现")

    captured = capsys.readouterr()
    assert "该出现" in captured.err
    assert "不该出现" not in captured.err


def test_日志带模块名(capsys):
    setup_logging("INFO")

    get_logger("ragamer.tests").info("自检通过")

    captured = capsys.readouterr()
    assert "ragamer.tests: 自检通过" in captured.err


def test_日志不往_root_冒泡():
    """统一走一个入口：root 上再挂 handler 也不会打印两遍。"""
    ours, theirs = io.StringIO(), io.StringIO()
    root = logging.getLogger()
    root_handler = logging.StreamHandler(theirs)
    root.addHandler(root_handler)
    try:
        setup_logging("INFO", stream=ours)
        get_logger("tests").info("只此一份")
    finally:
        root.removeHandler(root_handler)

    assert ours.getvalue().count("只此一份") == 1
    assert theirs.getvalue() == ""
