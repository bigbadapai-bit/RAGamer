"""`ragamer` 命令：启动自检。

配置装载 → 按配置构造组合根 → 三个存储逐个连通性自检。任一步失败都不放行，
退出码区分是哪一类问题。存储自检要连云端，测试里把 `build_container` 换成内存假件。
"""

from __future__ import annotations

import logging

from ragamer.config import ConfigError, Settings, get_settings
from ragamer.container import build_container
from ragamer.logging import get_logger, setup_logging
from ragamer.redaction import redact_address
from ragamer.stores.base import StoreError

logger = get_logger(__name__)

#: 退出码。配置问题与服务不可达分开，脚本里能分别处理。
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_STORE = 3


def describe(settings: Settings) -> str:
    """配置摘要。凭据只报「已配置」，地址只留主机，取值不落进日志。"""
    return "\n".join(
        [
            f"log_level={settings.log_level}",
            f"store_timeout_seconds={settings.store_timeout_seconds:g}",
            f"milvus: uri={redact_address(settings.milvus.uri)} db={settings.milvus.db}"
            " token=已配置",
            f"mongo: uri=已配置 db={settings.mongo.db}",
            f"minio: endpoint={redact_address(settings.minio.endpoint)}"
            f" bucket={settings.minio.bucket}"
            f" secure={settings.minio.secure} access_key=已配置 secret_key=已配置",
            f"llm: base_url={redact_address(settings.llm.base_url)} model={settings.llm.model}"
            " api_key=已配置",
            _vision_line(settings),
            f"mineru: base_url={redact_address(settings.mineru.base_url)}"
            f" model_version={settings.mineru.model_version}"
            f" poll_interval={settings.mineru.poll_interval_seconds:g}"
            f" poll_timeout={settings.mineru.poll_timeout_seconds:g} api_key=已配置",
            f"models: device={settings.models.device} fp16={settings.models.fp16}",
            f"embed: model={settings.embed.model} batch_size={settings.embed.batch_size}"
            f" max_length={settings.embed.max_length}",
            f"rerank: model={settings.rerank.model} batch_size={settings.rerank.batch_size}"
            f" max_length={settings.rerank.max_length}",
        ]
    )


def _vision_line(settings: Settings) -> str:
    """视觉模型那一行。「没配」是一种合法状态（补图只做二次 OCR），
    所以它要**明说**，而不是省略——省略了看起来就像这一项不存在。
    """
    if not settings.vision.enabled:
        return "vision: 未配置（图片只做二次 OCR，没有摘要）"
    return (
        f"vision: base_url={redact_address(settings.vision.base_url)}"
        f" model={settings.vision.model} api_key=已配置"
    )


def main() -> int:
    """装载配置、构造存储、自检。0 = 通过，2 = 配置有问题，3 = 服务不可达。"""
    setup_logging()  # 先按默认等级装上，配置装载失败时才有地方报错
    try:
        settings = get_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_CONFIG

    setup_logging(settings.log_level)
    # 自检报告是这条命令的输出，不该被 RAGAMER_LOG_LEVEL 吞掉
    logger.setLevel(logging.INFO)
    logger.info("配置自检通过\n%s", describe(settings))

    container = build_container(settings)
    try:
        container.check()
    except StoreError as exc:
        logger.error("%s", exc)
        return EXIT_STORE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
