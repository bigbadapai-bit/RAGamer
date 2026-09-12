"""`ragamer` 命令：启动自检。

阶段 0 能自检的只有配置本身——存储连通性检查随 T02 接到同一条启动路径上。
"""

from __future__ import annotations

import logging

from ragamer.config import ConfigError, Settings, get_settings
from ragamer.logging import get_logger, setup_logging

logger = get_logger(__name__)


def describe(settings: Settings) -> str:
    """配置摘要。凭据只报「已配置」，地址只留主机，取值不落进日志。"""
    return "\n".join(
        [
            f"log_level={settings.log_level}",
            f"milvus: uri={_redact(settings.milvus.uri)} db={settings.milvus.db} token=已配置",
            f"mongo: uri=已配置 db={settings.mongo.db}",
            f"minio: endpoint={_redact(settings.minio.endpoint)} bucket={settings.minio.bucket}"
            f" secure={settings.minio.secure} access_key=已配置 secret_key=已配置",
            f"llm: base_url={_redact(settings.llm.base_url)} model={settings.llm.model}"
            " api_key=已配置",
        ]
    )


def _redact(address: str) -> str:
    """抹掉地址里的账号密码与查询串：`user:pass@host`、`?api_key=…` 都可能带凭据。"""
    scheme, separator, rest = address.partition("//")
    if not separator:
        rest, scheme = scheme, ""
    return f"{scheme}{separator}{rest.partition('?')[0].rpartition('@')[2]}"


def main() -> int:
    """装载配置并输出自检结果。0 = 通过，2 = 配置有问题。"""
    setup_logging()  # 先按默认等级装上，配置装载失败时才有地方报错
    try:
        settings = get_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    setup_logging(settings.log_level)
    # 自检报告是这条命令的输出，不该被 RAGAMER_LOG_LEVEL 吞掉
    logger.setLevel(logging.INFO)
    logger.info("配置自检通过\n%s", describe(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
