"""地址脱敏。

地址里可能内嵌账号密码（`mongodb://user:pass@host`）或查询串（`?api_key=…`），
原样写进日志或异常信息等于把凭据落到日志文件里。凡是要把地址放进出错信息的地方
都先过这里——启动自检的配置摘要与存储客户端的连接错误共用同一个实现。
"""

from __future__ import annotations


def redact_address(address: str) -> str:
    """抹掉地址里的账号密码与查询串，只留 scheme 与主机端口。

    `http://root:pw@milvus.internal:19530` → `http://milvus.internal:19530`
    `minio.internal:9000` → `minio.internal:9000`（没有 scheme 也照抹）
    """
    scheme, separator, rest = address.partition("//")
    if not separator:
        rest, scheme = scheme, ""
    return f"{scheme}{separator}{rest.partition('?')[0].rpartition('@')[2]}"
