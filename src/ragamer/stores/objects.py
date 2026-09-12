"""对象存储的 MinIO 实现：原图等二进制内容。

桶名由配置给，与原项目错开（本项目默认 `ragamer-images`），两个项目共用实例互不干扰。

对象 key 与前缀共用 :func:`~ragamer.stores.base.normalize_prefix` 这一个归一函数，
所以写进去的 key 与列出来的 key 永远对得上。

MinIO 容器的卷属主不归这一层管——卷是部署时挂的，属主不对容器直接起不来，
应用只能看到"连不上"。启动自检会把这表现为超时，见 `Container.check`。
"""

from __future__ import annotations

import io

import urllib3
from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import MinioException, S3Error
from urllib3.exceptions import HTTPError

from ragamer.config import MinioSettings
from ragamer.logging import get_logger
from ragamer.redaction import redact_address
from ragamer.stores.base import MINIO, StoreError, normalize_prefix, unavailable

logger = get_logger(__name__)

#: 连不上时 minio 会把 urllib3 的异常直接抛出来（重试关掉之后没有包装层）。
_FAILURES = (MinioException, HTTPError, OSError, ValueError)

#: 取对象时"不存在"是正常结果，不该当成连不上。
_NOT_FOUND = "NoSuchKey"


class MinioObjectStore:
    """MinIO 上的对象存储。构造不连服务，第一次操作才连。"""

    def __init__(self, settings: MinioSettings, *, timeout: float) -> None:
        self.name = MINIO
        self.address = redact_address(settings.endpoint)
        self._endpoint = settings.endpoint
        self._access_key = settings.access_key
        self._secret_key = settings.secret_key.get_secret_value()
        self._bucket = settings.bucket
        self._secure = settings.secure
        self._timeout = timeout
        self._client: Minio | None = None

    def check(self) -> None:
        """连通性自检，顺带确保桶存在。

        查的是桶在不在，而不是某个固定路径：桶不存在返回 `False` 而不是报错，
        所以这一条同时验证了"地址通"与"凭据认"。
        """
        self.ensure_bucket()

    def ensure_bucket(self) -> None:
        """确保桶存在。已存在就不动它——桶上可能已经有归属与策略设置。

        失败按"这个服务用不了"报出：无论是连不上还是没权限建桶，
        结果是同一个——进程起来了也存不了原图。
        """
        try:
            client = self._connect()
            if not client.bucket_exists(self._bucket):
                client.make_bucket(self._bucket)
                logger.info("新建 MinIO 桶 %s", self._bucket)
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        try:
            self._connect().put_object(
                self._bucket, normalize_prefix(key), io.BytesIO(data), len(data), content_type
            )
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def get(self, key: str) -> bytes:
        try:
            response = self._connect().get_object(self._bucket, normalize_prefix(key))
        except S3Error as exc:
            if exc.code == _NOT_FOUND:
                raise StoreError(f"{self.name} 上没有这个对象：{key}") from exc
            raise unavailable(self.name, self.address, self._timeout, exc) from exc
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc
        # 连接是从池里借的，读完必须还回去，否则连接池会被慢慢占满
        with response:
            return response.read()

    def delete(self, key: str) -> None:
        # S3 的删除本身幂等：删一个不存在的 key 也返回成功
        try:
            self._connect().remove_object(self._bucket, normalize_prefix(key))
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def list_keys(self, prefix: str = "") -> list[str]:
        try:
            objects = self._connect().list_objects(
                self._bucket, prefix=normalize_prefix(prefix), recursive=True
            )
            return sorted(obj.object_name for obj in objects)
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc

    def delete_prefix(self, prefix: str) -> int:
        """按前缀批量删。

        前缀与 :meth:`list_keys` 走同一个归一函数：原项目 list 去前导 `/` 而 put 不去，
        清旧图时一个也匹配不上，静默失效。这里也对得上，做不到"列得出、删不掉"。
        """
        keys = self.list_keys(prefix)
        if not keys:
            return 0
        try:
            errors = list(
                self._connect().remove_objects(self._bucket, [DeleteObject(key) for key in keys])
            )
        except _FAILURES as exc:
            raise unavailable(self.name, self.address, self._timeout, exc) from exc
        if errors:
            raise StoreError(
                f"{self.name} 上有 {len(errors)} 个对象没删掉，第一个是 "
                f"{errors[0].object_name}：{errors[0].message}"
            )
        return len(keys)

    def _connect(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                # 超时与重试都设在传输层：minio 自己没有超时参数
                http_client=urllib3.PoolManager(
                    timeout=urllib3.Timeout(connect=self._timeout, read=self._timeout),
                    # 关掉重试：自检的意义是尽早报出问题，重试只会把启动拖长
                    retries=False,
                ),
            )
        return self._client
