"""对象存储（MinIO）。

桶的确保与按前缀清理用假客户端断言；真的连上 MinIO 属于云端行为，留给集成测试。
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
import urllib3
from minio.error import S3Error

from ragamer.config import MinioSettings, load_settings
from ragamer.stores import objects
from ragamer.stores.base import StoreError, StoreUnavailableError
from ragamer.stores.objects import MinioObjectStore


class FakeObject:
    def __init__(self, name: str) -> None:
        self.object_name = name


class FakeDeleteError:
    """批量删里单个对象的失败项。"""

    def __init__(self, name: str) -> None:
        self.object_name = name
        self.message = "拒绝访问"


class FakeResponse:
    """urllib3 的响应。用它验证"读完把连接还回去"。"""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.released = False

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.released = True


class FakeMinio:
    """替代 `minio.Minio`：记录构造参数与调用，不连服务。"""

    instances: ClassVar[list[FakeMinio]] = []

    def __init__(self, endpoint: str, **kwargs: Any) -> None:
        self.endpoint = endpoint
        self.init_kwargs = kwargs
        self.buckets: set[str] = set()
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.response: FakeResponse | None = None
        self.remove_errors: list[FakeDeleteError] = []
        FakeMinio.instances.append(self)

    @property
    def http_client(self) -> urllib3.PoolManager:
        return self.init_kwargs["http_client"]

    def bucket_exists(self, bucket: str) -> bool:
        self.calls.append(("bucket_exists", {"bucket": bucket}))
        return bucket in self.buckets

    def make_bucket(self, bucket: str) -> None:
        self.calls.append(("make_bucket", {"bucket": bucket}))
        self.buckets.add(bucket)

    def put_object(self, bucket: str, name: str, data: Any, length: int, content_type: str) -> None:
        self.calls.append(
            ("put_object", {"bucket": bucket, "name": name, "content_type": content_type})
        )
        self.objects[name] = data.read()

    def get_object(self, bucket: str, name: str) -> FakeResponse:
        self.calls.append(("get_object", {"bucket": bucket, "name": name}))
        if name not in self.objects:
            raise _missing(name)
        self.response = FakeResponse(self.objects[name])
        return self.response

    def remove_object(self, bucket: str, name: str) -> None:
        self.calls.append(("remove_object", {"bucket": bucket, "name": name}))
        self.objects.pop(name, None)

    def list_objects(self, bucket: str, prefix: str = "", recursive: bool = False) -> list[Any]:
        self.calls.append(("list_objects", {"bucket": bucket, "prefix": prefix}))
        return [FakeObject(name) for name in sorted(self.objects) if name.startswith(prefix)]

    def remove_objects(self, bucket: str, names: list[Any]) -> list[FakeDeleteError]:
        self.calls.append(("remove_objects", {"bucket": bucket, "names": [n.name for n in names]}))
        for name in [n.name for n in names]:
            self.objects.pop(name, None)
        return list(self.remove_errors)

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]


def _missing(name: str) -> S3Error:
    """minio 里"对象不存在"是抛错，不是返回 None。"""
    return S3Error(
        code="NoSuchKey",
        message="不存在",
        resource=name,
        request_id="",
        host_id="",
        response=None,
    )


@pytest.fixture
def minio(monkeypatch: pytest.MonkeyPatch) -> type[FakeMinio]:
    FakeMinio.instances = []
    monkeypatch.setattr(objects, "Minio", FakeMinio)
    return FakeMinio


@pytest.fixture
def store(minio: type[FakeMinio], settings_env) -> MinioObjectStore:
    return MinioObjectStore(load_settings(env_file=None).minio, timeout=2.5)


def _client(minio: type[FakeMinio]) -> FakeMinio:
    assert minio.instances, "还没有连过"
    return minio.instances[0]


def test_连接参数按配置传给_MinIO(store, minio):
    store.check()

    client = _client(minio)
    assert client.endpoint == "minio.test:9000"
    assert client.init_kwargs["access_key"] == "test-access-key"
    assert client.init_kwargs["secret_key"] == "test-secret-key"
    assert client.init_kwargs["secure"] is True


def test_超时与关掉重试设在传输层(store, minio):
    """minio 自己没有超时参数，不加就会挂在启动上。"""
    store.check()

    pool = _client(minio).http_client
    timeout = pool.connection_pool_kw["timeout"]
    assert timeout.connect_timeout == 2.5
    assert timeout.read_timeout == 2.5
    assert pool.connection_pool_kw["retries"].total is False


def test_自检时确保桶存在(store, minio):
    """查桶在不在同时验证了地址通与凭据认——桶不存在是返回 False，不是报错。"""
    store.check()

    assert _client(minio).called("bucket_exists") == [{"bucket": "ragamer-test"}]
    assert _client(minio).called("make_bucket") == [{"bucket": "ragamer-test"}]


def test_桶不存在时创建它(store, minio):
    store.ensure_bucket()

    assert _client(minio).called("make_bucket") == [{"bucket": "ragamer-test"}]
    assert _client(minio).buckets == {"ragamer-test"}


def test_桶已存在时不动它(store, minio):
    """桶上可能已经有归属与策略设置，重建会盖掉。"""
    store.ensure_bucket()
    _client(minio).calls.clear()

    store.ensure_bucket()

    assert _client(minio).called("make_bucket") == []


def test_写入与读取对象(store, minio):
    store.put("images/二郎神.png", b"a", content_type="image/png")

    assert _client(minio).called("put_object") == [
        {"bucket": "ragamer-test", "name": "images/二郎神.png", "content_type": "image/png"}
    ]
    assert store.get("images/二郎神.png") == b"a"


def test_读完对象把连接还回去(store, minio):
    store.put("images/二郎神.png", b"a")

    store.get("images/二郎神.png")

    assert _client(minio).response is not None
    assert _client(minio).response.released is True


def test_取不存在的对象报存储错误(store, minio):
    with pytest.raises(StoreError, match="没有这个对象"):
        store.get("images/没有这张.png")


def test_按前缀清理对象(store, minio):
    store.put("images/二郎神.png", b"a")
    store.put("images/寒江雪.png", b"b")
    store.put("docs/readme.md", b"c")

    assert store.delete_prefix("images/") == 2

    assert _client(minio).called("list_objects") == [
        {"bucket": "ragamer-test", "prefix": "images/"}
    ]
    assert set(_client(minio).objects) == {"docs/readme.md"}


def test_前导斜杠不影响前缀匹配(store, minio):
    """原项目的 list 去前导 `/` 而 put 不去，清旧图时一个也匹配不上、静默失效。"""
    store.put("/images/二郎神.png", b"a")

    assert store.list_keys("images/") == ["images/二郎神.png"]
    assert store.delete_prefix("/images/") == 1
    assert _client(minio).objects == {}


def test_没有匹配的前缀不必发删除请求(store, minio):
    assert store.delete_prefix("images/") == 0

    assert _client(minio).called("remove_objects") == []


def test_有对象没删掉时报错(store, minio):
    """按前缀删是批量接口，它把失败项当返回值而不是抛异常——不看就静默留下了。"""
    store.put("images/二郎神.png", b"a")
    _client(minio).remove_errors = [FakeDeleteError("images/二郎神.png")]

    with pytest.raises(StoreError, match="没删掉"):
        store.delete_prefix("images/")


def test_远端不可达时在超时内失败并点名服务与地址():
    settings = MinioSettings(
        endpoint="127.0.0.1:1", access_key="k", secret_key="s", bucket="ragamer-test"
    )
    store = MinioObjectStore(settings, timeout=1.0)

    start = time.monotonic()
    with pytest.raises(StoreUnavailableError) as excinfo:
        store.check()
    elapsed = time.monotonic() - start

    assert elapsed < 5, f"没有在设定的超时内失败：{elapsed:.1f} 秒"
    assert "MinIO" in str(excinfo.value)
    assert "127.0.0.1:1" in str(excinfo.value)
