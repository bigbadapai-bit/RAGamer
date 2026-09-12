"""共享 fixture。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ragamer.config import get_settings

#: 一组完整、合法的配置。每个键给不同的值，以便断言"读到的正是这个键"。
#: 键名与模型的对应关系由 `tests/test_config.py::test_env_keys_列出模型读取的全部键` 兜住。
COMPLETE_ENV: dict[str, str] = {
    "RAGAMER_LOG_LEVEL": "DEBUG",
    "RAGAMER_MILVUS_URI": "http://milvus.test:19530",
    "RAGAMER_MILVUS_TOKEN": "test-milvus-token",
    "RAGAMER_MILVUS_DB": "ragamer-test",
    "RAGAMER_MONGO_URI": "mongodb://mongo.test:27017/?authSource=admin",
    "RAGAMER_MONGO_DB": "ragamer-test",
    "RAGAMER_MINIO_ENDPOINT": "minio.test:9000",
    "RAGAMER_MINIO_ACCESS_KEY": "test-access-key",
    "RAGAMER_MINIO_SECRET_KEY": "test-secret-key",
    "RAGAMER_MINIO_BUCKET": "ragamer-test",
    "RAGAMER_MINIO_SECURE": "true",
    "RAGAMER_LLM_BASE_URL": "https://llm.test/v1",
    "RAGAMER_LLM_API_KEY": "test-llm-api-key",
    "RAGAMER_LLM_MODEL": "test-model",
    "RAGAMER_LLM_TIMEOUT": "12.5",
    "RAGAMER_LLM_MAX_ATTEMPTS": "5",
    "RAGAMER_LLM_BACKOFF_BASE": "0.25",
    "RAGAMER_LLM_BACKOFF_MAX": "4",
}


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, str]]:
    """完整配置进环境变量，工作目录里没有 `.env`，进程内配置缓存为空。

    切到 tmp_path 是因为默认装载路径是相对于工作目录的 `.env`——
    开发者本机的 `.env` 不该影响测试。
    """
    for key, value in COMPLETE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield COMPLETE_ENV
    get_settings.cache_clear()
