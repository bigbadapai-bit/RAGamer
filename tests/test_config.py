"""配置装载：一个入口，缺键、非法键与会被静默忽略的写法都在启动阶段被指名。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragamer.config import (
    ConfigError,
    Settings,
    env_keys,
    get_settings,
    load_settings,
    required_env_keys,
)

from .conftest import COMPLETE_ENV

REQUIRED_ENV = required_env_keys()


def _write_env_file(directory: Path, body: str) -> Path:
    path = directory / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def _env_file_body(extra: str = "") -> str:
    return "\n".join(f"{key}={value}" for key, value in COMPLETE_ENV.items()) + extra


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in COMPLETE_ENV:
        monkeypatch.delenv(key)


def test_每个配置键都能从环境变量读入(settings_env):
    settings = load_settings(env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.milvus.uri == "http://milvus.test:19530"
    assert settings.milvus.token.get_secret_value() == "test-milvus-token"
    assert settings.milvus.db == "ragamer-test"
    assert settings.mongo.uri.get_secret_value() == "mongodb://mongo.test:27017/?authSource=admin"
    assert settings.mongo.db == "ragamer-test"
    assert settings.minio.endpoint == "minio.test:9000"
    assert settings.minio.access_key == "test-access-key"
    assert settings.minio.secret_key.get_secret_value() == "test-secret-key"
    assert settings.minio.bucket == "ragamer-test"
    assert settings.minio.secure is True
    assert settings.llm.base_url == "https://llm.test/v1"
    assert settings.llm.api_key.get_secret_value() == "test-llm-api-key"
    assert settings.llm.model == "test-model"


def test_未给出可选项时取默认值(settings_env, monkeypatch):
    for key in ("RAGAMER_LOG_LEVEL", "RAGAMER_MILVUS_DB", "RAGAMER_MINIO_BUCKET"):
        monkeypatch.delenv(key)

    settings = load_settings(env_file=None)

    assert settings.log_level == "INFO"
    assert settings.milvus.db == "ragamer"
    assert settings.minio.bucket == "ragamer-images"


def test_取值为空时按没配处理(settings_env, monkeypatch):
    monkeypatch.setenv("RAGAMER_MILVUS_TOKEN", "")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    assert "RAGAMER_MILVUS_TOKEN" in str(excinfo.value)


def test_可选项留空时取默认值(settings_env, monkeypatch):
    monkeypatch.setenv("RAGAMER_MINIO_BUCKET", "")

    assert load_settings(env_file=None).minio.bucket == "ragamer-images"


@pytest.mark.parametrize("missing", REQUIRED_ENV)
def test_缺少必需键时报错并指出键名(settings_env, monkeypatch, missing):
    monkeypatch.delenv(missing)

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    assert missing in str(excinfo.value)


def test_整组缺省时报错列出该组的必需键(settings_env, monkeypatch):
    for key in list(COMPLETE_ENV):
        if key.startswith("RAGAMER_MINIO_"):
            monkeypatch.delenv(key)

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    message = str(excinfo.value)
    assert "RAGAMER_MINIO_ENDPOINT" in message
    assert "RAGAMER_MINIO_ACCESS_KEY" in message
    assert "RAGAMER_MINIO_SECRET_KEY" in message
    # 有默认值的键不该被要求填
    assert "RAGAMER_MINIO_BUCKET" not in message
    assert "RAGAMER_MINIO_SECURE" not in message


def test_取值非法时报错并指出键名(settings_env, monkeypatch):
    monkeypatch.setenv("RAGAMER_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    assert "RAGAMER_LOG_LEVEL" in str(excinfo.value)


def test_配置组被整体赋值时报错并说清要按组内逐键给(settings_env, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("RAGAMER_MINIO", "whatever")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    message = str(excinfo.value)
    assert "RAGAMER_MINIO：" in message
    assert "按组内每个键" in message


def test_没有前缀的键被忽略(settings_env, monkeypatch):
    """同机其它项目（如原项目）的环境变量不能串进本项目。"""
    monkeypatch.delenv("RAGAMER_MILVUS_TOKEN")
    monkeypatch.setenv("MILVUS_TOKEN", "other-project-token")
    monkeypatch.setenv("MILVUS_DB", "other-project-db")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    message = str(excinfo.value)
    assert "RAGAMER_MILVUS_TOKEN" in message
    assert "未知的配置键" not in message


def test_前缀内拼错的环境变量被指出来(settings_env, monkeypatch):
    """环境变量这条路径上 pydantic-settings 只会静默忽略，得自己点名。"""
    monkeypatch.setenv("RAGAMER_MINIO_BUCKETT", "typo")

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None)

    message = str(excinfo.value)
    assert "RAGAMER_MINIO_BUCKETT" in message
    assert "RAGAMER_MINIO_BUCKET" in message  # 给出相近的键名


def test_从_env_文件装载(settings_env, monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _write_env_file(tmp_path, _env_file_body())

    assert load_settings().llm.model == "test-model"

    # 反向对照：删掉文件后同样一组环境变量不存在，装载必须失败
    (tmp_path / ".env").unlink()
    with pytest.raises(ConfigError):
        load_settings()


def test_env_文件里的取值覆盖缺省值(settings_env, monkeypatch, tmp_path):
    monkeypatch.delenv("RAGAMER_LOG_LEVEL")
    monkeypatch.delenv("RAGAMER_MINIO_BUCKET")
    _write_env_file(tmp_path, "RAGAMER_LOG_LEVEL=WARNING\nRAGAMER_MINIO_BUCKET=from-file\n")

    settings = load_settings()

    assert settings.log_level == "WARNING"
    assert settings.minio.bucket == "from-file"


def test_环境变量优先于_env_文件(settings_env, tmp_path):
    """同一个键两处都有时以环境变量为准——部署时不必改文件。"""
    _write_env_file(tmp_path, "RAGAMER_LOG_LEVEL=WARNING\n")

    assert load_settings().log_level == "DEBUG"


def test_env_文件里无关的键被忽略(settings_env, tmp_path):
    _write_env_file(tmp_path, _env_file_body("\nTZ=Asia/Shanghai\nMILVUS_HOST=other-project\n"))

    assert load_settings().milvus.db == "ragamer-test"


@pytest.mark.parametrize("typo", ["RAGAMER_LOG_LEVL=DEBUG", "RAGAMER_MILVUS_DBB=kb"])
def test_env_文件里拼错的键被指出来(settings_env, tmp_path, typo):
    _write_env_file(tmp_path, _env_file_body(f"\n{typo}"))

    with pytest.raises(ConfigError) as excinfo:
        load_settings()

    assert typo.split("=")[0] in str(excinfo.value)


def test_env_文件里会被展开的取值被指出来(settings_env, tmp_path):
    """.env 会做 ${…} 插值：不拦下来，密钥会静默变成别的东西。"""
    _write_env_file(tmp_path, _env_file_body().replace("test-llm-api-key", "sk-${TOKEN}"))

    with pytest.raises(ConfigError) as excinfo:
        load_settings()

    assert "RAGAMER_LLM_API_KEY" in str(excinfo.value)
    assert "sk-" not in str(excinfo.value)  # 报错里不回显取值


def test_BOM_开头的_env_文件也能读(settings_env, monkeypatch, tmp_path):
    """Windows 记事本另存 UTF-8 会带 BOM。"""
    _clear_env(monkeypatch)
    (tmp_path / ".env").write_text("﻿" + _env_file_body(), encoding="utf-8")

    assert load_settings().log_level == "DEBUG"


def test_不是_UTF8_的_env_文件报编码问题(settings_env, tmp_path):
    """PowerShell 5.1 的 Set-Content 默认按 ANSI 写文件。"""
    (tmp_path / ".env").write_bytes((_env_file_body() + "\n# 中文注释\n").encode("gbk"))

    with pytest.raises(ConfigError) as excinfo:
        load_settings()

    assert "UTF-8" in str(excinfo.value)


def test_没有_env_文件时提示找不到(settings_env, monkeypatch):
    monkeypatch.delenv("RAGAMER_LLM_MODEL")

    with pytest.raises(ConfigError) as excinfo:
        load_settings()

    message = str(excinfo.value)
    assert "RAGAMER_LLM_MODEL" in message
    assert "没有找到" in message


def test_没有_env_文件也能装载(settings_env):
    assert load_settings().log_level == "DEBUG"


def test_进程内共享同一个配置对象(settings_env):
    assert get_settings() is get_settings()


def test_env_keys_列出模型读取的全部键():
    keys = env_keys(Settings)

    assert set(keys) == set(COMPLETE_ENV)
    assert len(keys) == len(set(keys))


def test_必需的是凭据与地址_有默认值的不在其中():
    required = set(required_env_keys())

    assert required <= set(env_keys(Settings))
    assert {
        "RAGAMER_MILVUS_TOKEN",
        "RAGAMER_MONGO_URI",
        "RAGAMER_MINIO_SECRET_KEY",
        "RAGAMER_LLM_API_KEY",
    } <= required
    assert not required & {
        "RAGAMER_LOG_LEVEL",
        "RAGAMER_MILVUS_DB",
        "RAGAMER_MINIO_BUCKET",
        "RAGAMER_MINIO_SECURE",
    }
