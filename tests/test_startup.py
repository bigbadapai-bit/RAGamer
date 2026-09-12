"""启动自检：配置合格、存储连得上才放行；失败时点名是哪个键、哪个服务。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, SecretStr

from ragamer import __main__ as startup
from ragamer.__main__ import main
from ragamer.config import Settings
from ragamer.container import Container
from ragamer.stores import InMemoryDocStore, InMemoryObjectStore

from .conftest import FailingStore


@pytest.fixture(autouse=True)
def 内存版的组合根(monkeypatch: pytest.MonkeyPatch, memory_container) -> None:
    """自检这条路径不该真的去连云端：把组合根换成内存假件。

    真要连云端的那部分（建表、检索）属于集成测试，见 pyproject 里的 integration 标记。
    """
    monkeypatch.setattr(startup, "build_container", lambda settings: memory_container)


def _secret_values(node: BaseModel) -> list[str]:
    """模型里所有密钥字段的取值——用模型的形状断言输出，而不是手抄一份清单。"""
    values: list[str] = []
    for name in type(node).model_fields:
        value = getattr(node, name)
        if isinstance(value, BaseModel):
            values.extend(_secret_values(value))
        elif isinstance(value, SecretStr):
            values.append(value.get_secret_value())
    return values


def test_配置完整时自检通过并报出生效的配置(settings_env, capsys):
    assert main() == 0

    captured = capsys.readouterr()
    assert "配置自检通过" in captured.err
    assert "milvus.test:19530" in captured.err
    assert "test-model" in captured.err


def test_缺少必需键时自检失败并指出键名(settings_env, monkeypatch, capsys):
    monkeypatch.delenv("RAGAMER_MINIO_SECRET_KEY")

    assert main() == 2

    assert "RAGAMER_MINIO_SECRET_KEY" in capsys.readouterr().err


def test_取值非法时自检失败并指出键名(settings_env, monkeypatch, capsys):
    monkeypatch.setenv("RAGAMER_LOG_LEVEL", "VERBOSE")

    assert main() == 2

    assert "RAGAMER_LOG_LEVEL" in capsys.readouterr().err


def test_日志等级高时自检报告照样打印(settings_env, monkeypatch, capsys):
    """报告是这条命令的输出，WARNING 也不该把它吞掉。"""
    monkeypatch.setenv("RAGAMER_LOG_LEVEL", "WARNING")

    assert main() == 0

    assert "配置自检通过" in capsys.readouterr().err


def test_自检输出里没有密钥(settings_env, capsys):
    settings = Settings()

    assert main() == 0

    captured = capsys.readouterr()
    for secret in _secret_values(settings):
        assert secret not in captured.err


def test_地址里内嵌的账号密码与查询串被抹掉(settings_env, monkeypatch, capsys):
    monkeypatch.setenv("RAGAMER_MILVUS_URI", "http://root:MILVUS-PW@milvus.internal:19530")
    monkeypatch.setenv("RAGAMER_LLM_BASE_URL", "https://gw.test/v1?api_key=LLM-QUERY-KEY")

    assert main() == 0

    captured = capsys.readouterr()
    assert "MILVUS-PW" not in captured.err
    assert "LLM-QUERY-KEY" not in captured.err
    # 主机留着，出问题时还能定位到打给了谁
    assert "milvus.internal:19530" in captured.err
    assert "https://gw.test/v1" in captured.err


def test_存储自检通过时报告里列出服务(settings_env, capsys):
    assert main() == 0

    assert "存储自检通过" in capsys.readouterr().err


def test_存储连不上时退出码为_3_并点名服务与地址(settings_env, monkeypatch, capsys):
    monkeypatch.setattr(
        startup,
        "build_container",
        lambda settings: Container(
            chunks=FailingStore("Milvus", "milvus.test:19530"),
            docs=InMemoryDocStore(),
            objects=InMemoryObjectStore(),
        ),
    )

    # 退出码把"存储不通"与"配置有问题"分开，脚本里能分别处理
    assert main() == 3

    captured = capsys.readouterr()
    assert "Milvus" in captured.err
    assert "milvus.test:19530" in captured.err
