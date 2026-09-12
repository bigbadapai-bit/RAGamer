"""启动自检：配置合格才放行，不合格时点名是哪个键，输出里不带凭据。"""

from __future__ import annotations

from pydantic import BaseModel, SecretStr

from ragamer.__main__ import main
from ragamer.config import Settings


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
