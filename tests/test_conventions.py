"""结构约束：把「不许再犯」清单变成机械拦截。

这几条不是行为测试，而是把项目明确写下的规矩钉在源码上——
旧项目正是踩了这些（20+ 模块各自 load_dotenv、日志用 print、.env.example 少列 19 个键）。
"""

from __future__ import annotations

import ast
from pathlib import Path

from ragamer.config import Settings, env_keys

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "src" / "ragamer"
_ENV_EXAMPLE = _ROOT / ".env.example"
#: 只有这个模块允许读环境变量
_CONFIG_MODULE = _PACKAGE / "config.py"

#: 读环境变量的名字。`from os import getenv` 之后是裸名字，只认属性会漏掉。
_ENV_NAMES = {"environ", "environb", "getenv", "putenv", "load_dotenv"}


def _sources() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_守则扫到了源码():
    assert {path.name for path in _sources()} >= {
        "__init__.py",
        "__main__.py",
        "config.py",
        "logging.py",
    }


def test_源码里没有_print():
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in _sources()
        for node in ast.walk(_tree(path))
        if _is_print_call(node)
    ]

    assert offenders == [], f"日志要走 ragamer.logging：{offenders}"


def _is_print_call(node: ast.AST) -> bool:
    """`print(…)` 与 `builtins.print(…)` 都算。"""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "print"
    return isinstance(node.func, ast.Attribute) and node.func.attr == "print"


def test_环境变量只在_config_模块读取():
    offenders = [
        path.name for path in _sources() if path != _CONFIG_MODULE and _touches_environment(path)
    ]

    assert offenders == [], f"配置只在 ragamer.config 装载：{offenders}"


def _touches_environment(path: Path) -> bool:
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.Name, ast.Attribute)) and (
            getattr(node, "id", None) in _ENV_NAMES or getattr(node, "attr", None) in _ENV_NAMES
        ):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "dotenv" for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "dotenv":
                return True
            if (node.module or "") == "os" and any(
                alias.name in _ENV_NAMES for alias in node.names
            ):
                return True
    return False


def test_env_example_与代码逐键对齐():
    documented = _documented_keys()
    declared = set(env_keys(Settings))

    assert documented - declared == set(), "模板里有代码不读的键"
    assert declared - documented == set(), "代码读了模板里没列的键"


def _documented_keys() -> set[str]:
    keys: set[str] = set()
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        key, separator, _ = entry.partition("=")
        assert separator, f"模板里这一行不是键值对：{entry}"
        keys.add(key.strip())
    return keys
