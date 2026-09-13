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
#: 只有这个模块允许构造存储客户端
_CONTAINER_MODULE = _PACKAGE / "container.py"

#: 读环境变量的名字。`from os import getenv` 之后是裸名字，只认属性会漏掉。
_ENV_NAMES = {"environ", "environb", "getenv", "putenv", "load_dotenv"}

#: 三个存储客户端、两个本地模型与语言模型的实现类。它们在组合根构造一次然后注入——
#: 模块级单例换不掉，测试缝也就没了。
_ADAPTERS = {
    "MilvusChunkStore",
    "MongoDocStore",
    "MinioObjectStore",
    "BgeM3Embedder",
    "BgeReranker",
    "OpenAiLlm",
}

#: 供应商库只允许出现在各自的适配器模块里：其余模块只认 ragamer.stores 与
#: ragamer.vectors 的协议，换后端、换模型都不动业务代码。
#: 测试不在扫描范围内——造假件要用到供应商的异常类型。
_VENDOR_MODULES = {
    "pymilvus": _PACKAGE / "stores" / "chunks.py",
    "pymongo": _PACKAGE / "stores" / "documents.py",
    "minio": _PACKAGE / "stores" / "objects.py",
    "urllib3": _PACKAGE / "stores" / "objects.py",
    # 真实模型是可选的 models 组，bge 里对它是懒导入（没装也能 import 本模块）
    "FlagEmbedding": _PACKAGE / "vectors" / "bge.py",
}


def _sources() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_守则扫到了源码():
    assert {path.name for path in _sources()} >= {
        "__init__.py",
        "__main__.py",
        "config.py",
        "container.py",
        "llm.py",
        "chunking.py",
        "tagging.py",
        "sources.py",
        "importing.py",
        "query.py",
        "routing.py",
        "retrieval.py",
        "answering.py",
        "conversations.py",
        "api.py",
        "logging.py",
        "memory.py",
        "chunks.py",
        "documents.py",
        "objects.py",
        "base.py",
        "bge.py",
        "fake.py",
    }


def test_存储客户端与模型适配器只在组合根构造():
    """没有任何模块级单例——单例在测试里换不成内存假件。"""
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in _sources()
        if path != _CONTAINER_MODULE
        for node in ast.walk(_tree(path))
        if _calls_adapter(node)
    ]

    assert offenders == [], f"客户端与模型适配器只在 ragamer.container 构造：{offenders}"


def _calls_adapter(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) in _ADAPTERS


def test_供应商库只在各自的适配器模块导入():
    """业务层只认协议，不认 pymilvus / pymongo / minio。"""
    offenders = [
        f"{path.name}:{node.lineno} → {vendor}"
        for path in _sources()
        for node in ast.walk(_tree(path))
        if (vendor := _imported_vendor(node)) is not None and path != _VENDOR_MODULES[vendor]
    ]

    assert offenders == [], f"供应商库只能出现在自己的适配器模块里：{offenders}"


def _imported_vendor(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        roots = [alias.name.split(".")[0] for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        roots = [(node.module or "").split(".")[0]]
    else:
        return None
    return next((root for root in roots if root in _VENDOR_MODULES), None)


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
