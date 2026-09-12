"""配置装载的唯一入口。

环境变量只在本模块读取（由 `tests/test_conventions.py` 拦截其它模块的 `os.environ`），
其余模块从同一个 `Settings` 对象取用。键名与 `.env.example` 一一对应，
多一个少一个都会被 `tests/test_conventions.py` 拦下。

会静默生效的错误一律在启动阶段拦下：缺键、取值非法、拼错的键名，
以及会被 `.env` 展开改写的取值。
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

#: 所有键的前缀。与原项目共享同一台机器时靠它避免环境变量串味。
ENV_PREFIX = "RAGAMER_"

#: 默认读取的配置文件，相对于运行时的工作目录。
DEFAULT_ENV_FILE = ".env"

#: utf-8-sig 与纯 UTF-8 都认——Windows 记事本另存 UTF-8 会带 BOM。
ENV_FILE_ENCODING = "utf-8-sig"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# 端点和凭据一律不允许为空：空串等于没配，但要等运行到一半才炸。
NonEmptyStr = Annotated[str, Field(min_length=1)]
NonEmptySecret = Annotated[SecretStr, Field(min_length=1)]


class ConfigError(Exception):
    """配置缺失或非法。信息里已经点名是哪个键。"""


class MilvusSettings(BaseModel):
    """Milvus 向量库。每款游戏一个 collection，用 database 与原项目隔离。"""

    uri: NonEmptyStr
    token: NonEmptySecret
    db: NonEmptyStr = "ragamer"


class MongoSettings(BaseModel):
    """MongoDB。存会话、知识库元数据、术语映射。"""

    # 连接串可能内嵌账号密码，按密钥对待，不落进日志
    uri: NonEmptySecret
    db: NonEmptyStr = "ragamer"


class MinioSettings(BaseModel):
    """MinIO 对象存储。存原图等二进制内容。"""

    endpoint: NonEmptyStr
    access_key: NonEmptyStr
    secret_key: NonEmptySecret
    bucket: NonEmptyStr = "ragamer-images"
    secure: bool = False


class LlmSettings(BaseModel):
    """语言模型。打标兜底、查询路由、多查询改写、生成都走它。"""

    base_url: NonEmptyStr
    api_key: NonEmptySecret
    model: NonEmptyStr


class Settings(BaseSettings):
    """全部配置。由 :func:`load_settings` 或 :func:`get_settings` 构造。"""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="_",
        # 默认会按每一个下划线拆层级（minio_access_key → minio.access.key），
        # 于是多词字段匹配不上、被静默丢掉——必须限成只拆一层。
        env_nested_max_split=1,
        # match_prefix：只认带前缀的键。`.env` 里属于别的项目的键被忽略。
        dotenv_filtering="match_prefix",
        extra="forbid",
        case_sensitive=False,
        # 空串按没配处理：有默认值的回落到默认值，没默认值的以「缺少该键」报出来
        env_ignore_empty=True,
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
    )

    log_level: LogLevel = "INFO"
    milvus: MilvusSettings
    mongo: MongoSettings
    minio: MinioSettings
    llm: LlmSettings


def load_settings(env_file: str | Path | None = DEFAULT_ENV_FILE) -> Settings:
    """装载并校验配置。这是环境变量被读到的唯一地方。

    :param env_file: 配置文件路径；`None` 表示只读环境变量（测试用）。
    :raises ConfigError: 缺少必需键、取值非法、键名拼错，或配置文件读不了。
    """
    path = None if env_file is None else Path(env_file).expanduser()
    try:
        problems = _env_file_problems(path)
        if problems:
            raise ConfigError(_format_problems(problems, path))
        return Settings(_env_file=env_file)
    except UnicodeDecodeError as exc:
        raise ConfigError(_encoding_problem(path, exc)) from exc
    except (ValidationError, SettingsError) as exc:
        raise ConfigError(_format_errors(exc, path)) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内共享同一个配置对象。测试里用 `get_settings.cache_clear()` 复位。

    装载路径是相对工作目录的 `.env`，所以进程中途换工作目录不会换配置。
    """
    return load_settings()


def env_keys(model: type[BaseModel] = Settings) -> list[str]:
    """列出模型会读取的全部环境变量键，按声明顺序、嵌套配置组递归展开。"""
    return [_env_key(path) for path in _field_paths(model)]


def required_env_keys(model: type[BaseModel] = Settings) -> list[str]:
    """列出没有默认值、必须由使用者填的键。"""
    return [_env_key(path) for path in _field_paths(model, required_only=True)]


def _field_paths(
    model: type[BaseModel],
    prefix: tuple[str, ...] = (),
    *,
    required_only: bool = False,
) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    for name, field in model.model_fields.items():
        path = prefix + (name,)
        nested = _nested_model(field.annotation)
        if nested is not None:
            paths.extend(_field_paths(nested, path, required_only=required_only))
        elif not required_only or field.is_required():
            paths.append(path)
    return paths


def _nested_model(annotation: object) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _model_at(path: Sequence[str]) -> type[BaseModel] | None:
    """按字段路径往下走，停在嵌套配置组上；不是配置组则返回 `None`。"""
    model: type[BaseModel] | None = Settings
    for part in path:
        if model is None:
            return None
        field = model.model_fields.get(part)
        if field is None:
            return None
        model = _nested_model(field.annotation)
    return model


def _env_key(path: Sequence[str]) -> str:
    return ENV_PREFIX + "_".join(part.upper() for part in path)


def _env_file_values(path: Path | None, *, interpolate: bool) -> dict[str, str | None]:
    """直接读配置文件，用来对比 pydantic-settings 读到的东西。

    它不汇报未知键、也不汇报取值被 `${…}` 改写——这两件事只能自己看。
    """
    if path is None or not path.is_file():
        return {}
    values = dotenv_values(path, encoding=ENV_FILE_ENCODING, interpolate=interpolate)
    return {key: value for key, value in values.items() if key is not None}


def _env_file_problems(path: Path | None) -> list[str]:
    """找出会静默生效的问题：未知的键、会被 `${…}` 展开改写的取值。"""
    raw = _env_file_values(path, interpolate=False)
    expanded = _env_file_values(path, interpolate=True)
    rewritten = [
        f"{key.upper()}：取值里的 ${{…}} 会被 .env 展开后再使用，"
        "实际生效的取值与写入的不一致——请直接写成最终取值"
        for key, value in raw.items()
        if expanded.get(key) != value
    ]
    return rewritten + _unknown_key_problems(set(raw) | set(os.environ))


def _unknown_key_problems(found: set[str]) -> list[str]:
    """键名拼错在 `.env` 与环境变量两条路径上都只会被静默忽略，得自己点名。"""
    known = {key.upper() for key in env_keys(Settings)}
    problems = []
    for key in sorted({key.upper() for key in found if key.upper().startswith(ENV_PREFIX)} - known):
        if any(known_key.startswith(f"{key}_") for known_key in known):
            problems.append(f"{key}：这是配置组名，要按组内每个键分别给")
        else:
            problems.append(f"{key}：未知的配置键{_suggestion(key, known)}")
    return problems


def _suggestion(key: str, known: set[str]) -> str:
    close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.7)
    return f"，是不是想写 {close[0]}？" if close else ""


#: 常见错误的中文提示；其余情况直接用 pydantic 的原文（更具体）。
_HINTS = {
    "missing": "缺少该键",
    "extra_forbidden": "未知的配置键，检查拼写",
    "string_too_short": "不能为空",
    "too_short": "不能为空",
    "int_parsing": "必须是整数",
    "bool_parsing": "必须是 true 或 false",
}


def _format_errors(exc: ValidationError | SettingsError, path: Path | None) -> str:
    if isinstance(exc, SettingsError):
        # 兜底：已知字段拿到非 JSON 取值之类。未知键与整组赋值在上面就拦下了。
        lines = [f"  - {exc}"]
    else:
        lines = [f"  - {key}：{_hint(error)}" for error in exc.errors() for key in _keys_of(error)]
    return _header(path) + "\n".join(lines)


def _format_problems(problems: Sequence[str], path: Path | None) -> str:
    return _header(path) + "\n".join(f"  - {problem}" for problem in problems)


def _header(path: Path | None) -> str:
    header = "配置校验失败，请对照 .env.example 修正（首次使用：cp .env.example .env）"
    if path is not None and not path.is_file():
        header += f"\n没有找到 {path}（配置文件相对工作目录查找，当前目录：{Path.cwd()}）"
    return header + "：\n"


def _encoding_problem(path: Path | None, exc: UnicodeDecodeError) -> str:
    return (
        f"配置文件 {path} 不是 UTF-8 编码（{exc.reason}）。"
        "本项目的 .env 一律用 UTF-8 保存——Windows 记事本选「UTF-8」另存即可。"
    )


def _keys_of(error: Mapping[str, Any]) -> list[str]:
    path = tuple(str(part) for part in error["loc"])
    if error["type"] == "missing":
        nested = _model_at(path)
        if nested is not None:
            # 整个配置组一个键都没给：点名这组里所有没有默认值的键
            return [_env_key(item) for item in _field_paths(nested, path, required_only=True)]
    return [_env_key(path)]


def _hint(error: Mapping[str, Any]) -> str:
    return _HINTS.get(str(error["type"]), str(error["msg"]))
