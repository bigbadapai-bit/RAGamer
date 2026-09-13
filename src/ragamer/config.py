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
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
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

    @field_validator("db")
    @classmethod
    def _reject_shared_namespace(cls, value: str) -> str:
        """`default` 是每个 Milvus 实例都自带的库，原项目八成就在里面。

        两个项目共用实例时撞进同一个库，检索就会捞到对方的数据——而且是静默的。
        """
        if value == "default":
            raise ValueError(
                "不能叫 default：那是 Milvus 自带的库，与原项目共用实例时会撞在一起。"
                "请换一个只属于本项目的库名"
            )
        return value


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


class RedisSettings(BaseModel):
    """Redis。答案缓存与提问计数共用它（docs/ARCHITECTURE.md §4）。

    **它是唯一一个不通也不影响作答的外部依赖**：缓存读不了、写不进去都只是这次没缓存，
    提问照常走完整链路。所以它不在启动自检里——自检失败会拦住进程起来，
    而少了缓存并不该拦住任何事。
    """

    #: 连接串，形如 redis://[:密码@]主机:6379/0。可能内嵌密码，按密钥对待
    url: NonEmptySecret
    #: 键前缀。与原项目共用一套 Redis 时靠它错开——按前缀批量失效**是真的在删键**，
    #: 撞上了就是删掉别人的数据
    prefix: NonEmptyStr = "ragamer"


class LlmSettings(BaseModel):
    """语言模型。打标兜底、查询路由、多查询改写、生成都走它。"""

    base_url: NonEmptyStr
    api_key: NonEmptySecret
    model: NonEmptyStr
    #: 单次请求的超时（秒）。流式按两次产出之间的间隔计，不是整段答案的时长。
    timeout: float = Field(default=60.0, gt=0, le=600)
    #: 一次调用的总尝试次数（含首次）。有上限，重试不可能变成无限循环。
    max_attempts: int = Field(default=3, ge=1, le=10)
    #: 重试退避：起始间隔与封顶（秒），中间按 2 的幂增长。
    backoff_base: float = Field(default=0.5, gt=0, le=60)
    backoff_max: float = Field(default=8.0, gt=0, le=120)


class VisionSettings(LlmSettings):
    """视觉模型：给每张图补一段摘要，写进替代文本（docs/ARCHITECTURE.md §1.3）。

    与语言模型同形——同一个 `OpenAiLlm` 适配器接它，只是地址、密钥、模型名各是各的：
    读图的模型和写字的模型通常不是同一个（纯文本模型接不了图）。

    **不配就是整组不启用。** 补图那一层没接视觉模型时只做二次 OCR——图里没有文字的
    那几张（立绘、示意图）会少掉可检索的文本，其余一切照旧。三个键要么都给、要么
    都不给：配了一半是最坏的一种，它看起来像配好了，实际到用的时候才炸。
    """

    # 这三项在父类里是必需项，这里给空默认值——空即「没配」
    base_url: str = ""
    api_key: SecretStr = SecretStr("")
    model: str = ""

    @property
    def enabled(self) -> bool:
        """配齐了没有。全有或全无由下面的校验器保证，所以看一个就够。"""
        return bool(self.base_url)

    @model_validator(mode="after")
    def _all_or_nothing(self) -> VisionSettings:
        keys = (
            ("RAGAMER_VISION_BASE_URL", self.base_url),
            ("RAGAMER_VISION_API_KEY", self.api_key.get_secret_value()),
            ("RAGAMER_VISION_MODEL", self.model),
        )
        missing = [key for key, value in keys if not value]
        if len(missing) == len(keys):
            return self
        if missing:
            raise ValueError(
                f"视觉模型要么三个键都给、要么都不给。还差 {'、'.join(missing)}"
                "（整组不配时补图只做二次 OCR）"
            )
        return self


class MineruSettings(BaseModel):
    """MinerU 云端解析：PDF 与图片 → Markdown（docs/ARCHITECTURE.md §1.1）。

    三个上限都是「不无限等」的落点：单次请求、轮询间隔、轮询总时长。任务本身失败
    （`state == failed`）当场抛错，不等满时长——那是失败，不是还没好。
    """

    #: 服务地址。批量上传解析接口挂在它下面的 `/api/v4`
    base_url: NonEmptyStr = "https://mineru.net"
    api_key: NonEmptySecret
    #: 解析后端。**默认 vlm 是有意的**：图内文字的分析只有 vlm 才默认开着（§1.2）
    model_version: Literal["pipeline", "vlm", "MinerU-HTML"] = "vlm"
    #: 轮询间隔（秒）。再密也只是空转，原项目标定的是 3 秒
    poll_interval_seconds: Annotated[float, Field(gt=0, le=60)] = 3.0
    #: 轮询的总时长上限（秒）。到点还没完就报错，原项目标定的是 600 秒
    poll_timeout_seconds: Annotated[float, Field(gt=0, le=3600)] = 600.0
    #: 单次 HTTP 请求的超时（秒）。上传与下载都是几十上百 MB，不能用几秒的默认值——
    #: 接口自己的上限是 200MB，按 1MB/s 估要 200 秒，故留到 300
    request_timeout_seconds: Annotated[float, Field(gt=0, le=3600)] = 300.0


class ModelSettings(BaseModel):
    """向量化与精排**共用**的加载参数。

    两个模型跑在同一台机器上，设备与精度没有分开配置的理由——分开只会带来
    「一个在 GPU、一个在 CPU」这种没人有意为之、出事也难查的组合。
    """

    #: 加载到哪个设备：cpu / cuda / cuda:0
    device: NonEmptyStr = "cpu"
    #: 半精度。GPU 上打开能省一半显存；CPU 上开着也没关系——两个库都会把模型转回单精度。
    fp16: bool = False


class EmbedSettings(BaseModel):
    """向量化模型。**一个模型同时产出稠密与稀疏两路**，这是混合检索的基础。"""

    #: HuggingFace 上的模型名，或本地权重目录
    model: NonEmptyStr = "BAAI/bge-m3"
    #: 一次喂给模型几条文本
    batch_size: int = Field(default=8, ge=1, le=1024)
    #: 单条文本的 token 上限。**不要往下调**——调低就是重新引入静默截断：
    #: 超出的部分被悄悄丢掉，内容永久缺失且不报错（原项目照库的默认值 512 用，
    #: 踩的正是这一个）。真正生效的上限由模型自己的上下文长度决定，这里只是防手滑。
    max_length: int = Field(default=8192, ge=1, le=32768)


class RerankSettings(BaseModel):
    """精排模型。

    选长上下文的是有意的：512 token 那种上限会把「超长就摘要压缩再重试」逼出来，
    而那条路径在长上下文模型上根本不需要存在（docs/ARCHITECTURE.md §3.3）。
    """

    model: NonEmptyStr = "BAAI/bge-reranker-v2-m3"
    batch_size: int = Field(default=8, ge=1, le=1024)
    #: 单条候选的 token 上限。同样**不要往下调**，理由见 `EmbedSettings.max_length`。
    max_length: int = Field(default=8192, ge=1, le=32768)


class CrawlSettings(BaseModel):
    """网页抓取。合规要的三件事在这里，另一件（robots）在 `ragamer.crawl` 里做。"""

    #: 请求头里的身份。**不要改成匿名的通用值**：站点按 UA 匹配 robots 规则，
    #: 而不肯说自己是谁的爬虫被拦下来是应该的。带上联系地址，站长才找得到人。
    user_agent: NonEmptyStr = "RAGamerBot/0.1 (+https://github.com/bigbadapai-bit/RAGamer)"
    #: 单次请求超时（秒）
    timeout: float = Field(default=15.0, gt=0, le=300)
    #: 同一台主机两次请求之间的最小间隔（秒）。限的是主机不是页面——并发抓一个站的
    #: 十个页面，压力全落在同一台服务器上。0 表示不限，只在自己搭的站上这么配。
    min_interval: float = Field(default=1.0, ge=0, le=60)
    #: 单个页面的字节上限。**超了当场报错而不是截断**：截断的 HTML 会安静地少掉后半篇，
    #: 而导入看着是成功的。
    max_bytes: int = Field(default=5_000_000, ge=64_000, le=64_000_000)


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
    # 外部存储的连接与自检超时。远端不可达时要在这一点时间内失败，
    # 而不是挂在启动上——三个客户端共用同一个值，原项目标定的是 5 秒。
    store_timeout_seconds: Annotated[float, Field(gt=0)] = 5.0
    milvus: MilvusSettings
    mongo: MongoSettings
    minio: MinioSettings
    redis: RedisSettings
    llm: LlmSettings
    mineru: MineruSettings
    # 四个模型配置组都有完整默认值：不配也能跑，配了才落进 .env
    vision: VisionSettings = Field(default_factory=VisionSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    # 抓取也只有完整默认值：不配也能跑，配了才落进 .env
    crawl: CrawlSettings = Field(default_factory=CrawlSettings)


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
    "float_parsing": "必须是数字",
    "bool_parsing": "必须是 true 或 false",
    "greater_than": "必须大于 0",
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
    hint = _HINTS.get(str(error["type"]))
    if hint is not None:
        return hint
    # 校验器自己抛的 ValueError，pydantic 会加一层 "Value error, " 前缀
    return str(error["msg"]).removeprefix("Value error, ")
