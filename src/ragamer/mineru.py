"""MinerU 云端解析：PDF 与图片 → Markdown。

一份资料走四步：申请上传链接 → PUT 上传 → 轮询结果 → 下载结果包。结果包解开就是
归一化文档（ADR-0006）——版面分析外包给 MinerU（本地不跑 magic-pdf），本模块只负责
把它接进归一化那条路（docs/ARCHITECTURE.md §1.1）。

四件事必须一次做对：

- **不无限等**：轮询有间隔与总时长两个上限（配置给，原项目标定的 3 秒 / 600 秒），
  到点报错并带上最后看到的状态；任务本身 `failed` 当场抛错——那是失败，不是还没好。
- **代理**：客户端 `trust_env=False`。大文件上传走系统代理会超时（坑 #6），
  结果包同样是几十上百 MB。
- **凭据不外流**：Token 只发给 MinerU 的接口。上传与下载走的是预签名地址，
  单独**不带** Bearer——把认证头挂在客户端上，就等于顺手把它发给了 CDN。
- **图内文字别指望它**：MinerU 不把图片区域里的文字 OCR 成正文，这是它的产品决策
  （§1.2）。二次 OCR 回填是下一张票的事，本模块只保证原图与条目级结构都拿得到。
"""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import posixpath
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from ragamer.config import MineruSettings
from ragamer.logging import get_logger
from ragamer.redaction import redact_address
from ragamer.sources import NormalizedDoc, SourceAsset, SourceDocument, SourceError, image_refs

logger = get_logger(__name__)

#: 批量上传解析接口挂在 base_url 下面的这一层。
API_PREFIX = "/api/v4"

#: 认得的格式。MinerU 还认 doc／ppt／html，那几种本项目走不到它：md／txt 直接读，
#: 网页走爬虫。列进来只会把不该走云端的格式引到这条路上。
PDF_SUFFIXES = (".pdf",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp")

#: 结果包里的正文与条目级结构。条目级结构按后缀认——不同版本的前缀不一样。
_MARKDOWN_NAME = "full.md"
_CONTENT_LIST_SUFFIX = "_content_list.json"

#: 解析没结束时服务端会报的中间状态。`done` 与 `failed` 之外都算「还在跑」。
_RUNNING_STATES = ("waiting-file", "pending", "running", "converting")

#: 一次请求最多试几次（只用于创建任务与下载结果包）。轮询不在此列：它本来就在反复问，
#: 上限是总时长。凭据被拒再试也还是被拒，所以不在这里重试。
_ATTEMPTS = 3


class MineruError(SourceError):
    """MinerU 这条路走不通。信息里已点名是哪个文件。"""


class MineruUnavailable(MineruError):
    """服务那一侧的问题：连不上、5xx、限流、返回的结构不认识。**可以再试**。"""


class MineruRejected(MineruError):
    """这次请求本身有问题：凭据或参数（HTTP 4xx、业务码非 0）。再试还是被拒。"""


class MineruTaskFailed(MineruError):
    """任务跑完了但失败了。服务端给的原因在信息里。"""


class MineruTimeout(MineruError):
    """等过了总时长上限还没完。信息里带最后看到的状态。"""


class MineruParser:
    """PDF 与图片 → MinerU → 归一化文档。

    端点、凭据、三个上限全部来自配置，这里不硬编码任何一项。传输层与时钟可注入：
    测试用 `httpx.MockTransport` 打真实代码路径（含轮询与解包），一次网络都不发。

    原图**不在这里进对象存储**——对象 key 要用到游戏 id，那在导入编排器里
    （`ragamer.sources.publish_assets`）。
    """

    SUFFIXES = PDF_SUFFIXES + IMAGE_SUFFIXES

    def __init__(
        self,
        config: MineruSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._base = config.base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}
        self._client = (
            client
            if client is not None
            else httpx.Client(
                # 不读环境里的代理：大文件上传走代理会超时（坑 #6）
                trust_env=False,
                timeout=config.request_timeout_seconds,
            )
        )
        self._sleep = sleep
        self._clock = clock

    def parse(self, source: SourceDocument) -> NormalizedDoc:
        """一份资料 → 归一化文档。原图作为附件留给导入编排器发布。

        :raises MineruError: 走不通。信息里已点名是哪个文件、卡在哪一步。
        """
        name = _upload_name(source.filename, source.data)
        batch_id, upload_url = self._request_upload(source, name)
        self._upload(source, upload_url)
        zip_url = self._await_result(source, batch_id, name)
        return _normalize(self._download(source, zip_url), source)

    def _request_upload(self, source: SourceDocument, name: str) -> tuple[str, str]:
        """申请上传链接。上传完不必再调提交接口——服务端扫到文件就自己排上。"""
        payload: dict[str, Any] = {
            "model_version": self._config.model_version,
            # 截图的文字全在图里，不开 OCR 就是一张白纸；PDF 反过来，
            # 开了 OCR 反而丢掉原有的文字层
            "files": [{"name": name, "is_ocr": _needs_ocr(source.filename)}],
        }
        url = f"{self._base}{API_PREFIX}/file-urls/batch"
        response = self._fetch("POST", url, source=source, json=payload)
        data = _business(response, source)
        batch_id = data.get("batch_id")
        urls = data.get("file_urls")
        if not isinstance(batch_id, str) or not batch_id:
            raise MineruUnavailable(_missing(source, "任务号", data))
        if not isinstance(urls, list) or not urls or not isinstance(urls[0], str) or not urls[0]:
            raise MineruUnavailable(_missing(source, "上传地址", data))
        return batch_id, urls[0]

    def _upload(self, source: SourceDocument, upload_url: str) -> None:
        # 预签名地址，裸字节上传：不带 Bearer，也不带 Content-Type
        self._fetch("PUT", upload_url, source=source, content=source.data, authorized=False)

    def _await_result(self, source: SourceDocument, batch_id: str, name: str) -> str:
        """轮询到解析完成，返回结果包的下载地址。

        两个上限一起兜住「不无限等」：间隔由配置给，总时长同样。单次失败（连不上、
        5xx）不判死刑——轮询本来就在反复问，死线是它的上限；但**任务 failed 当场抛错**，
        不等满时长。

        死线只管轮询这一段。在它之上还有上传、下载与各最多 `_ATTEMPTS` 次重试，
        每段都有自己的单请求超时，所以一份资料的最坏耗时是有限的，
        但不是 `poll_timeout_seconds` 秒——要卡整份资料的预算得从调用方那层来。
        """
        url = f"{self._base}{API_PREFIX}/extract-results/batch/{quote(batch_id, safe='')}"
        timeout = self._config.poll_timeout_seconds
        deadline = self._clock() + timeout
        state = "未知"
        reported: str | None = None
        while True:
            if self._clock() >= deadline:
                raise MineruTimeout(
                    f"{source.filename}：等待 MinerU 解析超过 {timeout:g} 秒（最后状态：{state}）"
                )
            try:
                # 单发不重试：这一层本来就在反复问，重试交给死线和间隔
                response = self._request("GET", url, source=source)
                entry = _entry(_business(response, source), source, name)
            except MineruUnavailable as exc:
                logger.warning("%s；%g 秒后再问一次", exc, self._config.poll_interval_seconds)
            else:
                state = str(entry.get("state") or "未知")
                if state != reported:
                    logger.info("%s：MinerU 状态 %s", source.filename, state)
                    reported = state
                if state == "done":
                    return _zip_url(entry, source)
                if state == "failed":
                    raise MineruTaskFailed(
                        f"{source.filename}：MinerU 解析失败"
                        f"（{entry.get('err_msg') or '服务端没给原因'}）"
                    )
                if state not in _RUNNING_STATES:
                    logger.warning(
                        "%s：MinerU 报了个没见过的状态 %r，继续等", source.filename, state
                    )
            self._sleep(self._config.poll_interval_seconds)

    def _download(self, source: SourceDocument, zip_url: str) -> bytes:
        # 同样是预签名地址，同样不带 Bearer
        response = self._fetch("GET", zip_url, source=source, authorized=False)
        return response.content

    def _fetch(
        self,
        method: str,
        url: str,
        *,
        source: SourceDocument,
        json: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        authorized: bool = True,
    ) -> httpx.Response:
        """发一次请求，可以再试的那种失败最多试 `_ATTEMPTS` 次。"""
        attempt = 1
        while True:
            try:
                return self._request(
                    method,
                    url,
                    source=source,
                    json=json,
                    content=content,
                    authorized=authorized,
                )
            except MineruUnavailable as exc:
                if attempt >= _ATTEMPTS:
                    raise
                logger.warning(
                    "%s；%g 秒后重试（第 %d/%d 次）",
                    exc,
                    self._config.poll_interval_seconds,
                    attempt,
                    _ATTEMPTS,
                )
                self._sleep(self._config.poll_interval_seconds)
                attempt += 1

    def _request(
        self,
        method: str,
        url: str,
        *,
        source: SourceDocument,
        json: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        authorized: bool = True,
    ) -> httpx.Response:
        """发一次请求。传输层与状态码在这里翻成项目自己的异常，重试是 `_fetch` 的事。"""
        try:
            response = self._client.request(
                method,
                url,
                json=json,
                content=content,
                headers=self._headers if authorized else None,
            )
        except httpx.HTTPError as exc:
            raise MineruUnavailable(
                f"{source.filename}：连不上 MinerU 的 {redact_address(url)}（{exc!r}）"
            ) from exc
        if response.status_code >= 400:
            raise _status_error(source, url, response)
        return response


def _status_error(source: SourceDocument, url: str, response: httpx.Response) -> MineruError:
    where = redact_address(url)
    if response.status_code == 429:
        return MineruUnavailable(f"{source.filename}：{where} 限流（HTTP 429）")
    if response.status_code >= 500:
        return MineruUnavailable(
            f"{source.filename}：{where} 服务端出错（HTTP {response.status_code}）"
        )
    return MineruRejected(
        f"{source.filename}：{where} 拒绝了请求"
        f"（HTTP {response.status_code}：{_excerpt(response.text)}）"
    )


def _business(response: httpx.Response, source: SourceDocument) -> Mapping[str, Any]:
    """取业务数据。业务码为 0 才算成功——非 0 是「这个请求本身有问题」，重试无用。"""
    try:
        payload = response.json()
    except ValueError as exc:
        raise MineruUnavailable(
            f"{source.filename}：MinerU 返回的不是 JSON（{_excerpt(response.text)}）"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MineruUnavailable(f"{source.filename}：MinerU 返回的不是对象（{_excerpt(payload)}）")
    if str(payload.get("code")) != "0":
        raise MineruRejected(
            f"{source.filename}：MinerU 拒绝了这次请求"
            f"（code={payload.get('code')}：{payload.get('msg') or '没给原因'}）"
        )
    data = payload.get("data")
    return data if isinstance(data, Mapping) else {}


def _missing(source: SourceDocument, what: str, data: Mapping[str, Any]) -> str:
    return f"{source.filename}：MinerU 没有返回{what}（{_excerpt(data)}）"


def _entry(data: Mapping[str, Any], source: SourceDocument, name: str) -> Mapping[str, Any]:
    """这个批次里属于这份文件的那一条结果。

    优先按文件名匹配，服务端只回一条时就用它（MinerU 可能给文件名加后缀）。
    一条都没有是「刚传上去还没被扫到」，属于还没好，由轮询继续等。
    """
    results = data.get("extract_result")
    entries = (
        [entry for entry in results if isinstance(entry, Mapping)]
        if isinstance(results, list)
        else []
    )
    matched = [entry for entry in entries if entry.get("file_name") == name]
    if matched:
        return matched[0]
    if len(entries) == 1:
        return entries[0]
    if entries:
        logger.warning(
            "%s：批次里有 %d 条结果，没有一条对得上 %s，继续等",
            source.filename,
            len(entries),
            name,
        )
    return {}


def _zip_url(entry: Mapping[str, Any], source: SourceDocument) -> str:
    url = entry.get("full_zip_url")
    if not isinstance(url, str) or not url:
        raise MineruUnavailable(f"{source.filename}：MinerU 说解析完了，却没给结果包地址")
    return url


def _normalize(payload: bytes, source: SourceDocument) -> NormalizedDoc:
    """结果包 → 归一化文档：正文 + 条目级结构 + 原图（附件）。

    只按名字读条目，**不落盘**：不写文件就没有目录穿越（zip slip）可谈。`..` 之类的
    越界条目直接跳过——它们只可能来自坏掉或伪造的包。
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise MineruUnavailable(f"{source.filename}：MinerU 的结果包不是个 zip（{exc}）") from exc
    with archive:
        names = [name for name in archive.namelist() if _readable(name)]
        markdown_name = _markdown_name(names, source)
        base = posixpath.dirname(markdown_name)
        markdown = _read_markdown(archive, markdown_name, source)
        return NormalizedDoc(
            markdown=markdown,
            images=image_refs(markdown),
            content_list=_read_content_list(archive, names, source),
            assets=tuple(
                SourceAsset(
                    name=_relative(name, base),
                    data=archive.read(name),
                    content_type=_content_type(name),
                )
                for name in sorted(names)
                if _is_image(name, base)
            ),
        )


def _readable(name: str) -> bool:
    """能读的条目：不是目录，也没有想往外爬的路径。"""
    parts = posixpath.normpath(name).split("/")
    return not name.endswith("/") and not posixpath.isabs(name) and ".." not in parts


def _markdown_name(names: Sequence[str], source: SourceDocument) -> str:
    """结果包里的正文。先按名字精确找，找不到才退一步认唯一的 `.md`。"""
    found = [name for name in names if posixpath.basename(name) == _MARKDOWN_NAME]
    if not found:
        found = [name for name in names if name.lower().endswith(".md")]
    if len(found) != 1:
        raise MineruUnavailable(
            f"{source.filename}：结果包里找不出正文（{_MARKDOWN_NAME}），看到的是 {_names(names)}"
        )
    return found[0]


def _read_markdown(archive: zipfile.ZipFile, name: str, source: SourceDocument) -> str:
    try:
        return archive.read(name).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MineruUnavailable(
            f"{source.filename}：结果包里的正文不是 UTF-8（{exc.reason}）"
        ) from exc


def _read_content_list(
    archive: zipfile.ZipFile, names: Sequence[str], source: SourceDocument
) -> tuple[Mapping[str, Any], ...]:
    """条目级结构。**缺了不阻断入库**：正文与原图都还在，只是二次 OCR 那一票少了依据。

    「哪个条目是图片」只有它说得清（§1.3），所以缺了要留一条痕，不能装着没这回事。
    """
    found = [
        name
        for name in names
        if posixpath.basename(name) == "content_list.json" or name.endswith(_CONTENT_LIST_SUFFIX)
    ]
    if not found:
        logger.warning(
            "%s：结果包里没有条目级结构（*%s），二次 OCR 少了依据；看到的是 %s",
            source.filename,
            _CONTENT_LIST_SUFFIX,
            _names(names),
        )
        return ()
    try:
        payload = json.loads(archive.read(sorted(found)[0]))
    except (ValueError, UnicodeDecodeError) as exc:
        raise MineruUnavailable(f"{source.filename}：结果包里的条目级结构读不了（{exc}）") from exc
    if not isinstance(payload, list):
        raise MineruUnavailable(f"{source.filename}：条目级结构不是一个数组（{_excerpt(payload)}）")
    return tuple(entry for entry in payload if isinstance(entry, Mapping))


def _is_image(name: str, base: str) -> bool:
    """结果包里的原图一律放在与正文同级的 `images/` 下面。"""
    prefix = f"{base}/" if base else ""
    return name.startswith(f"{prefix}images/")


def _relative(name: str, base: str) -> str:
    """包内的相对路径：正文引用图片时用的就是它，与正文所在的目录对齐。"""
    return posixpath.relpath(name, base) if base else name


def _content_type(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def _needs_ocr(filename: str) -> bool:
    """图片一律开 OCR。截图没有文字层可丢，不开就等于传了一张白纸上去。"""
    return Path(filename).suffix.lower() in IMAGE_SUFFIXES


def _upload_name(filename: str, data: bytes) -> str:
    """传给 MinerU 的文件名。

    扩展名要留着——服务端按它认格式。名字里带一段内容摘要：同一个批次里两份同名文件
    会互相覆盖，而且不会有任何提示。路径与引号一并去掉：这个名字要进请求体与查询串。
    """
    suffix = Path(filename).suffix
    stem = Path(filename).stem.replace('"', "").replace("\\", "").strip() or "source"
    return f"{stem}-{hashlib.sha256(data).hexdigest()[:8]}{suffix}"


def _names(names: Sequence[str]) -> str:
    """结果包里的文件名，进错误信息。封顶几条，坏包可能列几百个。"""
    shown = ", ".join(names[:10])
    return shown + ("…" if len(names) > 10 else "") if shown else "（空包）"


def _excerpt(value: Any, limit: int = 200) -> str:
    """压平并截断：服务端的错误页可能是几百行 HTML，不该整段进日志。"""
    flat = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    flat = " ".join(flat.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")
