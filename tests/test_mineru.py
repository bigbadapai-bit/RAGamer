"""MinerU 接入：四步链路、两个上限、代理与凭据、结果包解包。

外部服务用 `httpx.MockTransport` 顶掉——打的是真实代码路径（申请链接、上传、轮询、
下载、解包），一次网络都不发。时钟也换成假的：轮询的「还没到点」与「到点了」必须
是确定性的，真等 600 秒的测试没人会跑。
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from ragamer.config import MineruSettings
from ragamer.mineru import (
    MineruParser,
    MineruRejected,
    MineruTaskFailed,
    MineruTimeout,
    MineruUnavailable,
)
from ragamer.sources import SourceDocument, SourceError

PNG = b"\x89PNG\r\n\x1a\n" + "截图".encode()
PDF = b"%PDF-1.7\n" + "正文".encode()

#: 一份正文：一张 Markdown 引用的图，一张 HTML 引用的图（表格内嵌图片就是这种）。
MARKDOWN = """\
# 二郎神攻略

![立绘](images/a.jpg)

<table><tr><td><img src="images/b.png"/></td></tr></table>
"""

CONTENT_LIST: list[dict[str, Any]] = [
    {"type": "text", "text": "二郎神攻略", "page_idx": 0},
    {"type": "image", "img_path": "images/a.jpg", "image_caption": [], "page_idx": 0},
    {"type": "image", "img_path": "images/b.png", "image_caption": [], "page_idx": 0},
]

IMAGES = {"images/a.jpg": b"\xff\xd8jpeg-bytes", "images/b.png": b"\x89PNG-bytes"}


def make_settings(**overrides: Any) -> MineruSettings:
    values: dict[str, Any] = {
        "base_url": "https://mineru.test",
        "api_key": "test-mineru-token",
        "model_version": "vlm",
        "poll_interval_seconds": 3.0,
        "poll_timeout_seconds": 600.0,
        "request_timeout_seconds": 30.0,
    }
    return MineruSettings(**{**values, **overrides})


def make_bundle(
    *,
    prefix: str = "",
    markdown: str = MARKDOWN,
    content_list: Any = CONTENT_LIST,
    images: Mapping[str, bytes] = IMAGES,
    extra: Mapping[str, bytes] | None = None,
) -> bytes:
    """造一个结果包。`prefix` 用来验服务端把产物放进一层子目录的情况。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{prefix}full.md", markdown)
        if content_list is not None:
            archive.writestr(
                f"{prefix}content_list.json", json.dumps(content_list, ensure_ascii=False)
            )
        for name, data in {**images, **(extra or {})}.items():
            archive.writestr(f"{prefix}{name}", data)
    return buffer.getvalue()


class FakeTime:
    """假时钟：`sleep` 把时间往前推，于是「到点」由 sleep 的次数决定。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeMineru:
    """一个假的 MinerU 服务：按 URL 分派，并把每一次请求记下来。"""

    def __init__(self, payload: bytes | None = None) -> None:
        self.payload = make_bundle() if payload is None else payload
        #: 轮询依次回的状态。取到最后一个之后一直用它。
        self.states: list[str] = ["done"]
        self.err_msg = "文件损坏"
        #: 申请上传链接那一步要回的状态码，用完了就回 200。
        self.batch_status: list[int] = []
        #: 下载结果包那一步要回的状态码，用完了就回 200。
        self.download_status: list[int] = []
        self.code = 0
        self.msg = "ok"
        #: 覆盖轮询结果里的那一条；不给就按真实字段拼一条。
        self.entry: dict[str, Any] | None = None
        self.requests: list[httpx.Request] = []
        self.uploaded = b""
        self.uploaded_name = ""
        self.model_version = ""
        self.is_ocr: bool | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.endswith("/file-urls/batch"):
            return self._accept(request)
        if request.method == "PUT":
            self.uploaded = request.content
            return httpx.Response(200, request=request)
        if "/extract-results/batch/" in url:
            return self._progress(request)
        if url.startswith("https://cdn.test/"):
            return self._download(request)
        raise AssertionError(f"假服务不认这个请求：{request.method} {url}")

    def _accept(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.model_version = body["model_version"]
        self.uploaded_name = body["files"][0]["name"]
        self.is_ocr = body["files"][0]["is_ocr"]
        status = self.batch_status.pop(0) if self.batch_status else 200
        if status >= 400:
            return httpx.Response(status, text="boom", request=request)
        return httpx.Response(
            200,
            json={
                "code": self.code,
                "msg": self.msg,
                "data": {
                    "batch_id": "b-1",
                    "file_urls": ["https://upload.test/put?sign=abc"],
                },
            },
            request=request,
        )

    def _progress(self, request: httpx.Request) -> httpx.Response:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        entry = self.entry
        if entry is None:
            entry = {"file_name": self.uploaded_name, "state": state}
            if state == "done":
                entry["full_zip_url"] = "https://cdn.test/r.zip?sign=xyz"
            if state == "failed":
                entry["err_msg"] = self.err_msg
        return httpx.Response(
            200, json={"code": 0, "data": {"extract_result": [entry]}}, request=request
        )

    def _download(self, request: httpx.Request) -> httpx.Response:
        status = self.download_status.pop(0) if self.download_status else 200
        if status >= 400:
            return httpx.Response(status, text="boom", request=request)
        return httpx.Response(200, content=self.payload, request=request)

    # --- 断言用的视图 ---

    def calls(self, method: str) -> list[httpx.Request]:
        return [item for item in self.requests if item.method == method]

    def polls(self) -> list[httpx.Request]:
        return [item for item in self.requests if "/extract-results/batch/" in str(item.url)]


def make_parser(fake: FakeMineru, time: FakeTime, **overrides: Any) -> MineruParser:
    return MineruParser(
        make_settings(**overrides),
        client=httpx.Client(transport=httpx.MockTransport(fake), trust_env=False),
        sleep=time.sleep,
        clock=time.clock,
    )


def make_source(filename: str = "攻略.png", data: bytes = PNG) -> SourceDocument:
    return SourceDocument(filename=filename, data=data)


# --- 四步链路 ---


def test_一份截图从申请链接一路走到归一化文档():
    fake = FakeMineru()
    time = FakeTime()
    parser = make_parser(fake, time)

    doc = parser.parse(make_source())

    assert doc.markdown == MARKDOWN
    assert doc.content_list == tuple(CONTENT_LIST)
    # 正文里 Markdown 与 HTML 两种引用都要认出来（表格内嵌的图是 HTML 那种）
    assert doc.images == ("images/a.jpg", "images/b.png")
    assert [asset.name for asset in doc.assets] == ["images/a.jpg", "images/b.png"]
    assert doc.assets[0].data == IMAGES["images/a.jpg"]
    assert doc.assets[0].content_type == "image/jpeg"
    assert fake.uploaded == PNG
    # 后端与 OCR 开关进请求体：默认 vlm，截图必须开 OCR
    assert fake.model_version == "vlm"
    assert fake.is_ocr is True


def test_上传给_MinerU_的文件名带扩展名与内容摘要():
    """MinerU 按扩展名认格式；同名不同内容的两份文件在服务端会互相覆盖。"""
    first = FakeMineru()
    make_parser(first, FakeTime()).parse(make_source("攻略.png", PNG))
    second = FakeMineru()
    make_parser(second, FakeTime()).parse(make_source("攻略.png", PDF))

    assert first.uploaded_name.startswith("攻略-")
    assert first.uploaded_name.endswith(".png")
    assert first.uploaded_name != second.uploaded_name


def test_PDF_不开_OCR():
    """PDF 有自己的文字层，开 OCR 反而把它丢掉；截图反过来，不开就是一张白纸。"""
    fake = FakeMineru()
    make_parser(fake, FakeTime()).parse(make_source("手册.pdf", PDF))

    assert fake.is_ocr is False


def test_结果包放在子目录里也能解出来():
    fake = FakeMineru(make_bundle(prefix="二郎神/vlm/"))
    parser = make_parser(fake, FakeTime())

    doc = parser.parse(make_source())

    assert doc.markdown == MARKDOWN
    # 包内的相对路径要跟正文对齐，正文里引用的是 images/a.jpg
    assert [asset.name for asset in doc.assets] == ["images/a.jpg", "images/b.png"]


# --- 凭据与代理 ---


def test_预签名地址不带_token():
    """上传与下载走的是各自的签名地址，把 Bearer 顺手发过去就是把凭据给了 CDN。"""
    fake = FakeMineru()
    make_parser(fake, FakeTime()).parse(make_source())

    authorized = [item for item in fake.requests if item.headers.get("authorization")]
    assert [str(item.url) for item in authorized] == [
        "https://mineru.test/api/v4/file-urls/batch",
        "https://mineru.test/api/v4/extract-results/batch/b-1",
    ]
    assert fake.calls("PUT")[0].headers.get("authorization") is None
    assert fake.calls("GET")[-1].headers.get("authorization") is None


def test_出错信息里的地址抹掉签名():
    """预签名地址的查询串里就是签名，原样进日志等于把凭据落盘。"""
    fake = FakeMineru()
    fake.download_status = [403] * 3

    with pytest.raises(MineruRejected) as excinfo:
        make_parser(fake, FakeTime()).parse(make_source())

    assert "cdn.test" in str(excinfo.value)
    assert "sign=xyz" not in str(excinfo.value)


def test_适配器不读环境里的代理():
    """大文件上传走系统代理会超时（架构文档第六节坑 #6）。"""
    parser = MineruParser(make_settings())

    assert parser._client.trust_env is False


# --- 轮询的两个上限 ---


def test_轮询按间隔等_出结果就停():
    fake = FakeMineru()
    fake.states = ["running", "running", "done"]
    time = FakeTime()
    parser = make_parser(fake, time)

    parser.parse(make_source())

    assert len(fake.polls()) == 3
    # 第一次立刻问，之后每问一次等一个间隔
    assert time.slept == [3.0, 3.0]


def test_任务失败时当场抛错_不等满时长():
    fake = FakeMineru()
    fake.states = ["running", "failed", "done"]
    time = FakeTime()

    with pytest.raises(MineruTaskFailed) as excinfo:
        make_parser(fake, time).parse(make_source("攻略.png"))

    assert "文件损坏" in str(excinfo.value)
    assert "攻略.png" in str(excinfo.value)
    assert len(fake.polls()) == 2  # failed 之后不再问
    assert time.now == 3.0  # 没等到总时长上限


def test_轮询到点还没完就报错_并带上最后看到的状态():
    fake = FakeMineru()
    fake.states = ["running"]
    time = FakeTime()
    parser = make_parser(fake, time, poll_timeout_seconds=10.0, poll_interval_seconds=3.0)

    with pytest.raises(MineruTimeout) as excinfo:
        parser.parse(make_source("攻略.png"))

    message = str(excinfo.value)
    assert "10 秒" in message
    assert "running" in message  # 最后看到的状态，排障的唯一线索
    assert time.now >= 10.0
    assert len(fake.polls()) <= 5  # 死线兜住了，不是问不完


def test_一条结果都还没有时继续等而不是判失败():
    """刚上传完还没被服务端扫到就是这个样子，属于还没好，不是失败。"""
    fake = FakeMineru()
    fake.entry = {}  # 服务端回了空结果

    with pytest.raises(MineruTimeout) as excinfo:
        make_parser(fake, FakeTime(), poll_timeout_seconds=6.0).parse(make_source())

    assert "未知" in str(excinfo.value)  # 一条状态都没看到，如实说
    assert len(fake.polls()) >= 2


# --- 失败分类 ---


def test_服务端_5xx_会重试():
    fake = FakeMineru()
    fake.batch_status = [503, 500]
    time = FakeTime()

    doc = make_parser(fake, time).parse(make_source())

    assert doc.markdown == MARKDOWN
    assert len(fake.calls("POST")) == 3
    assert time.slept == [3.0, 3.0]


def test_下载结果包_5xx_也重试():
    fake = FakeMineru()
    fake.download_status = [502]
    time = FakeTime()

    assert make_parser(fake, time).parse(make_source()).markdown == MARKDOWN
    assert len(fake.calls("GET")) == 3  # 一次轮询 + 两次下载


def test_凭据被拒不重试():
    fake = FakeMineru()
    fake.batch_status = [401]

    with pytest.raises(MineruRejected) as excinfo:
        make_parser(fake, FakeTime()).parse(make_source("攻略.png"))

    assert "401" in str(excinfo.value)
    assert len(fake.calls("POST")) == 1


def test_业务码非_0_也算被拒():
    fake = FakeMineru()
    fake.code = -60005
    fake.msg = "文件大小超出限制"

    with pytest.raises(MineruRejected) as excinfo:
        make_parser(fake, FakeTime()).parse(make_source())

    assert "-60005" in str(excinfo.value)
    assert "文件大小超出限制" in str(excinfo.value)
    assert len(fake.calls("POST")) == 1


def test_结果包不是_zip_时报错并点名文件():
    fake = FakeMineru("这不是个 zip".encode())

    with pytest.raises(MineruUnavailable) as excinfo:
        make_parser(fake, FakeTime()).parse(make_source("攻略.png"))

    assert "攻略.png" in str(excinfo.value)


def test_结果包里没有正文时报错():
    """正文是整条链路唯一的产出物，缺了就没有可入库的东西。"""
    fake = FakeMineru(_without_full_md(make_bundle()))

    with pytest.raises(MineruUnavailable) as excinfo:
        make_parser(fake, FakeTime()).parse(make_source())

    assert "full.md" in str(excinfo.value)


def _without_full_md(payload: bytes) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(payload))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in source.namelist():
            if name != "full.md":
                archive.writestr(name, source.read(name))
    return buffer.getvalue()


def test_越界的条目被跳过():
    """`..` 只可能来自坏掉或伪造的包。这里不落盘，也没有往外写的路径给它走。"""
    fake = FakeMineru(make_bundle(extra={"../evil.txt": b"nope", "/etc/passwd": b"nope"}))
    parser = make_parser(fake, FakeTime())

    doc = parser.parse(make_source())

    assert [asset.name for asset in doc.assets] == ["images/a.jpg", "images/b.png"]


def test_结果包里没有条目级结构时留一条痕_不阻断(caplog):
    """正文与原图都还在，少的只是二次 OCR 的依据——所以要留痕，不能装着没这回事。"""
    fake = FakeMineru(make_bundle(content_list=None))
    parser = make_parser(fake, FakeTime())

    with caplog.at_level("WARNING"):
        doc = parser.parse(make_source("攻略.png"))

    assert doc.content_list == ()
    assert "条目级结构" in caplog.text


# --- 解析适配器 ---


def test_适配器认_PDF_与图片():
    parser = MineruParser(make_settings())

    assert ".pdf" in parser.SUFFIXES
    assert ".png" in parser.SUFFIXES
    assert ".md" not in parser.SUFFIXES


def test_适配器把_MinerU_的失败原样报出来():
    fake = FakeMineru()
    fake.states = ["failed"]
    parser = make_parser(fake, FakeTime())

    with pytest.raises(SourceError):  # 导入编排器按这个基类兜
        parser.parse(make_source())
