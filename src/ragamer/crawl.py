"""网页抓取：一个网址进，一份 :class:`~ragamer.sources.NormalizedDoc` 出。

两条路（docs/ARCHITECTURE.md §1.1），走哪条由**站点自己说了算**——页面上有没有
MediaWiki 的声明，看一眼 HTML 就知道：

- **MediaWiki 类站点**走它的开放接口（`api.php`）拿 **wikitext 原文**，再交给
  `ragamer.wikitext` 转 Markdown。选接口而不是抓页面 HTML，是因为下游读的正是
  wiki 标记：切分器认 `[[内链]]`／`Category:`／`{{模板}}` 判断词条页，打标器读
  `[[Category:角色]]` 与 Infobox 字段定主体类型。先渲染成 HTML 再转 Markdown
  会把这两处信号一起洗掉，而且不报错。
- **普通网页**走正文抽取：导航、页脚、侧边栏这些 boilerplate 由 trafilatura 识别并丢掉，
  正文直接转成 Markdown。

**合规的三件事全落在 :meth:`HttpCrawler._fetch` 一处**，因为任何一次出网都得经过它：
先读该主机的 robots.txt、同一主机两次请求之间留足间隔、请求头里标明自己是谁。
少走一处，那条路径就静默地不受约束——所以不给出网留第二条路。

🔴 **已知限制：不拦内网地址。** 这个端点收的是用户给的网址，多用户部署时它就是一个
SSRF 面（`http://169.254.169.254/…`、`http://<内网服务>/`）。本项目现在是自己给自己
导资料的单机工具，而拦内网要连 DNS 解析一起做（否则域名解析到内网就绕过去了），
半套防护比没有更容易让人放心。上多用户之前必须补上。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from ragamer.config import CrawlSettings
from ragamer.logging import get_logger
from ragamer.sources import NormalizedDoc, SourceError, image_refs
from ragamer.wikitext import file_titles, normalize_file_name, to_markdown

logger = get_logger(__name__)

#: 一次 `prop=imageinfo` 最多问几个文件名。接口自己限 50。
_IMAGE_BATCH = 50

#: 站点的接口地址发现方式（RSD）。MediaWiki 每个页面都会在 head 里声明它，
#: 脚本路径是 `/w/` 还是根目录、语言子域占了几段路径，它一律说得准。
_RSD_LINK = re.compile(r"<link\b[^>]*\brel=[\"']?EditURI[\"']?[^>]*>", re.IGNORECASE)
_HREF = re.compile(r"\bhref=[\"']([^\"']+)[\"']", re.IGNORECASE)
#: 引擎声明。只有它、没有 RSD 时说不出接口在哪，只能把两种常见脚本路径都试一遍。
_GENERATOR_META = re.compile(r"<meta\b[^>]*\bname=[\"']generator[\"'][^>]*>", re.IGNORECASE)
_CONTENT = re.compile(r"\bcontent=[\"']([^\"']*)[\"']", re.IGNORECASE)
#: 正文开头是不是已经有一级标题。只认一个井号——`##` 是小节，不是文档标题
_TOP_HEADING = re.compile(r"^#\s")


class CrawlError(SourceError):
    """一个网址抓不成文档。信息里已点名是哪个地址。"""


class SiteUnreachableError(CrawlError):
    """站点连不上或暂时不可用：域名解析不了、连接被拒、超时、5xx。"""


class PageNotFoundError(CrawlError):
    """页面不存在（HTTP 404／410，或接口回 `missingtitle`）。"""


class AccessDeniedError(CrawlError):
    """站点拒绝访问：HTTP 401／403／451，或接口回权限不足。"""


class RobotsDisallowedError(AccessDeniedError):
    """robots.txt 不允许抓这个地址。

    是 :class:`AccessDeniedError` 的一种——对调用方来说都是「站点不让我们拿」，
    分开成一个类型只是为了信息里能说清是 robots 挡的，而不是服务端 403。
    """


@dataclass(frozen=True)
class _Fetched:
    """一次出网的产物。`url` 是跟过重定向之后的最终地址。"""

    text: str
    url: str


class HttpCrawler:
    """真实抓取。构造不碰网络，第一次真的抓才连。

    :param client: 换掉传输层用。测试拿 `httpx.MockTransport` 打真实的请求构造、
        状态码翻译与 robots 判定，不碰网络。
    :param sleep: 限频用的等待。测试传一个只记账不真等的函数。
    """

    def __init__(
        self,
        settings: CrawlSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = settings
        self._sleep = sleep
        self._clock = clock
        self._client = client if client is not None else httpx.Client(timeout=settings.timeout)
        #: 每台主机上一次请求的时刻与它读到的 robots 规则。抓同一站的几百页时，
        #: robots.txt 只读一次——每页都读一遍是拿人家服务器当缓存用。
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    def crawl(self, url: str) -> NormalizedDoc:
        """一个网址 → 一份归一化文档。

        :raises CrawlError: 地址不可用、站点不可达、页面不存在、被拒绝访问。
        """
        address = checked_url(url)
        page = self._get(address)
        for api in mediawiki_api(page.text, page.url):
            wiki = self._try_wiki(api, page.url)
            if wiki is not None:
                return wiki
        return self._from_page(page)

    # ── 出网 ──────────────────────────────────────────────

    def _get(self, url: str) -> _Fetched:
        """先问 robots 再出网。**业务代码一律走它。**"""
        rules = self._robots_for(_origin(url))
        if rules is not None and not rules.can_fetch(self._config.user_agent, url):
            raise RobotsDisallowedError(
                f"{url}：该站点的 robots.txt 不允许抓这个地址。"
                "换个允许抓的来源，或直接找站点要 dump"
            )
        return self._fetch(url)

    def _fetch(self, url: str) -> _Fetched:
        """真正出网的那一层：限频 + 标明身份的 UA。robots.txt 自己走这里，免得自指。"""
        self._throttle(_origin(url))
        try:
            # 重定向在这里显式跟随，不靠客户端的构造参数：注入的客户端（测试用的
            # MockTransport）默认是不跟的，那样两处行为会不一样，而差别只在跳转过的页面上
            # 才看得出来——地址记成中途那一跳，用户点回去看到的是另一页
            with self._client.stream(
                "GET",
                url,
                headers={"User-Agent": self._config.user_agent},
                follow_redirects=True,
            ) as response:
                raise_for_status(response, url)
                body = _read_capped(response, self._config.max_bytes, url)
                return _Fetched(text=_decode(body, response.encoding), url=str(response.url))
        except httpx.HTTPError as exc:
            # 域名解析不了、连接被拒、读超时都落在这里。翻成一句话，别把 httpx 的
            # 异常类型甩给用户看
            raise SiteUnreachableError(f"{url}：连不上站点（{type(exc).__name__}：{exc}）") from exc

    def _throttle(self, origin: str) -> None:
        """同一台主机两次请求之间至少隔 `min_interval` 秒。

        限的是**主机**不是页面：并发抓同一站的十个页面，压力全落在同一台服务器上。
        """
        interval = self._config.min_interval
        previous = self._last_request.get(origin)
        now = self._clock()
        if previous is not None and interval > 0:
            wait = interval - (now - previous)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request[origin] = now

    def _robots_for(self, origin: str) -> RobotFileParser | None:
        """一台主机的 robots 规则；`None` 表示没有规则、不限制。

        规则本身取不到时按 RFC 9309 分档：404／410 是「这个站没有 robots.txt」，
        按不限制；其余（403、5xx、连不上）按全禁。保守的方向只能是**少抓**——
        猜错的另一边是替用户违反了站点的意愿，那一条没有回头路。
        """
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        return self._robots[origin]

    def _load_robots(self, origin: str) -> RobotFileParser | None:
        url = f"{origin}/robots.txt"
        try:
            body = self._fetch(url).text
        except PageNotFoundError:
            logger.info("%s 没有 robots.txt，按不限制处理", origin)
            return None
        except CrawlError as exc:
            logger.warning("读不到 %s 的 robots.txt（%s），按全禁处理", origin, exc)
            return _deny_all()
        parser = RobotFileParser()
        parser.parse(body.splitlines())
        return parser

    # ── 两条路 ────────────────────────────────────────────

    def _try_wiki(self, api: str, page_url: str) -> NormalizedDoc | None:
        """试一个候选接口地址；它压根不是接口时返回 `None`，由调用方换下一个试。"""
        title = page_title(page_url)
        if not title:
            logger.warning("%s 像是 MediaWiki，但从地址里取不出条目名，改走正文抽取", page_url)
            return None
        try:
            self._api(api, {"action": "query", "meta": "siteinfo"})
        except CrawlError as exc:
            # 两种情况都会落到这里：这个地址压根不是接口（试下一个候选），以及
            # 站点的 robots.txt 不许调接口（那就只剩正文抽取这一条合规的路）
            logger.warning("%s 走不通（%s），改按普通网页抽正文", api, exc)
            return None
        return self._wiki(api, page_url, title)

    def _wiki(self, api: str, page_url: str, title: str) -> NormalizedDoc:
        """接口这条路：拿 wikitext，转 Markdown，顺带把图片地址问回来。"""
        parsed = self._api(
            api,
            {"action": "parse", "page": title, "prop": "wikitext", "redirects": "1"},
        )
        payload = parsed.get("parse") or {}
        wikitext = str(payload.get("wikitext") or "")
        if not wikitext.strip():
            raise CrawlError(f"{page_url}：接口没给出正文")
        page = to_markdown(
            wikitext,
            # 用接口归一之后的条目名，不用地址里那一段：重定向页的地址是旧名，
            # 而文档标题同时是重导时的替换键，两个名字会变成两份文档
            title=str(payload.get("title") or title),
            image_urls=self._image_urls(api, file_titles(wikitext)),
        )
        return NormalizedDoc(markdown=page.markdown, images=page.images, source_url=page_url)

    def _image_urls(self, api: str, names: Sequence[str]) -> dict[str, str]:
        """图片名 → 地址。**要名字是因为地址只有接口知道**：wikitext 里写的是文件名。

        取不到就返回手上这部分：页面正文才是这一趟的交付，图片地址是加菜。
        为了加菜把整页导入失败划不来，而那些图在正文里会原样保留（见 `wikitext._image`）。
        """
        urls: dict[str, str] = {}
        for batch in _batched(names, _IMAGE_BATCH):
            try:
                payload = self._api(
                    api,
                    {
                        "action": "query",
                        "prop": "imageinfo",
                        "iiprop": "url",
                        "titles": "|".join(f"File:{name}" for name in batch),
                    },
                )
            except CrawlError as exc:
                logger.warning("%s 取图片地址失败（%s），这些图在正文里原样保留", api, exc)
                return urls
            for entry in (payload.get("query") or {}).get("pages") or ():
                info = (entry.get("imageinfo") or [{}])[0]
                address = str(info.get("url") or "")
                if address:
                    urls[normalize_file_name(str(entry.get("title") or ""))] = address
        return urls

    def _api(self, api: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        """问一次 MediaWiki 接口。地址与参数由本模块拼，取值范围不外泄。"""
        query = urlencode({**params, "format": "json", "formatversion": "2"})
        fetched = self._get(f"{api}?{query}")
        try:
            payload = json.loads(fetched.text)
        except ValueError as exc:
            raise CrawlError(f"{api}：接口返回的不是 JSON") from exc
        if not isinstance(payload, dict):
            raise CrawlError(f"{api}：接口返回的不是一个对象")
        error = payload.get("error")
        if error is not None:
            raise _api_error(api, error)
        return payload

    def _from_page(self, page: _Fetched) -> NormalizedDoc:
        """普通网页：抽出正文、丢掉导航与页脚。"""
        markdown = trafilatura.extract(
            page.text,
            url=page.url,
            output_format="markdown",
            # 表格与图片都要：前者是切分器做原子化的对象，后者是补图那一层的输入
            include_tables=True,
            include_images=True,
            # 链接目标不要：正文里一串 URL 对检索没有帮助，只会占掉 token
            include_links=False,
            include_formatting=True,
        )
        if not markdown or not markdown.strip():
            # 抽不出正文与「页面不存在」是两回事：地址是通的，只是没有可用的内容
            raise CrawlError(f"{page.url}：抽不出正文，这个页面可能整页都是导航或脚本")
        text = markdown.strip()
        title = _page_title_of(page)
        if title and not _TOP_HEADING.match(text):
            # 正文里没有一级标题时补一个：文档标题同时是重导时的替换键，
            # 让它回落到地址最后一段（`https://…/a/b.html` → `b`）读不出是什么文档
            text = f"# {title}\n\n{text}"
        return NormalizedDoc(markdown=text + "\n", images=image_refs(text), source_url=page.url)


def _page_title_of(page: _Fetched) -> str:
    """网页自己的标题。抽不出来就返回空串——宁可不补标题，也不要补一个错的。"""
    # 兜住全部异常：标题只是给文档加的抬头，抽不出来不该拖垮整页的导入
    try:
        metadata = trafilatura.extract_metadata(page.text, default_url=page.url)
    except Exception as exc:
        logger.warning("%s 读不出页面标题（%s）", page.url, exc)
        return ""
    return str(getattr(metadata, "title", "") or "").strip()


def checked_url(url: str) -> str:
    """只收 http／https。外部输入一律不可信：`file://`、`javascript:` 不该走到抓取层。"""
    address = url.strip()
    parts = urlsplit(address)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise CrawlError(f"{url!r} 不是一个 http／https 地址")
    return address


def mediawiki_api(html: str, url: str) -> tuple[str, ...]:
    """这个页面是不是 MediaWiki，是的话它的 `api.php` 可能在哪。

    认两种信号，按可靠性排：

    1. `<link rel="EditURI" href="…/api.php?action=rsd">` —— MediaWiki 自己声明的
       接口发现方式。它一个候选就说准了，所以只给这一个。
    2. `<meta name="generator" content="MediaWiki …">` —— 只说清了引擎、没说接口在哪，
       于是把两种常见的脚本路径都试一遍，由调用方逐个探。
    """
    rsd = _RSD_LINK.search(html)
    if rsd is not None:
        href = _attribute(rsd.group(0), _HREF)
        if href:
            return (_absolute(href, url).split("?")[0],)
    generator = _GENERATOR_META.search(html)
    if generator is not None and "mediawiki" in _attribute(generator.group(0), _CONTENT).lower():
        origin = _origin(url)
        return (f"{origin}/w/api.php", f"{origin}/api.php")
    return ()


def page_title(url: str) -> str:
    """页面地址 → 条目名。`/wiki/二郎神`、`/zh/wiki/二郎神`、`?title=二郎神` 三种写法都认。

    取不出来就返回空串——那时候改走正文抽取，比拿地址里的一段瞎猜强。
    """
    parts = urlsplit(url)
    wanted = parse_qs(parts.query).get("title")
    if wanted:
        return unquote(wanted[0]).replace("_", " ").strip()
    path = unquote(parts.path)
    marker = "/wiki/"
    if marker in path:
        return path.split(marker, 1)[1].replace("_", " ").strip()
    return ""


def raise_for_status(response: httpx.Response, url: str) -> None:
    """状态码 → 明确的失败。三种情况各有各的类型（验收要求「各自给出明确错误」）。"""
    status = response.status_code
    if status < 400:
        return
    if status in (404, 410):
        raise PageNotFoundError(f"{url}：页面不存在（HTTP {status}）")
    if status in (401, 403, 451):
        raise AccessDeniedError(f"{url}：站点拒绝访问（HTTP {status}）")
    if status >= 500:
        raise SiteUnreachableError(f"{url}：站点暂时不可用（HTTP {status}）")
    raise CrawlError(f"{url}：请求失败（HTTP {status}）")


def _api_error(api: str, error: Any) -> CrawlError:
    code = str(error.get("code", "")) if isinstance(error, Mapping) else ""
    info = str(error.get("info", "")) if isinstance(error, Mapping) else str(error)
    if code in ("missingtitle", "nosuchpageid"):
        return PageNotFoundError(f"{api}：页面不存在（{info or code}）")
    if code in ("readapidenied", "permissiondenied", "badaccess-groups"):
        return AccessDeniedError(f"{api}：站点拒绝访问（{info or code}）")
    return CrawlError(f"{api}：接口报错 {code}：{info}")


def _read_capped(response: httpx.Response, limit: int, url: str) -> bytes:
    """按上限读响应体。**超限当场报错，不截断**——截断的 HTML 会安静地少掉后半篇，
    而导入看着是成功的。"""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > limit:
            raise CrawlError(
                f"{url}：页面超过 {limit} 字节的上限。真需要它就调大 RAGAMER_CRAWL_MAX_BYTES"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode(body: bytes, encoding: str | None) -> str:
    """按响应头声明的编码读，读不了退回 UTF-8。

    只认响应头、不去猜编码：猜错的中文正文会变成一片乱码进库，而乱码的向量照样算得出来，
    查不出来也不报错。
    """
    for candidate in (encoding, "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("utf-8", errors="replace")


def _deny_all() -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(["User-agent: *", "Disallow: /"])
    return parser


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _absolute(href: str, url: str) -> str:
    """把 href 补成绝对地址。RSD 给的通常已经是绝对的，相对的就按当前页解析。"""
    return urljoin(url, href)


def _attribute(tag: str, pattern: re.Pattern[str]) -> str:
    found = pattern.search(tag)
    return unescape(found.group(1)) if found is not None else ""


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
