"""网页抓取：robots、限频、两条路与三类错误。

真实客户端用 `httpx.MockTransport` 打——走的是完整的请求构造、重定向跟随、状态码翻译
与 robots 判定，只有最后的字节不来自网络。等待也换成一个自己会走的假钟，于是
「有没有留间隔、隔在谁和谁之间」可以直接断言，而不是靠真的等下去。
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ragamer.config import CrawlSettings
from ragamer.crawl import (
    AccessDeniedError,
    CrawlError,
    HttpCrawler,
    PageNotFoundError,
    RobotsDisallowedError,
    SiteUnreachableError,
    mediawiki_api,
    page_title,
)

WIKI = "https://wiki.test"
WIKITEXT = """\
{{Infobox character
| 名称 = 二郎神
| 类型 = 角色
}}
'''二郎神'''是《黑神话：悟空》中的[[妖王]]。

== 打法 ==
第二阶段要注意闪避。

[[File:Erlang.jpg|缩略图|二郎神立绘]]
[[Category:角色]]
"""

#: 有 RSD 的页面：MediaWiki 自己声明接口地址的方式，一个候选就说准了
WIKI_HTML = f"""\
<html><head>
<meta name="generator" content="MediaWiki 1.43.0" />
<link rel="EditURI" type="application/rsd+xml" href="{WIKI}/api.php?action=rsd" />
<title>二郎神 - 黑神话 Wiki</title>
</head><body><div id="mw-content-text">二郎神</div></body></html>
"""


def _article(lead: str) -> str:
    body = "".join(
        f"<h2>第{number}节 打法</h2><p>{'二郎神第二阶段要注意闪避与蓄力。' * 6}</p>"
        for number in range(1, 7)
    )
    return f"""\
<html><head><title>二郎神怎么打 - 黑神话攻略站</title></head><body>
<nav><a href="/">首页</a><a href="/a">最新攻略</a><a href="/b">配装推荐</a>
<a href="/c">地图大全</a><a href="/d">联系我们</a></nav>
<div class="content">{lead}{body}</div>
<footer>版权所有 黑神话攻略站 京ICP备00000000号</footer></body></html>
"""


#: 正文里自带一级标题。trafilatura 会把它渲染成 `#`，链路因此不用再补
PAGE_HTML = _article("<h1>二郎神打法</h1>")
#: 正文里没有一级标题，标题只在 `<title>` 里
PAGE_WITHOUT_HEADING = _article("")


class FakeClock:
    """自己走的钟：`sleep` 让它前进，于是限频算出来的等待是真的会到期。

    光记账不前进的话，第二次限频又会把整段间隔再等一遍——断言出来的等待比真机上多。
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeSite:
    """一个假站点：按地址前缀路由，连同收到请求的时刻一起记下来。

    前缀**长的先匹配**：`https://page.test/private` 与 `https://page.test/` 同时在表里时，
    要的是更具体的那条，而字典的插入顺序说了不算——排错了的话「页面不存在」那条路
    永远测不到，因为总是先命中通配的那一条。
    """

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock or FakeClock()
        self.requests: list[tuple[str, str, float]] = []
        #: 原样的请求对象，断言请求头这类东西时用
        self.sent: list[httpx.Request] = []
        self.routes: dict[str, Any] = {}

    def route(self, prefix: str, reply: Any) -> FakeSite:
        self.routes[prefix] = reply
        return self

    def paths(self) -> list[str]:
        return [path for _, path, _ in self.requests]

    def timeline(self) -> list[tuple[str, float]]:
        """每次请求的主机与时刻。限频断的是它们之间的先后，不是等了多久。"""
        return [(host, at) for host, _, at in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.host, request.url.path, self.clock()))
        self.sent.append(request)
        for prefix in sorted(self.routes, key=len, reverse=True):
            if str(request.url).startswith(prefix):
                reply = self.routes[prefix]
                return reply(request) if callable(reply) else httpx.Response(200, text=str(reply))
        return httpx.Response(404, text="没有排这条路由")


def crawl_settings(**overrides: Any) -> CrawlSettings:
    values: dict[str, Any] = {
        "user_agent": "RAGamerTest/0.1 (+https://crawl.test/bot)",
        "timeout": 5.0,
        "min_interval": 1.0,
        "max_bytes": 1_000_000,
    }
    return CrawlSettings(**{**values, **overrides})


def make_crawler(site: FakeSite, *, settings: CrawlSettings | None = None) -> HttpCrawler:
    return HttpCrawler(
        settings or crawl_settings(),
        client=httpx.Client(transport=httpx.MockTransport(site)),
        sleep=site.clock.sleep,
        clock=site.clock,
    )


def wiki_api(request: httpx.Request) -> httpx.Response:
    """按参数分流的那一个 `api.php`。"""
    params = request.url.params
    if params.get("meta") == "siteinfo":
        return httpx.Response(200, json={"query": {"general": {"sitename": "黑神话 Wiki"}}})
    if params.get("action") == "parse":
        return httpx.Response(200, json={"parse": {"title": "二郎神", "wikitext": WIKITEXT}})
    if params.get("prop") == "imageinfo":
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "title": "File:Erlang.jpg",
                            "imageinfo": [{"url": "https://img.test/erlang.jpg"}],
                        }
                    ]
                }
            },
        )
    return httpx.Response(400, text="没排这个参数组合")


def wiki_site(**extra: Any) -> FakeSite:
    """一个 MediaWiki 类假站点：robots 放行、接口按参数分流。"""
    site = FakeSite()
    site.route(f"{WIKI}/robots.txt", "User-agent: *\nDisallow:\n")
    site.route(f"{WIKI}/api.php", wiki_api)
    site.route(f"{WIKI}/wiki/", WIKI_HTML)
    for prefix, reply in extra.items():
        site.route(prefix, reply)
    return site


def wiki_site_failing(parse_reply: Any) -> FakeSite:
    """接口只在 `action=parse` 上出问题。

    探活那一步（`meta=siteinfo`）必须照常通过——它要是也失败，链路会认为这个候选地址
    压根不是接口，转去按普通网页抽正文，于是接口自己的报错根本走不到调用方。
    """
    site = wiki_site()
    site.routes[f"{WIKI}/api.php"] = lambda request: (
        parse_reply(request) if request.url.params.get("action") == "parse" else wiki_api(request)
    )
    return site


def page_site(html: str = PAGE_HTML, **extra: Any) -> FakeSite:
    site = FakeSite()
    site.route("https://page.test/robots.txt", "User-agent: *\nDisallow:\n")
    site.route("https://page.test/", html)
    for prefix, reply in extra.items():
        site.route(prefix, reply)
    return site


# ── 合规：robots ──────────────────────────────────────────


def test_抓之前先读_robots():
    site = page_site()
    make_crawler(site).crawl("https://page.test/a")

    assert site.paths()[0] == "/robots.txt"


def test_同一台主机只读一次_robots():
    site = page_site()
    crawler = make_crawler(site)
    crawler.crawl("https://page.test/a")
    crawler.crawl("https://page.test/b")

    assert site.paths().count("/robots.txt") == 1


def test_robots_不允许时明确报错():
    """报的是 robots 挡的，不是服务端 403——两者要改的地方不一样。"""
    site = page_site(**{"https://page.test/forbidden": None})
    site.routes["https://page.test/robots.txt"] = "User-agent: *\nDisallow: /forbidden\n"

    with pytest.raises(RobotsDisallowedError) as excinfo:
        make_crawler(site).crawl("https://page.test/forbidden")

    assert "robots.txt" in str(excinfo.value)
    assert "https://page.test/forbidden" in str(excinfo.value)
    # 被拦下之后一个页面都不该请求
    assert site.paths() == ["/robots.txt"]


def test_robots_只禁了别的蜘蛛时照抓():
    """站点按 UA 分规则，别人的禁令不是我们的。"""
    site = page_site()
    site.routes["https://page.test/robots.txt"] = (
        "User-agent: BadBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    )

    assert make_crawler(site).crawl("https://page.test/a").markdown


def test_没有_robots_文件时按不限制处理():
    """RFC 9309：404 是「这个站没有 robots.txt」，不是「不让抓」。"""
    site = page_site()

    assert make_crawler(site).crawl("https://page.test/a").markdown


def test_robots_读不到时按全禁处理():
    """保守的方向只能是少抓：猜错的另一边是替用户违反了站点的意愿。"""
    site = page_site()
    site.routes["https://page.test/robots.txt"] = lambda request: httpx.Response(500)

    with pytest.raises(RobotsDisallowedError):
        make_crawler(site).crawl("https://page.test/a")


# ── 合规：限频 ────────────────────────────────────────────


def test_同一台主机上连着两次抓之间留间隔():
    site = page_site()
    make_crawler(site).crawl("https://page.test/a")

    # 读 robots 与抓页面之间也要留间隔——限的是主机，不是「不同的页面之间」
    assert site.clock.sleeps == [1.0]


def test_不同主机之间不留间隔():
    """限频按主机算：给 A 站留的间隔不该让 B 站的请求一起等着。"""
    site = page_site()
    site.route("https://other.test/robots.txt", "User-agent: *\nDisallow:\n")
    site.route("https://other.test/", PAGE_HTML)
    crawler = make_crawler(site)
    crawler.crawl("https://page.test/a")
    crawler.crawl("https://other.test/a")

    timeline = site.timeline()
    last_page_test = max(at for host, at in timeline if host == "page.test")
    first_other = min(at for host, at in timeline if host == "other.test")
    # B 站的第一跳与 A 站的最后一跳同时发生：中间一秒是给 A 站留的，B 站没等
    assert first_other == last_page_test


def test_间隔可以配成零():
    site = page_site()
    make_crawler(site, settings=crawl_settings(min_interval=0)).crawl("https://page.test/a")

    assert site.clock.sleeps == []


# ── 合规：标明身份 ────────────────────────────────────────


def test_每一次请求都带上配置里的身份():
    """站点按 UA 匹配 robots 规则，匿名抓取被拦下来是应该的。"""
    site = wiki_site()
    make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")

    assert site.sent
    assert {request.headers["user-agent"] for request in site.sent} == {
        "RAGamerTest/0.1 (+https://crawl.test/bot)"
    }


# ── 三类错误 ──────────────────────────────────────────────


def test_站点连不上时报不可达():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("连接被拒绝", request=request)

    site = page_site(**{"https://page.test/down": refuse})

    with pytest.raises(SiteUnreachableError) as excinfo:
        make_crawler(site).crawl("https://page.test/down")

    assert "https://page.test/down" in str(excinfo.value)


def test_页面不存在时报找不到():
    site = page_site(**{"https://page.test/missing": lambda request: httpx.Response(404)})

    with pytest.raises(PageNotFoundError) as excinfo:
        make_crawler(site).crawl("https://page.test/missing")

    assert "404" in str(excinfo.value)


def test_被拒绝访问时报拒绝():
    site = page_site(**{"https://page.test/private": lambda request: httpx.Response(403)})

    with pytest.raises(AccessDeniedError) as excinfo:
        make_crawler(site).crawl("https://page.test/private")

    assert "403" in str(excinfo.value)


def test_服务端出错时按不可达处理():
    site = page_site(**{"https://page.test/broken": lambda request: httpx.Response(503)})

    with pytest.raises(SiteUnreachableError):
        make_crawler(site).crawl("https://page.test/broken")


def test_三种情况互不混淆():
    """「各自给出明确错误」要求的是能分开——都翻成同一个类型，用户就只知道失败了。"""
    assert issubclass(SiteUnreachableError, CrawlError)
    assert issubclass(PageNotFoundError, CrawlError)
    assert issubclass(AccessDeniedError, CrawlError)
    assert not issubclass(PageNotFoundError, SiteUnreachableError)
    assert not issubclass(SiteUnreachableError, AccessDeniedError)


def test_robots_挡下算作被拒绝访问的一种():
    """对调用方来说都是「站点不让我们拿」，分开只是为了信息里能说清是谁挡的。"""
    assert issubclass(RobotsDisallowedError, AccessDeniedError)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "不是地址", ""])
def test_只收_http_与_https(url):
    """外部输入一律不可信：别的协议不该走到抓取层。"""
    with pytest.raises(CrawlError):
        make_crawler(page_site()).crawl(url)


def test_页面超过字节上限时当场报错而不是截断():
    """截断的网页会安静地少掉后半篇，而导入看着是成功的。"""
    site = page_site(html=PAGE_HTML + "<p>" + ("填充" * 50_000) + "</p>")

    with pytest.raises(CrawlError) as excinfo:
        make_crawler(site, settings=crawl_settings(max_bytes=100_000)).crawl("https://page.test/a")

    assert "RAGAMER_CRAWL_MAX_BYTES" in str(excinfo.value)


# ── MediaWiki 那条路 ──────────────────────────────────────


def test_认得出_MediaWiki_并走接口():
    site = wiki_site()
    make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")

    assert "/api.php" in site.paths()


def test_接口那条路拿的是_wikitext_不是渲染后的_HTML():
    """下游靠 `[[Category:角色]]` 与 `{{Infobox}}` 定主体类型，渲染成 HTML 就没了。"""
    markdown = make_crawler(wiki_site()).crawl(f"{WIKI}/wiki/二郎神").markdown

    assert "[[Category:角色]]" in markdown
    assert "{{Infobox character" in markdown
    assert "## 打法" in markdown


def test_文档标题取接口归一之后的条目名():
    """重定向页的地址是旧名：拿地址里那一段，同一份资料会变成两份文档。"""
    site = wiki_site_failing(
        lambda request: httpx.Response(
            200, json={"parse": {"title": "二郎神（黑神话：悟空）", "wikitext": WIKITEXT}}
        )
    )

    assert (
        make_crawler(site)
        .crawl(f"{WIKI}/wiki/旧名字")
        .markdown.startswith("# 二郎神（黑神话：悟空）")
    )


def test_图片地址由接口问回来():
    document = make_crawler(wiki_site()).crawl(f"{WIKI}/wiki/二郎神")

    assert "![二郎神立绘](https://img.test/erlang.jpg)" in document.markdown
    assert document.images == ("https://img.test/erlang.jpg",)


def test_取不到图片地址时正文照样导入():
    """页面正文才是这一趟的交付，图片地址是加菜——为加菜把整页失败划不来。"""
    site = wiki_site()
    site.routes[f"{WIKI}/api.php"] = lambda request: (
        httpx.Response(500) if request.url.params.get("prop") == "imageinfo" else wiki_api(request)
    )

    document = make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")

    assert "[[File:Erlang.jpg|缩略图|二郎神立绘]]" in document.markdown
    assert document.images == ()


def test_接口说页面不存在时报找不到():
    site = wiki_site_failing(
        lambda request: httpx.Response(200, json={"error": {"code": "missingtitle", "info": "无"}})
    )

    with pytest.raises(PageNotFoundError):
        make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")


def test_接口说没权限时报拒绝():
    site = wiki_site_failing(
        lambda request: httpx.Response(200, json={"error": {"code": "readapidenied", "info": "无"}})
    )

    with pytest.raises(AccessDeniedError):
        make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")


def test_有_rsd_时不去试别的脚本路径():
    """RSD 是 MediaWiki 自己声明的接口地址，它说准了就不该再去猜 `/w/api.php`。"""
    site = wiki_site()
    make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")

    assert "/w/api.php" not in site.paths()


def test_只有引擎声明时按两种脚本路径逐个试():
    """`generator` 只说清了引擎、没说接口在哪。脚本放在 `/w/` 下的站点（维基百科那一类）
    只能靠逐个探——RSD 缺失时这是唯一还认得出的信号。"""
    site = wiki_site()
    site.routes[f"{WIKI}/wiki/"] = WIKI_HTML.replace(
        f'<link rel="EditURI" type="application/rsd+xml" href="{WIKI}/api.php?action=rsd" />', ""
    )

    make_crawler(site).crawl(f"{WIKI}/wiki/二郎神")

    assert "/w/api.php" in site.paths()


def test_像_MediaWiki_但取不出条目名时改走正文抽取():
    """地址里没有 `/wiki/` 也没有 `title=`，与其瞎猜一个条目名，不如按普通网页抽。"""
    site = wiki_site()
    site.route(f"{WIKI}/", WIKI_HTML)
    document = make_crawler(site).crawl(f"{WIKI}/")

    assert document.markdown


# ── 普通网页那条路 ────────────────────────────────────────


def test_普通网页抽出正文_丢掉导航与页脚():
    document = make_crawler(page_site()).crawl("https://page.test/a")

    assert "闪避与蓄力" in document.markdown
    assert "版权所有" not in document.markdown
    assert "配装推荐" not in document.markdown


def test_正文没有标题时补上页面标题():
    """没有它，文档标题会回落到地址最后一段（`https://…/a` → `a`），读不出是什么文档。"""
    markdown = (
        make_crawler(page_site(html=PAGE_WITHOUT_HEADING)).crawl("https://page.test/a").markdown
    )

    assert markdown.startswith("# 二郎神怎么打\n")


def test_正文自带标题时不再补一个():
    """页面自己有一级标题时再补，同一份文档会出现两个大标题，而文档标题同时是重导时的
    替换键——两个来源打架时替换的范围就飘了。"""
    markdown = make_crawler(page_site()).crawl("https://page.test/a").markdown

    assert markdown.startswith("# 二郎神打法\n")
    assert "二郎神怎么打" not in markdown


def test_抽不出正文时明确报错():
    site = page_site(html="<html><body><script>var x = 1;</script></body></html>")

    with pytest.raises(CrawlError) as excinfo:
        make_crawler(site).crawl("https://page.test/a")

    assert "抽不出正文" in str(excinfo.value)


# ── 来源地址 ──────────────────────────────────────────────


def test_抓下来的文档带来源地址():
    assert make_crawler(page_site()).crawl("https://page.test/a").source_url == (
        "https://page.test/a"
    )


def test_跟过重定向之后记的是最终地址():
    """地址是引用的落点：记成中途那一跳，用户点回去看到的是另一页。"""
    site = page_site()
    site.route("https://page.test/old", redirect_to("https://page.test/new"))
    site.route("https://page.test/new", PAGE_HTML)

    assert make_crawler(site).crawl("https://page.test/old").source_url == ("https://page.test/new")


# ── 重定向不能成为绕过合规的口子 ──────────────────────────


def redirect_to(target: str) -> Any:
    return lambda request: httpx.Response(301, headers={"location": target})


def test_重定向到别的主机时那一跳也要过_robots():
    """跳过去之后就是对那台主机的一次新请求，它同样受对方 robots 的约束。

    跟着 httpx 一次跳到底会绕过这一层：robots.txt 里只写了入口那个地址。
    """
    site = page_site(**{"https://page.test/hop": redirect_to("https://other.test/offlimits")})
    site.route("https://other.test/robots.txt", "User-agent: *\nDisallow: /offlimits\n")
    site.route("https://other.test/", PAGE_HTML)

    with pytest.raises(RobotsDisallowedError) as excinfo:
        make_crawler(site).crawl("https://page.test/hop")

    assert "other.test" in str(excinfo.value)
    # 对方主机的 robots 读了，那一页本体没请求
    assert "/robots.txt" in site.paths()
    assert site.paths().count("/offlimits") == 0


def test_重定向到别的主机时那一跳也要限频():
    """限频按主机算：跳过去之后，那一跳与对方主机上此前那一次请求之间同样要隔开。"""
    site = page_site(**{"https://page.test/hop": redirect_to("https://other.test/page")})
    site.route("https://other.test/robots.txt", "User-agent: *\nDisallow:\n")
    site.route("https://other.test/", PAGE_HTML)

    make_crawler(site).crawl("https://page.test/hop")

    other = [(path, at) for host, path, at in site.requests if host == "other.test"]
    # 对方主机上先读 robots（不同主机，紧接着发），跳过去的那一跳则等满一秒
    assert other == [("/robots.txt", 1.0), ("/page", 2.0)]


def test_重定向有上限():
    """转圈圈的站点不该把这一趟挂住。"""
    site = page_site(**{"https://page.test/a1": redirect_to("https://page.test/a2")})
    site.route("https://page.test/a2", redirect_to("https://page.test/a1"))

    with pytest.raises(CrawlError) as excinfo:
        make_crawler(site).crawl("https://page.test/a1")

    assert "重定向" in str(excinfo.value)


def test_robots_文件自己重定向时跟着走():
    """`http://` 的地址被站点 301 到 `https://` 是常态，robots.txt 也一样——
    跟不过去就会把一份其实读得到的规则当成「读不到」，按全禁处理。"""
    site = FakeSite()
    site.route("http://page.test/robots.txt", redirect_to("https://page.test/robots.txt"))
    site.route("https://page.test/robots.txt", "User-agent: *\nDisallow: /private\n")
    site.route("http://page.test/", PAGE_HTML)

    crawler = make_crawler(site)
    assert crawler.crawl("http://page.test/a").markdown  # 放行的那一页照抓

    # 跳过去读到的规则确实生效了，不只是「读到了」而已
    with pytest.raises(RobotsDisallowedError):
        crawler.crawl("http://page.test/private")


# ── 编码 ──────────────────────────────────────────────────


def test_头里没说编码时按页面自己声明的读():
    """中文老站把 `charset=gbk` 写在 `meta` 里、响应头什么都不写是常态。

    按 UTF-8 硬读会让整篇正文变成乱码进库，而乱码的向量照样算得出来——查不出来也不报错。
    """
    html = (
        '<html><head><meta charset="gbk"><title>二郎神</title></head><body>'
        + "<h2>打法</h2><p>"
        + "二郎神第二阶段要注意闪避。" * 8
        + "</p>"
        + "</body></html>"
    ).encode("gbk")
    site = page_site()
    site.routes["https://page.test/"] = lambda request: httpx.Response(
        200, content=html, headers={"content-type": "text/html"}
    )

    markdown = make_crawler(site).crawl("https://page.test/a").markdown

    assert "闪避" in markdown
    assert "�" not in markdown  # 一个替换字符都不该有


# ── 两个纯函数 ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (f"{WIKI}/wiki/二郎神", "二郎神"),
        (f"{WIKI}/zh/wiki/二郎神", "二郎神"),
        (f"{WIKI}/wiki/%E4%BA%8C%E9%83%8E%E7%A5%9E", "二郎神"),
        (f"{WIKI}/wiki/Erlang_Shen", "Erlang Shen"),
        (f"{WIKI}/w/index.php?title=%E4%BA%8C%E9%83%8E%E7%A5%9E", "二郎神"),
        (f"{WIKI}/", ""),
    ],
)
def test_从地址里取条目名(url, expected):
    assert page_title(url) == expected


def test_普通网页认不出_MediaWiki():
    assert mediawiki_api(PAGE_HTML, "https://page.test/a") == ()


def test_认_MediaWiki_优先看_RSD():
    assert mediawiki_api(WIKI_HTML, f"{WIKI}/wiki/二郎神") == (f"{WIKI}/api.php",)


def test_只有引擎声明时给出两种脚本路径():
    html = WIKI_HTML.replace(
        f'<link rel="EditURI" type="application/rsd+xml" href="{WIKI}/api.php?action=rsd" />', ""
    )

    assert mediawiki_api(html, f"{WIKI}/wiki/二郎神") == (f"{WIKI}/w/api.php", f"{WIKI}/api.php")


# ── 取图 ──────────────────────────────────────────────────

#: 一张最小的 PNG。取的是字节，所以它长什么样不重要，重要的是**不能被当文本读**。
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
IMAGE_URL = "https://cdn.test/a.png"


def image_reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PNG_BYTES)


def test_取图拿到的是字节():
    """图片是二进制，按文本读会把它毁掉——所以 `_Fetched` 存的是字节。"""
    site = FakeSite().route(IMAGE_URL, image_reply)

    assert make_crawler(site).image(IMAGE_URL) == PNG_BYTES


def test_取图也先看_robots():
    """图常在另一台主机上（正文在一个域、图在另一个域），对那一台就是一次新请求。

    绕过 robots 等于悄悄多抓了一台站——而且不报错。
    """
    site = FakeSite()
    site.route("https://cdn.test/robots.txt", "User-agent: *\nDisallow: /\n")
    site.route(IMAGE_URL, image_reply)

    with pytest.raises(RobotsDisallowedError):
        make_crawler(site).image(IMAGE_URL)


def test_取图也受限频约束_且与页面共用同一个计时():
    """限的是主机：同一台站上抓完页面紧接着取图，中间一样要留间隔。"""
    site = page_site(**{IMAGE_URL: image_reply})
    crawler = make_crawler(site)

    crawler.crawl("https://page.test/a")
    crawler.image(IMAGE_URL)

    # 抓页面（读 robots + 取正文）之后取的图，落在同一台主机上
    assert site.paths()[-1] == "/a.png"
    assert site.clock.sleeps[-1] == 1.0


def test_取图被拒时报的是被拒绝():
    site = FakeSite().route(IMAGE_URL, lambda request: httpx.Response(403, text="no"))

    with pytest.raises(AccessDeniedError):
        make_crawler(site).image(IMAGE_URL)


def test_取图超过字节上限时当场报错():
    """与抓页面同一条口径：超限报错，不截断——截断的图是一张坏图，而且看着是成功的。"""
    site = FakeSite().route(IMAGE_URL, lambda request: httpx.Response(200, content=b"x" * 70_000))

    with pytest.raises(CrawlError) as excinfo:
        make_crawler(site, settings=crawl_settings(max_bytes=64_000)).image(IMAGE_URL)

    assert "上限" in str(excinfo.value)


def test_取图只收_http_地址():
    with pytest.raises(CrawlError):
        make_crawler(FakeSite()).image("file:///etc/passwd")
