# RAGamer

面向游戏攻略与资料的中文 RAG 助手。用户按游戏建库、用口语提问，系统先确认问的是哪款游戏、哪个版本，再检索并生成带引用的回答。

设计取舍见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，领域术语以 [`CONTEXT.md`](CONTEXT.md) 为准，不可逆决策见 [`docs/adr/`](docs/adr/)。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)（Python 3.12 由 uv 自己装，不必预装）。

```bash
uv sync                  # 建虚拟环境、装依赖
cp .env.example .env     # 然后按注释把值填成自己的
uv run ragamer           # 启动自检：配置合格、Milvus/Mongo/MinIO 都连得上才通过
```

退出码 `0` 通过、`2` 配置有问题、`3` 存储连不上——远端不可达时在
`RAGAMER_STORE_TIMEOUT_SECONDS` 秒内失败，错误信息里点名是哪个服务、哪个地址。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `uv run pytest` | 跑测试（集成测试默认不跑，加 `-m integration` 才跑） |
| `uv run ruff check .` | 跑 lint |
| `uv run ruff format .` | 格式化 |

CI 在每次 push 与 PR 上跑 lint、格式检查与测试，见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。

## 配置

全部配置键集中列在 [`.env.example`](.env.example)，每个键带一行说明。

- **装载只有一个入口**：`src/ragamer/config.py`。其余模块拿 `Settings` 对象用，不各自读环境变量——这条由 `tests/test_conventions.py` 机械拦截。
- **键名必须带 `RAGAMER_` 前缀**。同机其它项目的环境变量与 `.env` 里不带前缀的键会被忽略；带前缀却拼错的键名（`.env` 与环境变量两条路径都算）在启动时报错，不会静默生效。
- **缺键或取值非法在启动阶段就报出来**，错误信息直接点名是哪个键（如 `RAGAMER_MILVUS_TOKEN`），不认识的键名还会给出最接近的那个。
- **取值按字面读取**：`.env` 默认会做 `${…}` 插值、静默改写取值，因此这类写法一律在启动时报错。
- 日志统一走 `ragamer.logging`，源码里不出现 `print`（同样由 `tests/test_conventions.py` 拦截）。

## 存储

三个外部存储各自定义成一个协议（`ragamer.stores`），实现类**只在组合根 `ragamer.container` 构造一次**，
没有任何模块级单例——这条由 `tests/test_conventions.py` 机械拦截。

- 换后端不动业务代码：业务层只认协议，`pymilvus` / `pymongo` / `minio` 只允许出现在各自的适配器模块里。
- 测试不依赖云端：`ragamer.stores.memory` 里的内存假件实现同一组协议，测试里整体替换。
  真的连云端的那部分标了 `integration`，默认不跑，`uv run pytest -m integration` 才跑（需要填好的 `.env`）。
- 与原项目共用实例、**命名空间错开**：Milvus 用 database（不接受 `default`）、MinIO 换桶名、Mongo 换库名。
- 切片存储的过滤条件只收结构化对象，不接受字符串表达式——表达式的生成与取值转义都封在适配器里。

## 向量化与精排

两个模型适配器（`ragamer.vectors`），协议与组合根里的存储客户端是同一套打法：
业务层只认 `Embedder` / `Reranker` 两个协议，`ragamer.vectors.fake` 里的确定性假件实现同一组协议，
默认测试一行云端代码、一个权重都不碰。

- **一个模型同时产出稠密与稀疏两路**（BGE-M3），混合检索的两路因此必然同源。
  稠密向量一律归一化，配 Milvus 的 IP 度量等价于余弦相似度。
- **精排吃长文本：适配器自己不截断、不摘要**，切在哪里只由 `max_length` 决定，
  而那已经贴着模型自己的上下文上限。两个库自己给的 `max_length` 默认值都是 512，
  照默认值用就等于把原项目「超长就 LLM 摘要压缩再重试」那条路径搬回来，
  所以上限显式来自配置（`RAGAMER_EMBED_MAX_LENGTH` / `RAGAMER_RERANK_MAX_LENGTH`，默认 8192）。
- **模型只加载一次**：权重在第一次真的要用到时才加载，之后整个进程复用同一个实例。

真实模型（torch + transformers）放在可选的 `models` 组里——核心链路与默认测试都不需要它：

```bash
uv sync --extra models           # 装真实模型
uv run pytest -m integration     # 跑真模型的集成测试（首次会下载几个 G 的权重）
```

## 导入

写入侧两个端点，对应界面上的两处输入：

| 端点 | 收什么 |
| --- | --- |
| `POST /api/kb/{game_id}/import` | 上传若干份资料（multipart） |
| `POST /api/kb/{game_id}/import/urls` | 若干网页地址（JSON：`{"urls": [...], "version": ""}`） |

四个来源先归一为 Markdown（`ragamer.sources`），之后串起补图、切分、打标、向量化与入库
（`ragamer.importing`）。两条端点**在归一化那一步就合流**：此后切分与打标不感知来源。
现在接上了 md／txt 与网页两条，MinerU 在后面一张票里接。

- **一批里某一条失败不牵连其余**：响应的 `results` 逐条给结果，失败的那条带
  `source`（文件那一路是文件名，网址那一路是地址）、`stage`（卡在哪一步）与 `error`。
  整批都失败也是 200，不是 500。
- **重复导入不产生重复切片**：切片主键由导入侧按「游戏 + 文档标题 + 版本 + 切片序号」算出来，
  重导覆盖同一批 id；入库时再按文档整体替换，新切出来的片数变少也不会留下旧的那一截。
- **同一份资料的另一个版本并存**：新版本作为新文档导入，删除只作用于它自己那个版本（ADR-0004）。
- **入库的字段与建表时的显式声明一一对应**，不开动态字段。
- **抓回来的切片带来源地址**（`source_url`）：本地文件是空串。答案是带引用给的，引用的落点就是它。
- 打标用的词表来自知识库元数据（MongoDB 的 `knowledge_bases` 集合，id 即游戏 id）。
  库不存在时 404，不静默按默认词表建内容。

## 网页抓取

`ragamer.crawl` 一个网址进、一份归一化文档出，两条路（`docs/ARCHITECTURE.md` §1.1）：

- **MediaWiki 类站点**走它的开放接口（`api.php`）拿 **wikitext 原文**，由 `ragamer.wikitext`
  转成 Markdown。选原文而不是渲染后的 HTML，是因为下游读的正是 wiki 标记：切分器靠
  `[[内链]]`／`Category:`／`{{模板}}` 认词条页，打标器靠 `[[Category:角色]]` 与 Infobox 字段
  读主体类型。走哪条路由页面自己说了算——认 head 里的 RSD（`rel="EditURI"`）与
  `generator` 声明。
- **普通网页**走正文抽取（trafilatura）：导航、页脚、侧边栏丢掉，正文转成 Markdown。

**合规的三件事落在出网的那一条路上**（`HttpCrawler._get` 查 robots、`_fetch` 限频与标明
身份，而这条路只有一条）：先读该主机的 `robots.txt`、同一主机两次请求之间留足间隔、请求头里
标明身份。**重定向也不例外**——每一跳都重新过一遍 robots 与限频，跟着 `httpx` 一次跳到底会
绕过这一层（`robots.txt` 只写了入口那个地址）。**没有关掉 robots 的开关。**
参数在 `.env.example` 的「网页抓取」一节，失败分三类各有各的异常：站点不可达、页面不存在、
被拒绝访问（robots 挡下的算后者的子类）。

⚠️ **不拦内网地址**：这个端点收的是用户给的网址，多用户部署时它是一个 SSRF 面
（`http://169.254.169.254/…` 这类）。本项目现在是自己给自己导资料的单机工具，
上多用户之前必须补上——拦内网要连 DNS 解析一起做，否则域名解析到内网就绕过去了。

## 目录

```
src/ragamer/          应用代码（config 配置装载、logging 日志、llm 语言模型适配器、sources 归一化、
                      crawl 网页抓取、wikitext 维基语法转 Markdown、chunking 切分器、tagging 打标、
                      importing 导入编排器、api HTTP 端点、container 组合根、__main__ 启动自检）
src/ragamer/stores/   存储适配器：base 协议与共享类型、chunks Milvus、documents Mongo、objects MinIO、memory 内存假件
src/ragamer/vectors/  向量化与精排：base 协议与共享类型、bge 真实模型、fake 确定性假件
tests/                测试：行为测试 + 结构约束 + 集成测试
docs/                 架构文档、ADR、给 agent 的说明
```
