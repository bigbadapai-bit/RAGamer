# RAGamer

面向游戏攻略与资料的中文 RAG 助手。用户按游戏建库、用口语提问，系统先确认问的是哪款游戏、哪个版本，再检索并生成带引用的回答。

设计取舍见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，领域术语以 [`CONTEXT.md`](CONTEXT.md) 为准，不可逆决策见 [`docs/adr/`](docs/adr/)。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)（Python 3.12 由 uv 自己装，不必预装）。

```bash
uv sync                  # 建虚拟环境、装依赖
cp .env.example .env     # 然后按注释把值填成自己的
uv run ragamer           # 启动自检：配置合格、Milvus/Mongo/MinIO 都连得上才通过
uv run ragamer-web       # 起界面：浏览器打开 http://127.0.0.1:8000
```

退出码 `0` 通过、`2` 配置有问题、`3` 存储连不上——远端不可达时在
`RAGAMER_STORE_TIMEOUT_SECONDS` 秒内失败，错误信息里点名是哪个服务、哪个地址。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `uv run ragamer` | 启动自检：配置合格、Milvus/Mongo/MinIO 都连得上才通过 |
| `uv run ragamer-web` | 起界面与后端（先跑一遍同样的自检，通了才起；默认 `127.0.0.1:8000`） |
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

写入侧的入口是 `POST /api/kb/{game_id}/import`——上传若干份资料、每个文件独立处理。
四个来源先归一为 Markdown（`ragamer.sources`），之后串起补图、切分、打标、向量化与入库
（`ragamer.importing`）。现在只接上了 md／txt 一条来源，MinerU 与网页爬虫在后面两张票里接。

- **一批里某个文件失败不牵连其余**：响应的 `results` 逐文件给结果，失败的那个带
  `filename`、`stage`（卡在哪一步）与 `error`。整批都失败也是 200，不是 500。
- **重复导入不产生重复切片**：切片主键由导入侧按「游戏 + 文档标题 + 版本 + 切片序号」算出来，
  重导覆盖同一批 id；入库时再按文档整体替换，新切出来的片数变少也不会留下旧的那一截。
- **同一份资料的另一个版本并存**：新版本作为新文档导入，删除只作用于它自己那个版本（ADR-0004）。
- **入库的字段与建表时的显式声明一一对应**，不开动态字段。
- 打标用的词表来自知识库元数据（MongoDB 的 `knowledge_bases` 集合，id 即游戏 id）。
  库不存在时 404，不静默按默认词表建内容。

## 界面

**FastAPI + Jinja2 + htmx，零构建步骤**——没有 npm、没有打包产物（[ADR-0005](docs/adr/0005-htmx-frontend.md)）。
页面在 `ragamer.web`，模板跟着包走；`ragamer.app` 把 JSON 端点与页面装成同一个应用。

**导航在左侧栏，主区拿整幅宽度**——切分预览那张表要八列并排，导入结果也要同时摆开文件名、
文档、结果与切片数，横着挤不下。外壳是 `templates/base.html` 一份，各页只填 `content`。

| 路径 | 页面 |
| --- | --- |
| `/kb` | 知识库管理：建库（游戏 id + 显示名 + 勾选启用的主体类型）、列出已有的库 |
| `/import` | 导入：选库、传 md／txt、标注版本，逐文件的导入结果就地列出来 |
| `/kb/{game_id}/preview` | **切分预览（只读）**：正文、祖先标题路径、主体类型、内容性质、来源文档 |
| `/chat`、`/eval` | 占位，后面几张票接上 |

- **禁用 JavaScript 也能用**。每条写入路径都是普通的 HTML 表单；htmx 在时把结果那一块换掉，
  不在时浏览器自己提交、回整页——服务端按 `HX-Request` 决定回整页还是回片段，两条路走的是
  同一段处理逻辑。
- **切分预览页没有任何编辑入口**，它把库里存下来的东西读回来渲染。切片排成表——正文、祖先
  标题路径、主体类型、内容性质、切片类型各占一列，逐行对得齐才看得出切分质量。取切片走的是
  检索那条路（`fetch_document`），所以看到的就是检索时会看见的：该版本的内容与**未标注版本**
  的内容一并列出（[ADR-0004](docs/adr/0004-versioned-content-coexists.md)），后者在版本那一列
  挂个徽章说明来历。
- 页面上的 Tailwind 与 htmx 走 CDN。代价是首次加载要连得上外网；连不上时页面照常能用，
  只是没有样式、也不会局部刷新。

## 目录

```
src/ragamer/          应用代码（config 配置装载、logging 日志、llm 语言模型适配器、sources 归一化、
                      chunking 切分器、tagging 打标、importing 导入编排器、knowledge 知识库元数据、
                      api JSON 端点、web 页面、app 应用装配与起服务、container 组合根、
                      __main__ 启动自检）
src/ragamer/stores/   存储适配器：base 协议与共享类型、chunks Milvus、documents Mongo、objects MinIO、memory 内存假件
src/ragamer/vectors/  向量化与精排：base 协议与共享类型、bge 真实模型、fake 确定性假件
src/ragamer/web/      页面与模板（templates/ 跟着包走，装成 wheel 也在）
tests/                测试：行为测试 + 结构约束 + 集成测试
docs/                 架构文档、ADR、给 agent 的说明
```
