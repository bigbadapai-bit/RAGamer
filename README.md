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

## 目录

```
src/ragamer/          应用代码（config 配置装载、logging 日志、llm 语言模型适配器、chunking 切分器、
                      tagging 打标、container 组合根、__main__ 启动自检）
src/ragamer/stores/   存储适配器：base 协议与共享类型、chunks Milvus、documents Mongo、objects MinIO、memory 内存假件
src/ragamer/vectors/  向量化与精排：base 协议与共享类型、bge 真实模型、fake 确定性假件
tests/                测试：行为测试 + 结构约束 + 集成测试
docs/                 架构文档、ADR、给 agent 的说明
```
