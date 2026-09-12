# RAGamer

面向游戏攻略与资料的中文 RAG 助手。用户按游戏建库、用口语提问，系统先确认问的是哪款游戏、哪个版本，再检索并生成带引用的回答。

设计取舍见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，领域术语以 [`CONTEXT.md`](CONTEXT.md) 为准，不可逆决策见 [`docs/adr/`](docs/adr/)。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)（Python 3.12 由 uv 自己装，不必预装）。

```bash
uv sync                  # 建虚拟环境、装依赖
cp .env.example .env     # 然后按注释把值填成自己的
uv run ragamer           # 配置自检：通过则打印生效的配置摘要
```

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

## 目录

```
src/ragamer/    应用代码（config 配置装载、logging 日志、chunking 切分器、__main__ 启动自检）
tests/          测试：行为测试 + 结构约束
docs/           架构文档、ADR、给 agent 的说明
```
