"""知识库元数据：一个游戏知识库在库里长什么样，以及怎么把它读回来。

元数据存 MongoDB 的 `knowledge_bases` 集合，**文档 id 就是游戏 id**——它同时是 Milvus 的
collection 名（ADR-0002），所以这里不做另一套 id 规则，读到的 id 直接能用。

读取侧有两处要用它，用法不同但读的是同一份文档：

- 导入端点要**打标词表**（`ragamer.api`）：`subject_types` 与 `term_mapping` 在里面。
- 澄清反问要**可选的游戏**（`ragamer.clarifying`）：候选得是库里真有的游戏，
  而「库里有哪些游戏」就是这份集合里的 id。显示名一并读出来，按钮上摆的是它。

**配置读不出来与库不存在是两件事**，但都要能被调用方分开处理，所以这里不抛异常：
词表读不了记进 `problem`（那个库还在，只是配错了），库不存在返回 `None`
（游戏选错了）。各自该报 404 还是 422，由 HTTP 那层决定。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ragamer.logging import get_logger
from ragamer.stores.base import UNVERSIONED, DocStore

logger = get_logger(__name__)

#: 知识库元数据所在的集合，文档 id 就是游戏 id。
KB_COLLECTION = "knowledge_bases"


@dataclass(frozen=True)
class KnowledgeBase:
    """一个游戏知识库对外的样子。

    `name` 是界面上认得出的名字，退回游戏 id——建库时没填名字是常事，
    界面上显示 `black_myth` 总好过显示一片空白。

    `version` 是**现行版本**：问题里没点名版本时按它检索，检索与聚合父块都从这一处取，
    不各自维护一份（`ragamer.query.version_filter`）。空串表示这个库没标版本。
    """

    game_id: str
    name: str
    version: str = UNVERSIONED


def knowledge_bases(docs: DocStore) -> tuple[KnowledgeBase, ...]:
    """全部知识库，按游戏 id 字典序（即 `DocStore.list_ids` 的次序）。"""
    bases: list[KnowledgeBase] = []
    for game_id in docs.list_ids(KB_COLLECTION):
        payload = docs.get(KB_COLLECTION, game_id)
        if payload is None:  # 列出来之后被删了
            continue
        bases.append(_from_payload(game_id, payload))
    return tuple(bases)


def knowledge_base(docs: DocStore, game_id: str) -> KnowledgeBase | None:
    """读一个知识库；没有这个库返回 `None`，不抛异常（见模块说明）。"""
    payload = docs.get(KB_COLLECTION, game_id)
    return None if payload is None else _from_payload(game_id, payload)


def _from_payload(game_id: str, payload: Mapping[str, Any]) -> KnowledgeBase:
    """库里那份文档 → 这个对象。缺的键走默认值，不因为一个键没写就当整份配置坏了。"""
    return KnowledgeBase(
        game_id=game_id,
        name=str(payload.get("name") or game_id),
        version=str(payload.get("version") or UNVERSIONED),
    )
