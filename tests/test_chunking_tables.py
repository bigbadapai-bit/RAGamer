"""切分器：表格原子化与 `content_meta`。

T04 钉死了切分器的主干（结构探测、祖先标题路径、降级切分）。这一层是它的第二票：
表格与模板块不能像普通正文那样按语义边界任意切开，**长文本列也不能留在正文里**。
后者不是优化——超长行会在向量化阶段被静默截断，内容永久丢失且不报错；落进
`content_meta` 随结果返回，但不参与向量化，也就没有截断这回事（架构文档 §2.5）。

参数照 T04 的办法收窄：正文上限 100 字，单元格超过 20 字算长文本列，一路数得清——
样例里的长列都写到 20 字以上，免得断言跟着字数的错觉走。
"""

from __future__ import annotations

from ragamer.chunking import ChunkRules, chunk_document

#: 收窄后的切分参数。
RULES = ChunkRules(max_chars=100, min_chars=40, long_cell_chars=20, heading_density=0.02)

#: 一张装得下的短表。
SHORT_TABLE = """\
| 阶段 | 血量 |
| --- | --- |
| 一 | 12000 |
| 二 | 9000 |
"""

#: 一张带超长「说明」列的表：短列该留在正文，长列该整列落进 `content_meta`。
LONG_COLUMN_TABLE = """\
| 阶段 | 说明 |
| --- | --- |
| 一 | 横扫之后有两秒硬直，这段时间贴身输出最划算，注意别追得太远。 |
| 二 | 召唤天兵，先清掉小怪再打本体，被围住就往外撤，别硬吃伤害。 |
"""

#: 一个比正文上限还长的 Infobox——切开就不是 Infobox 了。
INFOBOX = """\
{{Infobox character
| name = 二郎神
| hp = 12000
| 技能 = 三尖两刃刀、法天象地、七十二变
| 掉落 = 根器、玲珑内丹
| 出现 = 第三回 黄风岭，需先打完守关的妖王
| 备注 = 第三阶段会召唤哮天犬，记得留好回复道具
}}
"""


def _types(chunks) -> list[str]:
    return [chunk.chunk_type for chunk in chunks]


# --- 整表成块 ---


def test_表格整体成一片():
    chunks = chunk_document(f"# 二郎神\n\n{SHORT_TABLE}", RULES)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
    assert chunks[0].ancestor_path == "二郎神"
    assert chunks[0].content.splitlines() == [
        "| 阶段 | 血量 |",
        "| --- | --- |",
        "| 一 | 12000 |",
        "| 二 | 9000 |",
    ]


def test_表格切片与正文切片区分开():
    markdown = f"# 二郎神\n\n这是正文。\n\n{SHORT_TABLE}\n\n这也是正文。\n"

    chunks = chunk_document(markdown, RULES)

    assert _types(chunks) == ["text", "table", "text"]
    assert chunks[1].content.startswith("| 阶段 |")
    assert "|" not in chunks[0].content
    assert "|" not in chunks[2].content


def test_表格按所在小节带祖先标题路径():
    markdown = f"# 二郎神\n\n## 数值\n\n### 阶段血量\n\n{SHORT_TABLE}"

    chunks = chunk_document(markdown, RULES)

    assert chunks[0].ancestor_path == "二郎神 › 数值 › 阶段血量"


def test_只有表格的文档也能切出表格切片():
    chunks = chunk_document(SHORT_TABLE, RULES)

    assert _types(chunks) == ["table"]
    assert chunks[0].ancestor_path == ""
    assert chunks[0].content_meta == ""


def test_只有表头没有数据行的表也留一片():
    # MinerU 偶尔吐出空表：表头本身就是内容，丢掉等于把这张表删了
    chunks = chunk_document("# 二郎神\n\n| 阶段 | 血量 |\n| --- | --- |\n", RULES)

    assert _types(chunks) == ["table"]
    assert chunks[0].content == "| 阶段 | 血量 |\n| --- | --- |"


def test_正文切片的_content_meta_是空的():
    chunks = chunk_document("# 二郎神\n\n他是灌江口的那位。\n", RULES)

    assert chunks[0].chunk_type == "text"
    assert chunks[0].content_meta == ""


def test_单元格里转义的竖线不是分隔符():
    markdown = "# 说明\n\n| 名称 | 值 |\n| --- | --- |\n| A\\|B | 1 |\n"

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) == 1
    assert "| A\\|B | 1 |" in chunks[0].content


# --- 按行组切分 ---


def test_超长表格按行组切且每块都带表头():
    rows = "\n".join(f"| 第{index}阶段 | 血量 {index}000 |" for index in range(12))
    markdown = f"# 二郎神\n\n| 阶段 | 血量 |\n| --- | --- |\n{rows}\n"

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) > 1
    assert _types(chunks) == ["table"] * len(chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        lines = chunk.content.splitlines()
        assert lines[:2] == ["| 阶段 | 血量 |", "| --- | --- |"]
        assert len(lines) > 2, "每一组至少带一行数据"
        assert len(chunk.content) <= RULES.max_chars
    # 切开只是换了分组，行一行不少
    joined = "".join(chunk.content for chunk in chunks)
    assert all(f"血量 {index}000" in joined for index in range(12))


def test_行组切分后仍带同一祖先标题路径():
    rows = "\n".join(f"| 第{index}阶段 | 血量 {index}000 |" for index in range(12))
    markdown = f"# 二郎神\n\n## 数值\n\n| 阶段 | 血量 |\n| --- | --- |\n{rows}\n"

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) > 1
    assert {chunk.ancestor_path for chunk in chunks} == {"二郎神 › 数值"}


# --- 长文本列落进 content_meta ---


def test_超长文本列不进正文():
    chunks = chunk_document(f"# 二郎神\n\n{LONG_COLUMN_TABLE}", RULES)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "table"
    assert "横扫之后有两秒硬直" not in chunk.content
    assert "召唤天兵，先清掉小怪" not in chunk.content
    # 短列还留在正文里，切片不是空的
    assert chunk.content.splitlines() == ["| 阶段 |", "| --- |", "| 一 |", "| 二 |"]


def test_超长文本列一字不少地落进_content_meta():
    chunks = chunk_document(f"# 二郎神\n\n{LONG_COLUMN_TABLE}", RULES)

    assert chunks[0].content_meta.splitlines() == [
        "| 阶段 | 说明 |",
        "| --- | --- |",
        "| 一 | 横扫之后有两秒硬直，这段时间贴身输出最划算，注意别追得太远。 |",
        "| 二 | 召唤天兵，先清掉小怪再打本体，被围住就往外撤，别硬吃伤害。 |",
    ]


def test_长列跟着它所在的行组分片():
    rows = [
        f"| 第{index}阶段 | {index} | 这一阶段的说明写得足够长，长到该降级进 content_meta。 |"
        for index in range(12)
    ]
    markdown = "# 二郎神\n\n| 阶段 | 血量 | 说明 |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) > 1
    assert all("这一阶段的说明写得足够长" not in chunk.content for chunk in chunks)
    for chunk in chunks:
        # 正文与 meta 的数据行一一对应：meta 跟着它那一片走，不整表重复一遍
        assert len(chunk.content.splitlines()) == len(chunk.content_meta.splitlines())
    assert "".join(chunk.content_meta for chunk in chunks).count("这一阶段的说明写得足够长") == 12


def test_整张表只有长列时第一列留在正文里做锚点():
    markdown = (
        "# 说明\n"
        "\n"
        "| 这一列的标题本身就写得很长，长到超过长文本列的门槛 | "
        "另一列的标题也一样长，也同样超过长文本列的门槛 |\n"
        "| --- | --- |\n"
        "| 这一格的内容也长，也同样超过长文本列的门槛 | "
        "另一格的内容也一样长，也超过长文本列的门槛 |\n"
    )

    chunks = chunk_document(markdown, RULES)

    chunk = chunks[0]
    # 正文不能是空的，第一列留着；剩下那列整列降级
    assert chunk.content.splitlines() == [
        "| 这一列的标题本身就写得很长，长到超过长文本列的门槛 |",
        "| --- |",
        "| 这一格的内容也长，也同样超过长文本列的门槛 |",
    ]
    assert "另一格的内容也一样长，也超过长文本列的门槛" not in chunk.content
    assert "另一格的内容也一样长，也超过长文本列的门槛" in chunk.content_meta


# --- 模板块 ---


def test_模板块整体保留不被切开():
    markdown = f"# 二郎神\n\n{INFOBOX}\n\n他是灌江口的那位。\n"

    chunks = chunk_document(markdown, RULES)
    infobox = [chunk for chunk in chunks if "Infobox" in chunk.content]

    assert len(infobox) == 1
    assert infobox[0].content == INFOBOX.strip()
    assert infobox[0].ancestor_path == "二郎神"
    # 模板块多是 Infobox 这类键值结构，与表格同归结构化块
    assert infobox[0].chunk_type == "table"


def test_模板块与正文各自成片():
    markdown = f"# 二郎神\n\n{INFOBOX}\n\n他是灌江口的那位。\n"

    chunks = chunk_document(markdown, RULES)

    assert _types(chunks) == ["table", "text"]
    assert chunks[1].content == "他是灌江口的那位。"


def test_行首之外的模板符号不当模板块():
    markdown = "# 说明\n\n这一行里有 {{模板}} 但不在行首。\n"

    chunks = chunk_document(markdown, RULES)

    assert _types(chunks) == ["text"]
    assert "{{模板}}" in chunks[0].content


def test_没配平的模板符号不当模板块():
    # 孤零零一个 `{{` 若当模板，会把后面整篇正文吞进一个原子块里
    markdown = "# 说明\n\n正文里夹着一个 {{ 未配平的模板开头。\n\n后面还有一大段正文。\n"

    chunks = chunk_document(markdown, RULES)

    assert "{{ 未配平的模板开头。" in chunks[0].content
    assert _types(chunks) == ["text"] * len(chunks)


def test_嵌套的模板整块保留():
    markdown = (
        "# 装备\n"
        "\n"
        "{{Infobox item\n"
        "| name = 根器\n"
        "| 效果 = {{伤害加成|30%}}\n"
        "| 来源 = 二郎神掉落\n"
        "}}\n"
    )

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) == 1
    assert chunks[0].content == (
        "{{Infobox item\n| name = 根器\n| 效果 = {{伤害加成|30%}}\n| 来源 = 二郎神掉落\n}}"
    )
