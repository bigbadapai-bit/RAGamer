"""切分器：结构探测、祖先标题路径与降级切分。

这是规格 #1 定下的「缝一」：纯函数、零 I/O、零假件，测试直接调它。
切分参数收窄到能一眼数清的程度，断言才写得实。

表格原子化与打标不在这里——它们各有各的票，测试也各有各的文件。
"""

from __future__ import annotations

from ragamer.chunking import ChunkRules, chunk_document, probe_structure

#: 收窄后的切分参数：100 / 40 字，标题密度门槛照默认。
RULES = ChunkRules(max_chars=100, min_chars=40, heading_density=0.02)

#: 一份有三级标题的词条页。
WIKI_ARTICLE = """\
# 二郎神

二郎神是隐藏 BOSS，需要三阶段打完。

## 打法

### 第一阶段

先定身，再贴身输出。

### 第二阶段

横扫之后有硬直。
"""


def _paths(chunks) -> list[str]:
    return [chunk.ancestor_path for chunk in chunks]


def _contents(chunks) -> list[str]:
    return [chunk.content for chunk in chunks]


def _plain_lines(count: int) -> str:
    """一段没有任何标题的正文，一行一句。"""
    return "\n".join(f"第 {index} 句：这一段只是正文，没有标题。" for index in range(count))


def _sizes_are_sane(chunks, rules: ChunkRules = RULES) -> bool:
    """每一片正文都在下限与上限之间。

    下限是硬的；上限是软的——装不下时宁可让某一片略微超长，也不留下读不成句的
    碎片（见 `chunking._merge`），所以断言放宽到 `max_chars + min_chars`。

    表格切片不在此列：它自带表头，只剩一行也是读得懂的（见 `test_chunking_tables.py`）。
    """
    return all(
        rules.min_chars <= len(chunk.content) <= rules.max_chars + rules.min_chars
        for chunk in chunks
        if chunk.chunk_type == "text"
    )


# --- 祖先标题路径 ---


def test_每个切片带完整祖先标题路径():
    chunks = chunk_document(WIKI_ARTICLE, RULES)

    assert _paths(chunks) == [
        "二郎神",
        "二郎神 › 打法 › 第一阶段",
        "二郎神 › 打法 › 第二阶段",
    ]


def test_标题层级跳跃时不补出中间层级():
    markdown = "# 二郎神\n\n## 打法\n\n##### 冷知识\n\n掉落有概率给根器。\n"

    assert _paths(chunk_document(markdown, RULES)) == ["二郎神 › 打法 › 冷知识"]


def test_同级标题之间不累积路径():
    markdown = "# 二郎神\n\n## 背景\n\n他是灌江口的那位。\n\n## 掉落\n\n掉根器。\n"

    chunks = chunk_document(markdown, RULES)

    assert _paths(chunks) == ["二郎神 › 背景", "二郎神 › 掉落"]


def test_标题之前的内容没有祖先标题路径():
    markdown = "这份资料抄自维基，通篇没有小标题。\n\n# 二郎神\n\n正文。\n"

    chunks = chunk_document(markdown, RULES)

    assert _paths(chunks)[0] == ""
    assert chunks[0].content == "这份资料抄自维基，通篇没有小标题。"


def test_只有标题没有正文的小节不产出切片():
    markdown = "# 二郎神\n\n## 打法\n\n先躲横扫。\n"

    assert _contents(chunk_document(markdown, RULES)) == ["先躲横扫。"]


# --- 代码块状态机 ---


def test_代码块里的井号不被当作标题():
    markdown = (
        "# 配置说明\n"
        "\n"
        "下面这段要原样保留。\n"
        "\n"
        "```bash\n"
        "# 这是注释，不是标题\n"
        "echo hello\n"
        "```\n"
        "\n"
        "改完重启即可。\n"
    )

    chunks = chunk_document(markdown, RULES)

    assert _paths(chunks) == ["配置说明"]
    assert "# 这是注释，不是标题" in chunks[0].content
    assert "echo hello" in chunks[0].content


def test_波浪线围栏同样翻转():
    markdown = "# 说明\n\n~~~python\n# 还是注释\n~~~\n\n正文。\n"

    assert _paths(chunk_document(markdown, RULES)) == ["说明"]


def test_没有闭合的围栏之后的内容也不被当作标题():
    markdown = "# 说明\n\n下面这段没写结尾的围栏。\n\n```python\nprint(1)\n\n# 这段还是代码\n"

    chunks = chunk_document(markdown, RULES)

    assert _paths(chunks) == ["说明"]
    assert "# 这段还是代码" in chunks[0].content


def test_井号后面没有空格的不是标题():
    markdown = "# 说明\n\n#话题标签 写在正文里。\n"

    assert _paths(chunk_document(markdown, RULES)) == ["说明"]


# --- HTML 折叠块 ---


def test_切片正文里不残留折叠块标记():
    markdown = (
        "# 二郎神\n"
        "\n"
        "![](images/1.jpg)\n"
        "<details>\n"
        "<summary>text_image</summary>\n"
        "图内写着：血量 12000。\n"
        "</details>\n"
    )

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) == 1
    for marker in ("<details>", "</details>", "<summary>", "</summary>"):
        assert marker not in chunks[0].content
    assert "图内写着：血量 12000。" in chunks[0].content
    assert "![](images/1.jpg)" in chunks[0].content


def test_代码块里的折叠块标记原样保留():
    markdown = "# 说明\n\n```html\n<details>\n<summary>示例</summary>\n</details>\n```\n"

    assert "<details>" in _contents(chunk_document(markdown, RULES))[0]


# --- 顺序号 ---


def test_切片顺序号从_0_起连续():
    chunks = chunk_document(WIKI_ARTICLE, RULES)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_切片顺序能还原文档顺序():
    paragraphs = "\n\n".join(f"第{name}段：只是正文。{name * 30}" for name in "一二三四")
    chunks = chunk_document(paragraphs, RULES)

    texts = "".join(_contents(chunks))
    first, second, third, fourth = (texts.index(f"第{name}段") for name in "一二三四")

    assert first < second < third < fourth


# --- 结构探测 ---


def test_标题稀疏的扁平文档退化为递归切分():
    markdown = f"# 资料\n\n{_plain_lines(60)}\n"
    probe = probe_structure(markdown, RULES)

    assert probe.structured is False
    assert probe.density < RULES.heading_density

    chunks = chunk_document(markdown, RULES)

    assert _paths(chunks) == [""] * len(chunks)
    assert _sizes_are_sane(chunks)


def test_标题稀疏但带表格块的文档仍按结构切():
    markdown = f"# 二郎神\n\n{_plain_lines(60)}\n\n| 阶段 | 血量 |\n| --- | --- |\n| 一 | 12000 |\n"
    probe = probe_structure(markdown, RULES)

    assert probe.has_table is True
    assert probe.structured is True

    chunks = chunk_document(markdown, RULES)

    assert set(_paths(chunks)) == {"二郎神"}
    assert _sizes_are_sane(chunks)


def test_标题稀疏但带_MediaWiki_特征的文档仍按结构切():
    markdown = f"# 二郎神\n\n{_plain_lines(60)}\n\n[[根器]] 是本地叫法。\n"
    probe = probe_structure(markdown, RULES)

    assert probe.mediawiki is True
    assert probe.structured is True


def test_探测结果带上判定依据():
    probe = probe_structure(WIKI_ARTICLE, RULES)

    assert probe.heading_count == 4
    assert probe.body_line_count == 3
    assert probe.density == 4 / 3


def test_正文全是代码块的文档仍按标题切():
    # 围栏里的行也算正文行，否则这份文档的密度恒为 0，会被误判成扁平——
    # 标题丢了路径，`#` 还漏回正文里
    markdown = "# 配置\n\n```sh\nuv sync\nuv run ragamer\n```\n"

    assert probe_structure(markdown, RULES).structured is True

    chunks = chunk_document(markdown, RULES)

    assert _paths(chunks) == ["配置"]
    assert "# 配置" not in chunks[0].content


def test_只有标题的文档不落进扁平切分():
    assert probe_structure("# 一\n\n## 二\n", RULES).structured is True


def test_空文档探不出结构():
    assert probe_structure("", RULES).structured is False


# --- 两种切法并存与降级 ---


def test_有标题的部分按标题切没标题的部分按语义切():
    preamble = "\n\n".join(
        f"引子第 {index} 段：这份资料抄自别处，通篇没有小标题，"
        "只能按句子与段落的边界切开，切出来仍是一段完整的话。"
        for index in range(4)
    )
    markdown = f"{preamble}\n\n# 二郎神\n\n## 打法\n\n先躲横扫，再贴身输出。\n"

    chunks = chunk_document(markdown, RULES)
    paths = _paths(chunks)

    assert paths[:4] == [""] * 4
    assert paths[4] == "二郎神 › 打法"
    assert all("引子" in content for content in _contents(chunks)[:4])


def test_超长小节被切开后每片仍带同一祖先标题路径():
    body = "\n\n".join(
        f"第 {index} 段：这一段讲的是第二阶段的完整打法，包括走位与输出节奏。" for index in range(6)
    )
    markdown = f"# 二郎神\n\n## 打法\n\n### 第二阶段\n\n{body}\n"

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) > 1
    assert _paths(chunks) == ["二郎神 › 打法 › 第二阶段"] * len(chunks)
    assert _sizes_are_sane(chunks)


def test_超长句子按句号边界切开():
    markdown = "# 打法\n\n" + "打完这一下要立刻走位。 " * 20

    chunks = chunk_document(markdown, RULES)

    assert len(chunks) > 1
    assert _sizes_are_sane(chunks)
    # 句号是切点，不是被吃掉的分隔符
    assert all(chunk.content.endswith("。") for chunk in chunks)


def test_没有任何边界的超长文本也能被切开():
    markdown = "# 打法\n\n" + "字" * 300

    chunks = chunk_document(markdown, RULES)

    assert _sizes_are_sane(chunks)
    assert len("".join(_contents(chunks))) == 300


def test_过短的碎片并进相邻切片():
    # 中间那段短得单独成片没有意义，应当并进挨着的那一片
    markdown = "# 打法\n\n" + "第一段。" * 26 + "\n\n" + "短。" + "\n\n" + "第二段。" * 26

    chunks = chunk_document(markdown, RULES)

    assert _sizes_are_sane(chunks)
    assert any("短。" in chunk.content for chunk in chunks)
    assert not any(chunk.content == "短。" for chunk in chunks)


def test_整段切完剩下的尾巴并回前一片():
    # 13 个等长段落：贪心切到末尾会剩一个段落孤零零成片，得与前一片重新分一次
    body = "\n\n".join(f"第 {index} 段：一句完整的话。" for index in range(13))
    markdown = f"# 打法\n\n{body}\n"

    chunks = chunk_document(markdown, RULES)

    assert _sizes_are_sane(chunks)
    # 重新分一次只是挪了边界，正文一字不少（片首尾的空白不计）
    assert "".join(_contents(chunks)).replace("\n", "") == body.replace("\n", "")


# --- 边界 ---


def test_空文档切不出切片():
    assert chunk_document("", RULES) == []


def test_纯空白文档切不出切片():
    assert chunk_document("   \n\n\t\n  \n", RULES) == []
