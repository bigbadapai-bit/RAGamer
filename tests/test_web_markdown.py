"""答案正文的排版：认一个很小的 Markdown 子集，渲染成 HTML。

这一层是**安全边界**，所以断言分两类：排版认不认得出那几种记号，以及认不出的东西
（输入里的标签、没实现的语法）会不会原样漏出去。模型输出按不可信对待——网络检索的
内容会进提示词，能诱导它吐 HTML。
"""

from __future__ import annotations

from ragamer.web.markdown import to_html

# --- 认得的记号 ---


def test_粗体():
    assert to_html("**自带被动效果：**见貌辨色") == "<p><strong>自带被动效果：</strong>见貌辨色</p>"


def test_记号后面没空格的不是列表项():
    """`**粗体**` 开头的行与 `* 列表项` 只差一个空格，认错了整段就散了。"""
    assert to_html("**甲**\n**乙**") == "<p><strong>甲</strong><br><strong>乙</strong></p>"


def test_空行分段():
    assert to_html("第一段。\n\n第二段。") == "<p>第一段。</p><p>第二段。</p>"


def test_单个换行按换行渲染():
    """不按 Markdown 的软换行合成空格：中文里合成空格会凭空多出一个空格。"""
    assert to_html("第一行\n第二行") == "<p>第一行<br>第二行</p>"


def test_无序列表():
    assert to_html("- 甲\n- 乙") == "<ul><li>甲</li><li>乙</li></ul>"


def test_有序列表():
    assert to_html("1. 甲\n2. 乙") == "<ol><li>甲</li><li>乙</li></ol>"


def test_子列表落在父项里面():
    """答案里「共 3 个」底下挂三条是常事，摊平成一层就看不出谁属于谁了。"""
    assert (
        to_html("- 根器天赋共3个：\n  - 眼乖手疾\n  - 慧眼圆睁\n- 其他")
        == "<ul><li>根器天赋共3个：<ul><li>眼乖手疾</li><li>慧眼圆睁</li></ul></li>"
        "<li>其他</li></ul>"
    )


def test_子列表之后回到上一层():
    """收子列表时要把它外面那条父项也收掉，否则后面同级的项会落进父项里。"""
    assert (
        to_html("- 甲\n  - 甲一\n- 乙\n- 丙")
        == "<ul><li>甲<ul><li>甲一</li></ul></li><li>乙</li><li>丙</li></ul>"
    )


def test_列表项里的粗体照样认():
    assert (
        to_html("- **眼乖手疾**：冷却时间-12.5秒")
        == "<ul><li><strong>眼乖手疾</strong>：冷却时间-12.5秒</li></ul>"
    )


def test_列表项自己换行时接着往下写():
    assert to_html("- 甲\n  接着写") == "<ul><li>甲<br>接着写</li></ul>"


def test_标题():
    """`#` 当三级标题：页面上面还有页面标题与「回答」那一行，直接当 `h1` 太重。"""
    assert to_html("# 打法") == "<h3>打法</h3>"
    assert to_html("### 第五阶段") == "<h5>第五阶段</h5>"


def test_标题之后正文另起一段():
    assert to_html("## 打法\n\n先定身。") == "<h4>打法</h4><p>先定身。</p>"


def test_空正文什么都不产出():
    assert to_html("") == ""
    assert to_html("\n\n") == ""


# --- 认不出的原样留着 ---


def test_没实现的记号原样留着():
    """表格、链接、行内代码、围栏：留着至少看得出它想写什么，认一半反而更乱。"""
    assert to_html("| 名称 | 说明 |") == "<p>| 名称 | 说明 |</p>"
    assert to_html("见 [官方说明](https://example.com/x)") == (
        "<p>见 [官方说明](https://example.com/x)</p>"
    )
    assert to_html("```\n代码\n```") == "<p>```<br>代码<br>```</p>"


# --- 安全边界 ---


def test_输入里的标签被转义():
    """先整段转义再放行白名单：出来能当标签用的只有渲染器自己插的那几种。"""
    assert to_html('<img src=x onerror="alert(1)">') == (
        "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt;</p>"
    )
    assert to_html("<script>alert(1)</script>") == "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>"


def test_列表项里的标签也被转义():
    assert to_html("- <b>甲</b>") == "<ul><li>&lt;b&gt;甲&lt;/b&gt;</li></ul>"


def test_与号与引号被转义():
    """`&` 不转义的话 `&lt;` 这种输入能拼出标签来。"""
    assert to_html("甲 & 乙") == "<p>甲 &amp; 乙</p>"


# --- 真实形状 ---


def test_一条真实答案的形状():
    """照实测拿到的那条答案缩一份：伪标题 + 列表 + 子列表 + 有序项 + 粗体。"""
    answer = (
        "「眼看喜」是六根之一。\n"
        "\n"
        "**自带被动效果：**\n"
        "见貌辨色，恃才生恶。\n"
        "\n"
        "**根器天赋（3个）：**\n"
        "- 见机强攻——资料中没有写明具体效果。\n"
        "- 眼乖手疾——冷却时间 -12.5 秒。\n"
        "  1. 只在装备该根器时生效。\n"
        "  2. 与天赋重修无关。\n"
        "- 慧眼圆睁——暴击伤害 +15%。\n"
    )

    assert to_html(answer) == (
        "<p>「眼看喜」是六根之一。</p>"
        "<p><strong>自带被动效果：</strong><br>见貌辨色，恃才生恶。</p>"
        "<p><strong>根器天赋（3个）：</strong></p>"
        "<ul>"
        "<li>见机强攻——资料中没有写明具体效果。</li>"
        "<li>眼乖手疾——冷却时间 -12.5 秒。"
        "<ol><li>只在装备该根器时生效。</li><li>与天赋重修无关。</li></ol>"
        "</li>"
        "<li>慧眼圆睁——暴击伤害 +15%。</li>"
        "</ul>"
    )
