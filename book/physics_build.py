import argparse
import importlib.util
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
CONTENT_SPEC = importlib.util.spec_from_file_location("rendertorch_physics_content", ROOT / "physics_content.py")
CONTENT = importlib.util.module_from_spec(CONTENT_SPEC)
sys.modules[CONTENT_SPEC.name] = CONTENT
CONTENT_SPEC.loader.exec_module(CONTENT)

LAYOUT_SPEC = importlib.util.spec_from_file_location("rendertorch_physics_layout", ROOT / "layout.py")
LAYOUT = importlib.util.module_from_spec(LAYOUT_SPEC)
sys.modules[LAYOUT_SPEC.name] = LAYOUT
LAYOUT_SPEC.loader.exec_module(LAYOUT)
write = LAYOUT.write
escape = LAYOUT.escape
header = LAYOUT.header
footer = LAYOUT.footer
document = LAYOUT.document

UNITS = (
    "单元一：运动会怎样发生",
    "单元二：物体为什么保持形状",
    "单元三：让求解器尊重材料与时间",
    "单元四：从点到刚体和软体",
)
REFERENCE_GROUPS = ("基础力学", "约束与碰撞", "PBD/XPBD", "工程参考")
REFERENCE_FIELDS = ("group", "title", "authors", "url", "license", "usage")


def copy_assets(output_dir: Path) -> None:
    source = ROOT / "assets"
    # Chapter pages keep the contracted ../assets paths inside physics/chapters/.
    for target in (output_dir / "assets", output_dir / "physics" / "assets"):
        if source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)


def navigation(active_slug: str = "", prefix: str = "") -> str:
    home_current = ' aria-current="page"' if active_slug == "index" else ""
    items = [
        f'<li><a href="{prefix}index.html"{home_current}>'
        '<span class="number">P</span><span>物理首页</span></a></li>'
    ]
    for row in CONTENT.chapters():
        current = ' aria-current="page"' if active_slug == row.slug else ""
        items.append(
            f'<li><a href="{prefix}chapters/{row.slug}.html"{current}>'
            f'<span class="number">{row.number:02d}</span><span>{escape(row.title)}</span></a></li>'
        )
    current = ' aria-current="page"' if active_slug == "references" else ""
    items.append(
        f'<li><a href="{prefix}references.html"{current}>'
        '<span class="number">R</span><span>物理参考资料</span></a></li>'
    )
    return '<ol class="toc">' + "".join(items) + "</ol>"


def sidebar(active_slug: str = "", prefix: str = "") -> str:
    return (
        '<aside class="sidebar" id="sidebar" data-sidebar data-open="false">'
        '<p class="sidebar-title">物理子书目录</p>'
        f'{navigation(active_slug, prefix)}'
        '<p class="sidebar-note">先理解状态与时间，再讨论约束、碰撞和材料。</p>'
        '</aside>'
    )


def render_demo(demo) -> str:
    return (
        f'<div class="demo-mount physics-demo" id="{escape(demo.id)}" data-demo="{escape(demo.id)}">'
        f'<h3>{escape(demo.title)}</h3><p>{escape(demo.description)}</p></div>'
    )


def render_objectives(row, number: str) -> str:
    objectives = "".join(f"<li>{escape(item)}</li>" for item in row.objectives)
    prerequisites = "、".join(escape(item) for item in row.prerequisites)
    return (
        '<section class="section chapter-section" id="objectives">'
        f'<div class="section-label">{escape(number)} · 学习目标</div><div class="prose">'
        '<h2>本章结束后你能够</h2>'
        f'<ol>{objectives}</ol><p><strong>先验知识：</strong>{prerequisites}。</p></div></section>'
    )


def render_sources(row, number: str) -> str:
    if row.sources:
        links = "".join(
            f'<a class="source-link" href="{escape(source.url)}"><span>{escape(source.path)}</span>'
            f'<small>{escape(source.status)}</small>↗</a>'
            for source in row.sources
        )
        content = f'<h2>阅读当前项目的真实实现</h2><div class="source-grid">{links}</div>'
    else:
        content = (
            '<h2>概念内容与真实实现</h2><p>本章以浏览器概念实验为主；'
            'RenderTorch 当前实现的对应文件会在确实存在时列出。</p>'
        )
    return (
        '<section class="section chapter-section" id="sources">'
        f'<div class="section-label">{escape(number)} · 源码映射</div>'
        f'<div class="prose">{content}</div></section>'
    )


def page_navigation(row) -> str:
    rows = CONTENT.chapters()
    previous = next((item for item in rows if item.number < row.number), None)
    following = next((item for item in rows if item.number > row.number), None)
    left = (
        f'<a href="{previous.slug}.html"><small>← 上一章</small>{escape(previous.title)}</a>'
        if previous else '<span></span>'
    )
    right = (
        f'<a href="{following.slug}.html"><small>下一章 →</small>{escape(following.title)}</a>'
        if following else '<span></span>'
    )
    return f'<nav class="page-nav" aria-label="章节导航">{left}{right}</nav>'


def render_chapter(row) -> str:
    CONTENT.validate(row)
    demos_by_section = {}
    for demo in row.demos:
        demos_by_section.setdefault(demo.section_id, []).append(demo)
    rendered = []
    nav_items = []

    def append_section(section_id: str, label: str, render) -> None:
        number = len(rendered) + 1
        rendered.append(render(f"{row.number}.{number}"))
        nav_items.append(f'<a href="#{escape(section_id)}">{row.number}.{number} {escape(label)}</a>')

    append_section("objectives", "学习目标", lambda number: render_objectives(row, number))
    for section in row.sections:
        def render_section(number: str, section=section) -> str:
            demos = "".join(render_demo(demo) for demo in demos_by_section.get(section.id, ()))
            return (
                f'<section class="section chapter-section" id="{escape(section.id)}">'
                f'<div class="section-label">{escape(number)} · {escape(section.label)}</div>'
                f'<div class="prose"><h2>{escape(section.title)}</h2>{section.html}</div>{demos}</section>'
            )
        append_section(section.id, section.label, render_section)
    append_section("sources", "源码映射", lambda number: render_sources(row, number))
    nav = '<nav class="chapter-section-nav" aria-label="本章导航">' + "".join(nav_items) + "</nav>"
    body = (
        '<article class="content"><header class="chapter-header">'
        f'<p class="eyebrow">Physics Chapter {row.number:02d} / 10 · {escape(row.unit)}</p>'
        f'<h1>{escape(row.title)}</h1><p class="lede">{escape(row.summary)}</p>'
        '<div class="meta-row"><span class="pill">教材式推导</span>'
        '<span class="pill">可操作实验</span><span class="pill">真实源码分层</span></div></header>'
        f'{nav}{"".join(rendered)}{page_navigation(row)}</article>'
    )
    scripts = ("../assets/physics-core.js", "../assets/physics.js") + tuple(
        f"../assets/{demo.script}" for demo in row.demos
    )
    return document(
        row.title,
        body,
        scripts,
        prefix="../",
        sidebar_html=sidebar(row.slug, "../"),
        header_html=header("RenderTorch · 物理模拟教材", "../", meta="PARTICLES · CONSTRAINTS · DYNAMICS", has_sidebar=True),
        footer_html=footer("RenderTorch · 从像素到物理的中文交互教材"),
    )


def render_index(rows=None) -> str:
    rows = tuple(CONTENT.chapters() if rows is None else rows)
    units = []
    for unit in UNITS:
        cards = "".join(
            f'<a class="chapter-card" href="chapters/{row.slug}.html">'
            f'<span class="chapter-number">{row.number:02d}</span>'
            f'<span><strong>{escape(row.title)}</strong><small>{escape(row.summary)}</small></span></a>'
            for row in rows if row.unit == unit
        )
        if not cards:
            cards = '<p class="lede">本章节将在后续任务中加入。</p>'
        units.append(
            f'<section class="section" id="unit-{len(units) + 1}">'
            f'<div class="section-label">{escape(unit)}</div><div class="prose"><h2>{escape(unit)}</h2></div>'
            f'<div class="chapter-grid">{cards}</div></section>'
        )
    body = (
        '<div class="content"><section class="hero">'
        '<p class="eyebrow">RenderTorch · 中文交互物理教材</p>'
        '<h1>从真实运动，到数字状态</h1>'
        '<p class="lede">从状态、速度与积分出发，逐步进入弹簧、约束、碰撞、XPBD、刚体与软体。</p>'
        '<p><a class="primary-link" href="../index.html">返回全书入口</a></p></section>'
        + "".join(units) + "</div>"
    )
    return document(
        "物理模拟 · RenderTorch 教材",
        body,
        prefix="../",
        sidebar_html=sidebar("index"),
        header_html=header("RenderTorch · 物理模拟教材", "../", meta="PARTICLES · CONSTRAINTS · DYNAMICS", has_sidebar=True),
        footer_html=footer("RenderTorch · 从像素到物理的中文交互教材"),
    )


def validate_references(rows) -> None:
    if not isinstance(rows, list):
        raise ValueError("physics references must be a list")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"physics reference {index} must be an object")
        for field in REFERENCE_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"physics reference {index} has an invalid or missing {field}")
        if row["group"] not in REFERENCE_GROUPS:
            raise ValueError(f"unknown reference group {row['group']!r} in physics reference {index}")
    missing = [group for group in REFERENCE_GROUPS if not any(row["group"] == group for row in rows)]
    if missing:
        raise ValueError(f"missing reference group: {', '.join(missing)}")


def render_reference_card(row) -> str:
    return (
        f'<article class="reference-card"><h2>{escape(row["title"])}</h2>'
        f'<p class="reference-authors">{escape(row["authors"])}</p>'
        f'<p><a href="{escape(row["url"])}">{escape(row["url"])}</a></p>'
        f'<p><span class="status-pill">许可：{escape(row["license"])}</span></p>'
        f'<p>{escape(row["usage"])}</p></article>'
    )


def render_references() -> str:
    rows = CONTENT.references()
    validate_references(rows)
    sections = []
    for number, group in enumerate(REFERENCE_GROUPS, 1):
        cards = "".join(render_reference_card(row) for row in rows if row["group"] == group)
        sections.append(
            f'<section class="section" id="reference-group-{number}">'
            f'<div class="section-label">{escape(group)}</div><div class="prose"><h2>{escape(group)}</h2></div>'
            f'<div class="reference-grid">{cards}</div></section>'
        )
    sections.append(
        '<section class="section" id="katex-license"><div class="section-label">本地公式资源</div>'
        '<div class="prose"><h2>KaTeX MIT 许可</h2>'
        '<p>本地 KaTeX 0.16.47 用于渲染公式，采用 MIT 许可；完整许可证见 '
        '<a href="../assets/vendor/katex/LICENSE">book/assets/vendor/katex/LICENSE</a>。</p></div></section>'
    )
    body = (
        '<article class="content"><header class="chapter-header"><p class="eyebrow">Physics References</p>'
        '<h1>物理参考资料</h1><p class="lede">只核对知识，不复制正文/图片/代码；'
        '正文、公式推导、伪代码与图示保持原创。</p></header>'
        + "".join(sections) + "</article>"
    )
    return document(
        "物理参考资料 · RenderTorch 教材",
        body,
        prefix="../",
        sidebar_html=sidebar("references"),
        header_html=header("RenderTorch · 物理模拟教材", "../", has_sidebar=True),
        footer_html=footer("RenderTorch · 从像素到物理的中文交互教材"),
    )


def build(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    rows = CONTENT.chapters()
    write(output_dir / "physics" / "index.html", render_index(rows))
    write(output_dir / "physics" / "references.html", render_references())
    for row in rows:
        write(output_dir / "physics" / "chapters" / f"{row.slug}.html", render_chapter(row))
    copy_assets(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RenderTorch physics textbook")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.output)
    print(f"Built physics subbook in {args.output.resolve()}")
