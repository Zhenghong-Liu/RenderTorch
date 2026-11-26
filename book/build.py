import argparse
import html
import importlib.util
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
CONTENT_PATH = ROOT / "content.py"
CONTENT_NAME = "rendertorch_book_v2_content"
SPEC = importlib.util.spec_from_file_location(CONTENT_NAME, CONTENT_PATH)
CONTENT = importlib.util.module_from_spec(SPEC)
sys.modules[CONTENT_NAME] = CONTENT
SPEC.loader.exec_module(CONTENT)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_assets(output_dir: Path) -> None:
    source = ROOT / "assets"
    target = output_dir / "assets"
    if source.resolve() == target.resolve():
        return
    shutil.copytree(source, target, dirs_exist_ok=True)


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def navigation(active_slug: str = "", prefix: str = "") -> str:
    rows = CONTENT.chapters()
    chapter_prefix = "" if prefix else "chapters/"
    home_current = ' aria-current="page"' if active_slug == "index" else ""
    items = [
        f'<li><a href="{prefix}index.html"{home_current}>'
        '<span class="number">00</span><span>书籍首页</span></a></li>'
    ]
    for row in rows:
        current = ' aria-current="page"' if row.slug == active_slug else ""
        items.append(
            f'<li><a href="{chapter_prefix}{row.slug}.html"{current}>'
            f'<span class="number">{row.number:02d}</span><span>{escape(row.title)}</span></a></li>'
        )
    reference_current = ' aria-current="page"' if active_slug == "references" else ""
    items.append(
        f'<li><a href="{prefix}references.html"{reference_current}>'
        '<span class="number">R</span><span>参考资料与许可</span></a></li>'
    )
    return '<ol class="toc">' + "".join(items) + "</ol>"


def sidebar(active_slug: str = "", prefix: str = "") -> str:
    return (
        '<aside class="sidebar" id="sidebar" data-sidebar data-open="false">'
        '<p class="sidebar-title">全书目录</p>'
        f"{navigation(active_slug, prefix)}"
        '<p class="sidebar-note">3D 图形学子书 v2<br>先建立数学与管线，再进入光照和光线追踪。</p>'
        "</aside>"
    )


def header(prefix: str = "") -> str:
    return (
        '<header class="site-header">'
        f'<a class="brand" href="{prefix}index.html"><span class="brand-mark">R</span><span>RenderTorch · 图形学教材</span></a>'
        '<span class="header-meta">PIXELS · GEOMETRY · RENDERING</span>'
        '<button class="menu-toggle" type="button" data-menu-toggle aria-expanded="false" '
        'aria-controls="sidebar">目录</button>'
        "</header>"
    )


def footer() -> str:
    return (
        '<footer class="site-footer">RenderTorch · 从像素到物理的中文交互教材 · '
        '<a href="https://github.com/Zhenghong-Liu/RenderTorch">GitHub 源码</a></footer>'
    )


def document(title: str, active_slug: str, body: str, scripts: tuple[str, ...] = (), prefix: str = "") -> str:
    script_tags = "".join(f'<script src="{escape(path)}" defer></script>' for path in scripts)
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="RenderTorch 中文图形学交互教材：从像素、向量与相机到光栅化、光照和光线追踪。">
  <meta name="theme-color" content="#132832">
  <link rel="icon" href="{prefix}assets/favicon.svg" type="image/svg+xml">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{prefix}assets/style.css">
  <link rel="stylesheet" href="{prefix}assets/vendor/katex/katex.min.css">
  <script src="{prefix}assets/vendor/katex/katex.min.js" defer></script>
  <script src="{prefix}assets/vendor/katex/auto-render.min.js" defer></script>
  <script src="{prefix}assets/book.js" defer></script>
{script_tags}</head>
<body>
  <a class="skip-link" href="#main-content">跳到正文</a>
  {header(prefix)}
  <div class="layout">
    {sidebar(active_slug, prefix)}
    <main id="main-content">
      {body}
    </main>
  </div>
  {footer()}
</body>
</html>
'''


def render_section(section) -> str:
    return (
        f'<section class="section chapter-section" id="{escape(section.id)}">'
        f'<div class="section-label">{escape(section.label)}</div>'
        f'<div class="prose"><h2>{escape(section.title)}</h2>{section.html}</div>'
        "</section>"
    )


def render_objectives(row) -> str:
    objectives = "".join(f"<li>{escape(item)}</li>" for item in row.objectives)
    prerequisites = "、".join(escape(item) for item in row.prerequisites)
    return (
        '<section class="section chapter-section" id="objectives">'
        '<div class="section-label">学习目标</div><div class="prose"><h2>本章结束后你能够</h2>'
        f"<ol>{objectives}</ol><p><strong>先验知识：</strong>{prerequisites}。</p></div></section>"
    )


def render_demo_mount(demo) -> str:
    return (
        f'<div class="demo-mount" id="{escape(demo.id)}" data-demo="{escape(demo.id)}">'
        f"<h3>{escape(demo.title)}</h3><p>{escape(demo.description)}</p></div>"
    )


def render_sources(row) -> str:
    if not row.sources:
        return (
            '<section class="section chapter-section" id="sources">'
            '<div class="section-label">源码映射</div><div class="prose">'
            '<h2>概念内容与真实实现</h2><p>本章以浏览器概念实验为主；'
            'RenderTorch 当前实现的对应文件会在确实存在时列出。</p></div></section>'
        )
    links = "".join(
        f'<a class="source-link" href="{escape(source.url)}"><span>{escape(source.path)}</span>'
        f'<small>{escape(source.status)}</small>↗</a>'
        for source in row.sources
    )
    return (
        '<section class="section chapter-section" id="sources">'
        '<div class="section-label">源码映射</div><div class="prose">'
        '<h2>阅读当前项目的真实实现</h2><div class="source-grid">'
        f"{links}</div></div></section>"
    )


def page_navigation(row) -> str:
    rows = CONTENT.chapters()
    previous = rows[row.number - 2] if row.number > 1 else None
    following = rows[row.number] if row.number < len(rows) else None
    left = (
        f'<a href="{previous.slug}.html"><small>← 上一章</small>{escape(previous.title)}</a>'
        if previous
        else '<span></span>'
    )
    right = (
        f'<a href="{following.slug}.html"><small>下一章 →</small>{escape(following.title)}</a>'
        if following
        else '<span></span>'
    )
    return f'<nav class="page-nav" aria-label="章节导航">{left}{right}</nav>'


def render_chapter(row) -> str:
    sections = list(row.sections)
    section_ids = {section.id for section in sections}
    rendered = []
    if "objectives" not in section_ids:
        rendered.append(render_objectives(row))
    for section in sections:
        rendered.append(render_section(section))
    if "sources" not in section_ids:
        rendered.append(render_sources(row))
    if row.demos:
        mounts = "".join(render_demo_mount(demo) for demo in row.demos)
        scripts = tuple(f"../assets/{demo.script}" for demo in row.demos)
    else:
        mounts = ""
        scripts = ()
    demo_html = (
        '<section class="section chapter-section" id="demo">'
        '<div class="section-label">交互实验</div><div class="prose">'
        f'<h2>让状态变化看得见</h2>{mounts}</div></section>'
    )
    rendered.append(demo_html)
    body = (
        '<article class="content">'
        '<header class="chapter-header">'
        f'<p class="eyebrow">Chapter {row.number:02d} / 15 · {escape(row.unit)}</p>'
        f'<h1>{escape(row.title)}</h1><p class="lede">{escape(row.summary)}</p>'
        '<div class="meta-row"><span class="pill">教材式推导</span>'
        '<span class="pill">可操作实验</span><span class="pill">真实源码分层</span></div>'
        "</header>"
        + "".join(rendered)
        + page_navigation(row)
        + "</article>"
    )
    return document(row.title, row.slug, body, scripts, prefix="../")


def render_index(all_chapters=None) -> str:
    rows = list(all_chapters) if all_chapters is not None else CONTENT.chapters()
    units = []
    for unit in dict.fromkeys(row.unit for row in rows):
        unit_rows = [row for row in rows if row.unit == unit]
        cards = "".join(
            f'<a class="chapter-card" href="chapters/{row.slug}.html">'
            f'<span class="chapter-number">{row.number:02d}</span>'
            f'<span><strong>{escape(row.title)}</strong><small>{escape(row.summary)}</small></span></a>'
            for row in unit_rows
        )
        units.append(
            f'<section class="section" id="unit-{len(units) + 1}">'
            f'<div class="section-label">{escape(unit)}</div><div class="prose">'
            f'<h2>{escape(unit)}</h2></div><div class="chapter-grid">{cards}</div></section>'
        )
    body = (
        '<div class="content">'
        '<section class="hero">'
        '<p class="eyebrow">RenderTorch · 中文交互图形学教材</p>'
        '<h1>从一个像素，走到完整三维画面</h1>'
        '<p class="lede">沿着真实数学与代码，从图像坐标、向量和齐次变换出发，'
        '逐步推导相机、网格、光栅化、材质、阴影与光线追踪。</p>'
        '<a class="primary-link" href="chapters/chapter01.html">从第 1 章开始 →</a>'
        '</section>'
        '<section class="section" id="overview"><div class="section-label">学习路线</div>'
        '<div class="prose"><h2>先建立共同语言，再替换渲染方法</h2>'
        '<p>全书 15 章分为四个单元。每章都从直觉开始，给出符号定义、逐步推导、'
        '数值算例、伪代码和可操作实验；现有 Python 实现与浏览器概念实验会明确分层。</p>'
        '<p>本阶段先重建 3D 图形学子书。质点弹簧、PBD、XPBD 等物理子书将在图形学内容发布后另行重构。</p>'
        '</div></section>'
        + "".join(units)
        + '<section class="section" id="references"><div class="section-label">开放许可优先</div>'
        '<div class="prose"><h2>参考资料与原创图解</h2>'
        '<p>知识点参考开放教材、官方文档和论文；本书正文与 Canvas/SVG 图解均为原创，'
        '不复制商业教材或论文插图。<a href="references.html">查看参考资料与许可边界。</a></p>'
        '</div></section></div>'
    )
    return document("从像素到物理 · RenderTorch 图形学教材", "index", body)


def render_references() -> str:
    rows = CONTENT.references()
    cards = "".join(
        f'<article class="reference-card"><h2>{escape(row["title"])}</h2>'
        f'<p class="reference-authors">{escape(row["authors"])}</p>'
        f'<p><a href="{escape(row["url"])}">{escape(row["url"])}</a></p>'
        f'<p><span class="status-pill">许可：{escape(row["license"])}</span></p>'
        f'<p>{escape(row["usage"])}</p></article>'
        for row in rows
    )
    body = (
        '<article class="content"><header><p class="eyebrow">References</p>'
        '<h1>参考资料与许可边界</h1><p class="lede">只记录思想来源与使用边界；'
        '新书正文和图示保持原创。</p></header>'
        '<section class="section" id="references"><div class="section-label">参考资料</div>'
        f'<div class="reference-grid">{cards}</div></section>'
        '<section class="section" id="license"><div class="section-label">使用原则</div><div class="prose">'
        '<h2>引用知识，不复制受保护内容</h2><ul>'
        '<li>LearnOpenGL、PBRT 与 Ray Tracing in One Weekend 只核对知识点，不使用其图片或正文。</li>'
        '<li>所有公式推导、中文讲解、伪代码和 Canvas/SVG 图解由本教材重新组织。</li>'
        '<li>论文只引用算法与 DOI，不转载论文插图或大段文字。</li>'
        '<li>后续如复用开放许可素材，必须在页面保留作者、来源和许可链接。</li>'
        '</ul></div></section></article>'
    )
    return document("参考资料与许可 · RenderTorch 图形学教材", "references", body)


def build(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    write(output_dir / "index.html", render_index())
    write(output_dir / "references.html", render_references())
    for row in CONTENT.chapters():
        write(output_dir / "chapters" / f"{row.slug}.html", render_chapter(row))
    copy_assets(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RenderTorch graphics textbook")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.output)
    print(f"Built 17 pages in {args.output.resolve()}")
