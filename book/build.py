import argparse
import html
import shutil
from pathlib import Path

from content import BRANCH, CHAPTERS, REPOSITORY, source_url


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent


def navigation(current, prefix):
    items = []
    for chapter in CHAPTERS:
        active = ' aria-current="page"' if chapter["number"] == current else ""
        items.append(
            f'<li><a href="{prefix}chapters/chapter{chapter["number"]:02d}.html"{active}>'
            f'<span class="number">{chapter["number"]:02d}</span><span>{html.escape(chapter["title"])}</span></a></li>'
        )
    return "\n".join(items)


def page_shell(title, description, body, current=0, prefix=""):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{html.escape(description, quote=True)}">
  <meta name="theme-color" content="#132832">
  <link rel="icon" href="{prefix}assets/favicon.svg" type="image/svg+xml">
  <title>{html.escape(title)} · RenderTorch 教学书</title>
  <link rel="stylesheet" href="{prefix}assets/style.css">
  <script src="{prefix}assets/book.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#main-content">跳到正文</a>
  <header class="site-header">
    <a class="brand" href="{prefix}index.html"><span class="brand-mark">R</span><span>RenderTorch · 教学书</span></a>
    <span class="header-meta">PYTORCH × SOFTWARE RENDERING × PHYSICS</span>
    <button class="menu-toggle" type="button" data-menu-toggle aria-expanded="false" aria-controls="sidebar">目录</button>
  </header>
  <div class="layout">
    <aside class="sidebar" id="sidebar" data-sidebar data-open="false">
      <p class="sidebar-title">全书目录</p>
      <ol class="toc">
        <li><a href="{prefix}index.html"{' aria-current="page"' if current == 0 else ""}><span class="number">00</span><span>书籍首页</span></a></li>
        {navigation(current, prefix)}
      </ol>
      <p class="sidebar-note">源码分支：<code>{BRANCH}</code><br>内容与当前 RenderTorch 教学实现对应。</p>
    </aside>
    <main id="main-content">{body}</main>
  </div>
  <footer class="site-footer">RenderTorch · 从像素到物理的中文教学书 · <a href="{REPOSITORY}">GitHub 源码</a></footer>
</body>
</html>
"""


def home_page(prefix=""):
    cards = []
    milestones = [
        ("M1–M2", "显示与时间基础", "Qt 最小闭环、软件渲染、固定时间步与碰撞投影。"),
        ("M3–M5", "三种布料算法", "质量弹簧、PBD、XPBD 共享拓扑并可直接切换。"),
        ("M6", "刚体与软体", "球体冲量、摩擦、距离约束和四面体体积保持。"),
    ]
    for code, title, description in milestones:
        cards.append(
            f'<article class="milestone"><span class="status">{code} · 已完成</span>'
            f'<h3>{title}</h3><p>{description}</p></article>'
        )
    chapter_cards = []
    for chapter in CHAPTERS:
        chapter_cards.append(
            f'<a class="chapter-card" href="chapters/chapter{chapter["number"]:02d}.html">'
            f'<span class="chapter-number">{chapter["number"]:02d}</span>'
            f'<span><strong>{html.escape(chapter["title"])}</strong>'
            f'<small>{html.escape(chapter["summary"])}</small></span></a>'
        )
    body = f"""
    <div class="content">
      <section class="hero">
        <p class="eyebrow">从像素到物理 · 中文交互教学书</p>
        <h1>看清 3D 画面与物理模拟的每一步</h1>
        <p class="lede">沿着真实 Python 源码，从轨道相机、齐次裁剪和深度缓冲出发，逐步走到三种布料算法、刚体碰撞与软体体积约束。</p>
        <a class="primary-link" href="chapters/chapter01.html">从第 1 章开始 →</a>
      </section>

      <section class="section" id="overview">
        <div class="section-label">项目全景</div>
        <div class="prose"><h2>不是黑盒演示，而是一条可检查的因果链</h2>
          <p>RenderTorch 是一个模块化单体：Qt 只负责界面，Scene 保存状态，PyTorch 执行数值计算，软件渲染器生成 FrameBuffer。每种物理方法都有独立实现、可重复初始状态和自动化测试。</p>
        </div>
        <div class="milestone-grid" style="margin-top:1.4rem">{''.join(cards)}</div>
      </section>

      <section class="section" id="metrics">
        <div class="section-label">当前基线</div>
        <dl class="metric-grid">
          <div class="metric"><dt>自动化测试</dt><dd>88 项通过</dd></div>
          <div class="metric"><dt>默认物理步长</dt><dd>1/120 s · 2 子步</dd></div>
          <div class="metric"><dt>显示模式</dt><dd>实体 / 线框 / 粒子</dd></div>
        </dl>
      </section>

      <section class="section" id="chapters">
        <div class="section-label">12 章阅读路线</div>
        <div class="prose"><h2>先建立最小闭环，再替换求解方法</h2>
          <p>第 1–5 章建立共同语言；第 6–9 章深入布料；第 10–12 章连接刚体、软体、性能与后续扩展。每章都包含公式、伪代码、真实源码、图示、指标和调试清单。</p>
        </div>
        <div class="chapter-grid" style="margin-top:1.4rem">{''.join(chapter_cards)}</div>
      </section>

      <section class="section" id="reading">
        <div class="section-label">使用方式</div>
        <div class="prose"><h2>把书与桌面程序放在一起读</h2>
          <p>先运行 <code>python main_qt.py</code>，在右侧面板切换模拟类型和布料算法。阅读第 6–9 章时保持相同初始状态，切换线框与实体视图，并把单步误差与书中指标对照。</p>
          <p>第 9 章的浏览器交互示例可以暂停、重置并调整迭代与柔度。它用于展示趋势；精确算法行为以 GitHub 上的 Python 源码和测试为准。</p>
        </div>
      </section>
    </div>
    """
    return page_shell(
        "从像素到物理",
        "RenderTorch 中文教学书：软件渲染、三种布料算法、刚体与软体。",
        body,
        prefix=prefix,
    )


def chapter_page(chapter, prefix=""):
    number = chapter["number"]
    source_links = "".join(
        f'<a class="source-link" href="{source_url(path)}"><span>{html.escape(path)}</span>↗</a>'
        for path in chapter["sources"]
    )
    metrics = "".join(
        f'<div class="metric"><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>'
        for label, value in chapter["metrics"]
    )
    debugging = "".join(f"<li>{item}</li>" for item in chapter["debugging"])
    previous = (
        f'<a href="chapter{number - 1:02d}.html"><small>← 上一章</small>{html.escape(CHAPTERS[number - 2]["title"])}</a>'
        if number > 1 else '<a href="../index.html"><small>← 返回</small>书籍首页</a>'
    )
    following = (
        f'<a href="chapter{number + 1:02d}.html"><small>下一章 →</small>{html.escape(CHAPTERS[number]["title"])}</a>'
        if number < len(CHAPTERS) else f'<a href="{REPOSITORY}"><small>继续探索 →</small>RenderTorch GitHub</a>'
    )
    body = f"""
    <article class="content">
      <header>
        <p class="eyebrow">Chapter {number:02d} / 12</p>
        <h1>{html.escape(chapter['title'])}</h1>
        <p class="lede">{html.escape(chapter['summary'])}</p>
        <div class="meta-row"><span class="pill">阅读约 8 分钟</span><span class="pill"><strong>{len(chapter['sources'])}</strong> 个源码入口</span><span class="pill">对应真实实现</span></div>
      </header>

      <section class="section" id="intuition"><div class="section-label">01 · 直觉</div><div class="prose"><h2>先建立可观察的因果</h2>{chapter['intuition']}</div></section>
      <section class="section" id="formula"><div class="section-label">02 · 公式</div><div class="prose"><h2>用数学压缩核心关系</h2>{chapter['formula']}</div></section>
      <section class="section" id="pseudocode"><div class="section-label">03 · 伪代码</div><div class="prose"><h2>把公式变成执行顺序</h2></div><div class="code-card"><div class="code-toolbar"><span>PSEUDOCODE</span><button class="copy-button" type="button" data-copy>复制代码</button></div><pre><code>{html.escape(chapter['pseudocode'].strip())}</code></pre></div></section>
      <section class="section" id="source"><div class="section-label">04 · 源码</div><div class="prose"><h2>阅读当前项目的真实实现</h2><p>链接固定到本书对应的源码分支。建议先看公开接口，再追踪一个完整时间步。</p></div><div class="source-grid">{source_links}</div></section>
      <section class="section" id="visual"><div class="section-label">05 · 图示 / 交互</div><div class="prose"><h2>让状态变化看得见</h2></div><div class="visual-card">{chapter['visual']}</div></section>
      <section class="section" id="metrics"><div class="section-label">06 · 参数与指标</div><div class="prose"><h2>用数字验证直觉</h2></div><dl class="metric-grid">{metrics}</dl></section>
      <section class="section" id="debugging"><div class="section-label">07 · 调试</div><div class="prose"><h2>常见问题与排查顺序</h2><ol class="debug-list">{debugging}</ol></div></section>
      <nav class="page-nav" aria-label="章节导航">{previous}{following}</nav>
    </article>
    """
    return page_shell(chapter["title"], chapter["summary"], body, current=number, prefix=prefix)


def build(output):
    output = output.resolve()
    chapters_dir = output / "chapters"
    assets_dir = output / "assets"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(home_page(), encoding="utf-8")
    for chapter in CHAPTERS:
        path = chapters_dir / f"chapter{chapter['number']:02d}.html"
        path.write_text(chapter_page(chapter, prefix="../"), encoding="utf-8")
    for name in ("style.css", "book.js", "favicon.svg"):
        source = SOURCE_DIR / "assets" / name
        target = assets_dir / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    print(f"Built {len(CHAPTERS) + 1} pages in {output}")


def main():
    parser = argparse.ArgumentParser(description="Build the RenderTorch static teaching book.")
    parser.add_argument("--output", type=Path, default=SOURCE_DIR)
    arguments = parser.parse_args()
    build(arguments.output)


if __name__ == "__main__":
    main()
