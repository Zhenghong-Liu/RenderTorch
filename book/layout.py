import html
from pathlib import Path


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def header(
    brand: str,
    prefix: str = "",
    meta: str = "PIXELS · GEOMETRY · RENDERING",
    has_sidebar: bool = False,
) -> str:
    menu_button = (
        '<button class="menu-toggle" type="button" data-menu-toggle aria-expanded="false" '
        'aria-controls="sidebar">目录</button>'
        if has_sidebar
        else ""
    )
    return (
        '<header class="site-header">'
        f'<a class="brand" href="{prefix}index.html"><span class="brand-mark">R</span><span>{escape(brand)}</span></a>'
        f'<span class="header-meta">{escape(meta)}</span>'
        f'{menu_button}</header>'
    )


def footer(brand: str) -> str:
    return (
        f'<footer class="site-footer">{escape(brand)} · '
        '<a href="https://github.com/Zhenghong-Liu/RenderTorch">GitHub 源码</a></footer>'
    )


def document(
    title: str,
    body: str,
    scripts: tuple[str, ...] = (),
    prefix: str = "",
    sidebar_html: str = "",
    header_html: str = "",
    footer_html: str = "",
) -> str:
    script_tags = "".join(f'<script src="{escape(path)}" defer></script>' for path in scripts)
    layout_class = "layout" if sidebar_html else "layout single-column"
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="RenderTorch 中文交互教材：从像素到物理，连接图形学与物理模拟。">
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
  {header_html}
  <div class="{layout_class}">
    {sidebar_html}
    <main id="main-content">
      {body}
    </main>
  </div>
  {footer_html}
</body>
</html>
'''
