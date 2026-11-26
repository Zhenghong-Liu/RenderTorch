# RenderTorch 图形学子书 v2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零构建并发布一本 15 章、带实时 Canvas 实验和真实源码映射的中文计算机图形学教材。

**Architecture:** 新书先在 `book-v2/` 中作为独立静态站点开发，内容由 Python 标准库结构化，HTML/CSS/JavaScript 在浏览器直接运行；全部验收通过后，才备份并替换旧 `book/`，最后分别更新源分支与 GitHub Pages。

**Tech Stack:** Python 3.11 标准库、HTML5、CSS3、原生 JavaScript、Canvas 2D、SVG、Node.js 语法检查、unittest、GitHub Pages。

**Spec:** [RenderTorch 图形学子书 v2 设计](2026-11-26-rendertorch-graphics-book-v2-design.md)

## Global Constraints

- 开发目录固定为 `book-v2/`；任务 8 之前不得修改、移动或删除旧 `book/`。
- 全书固定为 15 章，章节编号、标题和四个单元与设计规格完全一致。
- 不引入 WebGL、Three.js、React、npm 构建或任何前端运行时依赖。
- 每章必须包含 `objectives`、`intuition`、`definitions`、`derivation`、`example`、`pseudocode`、`sources`、`demo`、`self-test`、`debugging`、`status` 十一类内容。
- 实现状态只能使用“现有 RenderTorch”“浏览器概念实验”“后续实现路线”三种固定标签。
- 所有 Canvas/SVG 图解原创；不下载或复用 LearnOpenGL 图片，不使用旧书插图。
- 浏览器实验不得请求网络；外部链接只允许用户主动点击。
- 新书所有页面必须支持 `file://` 与 HTTP 两种打开方式。
- Python 环境使用 `pythonTry`；测试命令通过 `/opt/anaconda3/bin/conda run -n pythonTry python ...` 执行。
- 每个新增或修改的 JavaScript 文件必须通过 `node --check`。
- 任务 8 之前不创建 Git 提交，以免夹带工作区已有的未提交修改；唯一提交在任务 9 完成。
- 最终发布只暂存教材相关文件，不使用 `git add -A`、`git clean`、`git reset --hard` 或 `git push --force`。
- 旧书必须先复制到仓库外并校验，再替换 `book/`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `book-v2/content.py` | 章节元数据、正文、公式、伪代码、源码链接和 Demo 规格 |
| `book-v2/references.json` | 参考资料、作者、URL、许可与使用边界 |
| `book-v2/build.py` | 生成首页、15 章页面、参考资料页和侧栏导航 |
| `book-v2/assets/style.css` | 教材版式、公式/代码卡、实验工具栏、移动端和打印样式 |
| `book-v2/assets/book.js` | 移动目录、代码复制、Canvas 公共数学工具 |
| `book-v2/assets/demos/chapter01.js` 至 `chapter15.js` | 每章独立交互实验 |
| `book-v2/index.html` | 构建产物：全书首页 |
| `book-v2/chapters/chapter01.html` 至 `chapter15.html` | 构建产物：章节页面 |
| `book-v2/references.html` | 构建产物：参考资料与许可页 |
| `tests/test_book.py` | 内容、构建、链接、资源和许可契约测试 |
| `README.md` | 新书构建、运行、章节数量和发布说明 |
| `docs/references.md` | 补充 v2 参考资料和许可边界 |
| `docs/superpowers/specs/2026-11-26-rendertorch-graphics-book-v2-design.md` | 本计划依据的设计规格 |
| `docs/superpowers/plans/2026-11-26-rendertorch-graphics-book-v2.md` | 本实施计划 |

## Interfaces

### Python 内容模型

```python
@dataclass(frozen=True)
class Section:
    id: str
    label: str
    title: str
    html: str

@dataclass(frozen=True)
class DemoSpec:
    id: str
    title: str
    script: str
    description: str

@dataclass(frozen=True)
class SourceLink:
    path: str
    url: str
    status: str

@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    unit: str
    title: str
    summary: str
    objectives: tuple[str, ...]
    prerequisites: tuple[str, ...]
    sections: tuple[Section, ...]
    demos: tuple[DemoSpec, ...]
    sources: tuple[SourceLink, ...]

def chapters() -> list[Chapter]: raise NotImplementedError
def references() -> list[dict]: raise NotImplementedError
```

### 构建接口

```python
def render_chapter(chapter: Chapter) -> str: raise NotImplementedError
def render_index(all_chapters: list[Chapter]) -> str: raise NotImplementedError
def render_references() -> str: raise NotImplementedError
def build(output_dir: Path) -> None: raise NotImplementedError

if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
```

### 测试路径接口

```python
BOOK_DIR = Path(os.environ.get("BOOK_DIR", "book-v2"))
```

任务 8 只把默认值从 `book-v2` 改为 `book`。

### JavaScript 公共接口

```javascript
window.RTBook = {
  clamp(value, min, max),
  lerp(a, b, t),
  normalize2(x, y),
  setupCanvas(canvas),
  formatNumber(value, digits = 3)
};
```

每个 `chapterNN.js` 都是普通脚本，按 `assets/book.js` → `assets/demos/chapterNN.js` 顺序加载。

### 固定章节标识

```text
chapter01  像素、颜色与图像
chapter02  三维空间、向量、点与方向
chapter03  齐次坐标与几何变换
chapter04  相机、视图、投影与视口
chapter05  几何图元与网格表示
chapter06  网格拓扑、法线、绕序与三角化
chapter07  参数曲线：Bézier、B-spline 与 NURBS 概念
chapter08  参数曲面与细分
chapter09  场景、模型文件、场景图与包围体
chapter10  齐次裁剪与三角形光栅化
chapter11  重心坐标、属性插值、深度与遮挡
chapter12  光照模型与材质
chapter13  纹理、UV 与透视矫正
chapter14  阴影、gamma 与色调映射
chapter15  光线追踪与完整渲染管线比较
```

---

### Task 1: 建立内容模型与参考资料目录

**Files:**
- Create: `book-v2/content.py`
- Create: `book-v2/references.json`
- Replace: `tests/test_book.py`

**Interfaces:**
- Consumes: 无。
- Produces: `chapters() -> list[Chapter]`、`references() -> list[dict]`，以及 `BOOK_DIR = Path(os.environ.get("BOOK_DIR", "book-v2"))`。

- [ ] **Step 1: 用新契约替换旧教材测试**

将 `tests/test_book.py` 替换为以下内容，使旧 12 章契约先失败：

```python
import importlib.util
import os
import unittest
from pathlib import Path

BOOK_DIR = Path(os.environ.get("BOOK_DIR", "book-v2"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")

    def test_chapter_metadata_is_complete(self):
        rows = self.content.chapters()
        self.assertEqual([row.number for row in rows], list(range(1, 16)))
        self.assertEqual(len({row.slug for row in rows}), 15)
        for row in rows:
            self.assertTrue(row.title.strip())
            self.assertTrue(row.summary.strip())
            self.assertTrue(row.unit.strip())
            self.assertTrue(row.objectives)
            self.assertTrue(row.prerequisites)

    def test_reference_entries_have_license_and_usage(self):
        rows = self.content.references()
        self.assertGreaterEqual(len(rows), 8)
        for row in rows:
            for field in ("title", "authors", "url", "license", "usage"):
                self.assertTrue(row[field].strip(), (row.get("title"), field))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期失败：`book-v2/content.py` 不存在，测试在加载模块时报错。

- [ ] **Step 3: 实现内容模型与 15 章元数据**

在 `book-v2/content.py` 中实现设计规格中的四个 `dataclass`。`CHAPTER_METADATA` 必须逐项包含固定章节标识表中的 15 个 `number`、`slug`、`unit`、`title`、`summary`，并为每章写出至少两条学习目标和一条先验要求。此时 `sections`、`demos`、`sources` 允许为空元组，后续任务按章填充。

`references()` 使用 `json.loads(Path(__file__).with_name("references.json").read_text(encoding="utf-8"))` 读取参考资料。

- [ ] **Step 4: 写入参考资料目录**

`book-v2/references.json` 必须包含以下九项，且每项都有 `title`、`authors`、`url`、`license`、`usage`：

1. David J. Eck，*Introduction to Computer Graphics*，`https://math.hws.edu/graphicsbook/`，CC BY-NC-SA 4.0。
2. Joey de Vries，*LearnOpenGL*，`https://learnopengl.com/About`，图片 CC BY 4.0、代码 CC BY-NC 4.0。
3. Matt Pharr、Wenzel Jakob、Greg Humphreys，*Physically Based Rendering, 4th edition*，`https://pbr-book.org/4ed/contents`，CC BY-NC-ND 4.0。
4. Peter Shirley、Trevor David Black、Steve Hollasch，*Ray Tracing in One Weekend*，`https://raytracing.github.io/books/RayTracingInOneWeekend.html`，All rights reserved。
5. OpenStax，*University Physics Volume 1*，`https://openstax.org/books/university-physics-volume-1/pages/preface`，CC BY 4.0。
6. David Baraff、Andrew Witkin，*Physically Based Modeling*，SIGGRAPH 2001 Course Notes，课程资料。
7. John House、John Keyser，*Foundations of Physically Based Modeling and Animation*，商业出版物。
8. Matthias Müller 等，*Position Based Dynamics*，`https://doi.org/10.1016/j.jvcir.2007.01.005`，论文引用。
9. Miles Macklin 等，*XPBD: Position-Based Simulation of Compliant Constrained Dynamics*，`https://doi.org/10.1145/1399504.1360662`，论文引用。

LearnOpenGL、PBRT、Ray Tracing in One Weekend 的 `usage` 必须写明只核对知识点、不复制正文或图片。

- [ ] **Step 5: 运行测试并确认通过**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：2 项测试通过。

- [ ] **Step 6: 检查差异且不提交**

```bash
git -C RenderTorch diff --check
git -C RenderTorch status --short
```

确认只新增 `book-v2/content.py`、`book-v2/references.json` 并替换 `tests/test_book.py`；不暂存其他工作区改动。

---

### Task 2: 实现静态站点骨架、模板与公共交互

**Files:**
- Create: `book-v2/build.py`
- Create: `book-v2/assets/style.css`
- Create: `book-v2/assets/book.js`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 任务 1 的 `Chapter`、`Section`、`DemoSpec`、`SourceLink`、`chapters()`、`references()`。
- Produces: `render_chapter()`、`render_index()`、`render_references()`、`build()`，以及 `window.RTBook`。

- [ ] **Step 1: 添加构建失败测试**

在 `tests/test_book.py` 中加入：

```python
import tempfile


class BuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")
        cls.build = load_module("book_v2_build", BOOK_DIR / "build.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.build.build(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_writes_home_references_and_fifteen_chapters(self):
        self.assertTrue((self.output / "index.html").is_file())
        self.assertTrue((self.output / "references.html").is_file())
        for row in self.content.chapters():
            self.assertTrue((self.output / "chapters" / f"{row.slug}.html").is_file())

    def test_navigation_and_accessibility_markers_exist(self):
        home = (self.output / "index.html").read_text(encoding="utf-8")
        chapter = (self.output / "chapters" / "chapter01.html").read_text(encoding="utf-8")
        for marker in ('id="main-content"', "data-menu-toggle", 'class="skip-link"'):
            self.assertIn(marker, home)
            self.assertIn(marker, chapter)
        self.assertEqual(home.count('class="chapter-card"'), 15)
        self.assertIn("references.html", home)
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：`BuildContractTests` 因 `book-v2/build.py` 不存在而失败，任务 1 的测试仍通过。

- [ ] **Step 3: 实现模板与构建器**

`build.py` 只使用 `dataclasses`、`html`、`json`、`pathlib` 和字符串模板。页面必须包含：

- `<html lang="zh-CN">`、viewport、主题色、favicon、`<title>`。
- 跳到正文链接、固定顶栏、品牌链接、移动端目录按钮。
- 侧栏只列出首页、参考资料和 15 章，不出现旧 12 章标题。
- `<main id="main-content">`、上一章/下一章导航、页脚。
- 每章脚本顺序为 `../assets/book.js` 后跟该章 `DemoSpec.script`。
- 首页包含四个单元介绍、15 张章节卡和“先 3D、后物理”的阶段说明。
- 参考资料页输出作者、标题、URL、许可和用途。

构建器写入文件的核心逻辑必须保持简单：

```python
def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(output_dir: Path) -> None:
    rows = chapters()
    write(output_dir / "index.html", render_index(rows))
    write(output_dir / "references.html", render_references())
    for row in rows:
        write(output_dir / "chapters" / f"{row.slug}.html", render_chapter(row))
```

`render_chapter()` 对空 `sections` 也要生成合法页面，以便任务 1 的元数据先构建成功；任务 3–6 再替换为完整正文。

- [ ] **Step 4: 实现视觉系统**

`assets/style.css` 必须定义并使用以下类：

- 全局：`:root`、`body`、`a`、`button`、`:focus-visible`、`.skip-link`。
- 框架：`.site-header`、`.brand`、`.menu-toggle`、`.layout`、`.sidebar`、`.toc`、`.content`、`.site-footer`。
- 教材：`.eyebrow`、`.lede`、`.meta-row`、`.pill`、`.section`、`.section-label`、`.prose`、`.formula`、`.callout`、`.code-card`。
- 实验：`.visual-card`、`.demo-toolbar`、`.demo-grid`、`.demo-readout`、`.status-pill`、`.swatch`。
- 导航：`.chapter-grid`、`.chapter-card`、`.page-nav`。

响应式断点固定为 980px 和 720px；720px 以下侧栏变为可关闭抽屉，实验工具栏改为纵向布局。打印样式隐藏顶栏、侧栏、工具栏和翻页导航，但保留正文、公式、图示标题和实验说明。

- [ ] **Step 5: 实现公共 JavaScript**

`assets/book.js` 必须实现：

```javascript
window.RTBook = {
  clamp(value, min, max) { return Math.min(max, Math.max(min, value)); },
  lerp(a, b, t) { return a + (b - a) * t; },
  normalize2(x, y) {
    const length = Math.hypot(x, y) || 1;
    return { x: x / length, y: y / length };
  },
  setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  },
  formatNumber(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
  }
};
```

同时实现移动目录切换、`aria-expanded` 同步、代码复制按钮和 `prefers-reduced-motion` 检查。公共脚本不得启动自动播放循环。

- [ ] **Step 6: 运行测试与 JavaScript 语法检查**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
node --check book-v2/assets/book.js
```

预期：4 项测试通过，`node --check` 无输出。

- [ ] **Step 7: 启动临时预览检查首页**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m http.server 8000 --directory book-v2
```

在浏览器确认首页 15 张卡片、移动目录和参考资料链接可点击；检查控制台无错误。完成后停止该临时服务器。

---

### Task 3: 完成第一单元——像素、数学、变换与相机

**Files:**
- Modify: `book-v2/content.py`
- Create: `book-v2/assets/demos/chapter01.js`
- Create: `book-v2/assets/demos/chapter02.js`
- Create: `book-v2/assets/demos/chapter03.js`
- Create: `book-v2/assets/demos/chapter04.js`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 任务 2 的页面模板、`RTBook` 和 `DemoSpec`。
- Produces: 第 1–4 章完整正文，以及 `pixel-demo`、`vector-demo`、`transform-demo`、`camera-demo` 四个唯一实验。

- [ ] **Step 1: 添加第一单元内容契约测试**

在 `tests/test_book.py` 中加入：

```python
REQUIRED_SECTIONS = {
    "objectives", "intuition", "definitions", "derivation", "example",
    "pseudocode", "sources", "demo", "self-test", "debugging", "status",
}

UNIT_ONE = {
    1: ("pixel-demo", ("u = x / (W - 1)", "RGBA")),
    2: ("vector-demo", ("a × b", "normalize")),
    3: ("transform-demo", ("p' = M p", "M^{-T}")),
    4: ("camera-demo", ("x_ndc", "x_pixel")),
}


class UnitContentTestCase(unittest.TestCase):
    unit_map = {}

    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")
        cls.build = load_module("book_v2_build", BOOK_DIR / "build.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.build.build(cls.output)
        cls.rows = {row.number: row for row in cls.content.chapters()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_chapters_have_all_textbook_sections_and_formulas(self):
        for number in self.unit_map:
            row = self.rows[number]
            page = self.output / "chapters" / f"{row.slug}.html"
            text = page.read_text(encoding="utf-8")
            for section_id in REQUIRED_SECTIONS:
                self.assertIn(f'section id="{section_id}"', text, row.title)
            demo_id, markers = self.unit_map[number]
            self.assertIn(f'id="{demo_id}"', text)
            self.assertTrue((BOOK_DIR / row.demos[0].script).is_file())
            self.assertIn(row.demos[0].script, text)
            for marker in markers:
                self.assertIn(marker, text)

    def test_status_labels_do_not_overstate_implementation(self):
        for number in self.unit_map:
            row = self.rows[number]
            page = self.output / "chapters" / f"{row.slug}.html"
            text = page.read_text(encoding="utf-8")
            status_start = text.index('section id="status"')
            status_end = text.index("</section>", status_start)
            status = text[status_start:status_end]
            self.assertIn("浏览器概念实验", status)
            self.assertTrue(
                "现有 RenderTorch" in status or "后续实现路线" in status
            )


class UnitOneContentTests(UnitContentTestCase):
    unit_map = UNIT_ONE
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：第一单元测试因缺少十一类 section、Demo 规格或脚本文件而失败。

- [ ] **Step 3: 写入第 1–4 章正文**

每章按十一类 section 写入，公式必须与设计规格完全一致：

- 第 1 章：像素索引、图像坐标、宽高、RGB/A、采样、分辨率；数值例使用 `W=8,H=8,x=3,y=5`。
- 第 2 章：点/向量、左右手系、点积、叉积、长度、归一化；数值例使用 `a=(1,0,0), b=(0,1,0)`。
- 第 3 章：齐次坐标、平移、缩放、旋转、复合顺序、法线矩阵；数值例使用 `p=(1,2,0,1), t=(3,-1,0)`。
- 第 4 章：轨道相机、lookAt、透视矩阵、近远裁剪面、NDC、视口变换；数值例使用 `x_ndc=0.5,W=800` 得 `x_pixel=600`。

第 4 章源码映射必须链接现有 `viewing/camera.py`、`viewing/projection.py`、`renderer/pipeline.py`；不得声称已有纹理、阴影或光线追踪。

- [ ] **Step 4: 实现四个实验脚本**

第 1 章 `pixel-demo`：

```javascript
const u = x / (W - 1);
const v = y / (H - 1);
readout.textContent = `像素 (${x}, ${y}) · u=${u.toFixed(3)} · v=${v.toFixed(3)} · RGBA=${rgba}`;
```

第 2 章 `vector-demo` 必须使用：

```javascript
const cross = {
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x
};
```

第 3 章 `transform-demo` 必须按“缩放 → 旋转 → 平移”的顺序复合，并显示每一步 4×4 矩阵；法线只显示 `M^{-T}` 思路，不实现通用矩阵求逆库。

第 4 章 `camera-demo` 必须显示相机位置、目标点、距离、立方体投影后的 NDC 与像素坐标；只使用 `RTBook` 的二维 Canvas 工具绘制线框立方体。

每个脚本都要提供重置控件、`aria-live` 读数和零控制台错误。

- [ ] **Step 5: 运行测试与脚本语法检查**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
for file in book-v2/assets/demos/chapter0{1,2,3,4}.js; do node --check "$file"; done
```

预期：第一单元内容测试通过，四个脚本语法检查通过。

- [ ] **Step 6: 浏览器验收第一单元**

在 `http://localhost:8000` 逐章验证：

- 第 1 章点击像素后索引、`u/v` 和 RGBA 同步更新。
- 第 2 章拖动点后叉积、法线和夹角同步更新。
- 第 3 章三个滑块改变矩阵和顶点坐标，重置恢复初值。
- 第 4 章相机滑块改变线框投影，读数给出 NDC 与像素坐标。
- 390px 宽度下控件不横向溢出，键盘可操作。

---

### Task 4: 完成第二单元——几何建模

**Files:**
- Modify: `book-v2/content.py`
- Create: `book-v2/assets/demos/chapter05.js`
- Create: `book-v2/assets/demos/chapter06.js`
- Create: `book-v2/assets/demos/chapter07.js`
- Create: `book-v2/assets/demos/chapter08.js`
- Create: `book-v2/assets/demos/chapter09.js`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 第一单元的 `RTBook`、Canvas 坐标换算和 section 模板。
- Produces: 第 5–9 章完整正文，以及 `mesh-demo`、`topology-demo`、`curve-demo`、`surface-demo`、`scene-demo`。

- [ ] **Step 1: 添加第二单元内容契约测试**

在 `tests/test_book.py` 中加入：

```python
UNIT_TWO = {
    5: ("mesh-demo", ("索引网格", "三角形扇")),
    6: ("topology-demo", ("V − E + F = 2", "n_face")),
    7: ("curve-demo", ("de Casteljau", "B_i^n(t)")),
    8: ("surface-demo", ("S(u,v)", "∂S/∂u")),
    9: ("scene-demo", ("p_world = M_model p_local", "AABB")),
}


class UnitTwoContentTests(UnitContentTestCase):
    unit_map = UNIT_TWO
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：第二单元测试失败，第一单元与构建测试仍通过。

- [ ] **Step 3: 写入第 5–9 章正文**

必须覆盖的推导和算例：

- 第 5 章：四边形 `(0,0),(1,0),(1,1),(0,1)` 的 fan 与 strip 三角化；比较索引复用次数。
- 第 6 章：四面体 `V=4,E=6,F=4` 验证 `V−E+F=2`；翻转一个三角形绕序并观察法线反向。
- 第 7 章：四个控制点的三次 Bézier；`t=0.5` 的 de Casteljau 中间级坐标。
- 第 8 章：2×2 控制点双线性曲面；`u=v=0.5` 的采样点与有限差分法线。
- 第 9 章：父节点平移 `(2,0,0)`、子节点旋转 `90°` 后的世界坐标；盒子的 AABB 与视锥判断。

第 5–9 章必须以“浏览器概念实验”或“后续实现路线”标注；只有确实存在的 OBJ、场景或变换文件才链接源码。

- [ ] **Step 4: 实现五个实验脚本**

- `mesh-demo`：同一四边形切换 fan/strip/indexed，画出共享边并显示索引数。
- `topology-demo`：四面体网格切换面法线/顶点法线、翻转绕序，显示 `V/E/F` 与 Euler 结果。
- `curve-demo`：de Casteljau 使用递归或显式三级线性插值；显示每一级点和 `B'(t)`。
- `surface-demo`：显示原始网格与一次细分网格，使用 `RTBook.lerp` 计算边中点；显示顶点/面数量变化。
- `scene-demo`：两个父子节点，滑块控制父平移和子旋转；切换 AABB/包围球并显示世界坐标。

实验必须避免把二维投影说成完整三维 OBJ 查看器。

- [ ] **Step 5: 运行测试与脚本语法检查**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
for file in book-v2/assets/demos/chapter0{5,6,7,8,9}.js; do node --check "$file"; done
```

预期：第二单元内容测试通过，五个脚本语法检查通过。

- [ ] **Step 6: 浏览器验收第二单元**

验证五种切换控件、拖动控制点、法线翻转、曲面细分计数和场景图父子变换；390px 宽度下实验仍显示完整文字读数。

---

### Task 5: 完成第三单元——光栅管线

**Files:**
- Modify: `book-v2/content.py`
- Create: `book-v2/assets/demos/chapter10.js`
- Create: `book-v2/assets/demos/chapter11.js`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 前两单元内容契约、`RTBook`、旧书第 4 章已验证的交互范式。
- Produces: 第 10–11 章完整正文，以及 `raster-demo`、`occlusion-demo`。

- [ ] **Step 1: 添加第三单元内容契约测试**

在 `tests/test_book.py` 中加入：

```python
UNIT_THREE = {
    10: ("raster-demo", ("E(a,b,p)", "λ0")),
    11: ("occlusion-demo", ("1 / w_pixel", "w_pixel < depth")),
}


class UnitThreeContentTests(UnitContentTestCase):
    unit_map = UNIT_THREE
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：第三单元测试失败，前两单元测试通过。

- [ ] **Step 3: 写入第 10–11 章正文**

第 10 章必须逐步推导：

1. 近裁剪面为什么必须在透视除法之前处理。
2. 边函数与有符号面积。
3. 由三个子三角形面积比得到 λ。
4. 像素中心 `(x+0.5,y+0.5)` 与包围盒。
5. 退化三角形的面积保护。

第 11 章必须逐步推导：

1. `1/w` 在屏幕空间线性，`w` 本身不线性。
2. 逆深度插值与取倒数。
3. z-buffer 写入条件。
4. 画家算法为什么随提交顺序改变。
5. 背面剔除、top-left 规则和 MSAA 的概念。

真实源码映射必须链接 `renderer/clipping.py`、`renderer/rasterizer.py`、`renderer/pipeline.py`，并明确当前共享边仍使用容差、未实现严格 top-left 和 MSAA。

- [ ] **Step 4: 实现两个实验脚本**

`chapter10.js` 必须实现：

```javascript
const area = edge(v0, v1, v2);
const l0 = Math.abs(area) < 1e-12 ? NaN : edge(v1, v2, p) / area;
const l1 = Math.abs(area) < 1e-12 ? NaN : edge(v2, v0, p) / area;
const l2 = 1 - l0 - l1;
const covered = Number.isFinite(l0) && l0 >= 0 && l1 >= 0 && l2 >= 0;
```

`chapter11.js` 必须在同一脚本中提供两组并排 Canvas：线性 `w` 与逆深度插值；同时提供 A/B 提交顺序切换和 z-buffer 比较读数。颜色之外必须显示每个像素的 A 深度、B 深度和最终写入对象。

- [ ] **Step 5: 运行测试与脚本语法检查**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
node --check book-v2/assets/demos/chapter10.js
node --check book-v2/assets/demos/chapter11.js
```

预期：第三单元内容测试通过，两个脚本语法检查通过。

- [ ] **Step 6: 浏览器验收第三单元**

- 拖动顶点跨过退化状态时显示“面积过小”，不产生 NaN 读数。
- 交换 A/B 顺序后画家算法结果改变，深度缓冲结果不变。
- 调整远端顶点深度后，线性插值与逆深度插值出现可见差异。
- 键盘可以聚焦顶点与顺序控件，重置恢复初始三角形。

---

### Task 6: 完成第四单元——光照、纹理、阴影与光线追踪

**Files:**
- Modify: `book-v2/content.py`
- Create: `book-v2/assets/demos/chapter12.js`
- Create: `book-v2/assets/demos/chapter13.js`
- Create: `book-v2/assets/demos/chapter14.js`
- Create: `book-v2/assets/demos/chapter15.js`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 前三单元内容契约、`RTBook` 和 Canvas 像素写入方式。
- Produces: 第 12–15 章完整正文，以及 `lighting-demo`、`texture-demo`、`shadow-demo`、`raytrace-demo`。

- [ ] **Step 1: 添加第四单元内容契约测试**

在 `tests/test_book.py` 中加入：

```python
UNIT_FOUR = {
    12: ("lighting-demo", ("diffuse = max(0, N · L)", "H = normalize(L + V)")),
    13: ("texture-demo", ("u_pixel", "w_pixel")),
    14: ("shadow-demo", ("C_linear^(1/2.2)", "C_reinhard")),
    15: ("raytrace-demo", ("p(t) = o + t d", "r = d − 2(d · n)n")),
}


class UnitFourContentTests(UnitContentTestCase):
    unit_map = UNIT_FOUR
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：第四单元测试失败，前三单元测试通过。

- [ ] **Step 3: 写入第 12–15 章正文**

第 12 章：

- 由 `N·L` 推导 Lambert 漫反射。
- 由观察方向与光源方向的半向量推导 Blinn–Phong 高光。
- 用一张贡献分解图解释环境光、漫反射、高光和最终颜色。
- 明确 RenderTorch 当前只有逐面 `abs(N·L)`，没有完整材质和 PBR。

第 13 章：

- 由屏幕 λ 和 `1/w` 推导透视矫正 UV。
- 比较 nearest 与 bilinear 过滤。
- 解释 mipmap 只作为概念，不在本实验中实现 GPU 纹理链。
- 明确当前 RenderTorch 没有 UV/纹理采样实现。

第 14 章：

- 由光源相机和光源投影解释 shadow mapping 的两次渲染。
- 用深度比较与 bias 解释阴影失真。
- 推导 gamma 显示变换和 Reinhard 色调映射。
- 明确当前 RenderTorch 没有阴影和色调映射实现。

第 15 章：

- 由 `p(t)=o+td` 与球面方程推导 ray-sphere 二次方程的两个根。
- 推导镜面反射方向。
- 说明折射、Fresnel 和路径追踪只给概念与公式，不在浏览器实现完整 PBR。
- 用表格比较光栅化与光线追踪，并给出 RenderTorch 的最小后续路线。

- [ ] **Step 4: 实现四个实验脚本**

`chapter12.js` 必须显示每个像素或样本点的 `N`、`L`、`V`、漫反射、高光和最终 RGB；滑杆控制光位、环境光与高光指数。

`chapter13.js` 必须使用程序生成棋盘纹理，不加载外部图片：

```javascript
const checker = ((Math.floor(u * 8) + Math.floor(v * 8)) % 2) === 0;
```

同一三角形分别用屏幕线性 UV 与透视矫正 UV 采样；切换 nearest/linear。

`chapter14.js` 必须用 CPU 生成一张小型 shadow map 深度数组，按光源方向比较方块与球的遮挡；同时显示线性颜色、gamma 后颜色和 Reinhard 后颜色。

`chapter15.js` 必须实现 ray-sphere 求交：

```javascript
const oc = subtract(origin, center);
const a = dot(direction, direction);
const b = 2 * dot(oc, direction);
const c = dot(oc, oc) - radius * radius;
const discriminant = b * b - 4 * a * c;
if (discriminant >= 0) {
  const t = (-b - Math.sqrt(discriminant)) / (2 * a);
}
```

浏览器实验只追踪球与平面，最大反射次数限制为 0–3，光线数限制为 1–4；用户操作后才重绘。

- [ ] **Step 5: 运行测试与脚本语法检查**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
for file in book-v2/assets/demos/chapter{12,13,14,15}.js; do node --check "$file"; done
```

预期：第四单元内容测试通过，四个脚本语法检查通过。

- [ ] **Step 6: 浏览器验收第四单元**

- 光照实验各项贡献之和与显示颜色一致。
- 纹理实验在改变顶点深度后出现正确/错误 UV 差异。
- 阴影实验移动光源后 shadow map 与最终阴影同步变化。
- 光线追踪实验在最大反射次数为 0 时只显示直接颜色，增加次数后出现反射。
- 全部实验在 390px 宽度下可读、可键盘操作。

---

### Task 7: 全书链接、资源、许可与可访问性验收

**Files:**
- Modify: `tests/test_book.py`
- Modify: `book-v2/build.py`
- Modify: `book-v2/assets/style.css`
- Modify: `book-v2/assets/book.js`

**Interfaces:**
- Consumes: 15 章完整内容和 15 个实验脚本。
- Produces: 可重复执行的全书自动验收和浏览器验收清单。

- [ ] **Step 1: 添加本地链接与资源测试**

在 `tests/test_book.py` 中加入：

```python
from html.parser import HTMLParser


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.assets = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])
        if tag in {"script", "img"} and "src" in values:
            self.assets.append(values["src"])
        if tag == "link" and "href" in values:
            self.assets.append(values["href"])


class BuiltSiteTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")
        cls.build = load_module("book_v2_build", BOOK_DIR / "build.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.build.build(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class LinkAndAssetTests(BuiltSiteTestCase):
    def test_every_local_link_and_asset_resolves(self):
        for page in self.output.rglob("*.html"):
            parser = AssetParser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in parser.links + parser.assets:
                if reference.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                path = reference.split("#", 1)[0].split("?", 1)[0]
                target = (page.parent / path).resolve()
                self.assertTrue(target.exists(), (page, reference))

    def test_pages_do_not_load_external_runtime_assets(self):
        for page in self.output.rglob("*.html"):
            parser = AssetParser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in parser.assets:
                self.assertFalse(
                    reference.startswith(("http://", "https://")),
                    (page, reference),
                )
```

- [ ] **Step 2: 运行测试并确认失败或定位问题**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

若失败，只修复真实的本地路径、资源引用或模板错误，不放宽断言。

- [ ] **Step 3: 添加页面结构与实验契约检查**

继续加入：

```python
class PageStructureTests(BuiltSiteTestCase):
    def test_chapter_pages_have_one_main_and_unique_demo_ids(self):
        seen = set()
        for row in self.content.chapters():
            page = self.output / "chapters" / f"{row.slug}.html"
            text = page.read_text(encoding="utf-8")
            self.assertEqual(text.count('id="main-content"'), 1)
            for demo in row.demos:
                self.assertNotIn(demo.id, seen)
                seen.add(demo.id)
                self.assertIn(f'id="{demo.id}"', text)
                self.assertIn(demo.script, text)
```

- [ ] **Step 4: 运行全部 Python 测试**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
```

预期：元数据、参考资料、构建、四个单元、链接、资源和页面结构测试全部通过。

- [ ] **Step 5: 检查所有 JavaScript 语法**

```bash
find book-v2/assets -name '*.js' -print0 | xargs -0 -n1 node --check
```

预期：无输出，退出码为 0。

- [ ] **Step 6: 执行浏览器全站验收**

启动本地服务器：

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m http.server 8000 --directory book-v2
```

逐项确认：

- 首页、参考资料页和 15 章均无控制台错误。
- 1440px 与 390px 两种宽度无正文横向溢出。
- 移动目录打开后焦点可进入，关闭后 `aria-expanded="false"`。
- 所有实验有重置、文字读数和键盘操作。
- 所有 `aria-live` 读数不会在页面加载时疯狂刷新。
- `prefers-reduced-motion: reduce` 下无自动动画。
- 打印预览保留公式和正文，隐藏工具栏与翻页导航。
- 页面不发起任何外部资源请求。

完成后停止临时服务器。

---

### Task 8: 备份旧书、替换 `book/` 并更新仓库文档

**Files:**
- Move: `book-v2/` → `book/`
- Modify: `tests/test_book.py`
- Modify: `README.md`
- Modify: `docs/references.md`

**Interfaces:**
- Consumes: 任务 7 已通过的 `book-v2/`。
- Produces: 正式 `book/`、默认指向 `book/` 的测试、更新后的构建说明和参考资料索引。

- [ ] **Step 1: 再次运行完整验收**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
find book-v2/assets -name '*.js' -print0 | xargs -0 -n1 node --check
```

只有全部通过才继续。

- [ ] **Step 2: 备份旧书到仓库外**

```bash
mkdir -p ../RenderTorch-book-legacy-backup/2026-11-26
test ! -e ../RenderTorch-book-legacy-backup/2026-11-26/book
test ! -e ../RenderTorch-book-legacy-backup/2026-11-26/book-replaced
cp -a book ../RenderTorch-book-legacy-backup/2026-11-26/book
diff -qr book ../RenderTorch-book-legacy-backup/2026-11-26/book
```

预期：`diff -qr` 无输出。若输出任何差异，停止替换并调查。

- [ ] **Step 3: 替换目录**

```bash
mv book ../RenderTorch-book-legacy-backup/2026-11-26/book-replaced
mv book-v2 book
```

旧书完整保留在仓库外；新书成为仓库中的 `book/`。

- [ ] **Step 4: 更新测试默认路径**

把 `tests/test_book.py` 中的：

```python
BOOK_DIR = Path(os.environ.get("BOOK_DIR", "book-v2"))
```

改为：

```python
BOOK_DIR = Path(os.environ.get("BOOK_DIR", "book"))
```

- [ ] **Step 5: 更新 README**

将教学书段落改为：

- 15 章中文计算机图形学教材，四单元结构。
- 每章包含推导、算例、伪代码、真实源码映射和 Canvas/SVG 实验。
- 构建与运行命令保持：

```bash
python book/build.py
python -m http.server 8000 --directory book
```

- 明确当前线上首先发布 3D 子书，物理子书将在后续阶段重构。
- 发布命令保持正常 `git subtree push --prefix=book origin gh-pages`，不使用强制推送。

- [ ] **Step 6: 更新参考资料索引**

在 `docs/references.md` 中增加“v2 图形学子书参考资料”小节，列出任务 1 的参考资料和许可边界；保留现有物理参考资料，不删除。

- [ ] **Step 7: 重建并运行正式路径测试**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python book/build.py
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -p test_book.py -v
find book/assets -name '*.js' -print0 | xargs -0 -n1 node --check
```

预期：全部通过。

- [ ] **Step 8: 检查替换结果**

```bash
git -C RenderTorch status --short
git -C RenderTorch diff --stat
```

确认旧 12 章文件已从 `book/` 移除，新 15 章文件存在；不得出现 `book-v2/` 残留。

---

### Task 9: 最终审查、提交并发布 GitHub Pages

**Files:**
- Stage only: `README.md`、`book/`、`tests/test_book.py`、`docs/references.md`、设计规格、实施计划
- Exclude: `main.py`、`main2.py`、`render.py`、`solid/object.py`、`viewing/camera.py`、`viewing/projection.py`、`tests/test_rendering.py`、所有 `__pycache__/`

**Interfaces:**
- Consumes: 任务 8 的正式 `book/` 和全部测试结果。
- Produces: `feat/m1-qt-minimal-loop` 新提交、更新后的 `gh-pages`、可访问的 GitHub Pages 首页。

- [ ] **Step 1: 检查完整测试和差异**

```bash
/opt/anaconda3/bin/conda run -n pythonTry python -m unittest discover -s tests -v
find book/assets -name '*.js' -print0 | xargs -0 -n1 node --check
git -C RenderTorch diff --check
git -C RenderTorch status --short
git -C RenderTorch diff --stat
```

必须确认全套测试通过；只允许暂存教材相关改动。

- [ ] **Step 2: 暂存明确文件**

```bash
git -C RenderTorch add README.md book tests/test_book.py docs/references.md \
  docs/superpowers/specs/2026-11-26-rendertorch-graphics-book-v2-design.md \
  docs/superpowers/plans/2026-11-26-rendertorch-graphics-book-v2.md
```

随后检查：

```bash
git -C RenderTorch status --short
git -C RenderTorch diff --cached --stat
```

若出现任何非教材文件，立即从暂存区移除该文件，不删除工作区内容。

- [ ] **Step 3: 创建提交**

```bash
git -C RenderTorch commit -m "$(cat <<'EOF'
docs: rebuild graphics textbook with interactive lessons
EOF
)"
```

- [ ] **Step 4: 推送源分支**

```bash
git -C RenderTorch push origin feat/m1-qt-minimal-loop
```

不使用强制推送。

- [ ] **Step 5: 正常发布 GitHub Pages**

```bash
git -C RenderTorch subtree push --prefix=book origin gh-pages
```

若远端拒绝推送，停止并检查分支历史；不得用 `--force` 绕过。

- [ ] **Step 6: 核验 Pages 状态与首页**

```bash
gh api repos/Zhenghong-Liu/RenderTorch/pages
```

确认 `status` 为 `built`。然后访问：

```text
https://zhenghong-liu.github.io/RenderTorch/
```

确认首页显示 15 章、参考资料页可打开、第 1 章实验资源无 404。

- [ ] **Step 7: 最终工作区检查**

```bash
git -C RenderTorch status --short
```

确认提交成功；保留所有未被本任务授权的原有未提交修改，不做清理。

---

## Plan Self-Review

- 设计规格中的 15 章分别由任务 3–6 实现，无遗漏章节。
- 参考资料和许可由任务 1 实现，参考资料页由任务 2 实现。
- 每章实验由任务 3–6 的独立 JavaScript 文件实现。
- 实现状态分层由任务 3–6 的测试强制检查。
- 本地链接、资源、页面结构和许可字段由任务 7 检查。
- 旧书备份与替换由任务 8 执行，且发生在全部验收之后。
- 源分支与 `gh-pages` 的正常发布由任务 9 执行。
- 计划不包含占位符、未知接口或未定义文件路径。
