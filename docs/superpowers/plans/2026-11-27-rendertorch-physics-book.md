# RenderTorch 物理模拟子书实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 10 章、至少 20 个互动实验的高中可读物理模拟子书，并让图形学 15 章增加编号小节与章内导航。

**Architecture:** 内容按章拆成独立 Python 数据模块，构建器复用现有静态页面与本地 KaTeX；浏览器实验使用共享的可测试数值核心和 Canvas 组件，避免每个 Demo 重复实现积分、约束和碰撞。

**Tech Stack:** Python 3.11 标准库、HTML5、CSS3、原生 JavaScript、Canvas 2D、SVG、Node.js、unittest、GitHub Pages。

**Spec:** [RenderTorch 物理模拟子书设计](2026-11-27-rendertorch-physics-book-design.md)

## Global Constraints

- 不引入 WebGL、Three.js、React、Matter.js 运行时、npm 构建或外部 CDN。
- 所有新公式使用 LaTeX `\(...\)` 或 `\[...\]`，由本地 KaTeX 0.16.47 渲染。
- 每章至少 10 个编号正文小节、2 个独立实验、1 组数值算例、伪代码、自测、调试和实现状态。
- 每个实验必须有重置、可操作控件、`aria-live="polite"` 文字读数和至少一个诊断指标。
- 真实源码只能链接当前存在的 `physics/`、`app/` 文件；浏览器实验必须标为概念实验。
- 时间步必须写明 Python 默认 `fixed_dt=1/120`、`substeps=2`、实际 `h=1/240`；误差统一讲最大相对误差。
- 面积约束、旋转刚体、CCD、自碰撞只能标为概念实验；不得复刻公共粒子碰撞的分离接触恢复或旧法向速度切向计算。
- 不改写现有 Python 物理实现，不修改工作区既有未提交文件，不清理未跟踪缓存。
- 所有新/改 JavaScript 必须通过 `node --check`。
- 只在全部验收完成后暂存教材相关文件；不使用 `git add -A`、`git reset --hard`、`git clean` 或强制推送。
- 发布使用正常源分支推送和 `git subtree push --prefix=book origin gh-pages`。
- Python 使用 `/opt/anaconda3/envs/pythonTry/bin/python`，完整测试使用 `QT_QPA_PLATFORM=offscreen`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `book/layout.py` | 两书共用的 HTML 页面骨架、页眉、页脚、转义和脚本标签 |
| `book/build.py` | 图形学构建、总入口、小节编号、章内导航和物理子书链接 |
| `book/physics_build.py` | 物理首页、10 章页面、物理参考资料页 |
| `book/physics_content.py` | 元数据、按文件加载章节详情、构造 `Chapter` 对象 |
| `book/physics_content_chapter01.py` 至 `physics_content_chapter10.py` | 每章独立正文、公式、实验和源码规格 |
| `book/assets/physics-core.js` | Node/浏览器共用纯数值函数 |
| `book/assets/physics.js` | Canvas、控件、固定步长、暂停/单步与指标组件 |
| `book/assets/demos/physics01.js` 至 `physics20.js` | 每章两个独立互动实验 |
| `book/assets/style.css` | 编号小节、章内导航、物理实验和移动布局 |
| `tests/test_book.py` | 保留图形学契约，增加小节编号与章内导航测试 |
| `tests/test_physics_book.py` | 物理内容、构建、公式、资源、Demo 和数值契约 |
| `README.md`、`docs/references.md` | 子书入口、构建方式、章节数量和许可边界 |
| `docs/superpowers/specs/2026-11-27-rendertorch-physics-book-design.md` | 设计依据 |
| `docs/superpowers/plans/2026-11-27-rendertorch-physics-book.md` | 本计划 |

## Interfaces

### Python 内容模块

`physics_content.py` 提供：

```python
def chapters() -> list[Chapter]: ...
def references() -> list[dict]: ...
```

每个 `physics_content_chapterNN.py` 只提供：

```python
CHAPTER = {
    "number": 1,
    "slug": "physics01",
    "unit": "单元一：运动会怎样发生",
    "title": "从真实运动到数字状态",
    "summary": "...",
    "objectives": ("...", "..."),
    "prerequisites": ("...",),
    "sections": (
        ("state-intuition", "小球也是一组数字", "可观察的运动", "<p>...</p>"),
    ),
    "demos": (
        ("particle-lab", "拖动质点与速度向量", "demos/physics01.js", "description", "state-vectors"),
    ),
    "sources": (
        ("physics/common/state.py", "现有实现：质点位置、速度与逆质量"),
    ),
}
```

`section_id` 决定 Demo 插入哪个正文小节；`sources` 在聚合时转换为 `SourceLink`。

### 共享页面布局

`book/layout.py` 提供：

```python
def escape(value: str) -> str: ...
def write(path: Path, text: str) -> None: ...
def header(brand: str, prefix: str = "", meta: str = "PIXELS · GEOMETRY · RENDERING", has_sidebar: bool = False) -> str: ...
def footer(brand: str) -> str: ...
def document(title: str, body: str, scripts=(), prefix: str = "", sidebar_html: str = "", header_html: str = "", footer_html: str = "") -> str: ...
```

`build.py` 与 `physics_build.py` 只负责各自内容，不重复整套 HTML 骨架。

### JavaScript 数值核心

`physics-core.js` 以 UMD 形式暴露 `RTPhysics`，Node 可 `require`，浏览器挂到 `window.RTPhysics`：

```javascript
RTPhysics.vec2 = { add, sub, scale, dot, length, normalize };
RTPhysics.integrateProjectile(state, gravity, dt, method);
RTPhysics.springForce(extension, velocityAlongSpring, stiffness, damping);
RTPhysics.projectDistance(a, b, restLength, stiffness = 1);
RTPhysics.projectDistanceXPBD(a, b, restLength, compliance, dt, lambda);
RTPhysics.collideSpherePlane(particle, radius, planeY, restitution, friction);
RTPhysics.collideParticleSphere(particle, center, circleRadius, particleRadius, restitution, friction);
RTPhysics.energy1d(mass, velocity, stiffness, displacement);
RTPhysics.stableSpringStep(stiffness, mass);
RTPhysics.projectAreaXPBD(p0, p1, p2, p3, restArea, compliance, dt, lambda);
```

对象均使用 `{x, y, vx, vy, w}` 或明确返回新对象，不修改全局状态。`collideParticleSphere` 返回新粒子；`projectAreaXPBD` 返回 `{particles: [p0,p1,p2,p3], lambda}`，用有向 shoelace 面积配合输入旋向归一化几何面积，乘子在同一子步内累积、跨子步清零。

### 浏览器组件

`physics.js` 提供：

```javascript
window.RTPhysicsUI = {
  mount(rootId, options),
  reset(),
  step(),
  setRunning(running),
  draw(),
  report(text)
};
```

`mount(rootId, options)` 的 options 固定为 `{createState, controls, update, draw, metrics}`：`createState()` 返回可复现初始状态；`controls` 是 `{key,label,min,max,step,value}` 数组；`update(state, dt, values)` 推进一步；`draw(ctx, state, width, height)` 使用 CSS 像素坐标绘制；`metrics(state)` 返回至少一项 `{标签: 有限值}`。它返回独立控制器 `{canvas, getState, reset, step, setRunning, draw, report}`；每个 Demo 持有自己的控制器。顶层同名方法仅委托给最近挂载的控制器，跨 Demo 不共享状态。控件变化后刷新读数；reset 恢复控件默认值和初始状态。

每个 `physicsNN.js` 在 `DOMContentLoaded` 后查找自己的 `[data-demo]`，创建 Canvas、控件和读数，并注册重置。

---

### Task 1: 建立共享布局与图形学小节编号

**Files:**
- Create: `book/layout.py`
- Modify: `book/build.py`
- Modify: `book/assets/style.css`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 现有 `content.py` 的 `Section`、`Chapter`。
- Produces: 两书共用 `document()`；图形学章节自动编号与 `chapter-section-nav`。

- [ ] **Step 1: 在 `tests/test_book.py` 增加失败测试**

```python
def test_chapters_have_numbered_sections_and_in_chapter_navigation(self):
    page = (self.output / "chapters" / "chapter01.html").read_text(encoding="utf-8")
    self.assertIn('class="chapter-section-nav"', page)
    self.assertIn("1.1", page)
    self.assertIn("1.2", page)
    self.assertIn('href="#intuition"', page)
```

增加总入口物理链接测试：

```python
def test_home_links_to_physics_subbook(self):
    home = (self.output / "index.html").read_text(encoding="utf-8")
    self.assertIn('href="physics/index.html"', home)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_book.BuildContractTests.test_chapters_have_numbered_sections_and_in_chapter_navigation tests.test_book.BuildContractTests.test_home_links_to_physics_subbook -v
```

Expected: 两个测试因缺少标记而失败。

- [ ] **Step 3: 提取共享布局**

将 `build.py` 的 `write`、`escape`、`header`、`footer`、`document` 移入 `layout.py`，保留图形学专用 `navigation`、`sidebar`、`render_*` 在 `build.py`。`document()` 接受完整 `sidebar_html`、`header_html`、`footer_html`，默认行为保持现有页面结构。

- [ ] **Step 4: 实现自动编号与章内导航**

在 `render_chapter()` 中按实际输出顺序计数：学习目标、正文小节、源码映射、实验。`render_section()` 接收 `number` 并输出 `1.1` 标签；在章节头部生成指向每个 section id 的 `<nav class="chapter-section-nav">`。总入口在 hero 后增加物理子书卡片，链接 `physics/index.html`。

- [ ] **Step 5: 增加样式**

```css
.chapter-section-nav { display: flex; flex-wrap: wrap; gap: .45rem; margin: 0 0 2rem; }
.chapter-section-nav a { padding: .3rem .65rem; border: 1px solid var(--line); border-radius: 999px; text-decoration: none; font-size: .78rem; }
.subbook-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
```

移动端 `.subbook-grid` 改为单列，打印时保留编号和导航文字。

- [ ] **Step 6: 运行图形学测试并检查产物**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_book -v
/opt/anaconda3/envs/pythonTry/bin/python book/build.py
```

Expected: 图形学测试全部通过，首页出现物理入口，章节出现 `1.1` 与章内导航。

---

### Task 2: 用失败数值测试建立物理核心

**Files:**
- Create: `book/assets/physics-core.js`
- Create: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: 无。
- Produces: Task 3–12 全部 Demo 共用的纯函数。

- [ ] **Step 1: 写 Node 数值测试**

`tests/test_physics_book.py` 使用 `subprocess.run(["node", "-e", script, core_path])`，脚本直接 `require` 核心并输出 JSON。首批断言：

```javascript
const assert = require('node:assert/strict');
const P = require(process.argv[1]);
const exact = P.integrateProjectile({x:0,y:0,vx:10,vy:20}, {x:0,y:-9.81}, 1, 'exact');
assert.ok(Math.abs(exact.x - 10) < 1e-12);
assert.ok(Math.abs(exact.y - 15.095) < 1e-12);

const pbd = P.projectDistance(
  {x:0,y:0,w:1}, {x:3,y:0,w:1}, 2, 1
);
assert.ok(Math.abs(pbd.a.x - 0.5) < 1e-12);
assert.ok(Math.abs(pbd.b.x - 2.5) < 1e-12);

const fixed = P.projectDistance(
  {x:0,y:0,w:0}, {x:3,y:0,w:1}, 2, 1
);
assert.equal(fixed.a.x, 0);
assert.ok(Math.abs(fixed.b.x - 2) < 1e-12);
```

- [ ] **Step 2: 运行测试确认失败**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_physics_book.PhysicsCoreTests -v
```

Expected: `physics-core.js` 不存在或 `RTPhysics` 接口缺失。

- [ ] **Step 3: 实现积分与向量**

实现二维向量加减、数乘、点积、长度、归一化。`integrateProjectile` 支持 `forward`、`semi-implicit`、`exact`；exact 使用 `x=vx*t`、`y=vy*t+0.5*g*t^2` 且重力恒定。

- [ ] **Step 4: 实现弹簧、能量与稳定步长**

```javascript
springForce(extension, relativeVelocity, k, c) = -k * extension - c * relativeVelocity
energy1d(m, v, k, x) = {kinetic: 0.5*m*v*v, potential: 0.5*k*x*x}
stableSpringStep(k, m) = 2 / Math.sqrt(k/m)
```

- [ ] **Step 5: 实现 PBD 单次投影**

`C=|b-a|-L0`，`n=(b-a)/|b-a|`，`denom=wa+wb`。`stiffness` 缩放校正；两端都固定或长度接近零时返回原状态。固定点 `w=0` 不移动。

- [ ] **Step 6: 实现 XPBD 增量投影**

`alpha_tilde=alpha/(dt*dt)`，`delta_lambda=(-C-alpha_tilde*lambda)/(denom+alpha_tilde)`；位置修正使用 `-w*delta_lambda*n` 与 `w*delta_lambda*n`。返回新 `lambda`。`alpha=0` 且无迭代历史时退化为 PBD。

- [ ] **Step 7: 实现球体平面碰撞与 Node 扩展断言**

碰撞返回位置、法向速度和切向速度。仅当法向相对速度指向平面内部（向下）时应用恢复；位置投影到 `planeY+radius`；接近时法向速度乘以 `-restitution`；切向速度乘以 `max(0, 1-friction)` 作为教学简化，并明确不是完整库仑模型。已经在向外运动的分离接触不做恢复。

追加测试：

```javascript
const soft = P.projectDistanceXPBD({x:0,y:0,w:1},{x:3,y:0,w:1},2,1,1/60,0);
const hard = P.projectDistanceXPBD({x:0,y:0,w:1},{x:3,y:0,w:1},2,0,1/60,0);
assert.ok(Math.abs((soft.b.x-soft.a.x)-2) > Math.abs((hard.b.x-hard.a.x)-2));
const hit = P.collideSpherePlane({x:0,y:0.2,vx:4,vy:-5,w:1},0.5,0,0.5,0.2);
assert.equal(hit.y, 0.5);
assert.equal(hit.vy, 2.5);
assert.equal(hit.vx, 3.2);
```

- [ ] **Step 8: 用失败测试补充静态圆碰撞与面积 XPBD**

```javascript
const circle = P.collideParticleSphere(
  {x:0,y:0.2,vx:4,vy:-5,w:1}, {x:0,y:0}, 0.3, 0.5, 0.5, 0.2
);
assert.equal(circle.y, 0.8);
assert.equal(circle.vy, 2.5);
assert.equal(circle.vx, 3.2);
const separating = P.collideParticleSphere(
  {x:0,y:0.9,vx:4,vy:2,w:1}, {x:0,y:0}, 0.3, 0.5, 0.5, 0.2
);
assert.equal(separating.y, 0.9);
assert.equal(separating.vy, 2);
assert.equal(separating.vx, 4);

const square = [
  {x:0,y:0,w:1}, {x:1,y:0,w:1},
  {x:1,y:1,w:1}, {x:0,y:1,w:1},
];
const stretched = square.map((p, i) => i === 1 || i === 2 ? {...p, x:2} : {...p});
const areaHard = P.projectAreaXPBD(...stretched, 1, 0, 1/60, 0);
const areaSoft = P.projectAreaXPBD(...stretched, 1, 1, 1/60, 0);
const areaOf = (points) => Math.abs(points.reduce((sum, p, i) => {
  const q = points[(i + 1) % points.length];
  return sum + p.x*q.y - q.x*p.y;
}, 0) / 2);
assert.ok(Math.abs(areaOf(areaHard.particles) - 27 / 25) < 1e-12);
assert.ok(Math.abs(areaOf(areaSoft.particles) - 1) > Math.abs(areaOf(areaHard.particles) - 1));
const clockwise = P.projectAreaXPBD(
  stretched[0], stretched[3], stretched[2], stretched[1], 1, 0, 1/60, 0
);
assert.ok(Math.abs(areaOf(clockwise.particles) - 27 / 25) < 1e-12);
const zeroDistance = P.projectDistanceXPBD(
  {x:0,y:0,w:1}, {x:3,y:0,w:1}, 2, 0, 0, 0.25
);
const zeroArea = P.projectAreaXPBD(...stretched, 1, 1, 0, 0.25);
assert.equal(zeroDistance.a.x, 0);
assert.equal(zeroDistance.b.x, 3);
assert.equal(zeroDistance.lambda, 0.25);
assert.deepEqual(zeroArea.particles, stretched);
assert.equal(zeroArea.lambda, 0.25);
```

先运行并确认这两组 API 缺失导致失败，再实现：静态圆法线由圆心指向粒子，仅接近时应用恢复与切向衰减，分离接触只做必要的位置投影；面积梯度为
`g0=((y1-y3)/2,(x3-x1)/2)`、`g1=((y2-y0)/2,(x0-x2)/2)`、`g2=((y3-y1)/2,(x1-x3)/2)`、`g3=((y0-y2)/2,(x2-x0)/2)`，使用与距离 XPBD 相同的增量乘子更新。面积约束按几何面积处理：`orientation = signedArea < 0 ? -1 : 1`，`C=signedArea-orientation*restArea`，因此顺时针与逆时针输入都保持面积。`dt=0` 时距离和面积 XPBD 返回克隆状态和原乘子，不产生 `0/0`。该拉伸方形单次投影的独立手算面积为 `27/25`，不是 1；面积约束是非线性的，需要迭代收敛，Demo 不得宣称一次投影精确恢复任意形变。

- [ ] **Step 9: 运行核心测试与语法检查**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_physics_book.PhysicsCoreTests -v
node --check book/assets/physics-core.js
```

Expected: 全部通过。

---

### Task 3: 建立物理 Demo 公共组件与构建契约

**Files:**
- Create: `book/assets/physics.js`
- Create: `book/physics_content.py`
- Create: `book/physics_build.py`
- Create: `tests/test_physics_book.py`
- Modify: `book/content.py`
- Modify: `book/build.py`
- Modify: `book/assets/style.css`

**Interfaces:**
- Consumes: Task 1 `layout.document`，Task 2 `RTPhysics`。
- Produces: `physics_build.build(output_dir)` 与 `RTPhysicsUI.mount()`。

- [ ] **Step 1: 写物理构建失败测试**

本章数据在 Task 4–13 才创建，因此本任务用测试内的 `Chapter` fixture 验证渲染器，不提前要求 10 章正文存在：

```python
def test_physics_pages_embed_demos_in_sections(self):
    fixture = Chapter(
        number=1,
        slug="physics01",
        unit="单元一：运动会怎样发生",
        title="测试章",
        summary="用于验证构建器。",
        objectives=("识别状态",),
        prerequisites=(),
        sections=(
            Section("state-vectors", "速度向量", "可观察的运动", "<p>速度有方向和大小。</p>"),
            Section("units", "单位", "米每秒", "<p>速度单位是米每秒。</p>"),
        ),
        demos=(DemoSpec("particle-lab", "质点实验", "demos/physics01.js", "拖动质点。", "state-vectors"),),
        sources=(),
    )
    text = physics_build.render_chapter(fixture)
    self.assertIn('class="chapter-section-nav"', text)
    self.assertIn("1.1", text)
    self.assertIn("1.2", text)
    self.assertIn('id="particle-lab"', text)
    self.assertIn("../assets/physics-core.js", text)
    self.assertIn("../assets/physics.js", text)
    demo_at = text.index('id="particle-lab"')
    section_at = text.index('id="state-vectors"')
    self.assertLess(section_at, demo_at)
    scripts = [text.index(path) for path in (
        "../assets/vendor/katex/katex.min.js",
        "../assets/vendor/katex/auto-render.min.js",
        "../assets/book.js",
        "../assets/physics-core.js",
        "../assets/physics.js",
        "../assets/demos/physics01.js",
    )]
    self.assertEqual(scripts, sorted(scripts))
```

```python
def test_graphics_build_preserves_existing_physics_home(self):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        home = output / "physics" / "index.html"
        home.parent.mkdir(parents=True)
        home.write_text("formal physics home", encoding="utf-8")
        graphics_build.build(output)
        self.assertEqual(home.read_text(encoding="utf-8"), "formal physics home")
```

- [ ] **Step 2: 运行测试确认失败**

Expected: `physics_content.py`、`physics_build.py` 不存在或 `render_chapter` 接口缺失。

- [ ] **Step 3: 实现章节聚合器**

先在 `content.py` 的 `DemoSpec` 末尾增加默认字段 `section_id: str = ""`，保持四个既有位置参数与图形学数据兼容。`physics_content.py` 用 `importlib.util.spec_from_file_location` 按顺序加载当前已存在的 `physics_content_chapter01.py` 至 `chapter10.py`，将原始 tuple 转为 `Section`、`DemoSpec`、`SourceLink`、`Chapter`。`chapters()` 在中间任务返回已创建章节，Task 14 的最终契约要求恰好 10 章。`references()` 在资料文件存在时读取并返回记录，否则返回空列表供占位页使用。聚合与渲染时拒绝重复 section id、重复 Demo id，以及 Demo 指向不存在 section 的情况；单章最终 HTML 的保留 ID（`objectives`、`sources`）、section ID 与 Demo ID 合起来也必须唯一，章节之间允许重复 section ID。

- [ ] **Step 4: 实现物理构建器**

物理首页按四个单元展示当前已创建章节；侧栏包含物理首页、已创建章节和物理参考资料。章节页为学习目标、正文、源码映射和实验按实际输出顺序自动编号，生成 `chapter-section-nav`；Demo 插入 `section_id` 对应小节的 `.prose` 后。脚本顺序为 KaTeX、auto-render、book.js、physics-core.js、physics.js、本章 Demo；后三项由构建器传入 `layout.document()`，避免重复脚本。`build()` 同时生成物理首页、章节和临时参考资料占位页，并把 `book/assets/` 复制到临时输出根，确保中间任务的侧栏链接和本地资源有效；Task 14 用真实资料替换占位页。修改 `book/build.py`：仅在物理首页不存在时写 Task 1 占位页，避免图形学构建覆盖正式物理首页。

- [ ] **Step 5: 实现公共 Demo 组件**

`RTPhysicsUI.mount` 负责：Canvas 高分屏适配、坐标转换、范围控件与输出绑定、重置、暂停、单步、requestAnimationFrame 固定步长、有限值检查、`data-readout aria-live="polite"`。reduced-motion 时默认暂停，仍允许单步。options 固定为 `{createState, controls, update, draw, metrics}`：`createState()` 返回初始状态；`controls` 为 `{key,label,min,max,step,value}` 数组；`update(state, dt, values)` 推进一步；`draw(ctx, state, width, height)` 使用 CSS 像素坐标；`metrics(state)` 返回至少一项有限读数。`mount` 返回独立控制器 `{canvas, getState, reset, step, setRunning, draw, report}`；每个 Demo 持有自己的控制器，顶层同名方法只委托最近挂载实例。控件变化刷新读数，reset 同时恢复默认控件和状态；非有限状态暂停并显示重置提示。

- [ ] **Step 6: 扩展样式**

增加 `.physics-demo`、`.metric-grid`、`.metric`、`.legend`、`.energy-bar`、`.comparison-grid` 等少量语义类；390px 下所有网格使用 `minmax(0,1fr)` 或单列，长读数 `overflow-wrap:anywhere`。

- [ ] **Step 7: 运行构建契约测试**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_physics_book.PhysicsBuildContractTests -v
node --check book/assets/physics.js
```

Expected: 接口、脚本顺序和小节插入测试通过。

---

### Task 4: 第 1 章 从真实运动到数字状态

**Files:**
- Create: `book/physics_content_chapter01.py`
- Create: `book/assets/demos/physics01.js`
- Create: `book/assets/demos/physics02.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: Task 2 `RTPhysics.vec2`，Task 3 `RTPhysicsUI.mount`。
- Produces: `physics01` 章与 `particle-lab`、`frame-states` 实验；`section_id` 分别为 `velocity-vector`、`discrete-frames`。

- [ ] **Step 1: 写章节失败测试**

`PhysicsChapterContentTests.setUpClass` 建立统一临时构建环境，后续各章测试复用同一约定：

```python
@classmethod
def setUpClass(cls):
    cls.tempdir = tempfile.TemporaryDirectory()
    cls.addClassCleanup(cls.tempdir.cleanup)
    cls.output = Path(cls.tempdir.name)
    physics_build.build(cls.output)
    cls.rows = {row.number: row for row in physics_content.chapters()}
```

```python
def test_chapter_one_teaches_state_vectors_and_units(self):
    row = self.rows[1]
    text = (self.output / "physics" / "chapters" / "physics01.html").read_text(encoding="utf-8")
    for marker in ("质点", "单位", "速度", "\\mathbf v", "状态"):
        self.assertIn(marker, text)
    self.assertEqual({demo.id for demo in row.demos}, {"particle-lab", "frame-states"})
```

- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 11 个小节**

必须包含：小球数字化、米/秒单位、二维坐标、质点假设、`p/v/a` 状态、速度向量、离散帧、拖拽初值、可复现、算例 `p0=(1,2), v=(2,-1), dt=0.5 → p=(2,1.5)`、伪代码、自测和调试。所有公式写为 LaTeX。
- [ ] **Step 4: 实现 `particle-lab`**

Canvas 显示坐标网格、可拖拽小球、速度向量和预测下一帧位置。控件：速度 x、速度 y、时间步。读数：位置、速度、下一帧位置、速度大小。
- [ ] **Step 5: 实现 `frame-states`**

显示 5 帧状态表、实际帧位置和线性插值位置。控件：播放/暂停、单步、帧率、时间缩放。读数：帧号、模拟时间、当前状态、插值误差。
- [ ] **Step 6: 运行章节测试与语法检查**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_physics_book.PhysicsChapterContentTests.test_chapter_one_teaches_state_vectors_and_units -v
node --check book/assets/demos/physics01.js
node --check book/assets/demos/physics02.js
```

---

### Task 5: 第 2 章 速度、加速度与时间积分

**Files:**
- Create: `book/physics_content_chapter02.py`
- Create: `book/assets/demos/physics03.js`
- Create: `book/assets/demos/physics04.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `RTPhysics.integrateProjectile`。
- Produces: `integration-race`、`fixed-step`；`section_id` 分别为 `exact-projectile`、`fixed-timestep`。

- [ ] **Step 1: 写失败测试**：要求 `physics02` 含 `前向欧拉`、`半隐式欧拉`、`固定时间步`、`a=\\frac{\\Delta v}{\\Delta t}`，且 Demo ID 精确匹配。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 11 个小节**：加速度、牛顿第二定律、速度更新、前向欧拉、半隐式欧拉、精确自由落体、离散误差、固定步长、累加器、算例 `g=-9.81, v0=20, dt=0.1` 的首帧速度与位置、伪代码和调试。必须解释 RenderTorch 默认 `fixed_dt=1/120`、`substeps=2`，实际积分子步 `h=1/240`，并说明暂停时仍可单步。
- [ ] **Step 4: 实现 `integration-race`**：同一抛物线叠加 forward/semi-implicit/exact 三条轨迹；控件 dt、初速度、重力；读数终点位置和两条数值误差。
- [ ] **Step 5: 实现 `fixed-step`**：并排显示“每帧直接 dt”与“累加器固定 1/120 s”；控件显示帧率、固定步长、卡顿帧；读数模拟时间、子步数、最大追帧误差。
- [ ] **Step 6: 运行测试以及 `node --check book/assets/demos/physics03.js` 和 `node --check book/assets/demos/physics04.js`**

---

### Task 6: 第 3 章 力、能量与阻尼

**Files:**
- Create: `book/physics_content_chapter03.py`
- Create: `book/assets/demos/physics05.js`
- Create: `book/assets/demos/physics06.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `RTPhysics.springForce`、`energy1d`、`stableSpringStep`。
- Produces: `force-balance`、`energy-ledger`；`section_id` 分别为 `air-drag`、`total-energy`。

- [ ] **Step 1: 写失败测试**：要求 `重力`、`空气阻力`、`动能`、`势能`、`终端速度`、`E=\\frac12mv^2` 和两个 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 10 个小节**：重力、线性阻力、弹簧力、功和动能、势能、总能量、阻尼、终端速度 `v_t=mg/c`、算例 `m=1,g=9.81,c=2 → v_t=4.905`、伪代码和能量诊断。
- [ ] **Step 4: 实现 `force-balance`**：小球下落，绘制重力、阻力、合力向量；控件质量、阻力、浮力开关；读数合力、加速度、当前与终端速度。
- [ ] **Step 5: 实现 `energy-ledger`**：一维阻尼弹簧振子；控件质量、刚度、阻尼；读数动能、势能、总能量和相对能量损失，绘制三段能量条。
- [ ] **Step 6: 运行测试、数值核心测试和两个 Demo 语法检查**

---

### Task 7: 第 4 章 质量弹簧系统

**Files:**
- Create: `book/physics_content_chapter04.py`
- Create: `book/assets/demos/physics07.js`
- Create: `book/assets/demos/physics08.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `springForce`、`stableSpringStep`。
- Produces: `spring-oscillator`、`spring-cloth`；`section_id` 分别为 `oscillator`、`spring-mesh`。

- [ ] **Step 1: 写失败测试**：要求 `胡克定律`、`F=-kx`、`稳定步长`、`结构`、`剪切`、`弯曲` 和两个 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 11 个小节**：静止长度、胡克定律、单位方向、等大反向力、相对速度阻尼、单振子、显式刚度与 `dt<2/sqrt(k/m)`、网格、三类弹簧、固定点与碰撞、算例 `k=20,m=1,x0=0.5`、伪代码和调试。
- [ ] **Step 4: 实现 `spring-oscillator`**：竖直弹簧-质点；控件质量、刚度、阻尼、初始拉伸；读数周期近似、当前伸长、速度、总能量和稳定步长上限。
- [ ] **Step 5: 实现 `spring-cloth`**：6×4 结构弹簧布片，顶行固定，可调风力和刚度；读数最大伸长、最大速度、固定点数和有限值状态。
- [ ] **Step 6: 运行测试和语法检查**

---

### Task 8: 第 5 章 PBD：直接修正位置

**Files:**
- Create: `book/physics_content_chapter05.py`
- Create: `book/assets/demos/physics09.js`
- Create: `book/assets/demos/physics10.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `RTPhysics.projectDistance`。
- Produces: `pbd-rope`、`constraint-order`；`section_id` 分别为 `projection`、`gauss-seidel`。

- [ ] **Step 1: 写失败测试**：要求 `C=\\lVert\\mathbf p_b-\\mathbf p_a\\rVert-L_0`、`逆质量`、`Gauss–Seidel`、`速度重建` 和 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 11 个小节**：力与位置修正区别、距离约束、梯度、单次投影、逆质量、固定点、迭代、处理顺序、速度重建、算例 `a=(0,0),b=(3,0),L0=2,w=1` 得 `a=(0.5,0),b=(2.5,0)`、伪代码和调试。
- [ ] **Step 4: 实现 `pbd-rope`**：8 点绳子，端点固定，中点可拖拽；控件迭代数、stiffness、时间步；读数最大约束误差、迭代前后误差、固定端数量。
- [ ] **Step 5: 实现 `constraint-order`**：三粒子两约束，正序/逆序/交替顺序并排；控件初始偏移、迭代数；读数三种顺序的最大误差与最终中点位置。
- [ ] **Step 6: 运行 PBD 核心测试和两个 Demo 语法检查**

---

### Task 9: 第 6 章 碰撞、恢复与摩擦

**Files:**
- Create: `book/physics_content_chapter06.py`
- Create: `book/assets/demos/physics11.js`
- Create: `book/assets/demos/physics12.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `collideSpherePlane`、`collideParticleSphere`。
- Produces: `collision-friction`、`continuous-collision`；`section_id` 分别为 `tangential-friction`、`continuous-detection`。

- [ ] **Step 1: 写失败测试**：要求 `碰撞法线`、`冲量`、`恢复系数`、`摩擦`、`穿透` 和 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 11 个小节**：穿透约束、平面/球体距离、法线、位置投影、相对速度、冲量、恢复系数、切向摩擦、离散穿透、swept AABB 的进入/离开时间、算例 `vy=-5,e=0.5 → vy'=2.5`、伪代码与调试。必须区分当前 Python 地面/静态球的乘性速度衰减与球体—粒子耦合的切向冲量上限，并说明公共粒子碰撞路径不充当正确冲量范例。
- [ ] **Step 4: 实现 `collision-friction`**：小球与地面、静态圆；控件恢复系数、摩擦、初速度；读数穿透修正量、碰撞前后法向/切向速度、接触次数。
- [ ] **Step 5: 实现 `continuous-collision`**：高速方块穿越薄墙，并排显示离散检测与 swept AABB；控件速度、墙厚、步长；读数离散是否漏碰、连续进入/离开时间和命中位置。
- [ ] **Step 6: 运行碰撞数值测试和语法检查**

---

### Task 10: 第 7 章 XPBD：用柔度表达软硬

**Files:**
- Create: `book/physics_content_chapter07.py`
- Create: `book/assets/demos/physics13.js`
- Create: `book/assets/demos/physics14.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `projectDistanceXPBD`、`projectAreaXPBD`。
- Produces: `xpbd-compliance`、`area-constraint`；`section_id` 分别为 `compliance`、`volume-gradient`。

- [ ] **Step 1: 写失败测试**：要求 `compliance`、`\\tilde\\alpha=\\alpha/h^2`、`\\Delta\\lambda`、`子步不变性`、`体积` 和 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 11 个小节**：PBD stiffness 边界、约束势能、柔度、有效柔度、乘子、增量更新、生命周期、距离 XPBD、体积梯度、子步不变性、算例 `alpha=0.01,h=1/60 → alpha_tilde=36`、伪代码和调试。
- [ ] **Step 4: 实现 `xpbd-compliance`**：同一拉伸初值切换 PBD/XPBD，控件 compliance、子步、迭代；读数有效柔度、lambda、约束误差和最终长度。
- [ ] **Step 5: 实现 `area-constraint`**：四粒子四边形面积约束，拖拽顶点；控件柔度、迭代；读数当前/静止面积、相对面积误差和最大乘子。显著标注“浏览器概念实验”；当前 Python 软体没有面积约束。
- [ ] **Step 6: 运行 XPBD 数值测试和语法检查**

---

### Task 11: 第 8 章 三种算法同题比较

**Files:**
- Create: `book/physics_content_chapter08.py`
- Create: `book/assets/demos/physics15.js`
- Create: `book/assets/demos/physics16.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: 三种核心求解路径。
- Produces: `solver-race`、`substep-invariance`；`section_id` 分别为 `stretch-response`、`timestep-change`。

- [ ] **Step 1: 写失败测试**：要求 `公平比较`、`质量弹簧`、`PBD`、`XPBD`、`子步不变性`、`约束误差` 和 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 12 个小节**：共同初始条件、共同循环、拉伸响应、参数不可直接互换、迭代/子步、能量/阻尼、误差、时间步变化、性能指标、算例、决策表、伪代码与调试。
- [ ] **Step 4: 实现 `solver-race`**：三条两粒子弹簧并排运行；控件公共质量、目标长度、迭代；读数每种算法最终长度、最大误差和能量变化。
- [ ] **Step 5: 实现 `substep-invariance`**：相同总时间、不同子步下比较 PBD 与 XPBD 最终长度；控件子步数、迭代、柔度；读数长度差和误差条。
- [ ] **Step 6: 增加子步不变性 Node 测试**：XPBD 在 1/60 与 1/120 子步、足够迭代下，最终长度差小于设定容限；PBD 不要求相同容限。
- [ ] **Step 7: 运行测试和语法检查**

---

### Task 12: 第 9 章 刚体、旋转与接触

**Files:**
- Create: `book/physics_content_chapter09.py`
- Create: `book/assets/demos/physics17.js`
- Create: `book/assets/demos/physics18.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: `collideSpherePlane`；旋转公式仅在概念层实现。
- Produces: `rigid-contact`、`rotation-intuition`；`section_id` 分别为 `ground-contact`、`angular-velocity`。

- [ ] **Step 1: 写失败测试**：要求 `刚体`、`角速度`、`转动惯量`、`力矩`、`恢复系数`、`当前边界` 和 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 12 个小节**：质点/刚体、平移、角速度、惯量、力矩、球碰撞、地面、滚动/滑动、摩擦、休眠/抖动、RenderTorch 无旋转边界、算例与调试。明确 RenderTorch `SphereBody` 只有平移速度。
- [ ] **Step 4: 实现 `rigid-contact`**：球落地反弹并水平滑动；控件半径、恢复系数、摩擦、初速度；读数接触次数、反弹高度、滑行距离和当前速度。
- [ ] **Step 5: 实现 `rotation-intuition`**：圆盘受切向力，显示角速度、弧长和滚动；控件力矩、转动惯量、时间；读数角加速度、角速度、转角。标注为浏览器概念实验。
- [ ] **Step 6: 运行测试和语法检查**

---

### Task 13: 第 10 章 软体、布料与综合项目

**Files:**
- Create: `book/physics_content_chapter10.py`
- Create: `book/assets/demos/physics19.js`
- Create: `book/assets/demos/physics20.js`
- Modify: `tests/test_physics_book.py`

**Interfaces:**
- Consumes: 弹簧/PBD/XPBD、`projectAreaXPBD` 与 `collideParticleSphere`。
- Produces: `cloth-showcase`、`soft-area`；`section_id` 分别为 `integrated-scene`、`volume-gradient`。

- [ ] **Step 1: 写失败测试**：要求 `软体`、`布料`、`应变限制`、`四面体`、`体积约束`、`自碰撞` 和 Demo ID。
- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 编写 12 个小节**：多粒子软体、三角网格、三类约束、固定点、应变限制、四面体体积、体积梯度、粒子/刚体双向碰撞、自碰撞概念、综合场景、性能指标、后续路线和调试。必须写清当前 Python 软体只有结构/剪切距离约束与六个四面体，中心粒子不直接参与体积约束，没有面积、独立弯曲或自碰撞；布料—动态球每个完整固定步只耦合一次。
- [ ] **Step 4: 实现 `cloth-showcase`**：7×5 布料，可切换质量弹簧/PBD/XPBD，移动碰撞球；控件算法、迭代/刚度或柔度、风力；读数最大约束误差、接触数、固定点数和算法。
- [ ] **Step 5: 实现 `soft-area`**：可拖拽四粒子软块，切换硬/软面积约束；控件柔度、迭代；读数面积误差、周长和形变比例。
- [ ] **Step 6: 运行测试和语法检查**

---

### Task 14: 参考资料、文档与全站契约

**Files:**
- Create: `book/physics_references.json`
- Modify: `book/physics_build.py`
- Modify: `README.md`
- Modify: `docs/references.md`
- Modify: `tests/test_physics_book.py`
- Modify: `tests/test_book.py`

**Interfaces:**
- Consumes: 10 章内容与 20 个 Demo。
- Produces: 物理参考资料页、更新后的项目说明、全站链接和许可契约。

- [ ] **Step 1: 写失败测试**

```python
def test_physics_subbook_has_ten_chapters_and_two_demos_each(self):
    rows = self.content.chapters()
    self.assertEqual([row.number for row in rows], list(range(1, 11)))
    for row in rows:
        self.assertGreaterEqual(len(row.sections), 10)
        self.assertGreaterEqual(len(row.demos), 2)
```

```python
def test_physics_references_cover_all_units(self):
    rows = self.content.references()
    self.assertGreaterEqual(len(rows), 6)
    for row in rows:
        for field in ("title", "authors", "url", "license", "usage"):
            self.assertTrue(row[field].strip())
```

```python
def test_all_physics_sources_exist(self):
    root = BOOK_DIR.parent
    for row in self.content.chapters():
        for item in row.sources:
            self.assertTrue((root / item.path).is_file(), item.path)
            self.assertEqual(
                item.url,
                f"https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/{item.path}",
            )
```

- [ ] **Step 2: 运行并确认失败**
- [ ] **Step 3: 建立物理参考资料数据**

至少包含：OpenStax University Physics、Baraff/Witkin Physically Based Modeling、House/Keyser、Müller PBD、Macklin XPBD、Matter.js 官方文档、Blender 物理文档。标注许可与“只核对知识，不复制正文/图片/代码”的用途。
- [ ] **Step 4: 实现物理参考资料页**

按基础力学、约束与碰撞、PBD/XPBD、工程参考分组；显示作者、许可、用途和外链；保留本地 KaTeX MIT 说明。
- [ ] **Step 5: 更新文档**

README 写明 15 章图形学 + 10 章物理、20 个实验、两步构建命令和 Pages 发布命令；测试数量在 Task 16 按完整套件的实测结果更新，不填猜测值。`docs/references.md` 新增物理子书资料与使用边界。
- [ ] **Step 6: 运行全站书籍测试**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_book tests.test_physics_book -v
```

Expected: 所有内容、公式、链接、资源、重置与读数契约通过。

---

### Task 15: 浏览器交互与响应式验收

**Files:**
- Modify: 仅当验收发现真实缺陷时修改对应 Demo、内容或样式
- Test: 10 个物理章节 + 图形学首页和第 1 章

**Interfaces:**
- Consumes: 构建产物 `book/physics/`。
- Produces: 浏览器验收证据与缺陷修复。

- [ ] **Step 1: 先重建完整站点**

```bash
/opt/anaconda3/envs/pythonTry/bin/python book/build.py
/opt/anaconda3/envs/pythonTry/bin/python book/physics_build.py
```

浏览器检查必须针对刚生成的 10 章产物，不能依赖各章测试留下的临时目录。

- [ ] **Step 2: 启动本地静态服务器**

```bash
/opt/anaconda3/envs/pythonTry/bin/python -m http.server 8000 --directory book
```

使用后台任务，不把本地服务器作为读者运行教材的前提。

- [ ] **Step 3: 逐页检查 10 章**

每页等待 Demo 初始化后读取：公式数、Canvas 数、`[data-readout]` 文本、reset 数。断言每章至少 2 个 Canvas，所有读数非空且不含 `NaN`/`Infinity`。
- [ ] **Step 4: 操作每个 Demo**

对 20 个 Demo 分别修改第一个范围控件、确认读数变化、点击 reset、确认恢复默认。为每个真实缺陷先写可重复检查，再修复。
- [ ] **Step 5: 检查控制台与网络**

要求页面自身无 error/warn；所有脚本、字体、样式和 Demo 请求无 404；无 fetch/xhr 外部运行时请求。
- [ ] **Step 6: 检查两种宽度**

使用 390px 与 1440px iframe 检查 10 个物理页面无横向溢出；390px 打开目录后焦点进入侧栏，Escape 关闭并返回目录按钮。
- [ ] **Step 7: 检查 `file://`**

直接打开物理首页和第 1 章，确认 KaTeX 与 Demo 脚本使用本地相对资源且读数正常。
- [ ] **Step 8: 记录截图限制**

若 `NATIVE_BROWSER_VIEWPORT_UNAVAILABLE`，不绕过限制；最终明确写“未完成可见截图验收”。

---

### Task 16: 完整验证、提交与发布

**Files:**
- Modify: `README.md` 的实测测试数量
- Stage: `README.md`、`book/` 内明确的教材源文件与生成页面、`tests/test_book.py`、`tests/test_physics_book.py`、`docs/references.md`、本设计规格与实施计划；不得递归暂存 `book/__pycache__/`

**Interfaces:**
- Consumes: 全部实现与验收。
- Produces: 源分支提交、`gh-pages` 发布和线上核验。

- [ ] **Step 1: 重建全部页面**

```bash
/opt/anaconda3/envs/pythonTry/bin/python book/build.py
/opt/anaconda3/envs/pythonTry/bin/python book/physics_build.py
```

- [ ] **Step 2: 运行完整测试**

```bash
QT_QPA_PLATFORM=offscreen /opt/anaconda3/envs/pythonTry/bin/python -m unittest discover -s tests -t . -v
```

记录实际测试数量，更新 README；任何失败先修复再继续。
- [ ] **Step 3: 运行格式与语法检查**

```bash
find book/assets -name '*.js' -print0 | xargs -0 -n1 node --check
git diff --check
```

- [ ] **Step 4: 审查工作区并只暂存授权文件**

```bash
git status --short
git diff --stat
git diff -- README.md book tests/test_book.py tests/test_physics_book.py docs/references.md
git add README.md book/*.py book/*.json book/*.html book/chapters/*.html book/physics/*.html book/physics/chapters/*.html book/assets book/assets/demos/*.js tests/test_book.py tests/test_physics_book.py docs/references.md docs/superpowers/specs/2026-11-27-rendertorch-physics-book-design.md docs/superpowers/plans/2026-11-27-rendertorch-physics-book.md
git status --short
git diff --cached --stat
git diff --cached --check
```

审查 `git diff --cached` 的实际文件清单和内容，确认没有缓存、凭据或无关改动后再提交；不使用 `git add -A` 或 `git add .`。

不得暂存 `main.py`、`render.py`、`solid/`、`viewing/`、`tests/test_rendering.py` 或任何 `__pycache__/`。
- [ ] **Step 5: 创建新提交**

提交信息聚焦教材价值，例如：

```text
docs: add interactive physics simulation textbook
```

不得 amend 已有提交，不跳过 hooks。
- [ ] **Step 6: 正常推送源分支**

```bash
git push origin feat/m1-qt-minimal-loop
```

- [ ] **Step 7: 正常发布 Pages**

```bash
git subtree push --prefix=book origin gh-pages
```

- [ ] **Step 8: 核验线上**

等待 Pages `status=built`，浏览器检查总入口、`physics/index.html`、`physics/chapters/physics01.html`、公式、20 个 Demo 脚本和资源请求；再次 `git status --short` 确认既有未提交修改保留。

## Self-Review

- 设计规格的 10 章由 Task 4–13 逐章覆盖，每章至少 2 个实验由 Task 4–13 和 Task 3 契约共同保证。
- 数值真实性由 Task 2、6、8、9、10、11 的 Node 断言覆盖，不仅检查文字。
- 图形学小节化由 Task 1 覆盖，物理小节插入由 Task 3 覆盖。
- 许可、源码边界、响应式、浏览器、提交与发布由 Task 14–16 覆盖。
- 计划不包含 TODO/TBD 或未定义接口；所有 Demo ID、脚本路径和测试命令均明确。
