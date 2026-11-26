# RenderTorch 图形学子书 v2 设计

- 日期：2026-11-26
- 状态：已在对话中批准总体设计，等待书面复核
- 实施范围：只重建 3D 图形学子书；物理子书在 3D 子书发布后另行设计
- 暂存目录：`book-v2/`
- 最终目录：`book/`

## 1. 目标

从零构建一本中文、可交互、可验证的计算机图形学教材，使读者能从像素和向量逐步推导到相机、网格、光栅化、光照与光线追踪。教材必须：

1. 用教材结构组织内容，而不是功能列表或项目说明。
2. 每个核心原理给出直觉、符号定义、逐步推导、数值算例和伪代码。
3. 每个关键可视化概念提供浏览器内 Canvas/SVG 实验和实时读数。
4. 严格区分 RenderTorch 已实现、浏览器概念实验和后续实现路线。
5. 使用开放许可优先的参考资料；新正文与自制图解保持原创。
6. 可在 GitHub Pages 直接运行，不需要 Python 或前端构建工具。

## 2. 非目标

本阶段不实现以下内容：

- 不扩展 RenderTorch 的 Python 渲染器功能。
- 不引入 WebGL、Three.js、React 或其他前端框架。
- 不沿用旧书 12 章正文、目录或章节编号。
- 不把浏览器二维或三维概念实验描述成当前 Python 源码已经实现的能力。
- 不在本阶段重写物理教材。
- 不删除或覆盖旧 `book/`，直到新书完成并通过全部验收。
- 不提交与教材无关的现有工作区修改。
- 不强制推送 Git 分支。

## 3. 读者与先验要求

- 读者：具备基础 Python 与高中数学，希望理解 RenderTorch 背后图形学原理的学习者。
- 数学先验：实数运算、函数、基础线性代数；缺少矩阵基础的读者可从第 2–3 章补齐。
- 编程先验：能阅读 Python 伪代码；浏览器实验不需要读者写代码。
- 阅读方式：正文与桌面 Demo 并行；所有关键实验同时提供文字读数，不只依赖颜色。

## 4. 总体架构

新书是纯静态网站，内容源与生成器使用 Python 标准库：

```text
book-v2/
├── build.py                 # 读取内容与参考资料，生成静态 HTML
├── content.py               # 15 章结构化正文、公式、伪代码与实验规格
├── references.json          # 参考资料、许可、用途与署名
├── assets/
│   ├── style.css            # 教材版式、响应式与打印样式
│   ├── book.js              # 目录、复制、Canvas 公共工具
│   └── demos/
│       ├── chapter01.js     # 每章独立实验，避免一个巨型脚本
│       └── chapter15.js
├── chapters/
│   └── chapter01.html       # 构建产物
├── index.html               # 构建产物
├── references.html          # 构建产物
```

仓库级测试放在 `tests/test_book.py`，不随静态网站发布。

`book-v2/` 同时保存内容源和可直接发布的静态产物。验证通过后，将其整体移动为仓库中的 `book/`，并把构建测试默认路径从 `book-v2/` 改为 `book/`。

构建器只写自己管理的文件，不递归删除未知文件。所有 JavaScript 都是普通脚本，按 `book.js` → 本章实验脚本的顺序加载，不使用 ES module，保证 `file://` 与 GitHub Pages 两种打开方式都可运行。

## 5. 教材结构

全书分为四个单元，共 15 章：

| 单元 | 章节 | 主题 |
|---|---|---|
| 一、像素与数学 | 1–4 | 像素与图像、三维数学、齐次坐标、相机与投影 |
| 二、几何建模 | 5–9 | 图元与网格、拓扑与法线、参数曲线、参数曲面、场景与包围体 |
| 三、光栅管线 | 10–11 | 裁剪与光栅化、属性插值与遮挡 |
| 四、光照与完整渲染 | 12–15 | 光照材质、纹理 UV、阴影与颜色管理、光线追踪 |

每章固定包含以下部分：

1. 学习目标与先验知识。
2. 直觉：先说明可观察的现象。
3. 符号与定义：集中列出本章符号和约定。
4. 原理推导：一步一个公式，不使用“显然”跳过关键步骤。
5. 数值算例：给出一组可以手算或代入实验的数字。
6. 伪代码与源码映射：只链接真实存在的 Python 文件；概念内容标为“后续路线”。
7. 交互实验：至少一个可操作实验，并提供 `aria-live` 文字读数。
8. 自测、常见错误与调试清单。
9. 本章实现状态：现有 RenderTorch、浏览器概念实验、后续实现路线。

实现状态使用固定样式与措辞：

- `现有 RenderTorch`：给出当前仓库真实源码路径和 GitHub 链接。
- `浏览器概念实验`：说明浏览器脚本展示算法思想，不是 Python 实现。
- `后续实现路线`：说明 RenderTorch 当前没有该能力，并给出最小扩展方向。

## 6. 各章内容契约

### 第 1 章 像素、颜色与图像

必须讲解像素索引、图像坐标原点、宽高、RGB/A 通道、采样与分辨率。推导：

```text
u = x / (W - 1)
v = y / (H - 1)
C(x, y) = (R, G, B, A)
```

实验：8×8 可点击像素网格；点击后显示索引、坐标、RGBA，并可切换颜色通道。读数验证 `u,v ∈ [0,1]`。

### 第 2 章 三维空间、向量、点与方向

必须讲解坐标轴、左右手系、点与向量、加减、数乘、点积、叉积、归一化和长度。推导：

```text
a · b = |a||b| cos θ
a × b = (ay bz − az by, az bx − ax bz, ax by − ay bx)
n = normalize((b − a) × (c − a))
```

实验：可拖动三个点，显示两条边、叉积向量、法线与夹角；提供 2D 投影与三维坐标读数。

### 第 3 章 齐次坐标与几何变换

必须讲解二维/三维齐次点、平移矩阵、缩放、旋转、复合顺序、列向量约定与法线变换。推导：

```text
p' = M p
M = T(t) R(θ) S(s)          # 先缩放，再旋转，最后平移
n' = normalize(M^{-T} n)    # 适用于三维线性变换的法线思路
```

实验：一个正方形与可调平移、旋转、缩放；显示矩阵每一步和变换后的坐标。明确区分教材使用的列向量约定与 RenderTorch 现有实现约定。

### 第 4 章 相机、视图、投影与视口

必须讲解轨道相机、lookAt、透视投影、近远裁剪面、齐次除法、NDC 与视口变换。推导：

```text
z_camera = distance from camera
x_ndc = f x_camera / w_camera
y_ndc = f y_camera / w_camera
x_pixel = (x_ndc + 1) W / 2
y_pixel = (1 − y_ndc) H / 2
```

实验：相机绕点旋转并改变距离；画出一个线框立方体、近远平面和 NDC 读数。明确 `w` 是视图空间前向距离的约定。

### 第 5 章 几何图元与网格表示

必须讲解点、线、三角形、多边形、索引网格、三角形扇和三角形带。推导三角形重心坐标定义，并解释索引复用。实验：切换 fan/strip/indexed 三种四边形三角化方式，显示顶点复用次数和共享边。

### 第 6 章 网格拓扑、法线、绕序与三角化

必须讲解流形边、Euler 公式、绕序、面法线、顶点法线、平滑着色与平面着色的区别。推导：

```text
V − E + F = 2                # 闭合可定向球面网格
n_face = normalize((v1 − v0) × (v2 − v0))
n_vertex = normalize(Σ n_face)
```

实验：翻转三角形绕序并切换面法线/顶点法线；显示背面、法线和共享边数量。严格 top-left 留到第 11 章。

### 第 7 章 参数曲线：Bézier、B-spline 与 NURBS 概念

必须讲解参数域、控制点、de Casteljau、二次/三次 Bézier、连续性，以及 B-spline/NURBS 的控制点权重概念。推导：

```text
B(t) = Σ C_i B_i^n(t)
B_i^n(t) = (n choose i) t^i (1 − t)^(n − i)
```

实验：四个控制点的三次 Bézier 曲线；拖动控制点并显示 de Casteljau 各级点和切线。

### 第 8 章 参数曲面与细分

必须讲解双参数曲面 `S(u,v)`、Bézier patch、网格采样、法线叉积、Loop/Catmull–Clark 与三角/四边细分的区别。推导：

```text
S(u,v) = Σ_i Σ_j P_ij B_i(u) B_j(v)
∂S/∂u ≈ S(u+ε,v) − S(u−ε,v)
∂S/∂v ≈ S(u,v+ε) − S(u,v−ε)
n = normalize((∂S/∂u) × (∂S/∂v))
```

实验：在低分辨率四边形网格与一次细分网格之间切换，显示顶点/面数量变化和法线。

### 第 9 章 场景、模型文件、场景图与包围体

必须讲解局部/世界坐标、模型矩阵、场景图、OBJ 的 `v/vt/vn`、AABB、包围球、视锥裁剪。推导：

```text
p_world = M_model p_local
AABB = [min(x_i), max(x_i)] × [min(y_i), max(y_i)] × [min(z_i), max(z_i)]
```

实验：父子场景图变换两个盒子；切换 AABB/包围球并显示世界坐标和剔除判断。

### 第 10 章 齐次裁剪与三角形光栅化

必须讲解近裁剪、齐次空间直线/多边形裁剪、透视除法、边函数、包围盒与像素中心覆盖。推导：

```text
E(a,b,p) = (px−ax)(by−ay) − (py−ay)(bx−ax)
A2 = E(v0,v1,v2)
λ0 = E(v1,v2,p) / A2
λ1 = E(v2,v0,p) / A2
λ2 = 1 − λ0 − λ1
covered ⇔ λ0,λ1,λ2 ≥ 0
```

实验：拖动三角形顶点、切换包围盒和像素中心、悬停像素显示边函数与 λ；提供退化面积保护。

### 第 11 章 重心坐标、属性插值、深度与遮挡

必须讲解透视矫正、逆深度插值、z-buffer、画家算法对照、背面剔除、top-left 规则和 MSAA 概念。推导：

```text
1 / w_pixel = λ0 / w0 + λ1 / w1 + λ2 / w2
w_pixel = 1 / (λ0 / w0 + λ1 / w1 + λ2 / w2)
write ⇔ w_pixel < depth[x,y]
```

实验：并排显示线性深度与逆深度插值；两个三角形可交换提交顺序；显示每像素比较结果。

### 第 12 章 光照模型与材质

必须讲解法线、光源方向、观察方向、漫反射、Phong、Blinn–Phong、高光与 PBR 的粗糙度/金属度直觉。推导：

```text
L = normalize(light_position − surface_position)
N = normalize(surface_normal)
V = normalize(camera_position − surface_position)
diffuse = max(0, N · L)
H = normalize(L + V)
specular = pow(max(0, N · H), shininess)
```

实验：球体或曲面片段；调节光位、环境光、漫反射、高光指数，显示逐像素法线和各项贡献。

### 第 13 章 纹理、UV 与透视矫正

必须讲解 UV、纹理采样、放大/缩小过滤、mipmap 概念与透视矫正。推导：

```text
(u_pixel,v_pixel) = (λ0 u0 + λ1 u1 + λ2 u2) / (λ0/w0 + λ1/w1 + λ2/w2)
                    × w_pixel
```

实验：棋盘纹理贴到三角形；切换 nearest/linear，移动顶点深度并显示正确与错误 UV 对照。

### 第 14 章 阴影、gamma 与色调映射

必须讲解 shadow mapping 的两次渲染、深度比较、阴影失真、线性颜色、gamma 与 Reinhard/ACES 直觉。推导：

```text
shadow ⇔ light_depth_sample > fragment_depth + bias
C_display ≈ C_linear^(1/2.2)
C_reinhard = C_linear / (1 + C_linear)
```

实验：一个方块与一个球；调节光源、阴影偏移和 gamma，显示阴影贴图深度与线性/显示颜色。

### 第 15 章 光线追踪与完整渲染管线比较

必须讲解光线生成、ray-sphere 求交、ray-triangle、反射、折射概念、阴影射线与路径追踪直觉。推导：

```text
p(t) = o + t d
|o + t d − c|² = r²
r = d − 2(d · n)n
```

实验：可旋转视角追踪三个球和一个平面；切换光线数、最大反射次数，显示首次命中、法线与累计颜色。最后用表格比较光栅化与光线追踪的成本、能力和 RenderTorch 路线。

## 7. 交互实验通用要求

- 每个实验是一个普通 JavaScript 文件，挂载到唯一 `data-demo` 容器。
- Canvas 使用固定逻辑分辨率，CSS 自适应缩放；坐标换算按实际显示尺寸处理。
- 控件具备 `<label>`、键盘可操作性和可见焦点样式。
- 所有重要读数写入 `aria-live="polite"` 元素。
- 支持“重置”控件，恢复可重复初始状态。
- 遵循 `prefers-reduced-motion`：不自动播放动画，只有用户主动操作时才更新。
- 不依赖颜色传递唯一信息；同时提供形状、标签和数字。
- 不请求网络；参考资料外链由用户主动点击。

## 8. 参考资料与许可策略

参考资料只用于核对知识结构与算法出处，不复制商业教材正文。

| 资料 | 许可 | 使用方式 |
|---|---|---|
| David J. Eck, Introduction to Computer Graphics, v1.4 | CC BY-NC-SA 4.0 | 仅核对知识点；新书不复用其正文或图 |
| Joey de Vries, LearnOpenGL | 图片 CC BY 4.0；代码示例 CC BY-NC 4.0 | 只核对知识点；v2 图解全部原创，不下载或复用站点图片 |
| Physically Based Rendering, 4th edition | CC BY-NC-ND 4.0 | 只引用知识点，不复制图片或改编正文 |
| Ray Tracing in One Weekend | All rights reserved | 只作学习参考，不复用正文或图 |
| OpenStax, University Physics Volume 1 | CC BY 4.0 | 留给后续物理子书 |
| Baraff & Witkin, Physically Based Modeling | 课程资料 | 留给后续物理子书 |
| Müller et al., Position Based Dynamics | 论文 | 只引用算法与 DOI |
| Macklin et al., XPBD | 论文 | 只引用算法与 DOI |

v2 不下载或复用 LearnOpenGL 图片。所有 Canvas/SVG 图解由本教材新绘制，不使用旧书插图；参考资料页仍保留许可信息，便于后续扩展时核查。

## 9. 内容源与构建接口

`book-v2/content.py` 使用标准库 `dataclass`：

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

`book-v2/build.py` 提供：

```python
def render_chapter(chapter: Chapter) -> str: raise NotImplementedError
def render_index(all_chapters: list[Chapter]) -> str: raise NotImplementedError
def render_references() -> str: raise NotImplementedError
def build(output_dir: Path) -> None: raise NotImplementedError

if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
```

浏览器公共接口：

```javascript
window.RTBook = {
  clamp(value, min, max),
  lerp(a, b, t),
  normalize2(x, y),
  setupCanvas(canvas),
  formatNumber(value, digits = 3)
};
```

每个 Demo 脚本只读取自己容器的 DOM，使用 `requestAnimationFrame` 仅在用户操作后重绘。

## 10. 验收标准

### 内容验收

- 15 章全部生成，章节编号、标题、单元与批准目录一致。
- 每章包含学习目标、直觉、符号、推导、算例、伪代码、实验、自测和实现状态。
- 每章至少一个可操作实验，且实验说明与正文公式一致。
- 现有能力链接真实文件；不存在“浏览器有、Python 也有”的虚假表述。

### 自动验收

- Python 构建器可从干净输出生成全部页面。
- `tests/test_book.py` 检查章节数量、标题、必需节、唯一 Demo ID、本地链接/资源存在和许可字段。
- 所有 Demo JavaScript 通过 `node --check`。
- HTML 不引用旧 `chapter12.html` 之外被删除的旧页面；新书导航只含 15 章。
- 参考资料表具备作者、标题、URL、许可与用途字段。

### 浏览器验收

在本地 HTTP 服务器上验证：

- 首页与全部 15 章可在桌面宽度 1440 与移动宽度 390 阅读。
- 目录在移动端可打开/关闭，键盘 Tab 顺序合理。
- 每个实验可鼠标与键盘操作，读数更新且重置有效。
- 页面无控制台错误、横向正文溢出或失效资源。
- `prefers-reduced-motion: reduce` 下没有自动动画。
- 打印样式隐藏交互控件但保留正文、公式和实验说明。

### 发布验收

- 旧书先复制到仓库外的 `../RenderTorch-book-legacy-backup/`。
- 新书通过验收后才替换 `book/`。
- 最终提交只包含新教材、构建器、测试和必要的 README 更新。
- 不夹带现有未提交的渲染器、相机、入口脚本或缓存文件。
- 正常提交并推送 `feat/m1-qt-minimal-loop`，再正常更新 `gh-pages`。
- 检查 GitHub Pages 状态为 `built` 且首页可访问。
- 不使用 `git push --force`。

## 11. 风险与应对

- **旧书工作被覆盖**：开发全程在 `book-v2/`，替换前完整备份到仓库外。
- **浏览器实验被误认为 Python 能力**：固定状态标签与源码映射测试。
- **外部图片许可不清**：只从 LearnOpenGL 复用有明确署名的图片；其他图解原创。
- **单文件脚本膨胀**：每章独立 `chapterNN.js`，公共数学与 Canvas 工具集中在 `book.js`。
- **公式与实验漂移**：章节内容契约列出公式，实验读数必须能显示同一组量。
- **发布夹带无关改动**：提交前检查 `git status` 和 diff，只暂存明确文件。
- **移动端交互拥挤**：工具栏在 720px 以下纵向排列，Canvas 保持比例并允许文字读数换行。
