from dataclasses import dataclass, replace
import json
from pathlib import Path


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
    section_id: str = ""


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


CHAPTER_METADATA = (
    Chapter(
        1,
        "chapter01",
        "一、像素与数学",
        "像素、颜色与图像",
        "从像素索引、坐标和颜色通道出发，建立数字图像最基本的表示。",
        ("解释像素、分辨率和图像坐标之间的关系", "读取并修改一个像素的 RGBA 值"),
        ("基础 Python 算术",),
        (),
        (),
        (),
    ),
    Chapter(
        2,
        "chapter02",
        "一、像素与数学",
        "三维空间、向量、点与方向",
        "用点、向量、点积和叉积描述三维几何，并为法线计算建立工具。",
        ("区分点与向量，并解释左右手坐标系", "使用点积求夹角、使用叉积求法线"),
        ("基础三角函数", "第 1 章的坐标概念"),
        (),
        (),
        (),
    ),
    Chapter(
        3,
        "chapter03",
        "一、像素与数学",
        "齐次坐标与几何变换",
        "用统一矩阵表示平移、旋转和缩放，并理解变换顺序与法线矩阵。",
        ("写出三维齐次点和基本变换矩阵", "解释先缩放再旋转最后平移的复合顺序"),
        ("第 2 章的向量运算", "基础矩阵乘法"),
        (),
        (),
        (),
    ),
    Chapter(
        4,
        "chapter04",
        "一、像素与数学",
        "相机、视图、投影与视口",
        "把三维点经过视图、透视投影和视口变换送到屏幕像素。",
        ("解释轨道相机、近远裁剪面与 NDC", "从相机坐标推导屏幕像素坐标"),
        ("第 3 章的齐次变换", "基础三角函数"),
        (),
        (),
        (),
    ),
    Chapter(
        5,
        "chapter05",
        "二、几何建模",
        "几何图元与网格表示",
        "用点、线、三角形和索引网格组织三维几何，并比较不同三角化方式。",
        ("说明索引网格为什么能复用顶点", "比较三角形扇、三角形带和显式索引"),
        ("第 2 章的点和向量", "基础数组概念"),
        (),
        (),
        (),
    ),
    Chapter(
        6,
        "chapter06",
        "二、几何建模",
        "网格拓扑、法线、绕序与三角化",
        "从共享边、Euler 公式和绕序出发，区分面法线、顶点法线与拓扑结构。",
        ("用 Euler 公式检查简单闭合网格", "解释绕序如何影响面法线和背面判断"),
        ("第 2 章的叉积", "第 5 章的网格表示"),
        (),
        (),
        (),
    ),
    Chapter(
        7,
        "chapter07",
        "二、几何建模",
        "参数曲线：Bézier、B-spline 与 NURBS 概念",
        "从参数区间和控制点出发，推导 Bézier 曲线并理解样条与权重的作用。",
        ("用 de Casteljau 算法计算曲线点", "解释控制点、基函数和参数连续性的意义"),
        ("基础多项式", "第 2 章的向量加法与数乘"),
        (),
        (),
        (),
    ),
    Chapter(
        8,
        "chapter08",
        "二、几何建模",
        "参数曲面与细分",
        "把一维曲线扩展为双参数曲面，用采样和细分生成可渲染网格。",
        ("写出双参数曲面并计算采样点", "用有限差分估计曲面法线并解释细分"),
        ("第 7 章的参数曲线", "第 6 章的法线"),
        (),
        (),
        (),
    ),
    Chapter(
        9,
        "chapter09",
        "二、几何建模",
        "场景、模型文件、场景图与包围体",
        "用模型矩阵、场景图和包围体组织多物体场景，并建立粗粒度裁剪基础。",
        ("把局部坐标通过父子变换送到世界坐标", "解释 OBJ 索引、AABB 和视锥裁剪"),
        ("第 3 章的矩阵复合", "第 5 章的网格表示"),
        (),
        (),
        (),
    ),
    Chapter(
        10,
        "chapter10",
        "三、光栅管线",
        "齐次裁剪与三角形光栅化",
        "在齐次空间裁剪几何，再用边函数和像素中心判断三角形覆盖。",
        ("说明为什么透视除法前必须做近裁剪", "从边函数推导重心坐标和覆盖判断"),
        ("第 4 章的投影与视口", "第 2 章的二维叉积"),
        (),
        (),
        (),
    ),
    Chapter(
        11,
        "chapter11",
        "三、光栅管线",
        "重心坐标、属性插值、深度与遮挡",
        "使用透视矫正插值和深度缓冲解决属性偏差与遮挡顺序问题。",
        ("解释为什么深度要插值 1/w", "比较画家算法与 z-buffer 的逐像素结果"),
        ("第 10 章的重心坐标", "基础深度比较"),
        (),
        (),
        (),
    ),
    Chapter(
        12,
        "chapter12",
        "四、光照与完整渲染",
        "光照模型与材质",
        "由法线、光源和观察方向推导漫反射与高光，并建立 PBR 的直觉。",
        ("分解 Lambert 漫反射与 Blinn–Phong 高光", "说明粗糙度和金属度在材质中的意义"),
        ("第 6 章的法线", "第 2 章的点积与归一化"),
        (),
        (),
        (),
    ),
    Chapter(
        13,
        "chapter13",
        "四、光照与完整渲染",
        "纹理、UV 与透视矫正",
        "把二维纹理坐标绑定到三维表面，并推导透视矫正后的纹理采样。",
        ("解释 UV 坐标和最近邻、双线性过滤", "比较屏幕线性 UV 与透视矫正 UV"),
        ("第 11 章的透视矫正插值", "第 5 章的三角形表示"),
        (),
        (),
        (),
    ),
    Chapter(
        14,
        "chapter14",
        "四、光照与完整渲染",
        "阴影、gamma 与色调映射",
        "用阴影贴图比较光源空间深度，并在线性颜色与显示颜色之间正确转换。",
        ("解释 shadow mapping 的两次渲染和深度偏移", "区分 gamma 编码与 Reinhard 色调映射"),
        ("第 12 章的光照", "第 11 章的深度缓冲"),
        (),
        (),
        (),
    ),
    Chapter(
        15,
        "chapter15",
        "四、光照与完整渲染",
        "光线追踪与完整渲染管线比较",
        "从光线与球面求交出发理解光线追踪，并比较它与光栅管线的能力与成本。",
        ("推导 ray-sphere 二次方程并选择有效交点", "比较光栅化、实时光线和路径追踪"),
        ("第 4 章的相机射线", "第 12 章的光照"),
        (),
        (),
        (),
    ),
)


def section(section_id: str, label: str, title: str, html: str) -> Section:
    return Section(section_id, label, title, html)


def source(path: str, status: str) -> SourceLink:
    branch = "feat/m1-qt-minimal-loop"
    return SourceLink(
        path,
        f"https://github.com/Zhenghong-Liu/RenderTorch/blob/{branch}/{path}",
        status,
    )


CHAPTER_DETAILS = {
    "chapter01": {
        "sections": (
            section(
                "intuition",
                "从屏幕上的小方格开始",
                "图像是有地址的颜色样本",
                r"""<p>把屏幕放大到足够大，会看到一个由小方格组成的阵列。每个方格叫作一个
<strong>像素</strong>；计算机不直接保存“红色看起来像什么”，而是保存一组可重复计算的数值。
渲染器的全部工作，最终都要落实到：哪个像素、写入什么颜色、是否更新深度。</p>
<p>本章暂时不讨论三维物体。先假设图像已经是一张表，并建立三个层次的坐标：用于遍历的
整数索引 \((x,y)\)、用于采样的归一化坐标 \((u,v)\)，以及用于数值计算的
通道值。混淆这三者，是图像翻转、越界和颜色错误最常见的根源。</p>
<div class="visual-card"><svg viewBox="0 0 640 250" role="img" aria-label="八列六行的像素网格，突出第五列第三行">
<defs><pattern id="pixel-grid" width="64" height="32" patternUnits="userSpaceOnUse"><rect width="64" height="32" fill="#f6f3eb" stroke="#cbd5d1"/></pattern></defs>
<rect x="24" y="24" width="512" height="192" rx="10" fill="url(#pixel-grid)"/>
<rect x="280" y="88" width="64" height="32" rx="5" fill="#087f83"/>
<text x="312" y="110" text-anchor="middle" fill="white" font-size="14">(5, 2)</text>
<path d="M312 122v70" stroke="#c66b35" stroke-width="2"/><text x="326" y="214" fill="#854721" font-size="14">v = 2 / (H - 1)</text>
<path d="M344 104h110" stroke="#c66b35" stroke-width="2"/><text x="466" y="98" fill="#854721" font-size="14">u = 5 / 7</text>
</svg><p><strong>图 1.1</strong>　坐标原点位于左上角，x 向右，y 向下；这是图像缓冲区常用的约定。</p></div>
<div class="callout"><strong>约定先于公式。</strong> 本章采用左上角原点、x 向右、y 向下。
后面相机视口公式会把 OpenGL 风格的左下角 NDC 转换成这套屏幕坐标。</div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "分辨率、坐标和 RGBA",
                r"""<p>设图像宽度为 <code>W</code>、高度为 <code>H</code>，单位都是像素。
有效像素索引是 \(x\in\{0,\ldots,W-1\}\)、\(y\in\{0,\ldots,H-1\}\)。
因此尺寸为 \(W\times H\) 的图像一共有 \(WH\) 个像素，而不是坐标中的
<code>W+1</code> 或 <code>H+1</code> 个位置。</p>
<div class="formula">\[\begin{aligned}
u &= \frac{x}{W-1}, & v &= \frac{y}{H-1}, \\
0 &\le u \le 1, & 0 &\le v \le 1.
\end{aligned}\]</div>
<p><code>u</code> 和 <code>v</code> 把像素中心映射到两端都包含的连续区间。
当纹理真正采样时，也可以采用像素中心约定 \((x+0.5)/W\)；两者解决的是不同问题。
本章使用 \(W-1\)，是为了让第一个和最后一个像素精确落在 0 与 1。</p>
<p>一个彩色像素通常保存红、绿、蓝和透明度四个通道，记作 RGBA。
通道有两种常见表示：</p>
<ul><li><strong>字节表示：</strong>整数 0–255，适合帧缓冲区和 PNG 文件；</li>
<li><strong>浮点表示：</strong>约 0–1，适合光照、纹理过滤和色调映射。</li></ul>
<p>通道数值描述的是颜色编码或强度，并不自动等同于人眼感受到的亮度。
Gamma 与线性光的关系将在第 14 章展开。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "从二维索引到一维内存地址",
                r"""<p>假设像素按行优先存储，每一行连续排列，RGBA 四个通道也紧密相连。
要找到第 <code>y</code> 行第 <code>x</code> 列，需要先跳过前面的 <code>y</code> 行，
再跳过当前行的 <code>x</code> 个像素。于是：</p>
<div class="formula">\[\begin{aligned}
\mathrm{pixel\_index} &= yW+x, \\
\mathrm{byte\_offset} &= 4(yW+x).
\end{aligned}\]</div>
<p>地址 <code>byte_offset</code> 指向红色通道；随后三个字节依次是绿、蓝、透明度。
如果图像还带有行填充，则每行步长可能大于 <code>4W</code>，这时应使用实际的 stride，
而不能把公式中的 <code>W</code> 当作 stride。</p>
<p>归一化坐标则提供另一种访问方式。给定连续的 <code>(u,v)</code>，在最近邻采样下先取整：</p>
<div class="formula">\[x=\operatorname{round}\!\bigl(u(W-1)\bigr),\qquad
y=\operatorname{round}\!\bigl(v(H-1)\bigr).\]</div>
<p>取整之后仍要检查范围。边界外的 <code>u,v</code> 可以采用截断、重复边缘像素或镜像重复，
但这些属于纹理寻址策略，不应让它们悄悄造成数组越界。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "定位 8 × 6 图像中的一个像素",
                r"""<p>取 <code>W=8</code>、<code>H=6</code>，检查 <code>(x,y)=(5,2)</code>。
它的归一化坐标是：</p>
<div class="formula">\[\begin{aligned}
u &= \frac{5}{8-1}=\frac{5}{7}\approx 0.714286, \\
v &= \frac{2}{6-1}=\frac{2}{5}=0.4.
\end{aligned}\]</div>
<p>该像素之前有 2 个完整行，每行 8 个像素，因此像素索引为：</p>
<div class="formula">\[\mathrm{index}=2\cdot8+5=21,\qquad
\mathrm{byte\_offset}=21\cdot4=84.\]</div>
<p>也就是说，字节 84–87 存储这个像素的 RGBA。若值为
<code>(12, 200, 64, 255)</code>，转换成浮点通道就是
<code>(0.0471, 0.7843, 0.2510, 1)</code>。透明度 255 表示完全不透明。</p>
<div class="callout"><strong>手算检查：</strong> <code>u</code> 从 0 到 1 时，最近邻索引应从 0 到 7；
<code>v</code> 从 0 到 1 时，索引应从 0 到 5。使用 <code>W</code> 而不是 \(W-1\)
会让端点无法到达最后一个像素。</div>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "生成一张可验证的测试图",
                r"""<p>下面的程序让灰度随 x 增加，并让透明度始终不透明。
它适合作为帧缓冲区、Canvas 上传或图像保存函数的第一个测试。</p>
<div class="code-card"><div class="code-toolbar"><span>PYTHON · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
<pre><code>def make_image(width, height):
    image = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            gray = round(255 * x / (width - 1))
            offset = (y * width + x) * 4
            image[offset:offset + 4] = bytes((gray, gray, gray, 255))
    return image

pixel = (5, 2)
offset = (pixel[1] * 8 + pixel[0]) * 4
rgba = tuple(image[offset:offset + 4])</code></pre></div>
<p>真实项目中，帧缓冲区通常直接分配 NumPy 数组；浏览器实验则用
<code>ImageData.data</code> 写入字节，再用 <code>putImageData</code> 显示。
两种实现的索引公式完全相同。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "确认坐标与地址没有脱节",
                r"""<details><summary>问题 1：4 × 3 图像中，(0,2) 的像素索引是多少？</summary>
<p><strong>答案：8。</strong> 前面有两行，每行 4 个像素，所以 <code>2×4+0=8</code>。</p></details>
<details><summary>问题 2：为什么 x 最大只能是 W−1，而不是 W？</summary>
<p><strong>答案：</strong>索引从 0 开始计数，W 个像素的最后一个索引是 W−1。</p></details>
<details><summary>问题 3：RGBA=(255,0,0,0) 一定不可见吗？</summary>
<p><strong>答案：</strong>不透明度和最终合成规则共同决定可见性。Alpha 为 0 通常表示完全透明，
但若混合或保存方式不正确，也可能出现误写；不能只凭一个通道推断最终显示。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "图像问题的排查顺序",
                r"""<ol class="debug-list"><li><strong>先检查范围：</strong>是否允许 x=W 或 y=H？最后一个有效索引应为 W−1、H−1。</li>
<li><strong>再检查原点：</strong>图像坐标向下，NDC 通常向上；y 翻转常表现为整张图倒置。</li>
<li><strong>检查通道：</strong>RGBA 是否误按 RGB 三个字节步进？透明通道是否始终为 0？</li>
<li><strong>检查编码：</strong>写入 0–1 的浮点到字节数组会接近黑色；写入 0–255 到归一化缓冲区会饱和。</li>
<li><strong>最后检查 stride：</strong>外部图像可能带行对齐，不应假定一行恰好是 4W 字节。</li></ol>
<p>最有效的调试图像是单像素变化、水平灰阶和四角标记。它们能把坐标、寻址和翻转错误直接显示出来。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "概念实验与当前 RenderTorch 实现",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验把图像放大为可点击的像素网格，
允许改变通道值并实时计算归一化坐标与字节地址。它不依赖 WebGL。</p>
<p><span class="status-pill">现有 RenderTorch</span>
<code>renderer/framebuffer.py</code> 已用 <code>numpy.uint8</code> 存储 RGBA 颜色缓冲区，
并用独立数组保存深度和物体编号。该实现与本章的行优先地址模型一致。</p>
<p><span class="status-pill">后续实现路线</span> 可在真实渲染循环中加入缩放显示、
像素拾取和颜色编码转换测试；这些扩展不应改变帧缓冲区的基本索引契约。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "pixel-demo",
                "像素检查器：索引、坐标与 RGBA",
                "demos/chapter01.js",
                "选择像素并改变通道，观察索引、u/v、字节偏移与图像结果。",
            ),
        ),
        "sources": (source("renderer/framebuffer.py", "现有实现：RGBA 与深度缓冲区"),),
    },
    "chapter02": {
        "sections": (
            section(
                "intuition",
                "位置与位移不是同一件事",
                "用向量描述方向和大小",
                r"""<p>点回答“在哪里”，向量回答“向哪里、走多远”。
两个点相减得到位移向量；一个向量乘以标量会改变大小，但不改变方向所在的那条直线。
三维渲染中的相机视线、表面法线、光线方向和三角形边，都是向量。</p>
<p>本章采用右手坐标系：食指指向 +x，中指指向 +y，拇指指向 +z。
从 +x 转到 +y 时，右手定则给出的轴向是 +z。
换用左手系并不会让数学失效，但会改变叉积符号，因此整条管线必须保持同一约定。</p>
<div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="右手坐标轴与两个向量点积和叉积示意">
<defs><marker id="arrow-a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#087f83"/></marker><marker id="arrow-b" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#c66b35"/></marker></defs>
<path d="M120 220l330 0M120 220l0-150M120 220l-90 75" stroke="#42545c" stroke-width="2"/>
<text x="465" y="225">x</text><text x="115" y="58">y</text><text x="15" y="312">z</text>
<path d="M120 220L290 85" stroke="#087f83" stroke-width="4" marker-end="url(#arrow-a)"/>
<path d="M120 220L410 190" stroke="#c66b35" stroke-width="4" marker-end="url(#arrow-b)"/>
<text x="292" y="72" fill="#087f83">a</text><text x="417" y="185" fill="#c66b35">b</text>
<path d="M290 85A220 220 0 0 1 410 190" fill="none" stroke="#8a9a94" stroke-dasharray="6 5"/>
<text x="348" y="120" fill="#62727a">θ</text>
</svg><p><strong>图 2.1</strong>　点积衡量方向一致性；叉积生成同时垂直于 a 与 b 的方向。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "三种最基本的向量运算",
                r"""<p>设 \(\mathbf a=(a_x,a_y,a_z)\)、\(\mathbf b=(b_x,b_y,b_z)\)。</p>
<h3>点积：得到一个标量</h3>
<div class="formula">\[\begin{aligned}
\mathbf a\cdot\mathbf b &= a_xb_x+a_yb_y+a_zb_z, \\
\mathbf a\cdot\mathbf b &= \|\mathbf a\|\,\|\mathbf b\|\cos\theta.
\end{aligned}\]</div>
<h3>叉积：得到一个垂直于两个输入的方向</h3>
<div class="formula">\[\mathbf a\times\mathbf b=
\begin{pmatrix}
a_yb_z-a_zb_y\\
a_zb_x-a_xb_z\\
a_xb_y-a_yb_x
\end{pmatrix}.\]</div>
<h3>归一化：保持方向，把长度变为 1</h3>
<div class="formula">\[\hat{\mathbf a}=\frac{\mathbf a}{\|\mathbf a\|},\qquad
\|\mathbf a\|=\sqrt{a_x^2+a_y^2+a_z^2}.\]</div>
<p>叉积不满足交换律：\(\mathbf a\times\mathbf b=-(\mathbf b\times\mathbf a)\)。
它的长度满足 \(\|\mathbf a\times\mathbf b\|=\|\mathbf a\|\,\|\mathbf b\|\sin\theta\)，恰好是以 a、b 为邻边的平行四边形面积。</p>
<div class="callout"><strong>零向量不能归一化。</strong> 当 \(\|\mathbf a\|=0\) 时，方向没有定义。
真实代码应在这一系统边界返回明确状态，而不是让 NaN 扩散到后续矩阵。</div>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "从点积得到夹角与投影",
                r"""<p>把向量 b 分解为“沿 a 的部分”和“垂直 a 的部分”。
沿 a 的单位方向是 \(\hat{\mathbf a}\)，投影标量为 \(\mathbf a\cdot\hat{\mathbf a}\)，投影向量为
\((\mathbf a\cdot\hat{\mathbf a})\hat{\mathbf a}\)。代入 \(\mathbf a\cdot\mathbf b=\|\mathbf a\|\|\mathbf b\|\cos\theta\)：</p>
<div class="formula">\[\begin{aligned}
\operatorname{proj}_{\mathbf a}(\mathbf b)
&=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|^2}\mathbf a, \\
\cos\theta
&=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\,\|\mathbf b\|}.
\end{aligned}\]</div>
<p>因此，同向单位向量的点积为 1，反向为 −1，垂直为 0。
光照中的 \(\mathbf n\cdot\mathbf l\)、背面判断和视锥点乘，都建立在这一思想上。</p>
<p>叉积的垂直性可以直接验证。对任意向量 c：</p>
<div class="formula">\[(\mathbf a\times\mathbf b)\cdot\mathbf a=0,\qquad
(\mathbf a\times\mathbf b)\cdot\mathbf b=0.\]</div>
<p>展开第一式会得到
\(a_yb_za_x-a_zb_ya_x+a_zb_xa_y-a_xb_za_y+a_xb_ya_z-a_yb_xa_z=0\)，
所有项两两抵消。由长度公式还可知，两个共线向量的叉积为零向量。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "同时计算点积、夹角和法线",
                r"""<p>取 \(\mathbf a=(1,2,3)\)、\(\mathbf b=(2,-1,2)\)。</p>
<div class="formula">\[\begin{aligned}
\mathbf a\cdot\mathbf b &= 1\cdot2+2\cdot(-1)+3\cdot2=1, \\
\|\mathbf a\| &= \sqrt{14}\approx3.742, & \|\mathbf b\| &=3, \\
\cos\theta &= \frac{1}{3\sqrt{14}}\approx0.08909, &
\theta&\approx84.89^\circ.
\end{aligned}\]</div>
<p>叉积为：</p>
<div class="formula">\[\begin{aligned}
\mathbf a\times\mathbf b
&=(2\cdot2-3\cdot(-1),\;3\cdot2-1\cdot2,\;1\cdot(-1)-2\cdot2) \\
&=(7,4,-5).
\end{aligned}\]</div>
<p>其长度为 \(\sqrt{90}\approx9.487\)。用 \(\sqrt{14}\cdot3\cdot\sin84.89^\circ\)
计算也得到相同结果。归一化后法线为
\((7,4,-5)/\sqrt{90}\approx(0.738,0.422,-0.527)\)。</p>
<div class="callout"><strong>符号检查：</strong> 交换 a、b 后，叉积应变为 (−7,−4,5)，
点积、长度和夹角不变。</div>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "渲染器需要反复使用的向量原语",
                r"""<div class="code-card"><div class="code-toolbar"><span>PYTHON · 三维向量</span><button class="copy-button" type="button" data-copy>复制</button></div>
<pre><code>def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def normalize(a):
    length = dot(a, a) ** 0.5
    if length == 0:
        raise ValueError("zero vector has no direction")
    return tuple(component / length for component in a)

cos_theta = dot(a, b) / (dot(a, a)**0.5 * dot(b, b)**0.5)</code></pre></div>
<p>GPU 着色器通常直接提供 <code>dot</code>、<code>cross</code> 和 <code>normalize</code>。
自己实现一次仍有价值：它能揭示数据类型、精度和零向量处理等问题。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "用符号性质检查结果",
                r"""<details><summary>问题 1：(1,0,0) 与 (0,1,0) 的点积和叉积是什么？</summary>
<p><strong>答案：</strong>点积为 0；叉积为 (0,0,1)。</p></details>
<details><summary>问题 2：为什么 (2,0,0) 与 (4,0,0) 的叉积为零？</summary>
<p><strong>答案：</strong>它们共线，sin θ=0，无法围成非零平行四边形。</p></details>
<details><summary>问题 3：单位向量点积能否略大于 1？</summary>
<p><strong>答案：</strong>数学上不能；浮点舍入可能产生 1.0000001，计算夹角前应截断到 [−1,1]。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "向量错误往往来自约定",
                r"""<ol class="debug-list"><li><strong>检查手系：</strong>同一组坐标在左手系和右手系中，叉积方向相反。</li>
<li><strong>检查分量顺序：</strong>叉积第二项常被误写成 \(a_zb_y-a_yb_z\)，导致整体符号错误。</li>
<li><strong>区分点与方向：</strong>法线、光线用向量；表面位置用点。点与点不能直接点积。</li>
<li><strong>检查长度：</strong>用归一化后的向量做方向判断，而不是任意长度的向量。</li>
<li><strong>观察退化：</strong>三角形面积接近零时，法线会剧烈变化，应检查重复顶点或共线边。</li></ol>
<p>固定测试向量 \(\mathbf x=(1,0,0)\)、\(\mathbf y=(0,1,0)\)、\(\mathbf z=(0,0,1)\)，
验证 \(\mathbf x\times\mathbf y=\mathbf z\)、\(\mathbf x\cdot\mathbf z=0\)，比随意画图更快定位符号问题。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "浏览器向量实验与项目边界",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章提供二维分量控制、点积、夹角、
叉积和归一化读数，并用 SVG 显示投影关系。它是理解渲染数学的概念工具。</p>
<p><span class="status-pill">现有 RenderTorch</span> 项目中的相机和投影大量使用 PyTorch 张量，
但尚没有独立、通用的向量运算模块。本章不把零散矩阵片段包装成已经完成的数学库。</p>
<p><span class="status-pill">后续实现路线</span> 若后续确实需要统一向量层，应先明确点/向量类型、
手系和设备张量约定，再用第 6 章的法线与第 15 章的射线求交验证。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "vector-demo",
                "点积、叉积与归一化实验室",
                "demos/chapter02.js",
                "拖动两个向量的分量，观察夹角、投影、叉积方向与长度关系。",
            ),
        ),
        "sources": (),
    },
    "chapter03": {
        "sections": (
            section(
                "intuition",
                "让几何移动而不修改每个顶点",
                "矩阵是坐标变换的规则",
                r"""<p>一个模型可能包含数十万个顶点。逐个修改坐标既慢又容易不一致。
更高效的方法是构造一个矩阵，对每个点执行同一套线性变换，再统一加上平移。
模型移动、相机旋转和物体缩放，都可以写成这种规则。</p>
<p>本章统一采用<strong>列向量</strong>约定：点是四元素列向量，矩阵写在左边，
\(\mathbf p'=M\mathbf p\)。有些 API 使用行向量，此时矩阵需要转置、书写顺序也要反过来。
约定本身没有优劣，但全篇必须一致。</p>
<div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="一个正方形先缩放再旋转最后平移">
<defs><marker id="arrow-a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#c66b35"/></marker></defs>
<rect x="70" y="100" width="80" height="80" fill="none" stroke="#087f83" stroke-width="4"/>
<rect x="190" y="70" width="160" height="160" fill="none" stroke="#c66b35" stroke-width="3" transform="rotate(-25 270 150)"/>
<path d="M260 170l120 15" stroke="#62727a" stroke-width="3" stroke-dasharray="7 5" marker-end="url(#arrow-a)"/>
<text x="75" y="210" fill="#087f83">局部形状</text><text x="210" y="250" fill="#c66b35">缩放 → 旋转 → 平移</text>
</svg><p><strong>图 3.1</strong>　复合变换从右向左作用于列向量，几何顺序则是先缩放、再旋转、最后平移。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "齐次点与四种基本矩阵",
                r"""<p>三维点写成 \(\mathbf p=(x,y,z,1)^\mathsf T\)。第四个分量用于区分平移：
点的 w=1，方向向量的 w=0。乘以含平移的矩阵时，方向不会受到位置平移影响。</p>
<div class="formula">\[\begin{aligned}
T(\mathbf t)&=
\begin{pmatrix}
1&0&0&t_x\\
0&1&0&t_y\\
0&0&1&t_z\\
0&0&0&1
\end{pmatrix}, &
S(\mathbf s)&=
\begin{pmatrix}
s_x&0&0&0\\
0&s_y&0&0\\
0&0&s_z&0\\
0&0&0&1
\end{pmatrix}.
\end{aligned}\]</div>
<p>绕三个坐标轴旋转的矩阵为：</p>
<div class="formula">\[\begin{aligned}
R_x(\alpha)&=
\begin{pmatrix}
1&0&0&0\\
0&c&-s&0\\
0&s&c&0\\
0&0&0&1
\end{pmatrix}, &
R_y(\alpha)&=
\begin{pmatrix}
c&0&s&0\\
0&1&0&0\\
-s&0&c&0\\
0&0&0&1
\end{pmatrix}, \\
R_z(\alpha)&=
\begin{pmatrix}
c&-s&0&0\\
s&c&0&0\\
0&0&1&0\\
0&0&0&1
\end{pmatrix}. &&
\end{aligned}\]</div>
<p>其中 \(c=\cos\alpha\)、\(s=\sin\alpha\)。列向量约定下，绕 z 轴正向旋转把
+x 转向 +y，符合右手定则。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "复合顺序与法线变换",
                r"""<p>如果先把点缩放 S，再旋转 R，最后平移 T，那么：</p>
<div class="formula">\[\begin{aligned}
\mathbf p_1&=S\mathbf p, \\
\mathbf p_2&=R\mathbf p_1=RS\mathbf p, \\
\mathbf p'&=T\mathbf p_2=TRS\mathbf p, \\
M&=TRS.
\end{aligned}\]</div>
<p>矩阵乘法不满足交换律，所以 \(TRS\) 一般不等于 \(SRT\)。
观察平移能否被后续旋转带走，最容易发现顺序写反。</p>
<p>位置使用 M，法线却通常不能直接使用 M。设模型变换的线性部分为 A，
表面切向量 t 变换为 \(\mathbf t'=A\mathbf t\)。法线 n 必须继续满足
\(\mathbf n'\cdot\mathbf t'=0\)，因此寻找矩阵 N，使：</p>
<div class="formula">\[\begin{aligned}
(N\mathbf n)^\mathsf T(A\mathbf t)
&=\mathbf n^\mathsf T N^\mathsf T A\mathbf t=0, \\
N^\mathsf T A&=I \quad\Longrightarrow\quad N=A^{-\mathsf T}.
\end{aligned}\]</div>
<p>若把整个仿射模型矩阵记作 M，并对方向使用 w=0，则也常把法线矩阵写作
\(M^{-\mathsf T}\)；求值时通常只取其左上角 3×3 线性部分。对于非均匀缩放，
直接使用 M 会让法线方向错误。通常先求逆转置，再归一化结果；若 A 不可逆，
说明变换把几何压扁，需要另行处理。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "一个点与一条法线经过复合变换",
                r"""<p>取 \(\mathbf p=(1,0,0,1)^\mathsf T\)、\(S=\operatorname{diag}(2,1,1)\)、
绕 y 轴旋转 90°，以及 \(\mathbf t=(1,2,3)\)。</p>
<div class="formula">\[\begin{aligned}
S\mathbf p&=(2,0,0,1)^\mathsf T, \\
R_y(90^\circ)S\mathbf p&=(0,0,-2,1)^\mathsf T, \\
TR_y(90^\circ)S\mathbf p&=(1,2,1,1)^\mathsf T.
\end{aligned}\]</div>
<p>若局部法线是 \(\mathbf n=(1,0,0)\)，先求 \(A=R_y(90^\circ)S\)。
缩放部分的逆是 \(\operatorname{diag}(1/2,1,1)\)，旋转后再逆转置，最终法线方向为
\((0,0,-1)\)。这与旋转后的 −z 面一致。</p>
<div class="callout"><strong>错误对照：</strong> 如果直接把 \(\mathbf n=(1,0,0)\) 乘 M，
会得到带平移的奇怪结果；即使忽略平移，先缩放再旋转也会错误地保留 x 方向。</div>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "先建立约定，再封装矩阵",
                r"""<div class="code-card"><div class="code-toolbar"><span>PYTHON · 列向量约定</span><button class="copy-button" type="button" data-copy>复制</button></div>
<pre><code>def translation(tx, ty, tz):
    return [[1, 0, 0, tx], [0, 1, 0, ty],
            [0, 0, 1, tz], [0, 0, 0, 1]]

def scale(sx, sy, sz):
    return [[sx, 0, 0, 0], [0, sy, 0, 0],
            [0, 0, sz, 0], [0, 0, 0, 1]]

def transform_point(matrix, point):
    x, y, z, w = matrix @ [*point, 1.0]
    return (x / w, y / w, z / w)

def transform_normal(matrix, normal):
    linear = matrix[:3, :3]
    result = inverse(linear).T @ normal
    return result / norm(result)</code></pre></div>
<p>本章实验只组合缩放、绕 z 轴旋转和平移，便于把每个中间结果与手算对应。
项目源码中的相机旋转采用张量乘法，阅读时必须先识别它使用的是行向量还是列向量。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "检查矩阵作用的顺序",
                r"""<details><summary>问题 1：列向量约定下，T R S 对应什么几何顺序？</summary>
<p><strong>答案：</strong>先缩放 S，再旋转 R，最后平移 T。</p></details>
<details><summary>问题 2：方向向量的 w 应设为 0 还是 1？</summary>
<p><strong>答案：</strong>0，这样平移不会作用在方向上。</p></details>
<details><summary>问题 3：非均匀缩放后，法线为什么需要逆转置？</summary>
<p><strong>答案：</strong>法线必须与变换后的切向量保持垂直；直接使用 M 一般不满足这个条件。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "变换错误的快速定位",
                r"""<ol class="debug-list"><li><strong>一次只改变一种变换：</strong>先验证纯平移、纯旋转或纯缩放，再组合。</li>
<li><strong>使用轴对齐测试点：</strong>(1,0,0) 经过 90° 旋转应落到坐标轴上，符号一目了然。</li>
<li><strong>检查齐次 w：</strong>点的 w 通常为 1，方向为 0；投影变换后则需要透视除法。</li>
<li><strong>打印中间结果：</strong>分别输出 S p、R S p、T R S p，不要只检查最终矩阵。</li>
<li><strong>法线单独测试：</strong>用非均匀缩放观察法线是否仍与对应边垂直。</li></ol>""",
            ),
            section(
                "status",
                "实现状态",
                "概念变换器与现有相机代码",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验实时显示点、法线和物体轮廓的变换，
并列出点与法线的变换结果。它用于建立矩阵顺序的直觉，不是完整场景图。</p>
<p><span class="status-pill">现有 RenderTorch</span>
<code>viewing/camera.py</code> 已包含相机旋转、平移和视图矩阵片段，但其张量排列采用行方向计算。
阅读第 9 章场景图时需要先做一次约定核对。</p>
<p><span class="status-pill">后续实现路线</span> 模型矩阵、场景图、法线矩阵应在统一坐标约定后实现，
并用旋转可逆性、平移不改变法线、非均匀缩放垂直性作为回归测试。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "transform-demo",
                "变换合成器：缩放、旋转与平移",
                "demos/chapter03.js",
                "调整变换参数，观察列向量公式 M = T R S 与法线矩阵 A⁻ᵀ。",
            ),
        ),
        "sources": (source("viewing/camera.py", "现有实现：相机旋转与平移矩阵"),),
    },
    "chapter04": {
        "sections": (
            section(
                "intuition",
                "从世界中的一个点到屏幕上的一个位置",
                "相机把三维空间压缩到二维图像",
                r"""<p>模型坐标描述物体自身，世界坐标描述整个场景，相机坐标描述“从相机看过去”的位置。
把相机放在原点、视线朝向 −z，很多计算会变得简单，因此渲染器先用视图变换把世界点搬到相机空间。</p>
<p>相机空间仍不是屏幕。透视投影让远处的物体看起来更小，把视锥体变成规则裁剪体。
随后除以 w 得到 NDC，再通过视口变换映射到实际像素。整条链路是：</p>
<div class="formula">\[\text{world point}\;\longrightarrow\;\text{view point}
\;\longrightarrow\;\text{clip point}\;\longrightarrow\;\text{NDC}
\;\longrightarrow\;\text{pixel}.\]</div>
<div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="世界点到相机空间、裁剪体和屏幕的变换链">
<defs><marker id="arrow-a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#c66b35"/></marker></defs>
<path d="M85 225L225 75L390 225Z" fill="#d8eeea" stroke="#087f83" stroke-width="2"/>
<circle cx="85" cy="225" r="7" fill="#c66b35"/><text x="45" y="255" fill="#854721">相机</text>
<path d="M285 75L285 225" stroke="#62727a" stroke-dasharray="5 5"/><text x="275" y="55">近</text>
<path d="M390 75L390 225" stroke="#62727a" stroke-dasharray="5 5"/><text x="380" y="55">远</text>
<path d="M390 150h95" stroke="#c66b35" stroke-width="3" marker-end="url(#arrow-a)"/>
<rect x="505" y="95" width="105" height="105" rx="8" fill="#132832"/><circle cx="557" cy="148" r="8" fill="#9ed9d0"/>
<text x="90" y="40">视图</text><text x="310" y="40">投影</text><text x="505" y="230">视口</text>
</svg><p><strong>图 4.1</strong>　每个阶段只做一件事；调试时可以分别检查各阶段坐标。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "轨道相机与视锥参数",
                r"""<p>轨道相机由眼睛位置 E、观察目标 C、世界上方向 U、垂直视场角 f 和宽高比 a 描述。
本章所有方向都先归一化。</p>
<div class="formula">\[\begin{aligned}
\mathbf f&=\operatorname{normalize}(\mathbf C-\mathbf E)
&&\text{（相机前向）},\\
\mathbf r&=\operatorname{normalize}(\mathbf f\times\mathbf U)
&&\text{（相机右向）},\\
\mathbf u&=\mathbf r\times\mathbf f
&&\text{（相机上向）},\\
a&=\frac{W}{H}. &&
\end{aligned}\]</div>
<p>当相机恰好与世界上方向共线时，\(\operatorname{cross}(\mathbf f,\mathbf U)\) 会接近零向量。
因此 up 不能任意选取，实际实现应选择不与视线平行的参考方向。</p>
<p>视图矩阵以 \(\mathbf r\)、\(\mathbf u\)、\(-\mathbf f\) 为行，并用相机位置平移。
第三行取负，使相机前方点满足 \(z_v&lt;0\)：</p>
<div class="formula">\[V=
\begin{pmatrix}
\mathbf r^\mathsf T&-\mathbf r\cdot\mathbf E\\
\mathbf u^\mathsf T&-\mathbf u\cdot\mathbf E\\
-\mathbf f^\mathsf T&\mathbf f\cdot\mathbf E\\
0&1
\end{pmatrix}.\]</div>
<p>于是相机位置变为原点，前向变为 −z。投影矩阵还需要近裁剪距离 n 和远裁剪距离 far：</p>
<div class="formula">\[\begin{aligned}
P&=
\begin{pmatrix}
f/a&0&0&0\\
0&f&0&0\\
0&0&\dfrac{\mathrm{far}+n}{n-\mathrm{far}}&\dfrac{2\,\mathrm{far}\,n}{n-\mathrm{far}}\\
0&0&-1&0
\end{pmatrix},\\
f&=\frac{1}{\tan(\mathrm{vertical\_fov}/2)}. &&
\end{aligned}\]</div>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "从裁剪坐标推导 NDC 与像素",
                r"""<p>设视图空间点为 \((x_v,y_v,z_v)\)，相机看向 −z，因此前方点满足
\(z_v&lt;0\)。投影后：</p>
<div class="formula">\[\begin{aligned}
x_{\mathrm{clip}}&=(f/a)x_v, & y_{\mathrm{clip}}&=fy_v, \\
z_{\mathrm{clip}}&=\frac{\mathrm{far}+n}{n-\mathrm{far}}z_v
+\frac{2\,\mathrm{far}\,n}{n-\mathrm{far}}, &
w_{\mathrm{clip}}&=-z_v.
\end{aligned}\]</div>
<p>透视除法把裁剪空间变成 NDC：</p>
<div class="formula">\[\begin{aligned}
x_{\mathrm{ndc}}&=\frac{x_{\mathrm{clip}}}{w_{\mathrm{clip}}},\\
y_{\mathrm{ndc}}&=\frac{y_{\mathrm{clip}}}{w_{\mathrm{clip}}},\\
z_{\mathrm{ndc}}&=\frac{z_{\mathrm{clip}}}{w_{\mathrm{clip}}}.
\end{aligned}\]</div>
<p>本章采用 OpenGL 风格 NDC：x、y 位于 [−1,1]，z 位于 [−1,1]。
而图像左上角是 (0,0)，所以视口变换为：</p>
<div class="formula">\[\begin{aligned}
x_{\mathrm{pixel}}&=\frac{(x_{\mathrm{ndc}}+1)W}{2},\\
y_{\mathrm{pixel}}&=\frac{(1-y_{\mathrm{ndc}})H}{2}.
\end{aligned}\]</div>
<p>注意 y 前的负号：NDC 向上，图像 y 向下。
深度 z_ndc 并不随距离线性变化，这就是为什么光栅化需要深度缓冲和透视矫正插值。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "把一个相机前方点映射到屏幕",
                r"""<p>取 \(W=512\)、\(H=256\)、垂直 fov=60°，
相机位于 \((0,0,5)\) 并看向原点。此时
\(\mathrm{aspect}=2\)、\(f=1/\tan30^\circ=\sqrt{3}\approx1.732\)。</p>
<p>世界点 \(\mathbf p=(1,0,0)\) 的视图坐标是 \((1,0,-5)\)：</p>
<div class="formula">\[\begin{aligned}
w_{\mathrm{clip}}&=5,\\
x_{\mathrm{ndc}}&=\frac{(\sqrt{3}/2)\cdot1}{5}\approx0.173205, &
y_{\mathrm{ndc}}&=0,\\
x_{\mathrm{pixel}}&=\frac{(0.173205+1)\cdot512}{2}\approx300.34, &
y_{\mathrm{pixel}}&=128.
\end{aligned}\]</div>
<p>这个点位于画面中心右侧约 44.34 像素处。
若相机后退到 z=10，同一世界点的 x_ndc 约减半，屏幕位置会更靠近中心，这正是透视效果。</p>
<div class="callout"><strong>近裁剪检查：</strong> 位于相机平面上或相机背后的点不能直接做透视除法，
必须先做齐次裁剪，否则 w≤0 会破坏投影。</div>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "按阶段实现投影函数",
                r"""<div class="code-card"><div class="code-toolbar"><span>PYTHON · 列向量管线</span><button class="copy-button" type="button" data-copy>复制</button></div>
<pre><code>def look_at(eye, target, up):
    forward = normalize(target - eye)
    right = normalize(cross(forward, up))
    camera_up = cross(right, forward)
    return rows(
        [*right, -dot(right, eye)],
        [*camera_up, -dot(camera_up, eye)],
        [-forward[0], -forward[1], -forward[2], dot(forward, eye)],
        [0, 0, 0, 1],
    )

def project_to_pixel(point, view, projection, width, height):
    clip = projection @ (view @ [*point, 1.0])
    if clip[3] <= 0:
        return None  # 相机平面或背后，需进入裁剪流程
    ndc = clip[:3] / clip[3]
    x = (ndc[0] + 1) * width / 2
    y = (1 - ndc[1]) * height / 2
    return x, y, ndc[2]</code></pre></div>
<p>RenderTorch 的 <code>Projection</code> 已包含透视矩阵和屏幕变换矩阵；
相机矩阵由旋转矩阵乘平移矩阵得到。阅读源码时要把本章列向量公式与项目的张量排列逐项对照。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "检查视图、投影和视口方向",
                r"""<details><summary>问题 1：为什么视口变换中 y 前面有负号？</summary>
<p><strong>答案：</strong>NDC 的 y 向上，图像像素 y 向下，需要翻转。</p></details>
<details><summary>问题 2：物体远离相机时，投影大小通常怎样变化？</summary>
<p><strong>答案：</strong>在固定 fov 下，x_ndc 和 y_ndc 的绝对值减小，因此屏幕尺寸变小。</p></details>
<details><summary>问题 3：相机背后的点为什么不能只做一次透视除法？</summary>
<p><strong>答案：</strong>其 w_clip≤0，直接除法会得到错误的坐标或符号，必须先裁剪。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "沿变换链逐段排查",
                r"""<ol class="debug-list"><li><strong>检查相机位置：</strong>视图变换后，眼睛应恰好位于原点。</li>
<li><strong>检查视线：</strong>观察目标应位于相机前方，即视图 z 为负。</li>
<li><strong>检查 up：</strong>不要把世界上方向直接当相机右向；右向由前向与 up 的叉积得到。</li>
<li><strong>检查宽高比：</strong>垂直 fov 固定时，x 投影应除以 aspect，否则画面被拉伸。</li>
<li><strong>检查像素原点：</strong>若内容上下颠倒，优先检查视口 y 符号。</li>
<li><strong>检查近远平面：</strong>单位、交换 n/far 或使用 DirectX 深度范围都会造成错误遮挡。</li></ol>""",
            ),
            section(
                "status",
                "实现状态",
                "浏览器相机实验与真实源码分层",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验用轨道相机参数实时计算 lookAt、
透视投影和像素坐标，并在二维画布中显示世界点与投影位置。它是数学演示，不是完整三维渲染器。</p>
<p><span class="status-pill">现有 RenderTorch</span>
<code>viewing/camera.py</code> 提供相机位置、偏航/俯仰控制和相机矩阵；
<code>viewing/projection.py</code> 提供透视投影与屏幕变换；
<code>renderer/pipeline.py</code> 将这些矩阵用于顶点管线。</p>
<p><span class="status-pill">后续实现路线</span> 下一单元将把变换后的三角形送入裁剪与光栅化，
并在此基础上解释覆盖、深度和属性插值。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "camera-demo",
                "轨道相机与投影链",
                "demos/chapter04.js",
                "调整相机、fov 和世界点，逐步查看视图坐标、NDC 与屏幕像素。",
            ),
        ),
        "sources": (
            source("viewing/camera.py", "现有实现：相机与 look-at 方向"),
            source("viewing/projection.py", "现有实现：透视与视口矩阵"),
            source("renderer/pipeline.py", "现有实现：顶点变换管线"),
        ),
    },
    "chapter05": {
        "sections": (
            section(
                "intuition",
                "从三个顶点开始",
                "网格是对几何的共享与连接",
                r"""<p>一个点只有位置，一条线段连接两个点，一个三角形由三个顶点围成。
                渲染器偏爱三角形，因为三个顶点总能确定一个平面，而且边函数、重心坐标和裁剪算法都很成熟。
                但场景不会只有少数三角形：球、角色和地形通常包含成千上万个面。</p>
                <p>如果每个三角形都重复保存三个坐标，相邻三角形之间的公共顶点就会存很多遍。
                <strong>索引网格</strong>把顶点位置集中保存一次，再用整数索引说明每个面引用了哪些顶点。
                索引不是另一套坐标，而是指向顶点数组的下标。</p>
                <div class="visual-card"><svg viewBox="0 0 640 280" role="img" aria-label="四边形分别用三角形扇和三角形带连接">
                <defs><marker id="mesh-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0 0L7 3L0 6Z" fill="#62727a"/></marker></defs>
                <g transform="translate(30 25)"><path d="M50 180L230 180L230 40L50 40Z" fill="#d8eeea" stroke="#087f83" stroke-width="3"/>
                <path d="M50 40L230 180" stroke="#c66b35" stroke-width="3"/><path d="M50 180A253 253 0 0 0 230 40" fill="none" stroke="#62727a" stroke-dasharray="5 5" marker-end="url(#mesh-arrow)"/>
                <g fill="#17252d" font-size="15"><text x="30" y="205">p0</text><text x="240" y="205">p1</text><text x="240" y="35">p2</text><text x="25" y="35">p3</text></g>
                <text x="88" y="225" fill="#854721">共享对角线</text></g>
                <g transform="translate(355 25)"><path d="M40 180L220 180L220 40L40 40Z" fill="#f2dfcf" stroke="#c66b35" stroke-width="3"/>
                <path d="M40 180L220 40" stroke="#087f83" stroke-width="3"/><g fill="#17252d" font-size="15"><text x="20" y="205">v0</text><text x="230" y="205">v1</text><text x="230" y="35">v3</text><text x="20" y="35">v2</text></g>
                <text x="60" y="225" fill="#854721">沿条带依次连接</text></g></svg>
                <p><strong>图 5.1</strong>　左侧是扇形三角化；右侧把四边形顶点按条带顺序重新排列。
                两者几何相同，但描述三角形的索引模式不同。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "顶点、面、边与索引缓冲",
                r"""<p>用 \(\mathbf p_i=(x_i,y_i,z_i)\) 表示第 \(i\) 个顶点。
                一个三角形面保存三个整数，例如 \(f=(i_0,i_1,i_2)\)。
                边通常按无序对 \(\{i,j\}\) 记录，因此共享边不因端点顺序相反而被当成两条边。</p>
                <div class="formula">\[\begin{aligned}
                \text{vertices}&=[\mathbf p_0,\mathbf p_1,\ldots,\mathbf p_{V-1}],\\
                \text{indices}&=[i_0,i_1,i_2,\ldots],\\
                f_k&=(i_{3k},i_{3k+1},i_{3k+2}).
                \end{aligned}\]</div>
                <p>对四边形 \(\mathbf p_0=(0,0)\)、\(\mathbf p_1=(1,0)\)、
                \(\mathbf p_2=(1,1)\)、\(\mathbf p_3=(0,1)\)，扇形三角化得到：</p>
                <div class="formula">\[f_0=(0,1,2),\qquad f_1=(0,2,3).\]</div>
                <p>两个面共享边 \(\{0,2\}\)。如果改成无索引的三角形汤，就要保存
                \(\mathbf p_0,\mathbf p_1,\mathbf p_2,\mathbf p_0,\mathbf p_2,\mathbf p_3\)
                六个顶点角点；索引网格只保存四个独立顶点。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "三角形扇、三角形带与索引复用",
                r"""<p>三角形扇固定一个锚点。给定 \(m+1\) 个顶点，它生成 \(m-1\) 个三角形：</p>
                <div class="formula">\[f_k=(0,k+1,k+2),\qquad k=0,\ldots,m-2.\]</div>
                <p>三角形带沿一条折线前进，每新增一个顶点就与上一对相邻顶点组成新三角形。
                若条带顶点依次为 \(\mathbf v_0,\mathbf v_1,\ldots,\mathbf v_m\)，则：</p>
                <div class="formula">\[T_k=\begin{cases}
                (\mathbf v_0,\mathbf v_1,\mathbf v_2),&k=0,\\
                (\mathbf v_{k-1},\mathbf v_k,\mathbf v_{k+1}),&k\ge1,
                \end{cases}\]</div>
                <p>奇偶三角形的绕序可能需要交替，才能保持连续朝外；这只是索引排列问题，不改变几何形状。
                本章的四边形条带顺序是 \(\mathbf p_0,\mathbf p_1,\mathbf p_3,\mathbf p_2\)，
                两个三角形是 \((0,1,2)\) 与 \((2,1,3)\)，其中索引仍指向原来的四个顶点。</p>
                <p>索引节省的不只是内存，更是重复计算。考虑 \(m\times n\) 个四边形的规则网格，
                索引网格的顶点数和三角形数分别为：</p>
                <div class="formula">\[\begin{aligned}
                V_{\text{indexed}}&=(m+1)(n+1),\\
                F&=2mn,\\
                V_{\text{soup}}&=6mn.
                \end{aligned}\]</div>
                <p>当 \(m=n=2\) 时，索引网格只需变换 9 个顶点，三角形汤却要处理 24 个重复角点。
                索引缓冲仍有 \(6mn\) 个整数，因此它不是“零成本”，而是把顶点变换、内存和缓存成本从面数中分离出来。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "一个四边形的三种连接方式",
                r"""<p>同一个四边形的独立顶点始终是 4 个。三角形扇需要 2 个面描述，
                条带需要 2 个面描述，显式索引数组需要 6 个整数。它们的三角形—顶点引用总数都是
                \(2\times3=6\)：</p>
                <div class="formula">\[\begin{array}{c|c|c|c}
                \text{方式}&\text{顶点记录}&\text{面/条带指令}&\text{共享边}\\
                \hline
                \text{fan}&4&2&1\\
                \text{strip}&4&2&1\\
                \text{indexed}&4&6\text{ 个索引}&1
                \end{array}\]</div>
                <p>不要把“顶点数 4”与“索引项数 6”混为一谈。GPU 读取每个面时仍会取出三个索引，
                只是这三个索引可能指向同一个缓存中的顶点记录。</p>
                <div class="callout"><strong>检查绕序：</strong>四边形的顶点若按矩形边界排列为
                \((0,1,2,3)\)，则两个三角形都应保持一致的逆时针方向；否则背面剔除会得到裂缝。</div>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "用共享索引构造四边形网格",
                r"""<div class="code-card"><div class="code-toolbar"><span>PYTHON · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
                faces = [(0, 1, 2), (0, 2, 3)]

                # 条带模式只改变顶点进入条带的顺序：
                strip_order = [0, 1, 3, 2]
                strip_triangles = [(0, 1, 2), (2, 1, 3)]

                edges = set()
                for a, b, c in faces:
                    edges.update({tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))})</code></pre></div>
                <p>RenderTorch 的 <code>MeshResource</code> 已经采用“顶点数组 + 面索引元组”的组织方式。
                当前 <code>faces</code> 允许三边以上的多边形，因此第 10 章还会看到裁剪后的多边形如何重新扇形成三角形。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "确认你理解的是连接关系，而不只是坐标",
                r"""<details><summary>问题 1：两个三角形共享三个顶点，一定共享几何位置吗？</summary>
                <p><strong>答案：</strong>不一定。无索引存储中，数值相同的坐标也可能属于不同顶点记录。
                是否共享取决于索引和拓扑合并策略。</p></details>
                <details><summary>问题 2：一个四边形三角化后有几个索引项？</summary>
                <p><strong>答案：</strong>显式索引方式有 6 个；扇形或条带 API 可以用 4 个顶点和 2 个指令隐式描述。</p></details>
                <details><summary>问题 3：为什么规则网格的三角形汤顶点数是 6mn？</summary>
                <p><strong>答案：</strong>共有 \(2mn\) 个三角形，每个三角形重复保存 3 个角点。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "网格错误的定位清单",
                r"""<ol class="debug-list"><li><strong>索引越界：</strong>检查索引是否从 1 开始。OBJ 的索引从 1 开始，内部数组通常从 0 开始。</li>
                <li><strong>三角形反向：</strong>交换一个面的第二、第三个索引，观察法线是否翻转。</li>
                <li><strong>共享边重复：</strong>用无序对建立边集合，不要依赖字符串或端点顺序。</li>
                <li><strong>重复顶点未合并：</strong>检查位置、法线、UV 是否本应共享；纹理接缝处有时反而必须拆顶点。</li>
                <li><strong>条带奇偶错误：</strong>检查每加入一个顶点后的绕序，不要把世界坐标随手一起重排。</li></ol>
                <p>最小调试网格应当是 1 个三角形、1 个四边形和 1 个四面体。它们分别能暴露基础索引、面拆分为三角形和闭合拓扑问题。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "浏览器网格实验与 RenderTorch 数据结构",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验在同一个四边形上切换扇形、条带和显式索引，
                高亮共享边并统计顶点记录与索引项。它是二维拓扑演示，不是完整 OBJ 查看器。</p>
                <p><span class="status-pill">现有 RenderTorch</span>
                <code>core/scene.py</code> 的 <code>MeshResource</code> 保存顶点和面索引；
                <code>core/primitives.py</code> 用共享顶点构造立方体、平面和球；
                <code>renderer/pipeline.py</code> 遍历面索引并送入裁剪与光栅化。</p>
                <p><span class="status-pill">后续实现路线</span> 可为网格添加边邻接表、退化三角形检查和属性级顶点拆分，
                但这些都应建立在明确的索引契约之上。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "mesh-demo",
                "四边形三角化与索引复用",
                "demos/chapter05.js",
                "切换 fan、strip 与 indexed，比较共享边、顶点记录和索引项。",
            ),
        ),
        "sources": (
            source("core/scene.py", "现有实现：MeshResource 顶点与面索引"),
            source("core/primitives.py", "现有实现：共享顶点构造基本网格"),
            source("renderer/pipeline.py", "现有实现：按面索引遍历三角形"),
        ),
    },
    "chapter06": {
        "sections": (
            section(
                "intuition",
                "让表面有正反两面",
                "拓扑决定连接，绕序决定朝向",
                r"""<p>三个位置相同的三角形，可以因为索引连接方式不同而成为完全不同的网格。
                拓扑研究的是“谁与谁相连”：哪些边共享、每个顶点周围有哪些面、表面有没有破洞。
                法线则进一步要求我们知道表面朝哪一面。</p>
                <p>一个三角形没有先天的“正面”。叉积 \(\mathbf n=(\mathbf p_1-\mathbf p_0)\times
                (\mathbf p_2-\mathbf p_0)\) 只是按照顶点排列计算出一个方向。
                交换任意两个顶点，法线就会反向。因此，绕序是几何数据的一部分，不是绘图顺序的装饰。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="四面体线框与一个面的法线">
                <defs><marker id="normal-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#c66b35"/></marker></defs>
                <path d="M150 240L430 240L290 55L150 240M430 240L510 105L290 55M510 105L150 240" fill="none" stroke="#42545c" stroke-width="3"/>
                <path d="M150 240L430 240L290 55Z" fill="#087f83" fill-opacity=".18" stroke="#087f83" stroke-width="3"/>
                <path d="M290 178L355 115" stroke="#c66b35" stroke-width="4" marker-end="url(#normal-arrow)"/>
                <text x="363" y="109" fill="#854721">n_face</text><text x="120" y="266">A</text><text x="435" y="266">B</text><text x="295" y="42">C</text><text x="518" y="110">D</text>
                </svg><p><strong>图 6.1</strong>　四面体有 4 个顶点、6 条边和 4 个三角面。高亮面的法线由该面绕序决定。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "边、流形、Euler 示性数与法线",
                r"""<p>边 \(\{i,j\}\) 是无序顶点对。一个闭合三角网格如果每条边恰好邻接两个面，
                通常称为二维流形；三条以上面共享同一条边则会出现非流形连接。</p>
                <p>Euler 公式把顶点、边、面的数量联系起来：</p>
                <div class="formula">\[V-E+F=\chi.\]</div>
                <p>其中 \(\chi\) 是 Euler 示性数。球面拓扑满足 \(\chi=2\)；
                有一个手柄的环面满足 \(\chi=0\)。公式不是“任何三个数凑出 2 就对”的充分条件，
                它只是检查拓扑一致性的必要条件之一。</p>
                <p>三角形面法线和顶点法线分别记为：</p>
                <div class="formula">\[\begin{aligned}
                \mathbf n_{\mathrm{face}}&=
                \frac{(\mathbf p_1-\mathbf p_0)\times(\mathbf p_2-\mathbf p_0)}
                {\lVert(\mathbf p_1-\mathbf p_0)\times(\mathbf p_2-\mathbf p_0)\rVert},\\
                \mathbf n_{\mathrm{vertex}}&=
                \operatorname{normalize}\!\left(\sum_{f\ni i}A_f\mathbf n_f\right).
                \end{aligned}\]</div>
                <p>\(A_f\) 是三角形面积。面积加权能避免大三角形在平滑法线中被小三角形“等权淹没”。
                面法线描述平面朝向，顶点法线描述局部平滑朝向。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "用四面体验证拓扑，再翻转一个面",
                r"""<p>四面体的四个顶点两两相连，共有 \(\binom{4}{2}=6\) 条边。
                每个面都是三角形，共 4 个面。因此：</p>
                <div class="formula">\[V-E+F=4-6+4=2.\]</div>
                <p>这与球面拓扑一致。若把面 \((A,B,C)\) 改成 \((A,C,B)\)，顶点、边和面的数量完全不变，
                Euler 检查仍通过，但面法线变为 \(-\mathbf n_{\mathrm{face}}\)。
                这说明拓扑检查不能替代绕序检查。</p>
                <p>面积加权顶点法线的推导很直接。第 \(f\) 个面的未归一化叉积模长是面积的两倍，
                因此可以用 \(\mathbf m_f=(\mathbf p_1-\mathbf p_0)\times(\mathbf p_2-\mathbf p_0)\)
                作为面积加权的方向贡献：</p>
                <div class="formula">\[\mathbf n_i=
                \operatorname{normalize}\!\left(\sum_{f\ni i}\mathbf m_f\right).\]</div>
                <p>相邻面夹角越平缓，求和后的方向越接近公共趋势；尖锐折痕处则可能更接近某一个面。
                若需要硬边，应复制该边两侧的顶点，使法线属性不共享。</p>
                <div class="callout"><strong>开口网格怎么办？</strong> 平面网格在拓扑上是圆盘，满足 \(V-E+F=1\)。
                不能把开口表面的 \(\chi=1\) 误判为建模错误。</div>""",
            ),
            section(
                "example",
                "数值算例",
                "翻转面后法线如何变化",
                r"""<p>取三角形 \(\mathbf p_0=(0,0,0)\)、\(\mathbf p_1=(1,0,0)\)、
                \(\mathbf p_2=(0,1,0)\)，则：</p>
                <div class="formula">\[\mathbf m=(1,0,0)\times(0,1,0)=(0,0,1),\qquad
                A=\frac12.\]</div>
                <p>未归一化面法线是 \((0,0,1)\)，归一化后仍是 \((0,0,1)\)。
                把面索引改成 \((0,2,1)\)，叉积变成 \((0,0,-1)\)，面积不变。
                若背面剔除规定正面为逆时针，那么这一面会从正面集合中消失。</p>
                <p>四面体中每个顶点邻接 3 个面。正四面体的三个相邻面法线对称求和后，
                方向从质心指向该顶点；这就是平滑顶点法线的几何直觉。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "一次遍历同时检查边与法线",
                r"""<div class="code-card"><div class="code-toolbar"><span>PYTHON · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>edges = {}
vertex_sums = [zero3() for _ in vertices]
for face in faces:
    a, b, c = (vertices[i] for i in face)
    raw_normal = cross(sub(b, a), sub(c, a))
    for i in face:
        vertex_sums[i] = add(vertex_sums[i], raw_normal)
    for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
        key = tuple(sorted((u, v)))
        edges.setdefault(key, []).append(face)

manifold = all(len(incident) == 2 for incident in edges.values())</code></pre></div>
                <p>这里直接用未归一化叉积累加，已经隐式包含面积权重。
                若面已经反向，应先决定是否修复索引，还是把反向保留为双面材质的特征；不要偷偷取绝对值掩盖问题。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "区分拓扑、几何与属性",
                r"""<details><summary>问题 1：翻转一个三角形的索引会改变 V、E、F 吗？</summary>
                <p><strong>答案：</strong>不会，但会改变绕序、面法线和背面可见性。</p></details>
                <details><summary>问题 2：球面网格为什么可能不满足每条边恰好两个面？</summary>
                <p><strong>答案：</strong>如果极点或纹理接缝处复制了顶点，却没有正确连接索引，就会出现边界边或非流形边。</p></details>
                <details><summary>问题 3：为什么顶点法线常用面积加权？</summary>
                <p><strong>答案：</strong>大三角形覆盖更大表面，对局部方向应贡献更多；叉积模长天然包含两倍面积。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "黑斑、裂缝与法线翻转",
                r"""<ol class="debug-list"><li><strong>法线随机闪烁：</strong>检查退化面积是否接近零，并检查面索引是否重复。</li>
                <li><strong>相邻面明暗断裂：</strong>可能是有意硬边，也可能共享了本不该共享的顶点法线。</li>
                <li><strong>表面出现黑洞：</strong>优先检查背面剔除方向与统一绕序。</li>
                <li><strong>Euler 对不上：</strong>统计无序边，检查重复面、缺失面和边界边。</li>
                <li><strong>平滑球出现折痕：</strong>检查极点是否复制了顶点却没有合并法线。</li></ol>
                <p>把法线画成短箭头，通常比只观察颜色更容易定位问题。颜色还会受到光照、材质和深度影响，法线箭头则直接显示数据本身。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "拓扑概念实验与当前逐面着色",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验用四面体展示 \(V-E+F=2\)，
                可翻转面绕序，并比较面法线与面积加权顶点法线。投影仅用于观察，不改变三维叉积。</p>
                <p><span class="status-pill">现有 RenderTorch</span>
                <code>renderer/pipeline.py</code> 在着色时对每个面前三个世界坐标计算叉积并归一化，
                当前是逐面法线；<code>MeshResource</code> 尚不保存独立法线属性。</p>
                <p><span class="status-pill">后续实现路线</span> 可为 <code>MeshResource</code> 增加顶点法线、UV 和硬边标记，
                并在导入阶段验证边邻接与退化三角形。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "topology-demo",
                "四面体拓扑、绕序与法线",
                "demos/chapter06.js",
                "翻转一个面的绕序，切换面法线和顶点法线，核对 V/E/F 与 Euler 示性数。",
            ),
        ),
        "sources": (
            source("core/scene.py", "现有实现：顶点与面拓扑，无独立法线属性"),
            source("renderer/pipeline.py", "现有实现：逐面叉积法线与着色"),
        ),
    },
    "chapter07": {
        "sections": (
            section(
                "intuition",
                "控制点不都在曲线上",
                "用加权平均画出一条平滑曲线",
                r"""<p>折线只由线段组成，而设计曲线需要一种可预测、可局部修改的平滑形状。
                Bézier 曲线把控制点当作“引力”：参数 \(t\) 从 0 走到 1，
                曲线在每个时刻取控制点附近的一个加权平均。除端点外，控制点通常并不落在曲线上。</p>
                <p>控制点数量决定次数：两个控制点得到直线，三个得到二次曲线，四个得到三次曲线。
                本章先掌握三次 Bézier 和 de Casteljau 算法，再把同一思想推广到 B-spline 与 NURBS。</p>
                <div class="visual-card"><svg viewBox="0 0 640 290" role="img" aria-label="四个控制点与三次贝塞尔曲线">
                <defs><marker id="curve-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0 0L7 3L0 6Z" fill="#087f83"/></marker></defs>
                <path d="M70 220L210 70L400 80L570 210" fill="none" stroke="#9aa8a2" stroke-width="2" stroke-dasharray="7 6"/>
                <path d="M70 220C180 20 330 190 570 210" fill="none" stroke="#087f83" stroke-width="4" marker-end="url(#curve-arrow)"/>
                <g fill="#c66b35"><circle cx="70" cy="220" r="8"/><circle cx="210" cy="70" r="8"/><circle cx="400" cy="80" r="8"/><circle cx="570" cy="210" r="8"/></g>
                <g fill="#17252d" font-size="15"><text x="48" y="245">P0</text><text x="190" y="55">P1</text><text x="405" y="65">P2</text><text x="560" y="190">P3</text></g>
                <circle cx="320" cy="126" r="6" fill="#132832"/><text x="330" y="120">t = 0.5</text>
                </svg><p><strong>图 7.1</strong>　虚线连接控制点形成控制多边形；绿色曲线是实际 Bézier 曲线。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "次数、Bernstein 基与参数区间",
                r"""<p>设 \(n+1\) 个控制点为 \(\mathbf P_0,\ldots,\mathbf P_n\)。
                次数为 \(n\) 的 Bézier 曲线定义为：</p>
                <div class="formula">\[\mathbf B(t)=\sum_{i=0}^{n}B_i^n(t)\mathbf P_i,\qquad 0\le t\le1,\]</div>
                <p>其中 Bernstein 基函数为：</p>
                <div class="formula">\[B_i^n(t)=\binom{n}{i}t^i(1-t)^{n-i}.\]</div>
                <p>基函数非负且总和为 1，所以曲线位于控制点的凸包内。
                当 \(t=0\) 时只有 \(B_0^n=1\)，当 \(t=1\) 时只有 \(B_n^n=1\)，
                因此曲线精确经过首尾控制点，但不保证经过中间控制点。</p>
                <p>三次 Bézier 常写成工程中熟悉的形式：</p>
                <div class="formula">\[\begin{aligned}
                \mathbf B(t)={}&(1-t)^3\mathbf P_0+3t(1-t)^2\mathbf P_1\\
                &+3t^2(1-t)\mathbf P_2+t^3\mathbf P_3.
                \end{aligned}\]</div>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "de Casteljau：反复取相邻中点",
                r"""<p>直接展开 Bernstein 多项式可以计算曲线，但 de Casteljau 更容易手算，也更容易画出中间过程。
                第一步在每对相邻控制点之间按比例 \(t\) 插值：</p>
                <div class="formula">\[\mathbf Q_i=(1-t)\mathbf P_i+t\mathbf P_{i+1}.\]</div>
                <p>对 \(\mathbf Q_i\) 重复同一操作，点数每轮减少 1；最后只剩一个点，就是 \(\mathbf B(t)\)。
                对四个控制点，完整过程是：</p>
                <div class="formula">\[\begin{aligned}
                \mathbf Q_0&=(1-t)\mathbf P_0+t\mathbf P_1,&
                \mathbf Q_1&=(1-t)\mathbf P_1+t\mathbf P_2,\\
                \mathbf Q_2&=(1-t)\mathbf P_2+t\mathbf P_3,&
                \mathbf R_0&=(1-t)\mathbf Q_0+t\mathbf Q_1,\\
                \mathbf R_1&=(1-t)\mathbf Q_1+t\mathbf Q_2,&
                \mathbf B(t)&=(1-t)\mathbf R_0+t\mathbf R_1.
                \end{aligned}\]</div>
                <p>导数可以由控制多边形得到。一般地：</p>
                <div class="formula">\[\frac{d\mathbf B}{dt}
                =n\sum_{i=0}^{n-1}B_i^{n-1}(t)(\mathbf P_{i+1}-\mathbf P_i).\]</div>
                <p>因此切线方向来自控制点差分的加权和，线速度则包含参数区间的缩放。
                若用曲线弧长重新参数化，才能得到近似匀速运动。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "四个控制点在 t = 0.5 的全部中间级",
                r"""<p>取二维控制点：</p>
                <div class="formula">\[\mathbf P_0=(0,0),\quad \mathbf P_1=(1,2),\quad
                \mathbf P_2=(3,2),\quad \mathbf P_3=(4,0).\]</div>
                <p>当 \(t=0.5\) 时，每一级都取相邻点的中点：</p>
                <div class="formula">\[\begin{aligned}
                \mathbf Q_0&=(0.5,1),&\mathbf Q_1&=(2,2),&\mathbf Q_2&=(3.5,1),\\
                \mathbf R_0&=(1.25,1.5),&\mathbf R_1&=(2.75,1.5),\\
                \mathbf B(0.5)&=(2,1.5).
                \end{aligned}\]</div>
                <p>导数使用二次 Bernstein 权重 \(0.25,0.5,0.25\)：</p>
                <div class="formula">\[\mathbf B'(0.5)=3\left[0.25(1,2)+0.5(2,0)+0.25(1,-2)\right]=(4.5,0).\]</div>
                <p>所以中点处的切线水平向右。实验会显示这 7 个点以及导数向量，便于逐项核对。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "显式三级插值",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 三次 de Casteljau</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>function bezierPoint(points, t) {
  const q = points.slice(0, -1).map((p, i) =&gt;
    lerpPoint(p, points[i + 1], t));
  const r = [lerpPoint(q[0], q[1], t), lerpPoint(q[1], q[2], t)];
  return { point: lerpPoint(r[0], r[1], t), levels: [points, q, r] };
}

function lerpPoint(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}</code></pre></div>
                <p>三次公式只是固定次数的特例。实际建模库通常实现任意次数，或直接提供 B-spline 求值器，
                以便用更多控制点表达局部细节，而不必把次数提高到数值不稳定的程度。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "参数曲线不是按弧长等速前进",
                r"""<details><summary>问题 1：二次 Bézier 在 t=0.5 时一定经过中间控制点吗？</summary>
                <p><strong>答案：</strong>不一定。它等于
                \(0.25\mathbf P_0+0.5\mathbf P_1+0.25\mathbf P_2\)，中间点权重只有 0.5。</p></details>
                <details><summary>问题 2：为什么曲线总在控制点凸包内？</summary>
                <p><strong>答案：</strong>Bernstein 基函数非负且总和为 1，曲线是控制点的凸组合。</p></details>
                <details><summary>问题 3：把参数 t 从 0 到 1 线性推进，为什么看起来可能时快时慢？</summary>
                <p><strong>答案：</strong>\(|\mathbf B'(t)|\) 随控制点布局改变，参数速度不等于弧长速度。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "曲线形状与速度问题",
                r"""<ol class="debug-list"><li><strong>端点接不上：</strong>检查曲线是否在 \(t=0\) 和 \(t=1\) 精确经过首尾控制点。</li>
                <li><strong>形状过冲：</strong>检查是否误用了高次多项式或错误节点向量；Bézier 本身位于凸包内。</li>
                <li><strong>动画忽快忽慢：</strong>区分参数时间与弧长时间，必要时建立弧长查找表。</li>
                <li><strong>导数方向错误：</strong>确认差分是 \(\mathbf P_{i+1}-\mathbf P_i\)，而不是反向。</li>
                <li><strong>拼接处有折角：</strong>检查相邻曲线端点的位置和切线是否按 \(C^0\)、\(C^1\) 或 \(C^2\) 要求匹配。</li></ol>""",
            ),
            section(
                "status",
                "实现状态",
                "曲线概念实验与样条扩展",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验允许拖动四个控制点，
                实时显示三次 Bézier 曲线、de Casteljau 每一级和导数。二维画布用于展示参数关系。</p>
                <p><span class="status-pill">后续实现路线</span> RenderTorch 当前没有独立曲线或 NURBS 模块。
                可以增加 B-spline 求值、节点向量校验和曲线采样为网格的函数；这些属于建模前端，
                不应侵入光栅化核心。</p>
                <p><strong>从 Bézier 到 B-spline：</strong>B-spline 用局部支撑的基函数组合控制点：
                \(\mathbf S(u)=\sum_iN_{i,p}(u)\mathbf P_i\)。NURBS 再引入非负权重：</p>
                <div class="formula">\[\mathbf R(u)=
                \frac{\sum_iN_{i,p}(u)w_i\mathbf P_i}{\sum_iN_{i,p}(u)w_i}.\]</div>
                <p>均匀权重时 NURBS 退化为普通 B-spline；改变权重可以精确表示圆锥曲线等有理形状。
                本章浏览器实验只实现 Bézier，不声称已实现完整 NURBS。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "curve-demo",
                "可拖动控制点的三次 Bézier 曲线",
                "demos/chapter07.js",
                "拖动控制点或使用键盘滑块，查看 de Casteljau 中间级、曲线点和导数。",
            ),
        ),
        "sources": (),
    },
    "chapter08": {
        "sections": (
            section(
                "intuition",
                "把一条曲线沿另一个方向铺开",
                "双参数曲面与可渲染网格",
                r"""<p>曲线只有一个参数 \(u\)，曲面需要两个参数 \(u,v\)。
                固定 \(v\)，沿 \(u\) 方向得到一条曲线；固定 \(u\)，沿 \(v\) 方向得到另一条曲线。
                两组曲线交织成一张连续表面。</p>
                <p>渲染器不能直接绘制“连续曲面”，必须先把参数区域采样为有限个顶点，
                再用三角形连接它们。采样太稀会失去形状，采样太密会增加顶点、三角形和内存成本。
                细分则从已有网格生成更多顶点，以近似更平滑的表面。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="双线性参数曲面网格">
                <defs><pattern id="surface-grid" width="36" height="36" patternUnits="userSpaceOnUse" patternTransform="skewY(-18)"><path d="M36 0H0V36" fill="none" stroke="#8fb5b0" stroke-width="1"/></pattern></defs>
                <path d="M100 180L340 240L555 135L310 75Z" fill="url(#surface-grid)" stroke="#087f83" stroke-width="3"/>
                <path d="M100 180L555 135M340 240L310 75" fill="none" stroke="#c66b35" stroke-width="2"/>
                <text x="375" y="265" fill="#854721">u</text><text x="300" y="55" fill="#854721">v</text>
                <text x="78" y="205">S(0,1)</text><text x="325" y="275">S(1,1)</text><text x="565" y="145">S(1,0)</text><text x="295" y="60">S(0,0)</text>
                </svg><p><strong>图 8.1</strong>　参数矩形被采样为网格。图中网格是拓扑结构，实际曲面可以弯曲。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "张量积曲面、采样与法线",
                r"""<p>给定两个方向上的控制网格 \(\mathbf P_{ij}\)，可用 Bernstein 基构造张量积 Bézier 曲面：</p>
                <div class="formula">\[S(u,v)=\sum_{i=0}^{m}\sum_{j=0}^{n}
                B_i^m(u)B_j^n(v)\mathbf P_{ij}.\]</div>
                <p>当两个方向都只有两个控制点时，就是双线性曲面：
                \(S(u,v)=(1-u)(1-v)\mathbf P_{00}+u(1-v)\mathbf P_{10}+
                (1-u)v\mathbf P_{01}+uv\mathbf P_{11}\)。名字中的“双线性”指对 \(u\) 和 \(v\) 分别线性。</p>
                <p>曲面的两个切向量是偏导数：</p>
                <div class="formula">\[\mathbf S_u=\frac{\partial S}{\partial u},\qquad
                \mathbf S_v=\frac{\partial S}{\partial v},\qquad
                \mathbf n=\operatorname{normalize}(\mathbf S_u\times\mathbf S_v).\]</div>
                <p>如果偏导数为零，该点是奇异点，法线无法由这个公式定义。
                这与三角形面积为 0 的情况类似：几何本身没有可用的局部朝向。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "从解析偏导到有限差分",
                r"""<p>对双线性曲面直接求偏导：</p>
                <div class="formula">\[\begin{aligned}
                \mathbf S_u&=(1-v)(\mathbf P_{10}-\mathbf P_{00})
                +v(\mathbf P_{11}-\mathbf P_{01}),\\
                \mathbf S_v&=(1-u)(\mathbf P_{01}-\mathbf P_{00})
                +u(\mathbf P_{11}-\mathbf P_{10}).
                \end{aligned}\]</div>
                <p>如果只有可求值的黑盒函数，可以用中心有限差分近似：</p>
                <div class="formula">\[\mathbf S_u\approx
                \frac{S(u+h,v)-S(u-h,v)}{2h},\qquad
                \mathbf S_v\approx
                \frac{S(u,v+h)-S(u,v-h)}{2h}.\]</div>
                <p>\(h\) 太大时截断误差明显，\(h\) 太小时浮点相消会放大误差。
                实际实现常选相对于参数尺度的 \(h\approx10^{-3}\)，再针对问题验证。</p>
                <p>一次边中点细分把一个四边形拆成四个：保留 4 个角点，加入 4 个边中点和 1 个面中心。
                因而计数从 \((V,F)=(4,1)\) 变为 \((9,4)\)：</p>
                <div class="formula">\[V' = V+E+F,\qquad F'=4F.\]</div>
                <p>这里没有移动旧顶点，所以只是几何加密，不是 Loop 或 Catmull–Clark 的平滑细分。
                对双线性曲面，边中点加密能更好地逼近原曲面，但角点处的折痕仍由三角形近似决定。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "双线性曲面中心点与法线",
                r"""<p>取四个三维控制点：</p>
                <div class="formula">\[\begin{aligned}
                \mathbf P_{00}&=(0,0,0),&\mathbf P_{10}&=(2,0,0),\\
                \mathbf P_{01}&=(0,1,0),&\mathbf P_{11}&=(2,1,1).
                \end{aligned}\]</div>
                <p>在 \(u=v=0.5\) 处，四个权重都是 0.25：</p>
                <div class="formula">\[S(0.5,0.5)=(1,0.5,0.25).\]</div>
                <p>两个切向量为：</p>
                <div class="formula">\[\mathbf S_u=(2,0,0.5),\qquad
                \mathbf S_v=(0,1,0.5).\]</div>
                <p>叉积得到：</p>
                <div class="formula">\[\mathbf S_u\times\mathbf S_v=(-0.5,-1,2).\]</div>
                <p>归一化前不必先强行把法线长度变成 1；叉积方向已经正确，
                光照使用时再归一化即可。反转参数方向或控制点顺序都会改变法线符号。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "采样、差分和边中点细分",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>function bilinear(p, u, v) {
  const a = lerpPoint(p[0][0], p[1][0], u);
  const b = lerpPoint(p[0][1], p[1][1], u);
  return lerpPoint(a, b, v);
}

const h = 0.001;
const du = scale(sub(surface(u + h, v), surface(u - h, v)), 1 / (2 * h));
const dv = scale(sub(surface(u, v + h), surface(u, v - h)), 1 / (2 * h));
const normal = normalize(cross(du, dv));

const midpointAB = lerpPoint(a, b, 0.5);</code></pre></div>
                <p>实际网格还需要记录哪些边中点是共享的。若每个面独立创建边中点，
                相邻面之间会出现 T 型接缝；正确的一层细分通常先建立边到中点的映射。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "参数方向、细分与法线",
                r"""<details><summary>问题 1：固定 u，S(u,v) 是什么？</summary>
                <p><strong>答案：</strong>一条沿 \(v\) 方向的参数曲线。</p></details>
                <details><summary>问题 2：一次边中点细分后，原来 1 个四边形变成几个三角形？</summary>
                <p><strong>答案：</strong>4 个，同时得到 4 个边中点和 1 个中心点。</p></details>
                <details><summary>问题 3：如果 S_u 与 S_v 平行，会发生什么？</summary>
                <p><strong>答案：</strong>叉积为零，参数映射发生退化，局部法线无法定义。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "曲面褶皱、裂缝和法线跳动",
                r"""<ol class="debug-list"><li><strong>采样出现交叉：</strong>检查控制点顺序是否形成自交参数网格，或 UV 方向是否互换。</li>
                <li><strong>细分出现裂缝：</strong>确认相邻面共享边中点，而不是各自计算后得到不同顶点。</li>
                <li><strong>法线正反随机：</strong>检查 \(u,v\) 参数方向与面绕序是否一致。</li>
                <li><strong>极点闪烁：</strong>检查奇异点，并在极点使用显式法线规则。</li>
                <li><strong>差分误差大：</strong>打印 \(h\)、两个切向量和叉积长度，区分几何退化与数值误差。</li></ol>""",
            ),
            section(
                "status",
                "实现状态",
                "细分概念实验与当前参数曲面能力",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验显示 \(2\times2\) 双线性控制网格、
                一次边中点细分、采样点、解析切向量和有限差分法线，并核对顶点与面计数。</p>
                <p><span class="status-pill">现有 RenderTorch</span>
                <code>core/primitives.py</code> 的球体通过经纬采样直接生成三角网格，
                但没有独立 Bézier 曲面求值器、UV 属性或通用细分器。</p>
                <p><span class="status-pill">后续实现路线</span> 可增加统一网格结构、边邻接表、
                参数曲面采样和带属性共享的一层细分；之后再考虑 Loop/Catmull–Clark 等不同细分方案。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "surface-demo",
                "双线性曲面采样与一次细分",
                "demos/chapter08.js",
                "改变 u/v，比较解析切向量与有限差分，并查看一次细分前后的网格计数。",
            ),
        ),
        "sources": (
            source("core/primitives.py", "现有实现：球体经纬采样与三角面连接"),
            source("core/scene.py", "现有实现：三角网格资源，无 UV 与法线属性"),
        ),
    },
    "chapter09": {
        "sections": (
            section(
                "intuition",
                "局部模型放进全局场景",
                "场景图把变换组织成父子链",
                r"""<p>一个角色的手臂不应把“角色已经走到哪里”写进每一个局部顶点。
                更合理的做法是：模型保留局部坐标，父节点记录角色整体的平移和旋转，
                子节点记录手臂相对肩部的运动。渲染时再沿父子链组合变换。</p>
                <p>模型文件负责几何与属性，场景图负责实例化和层级关系，包围体负责快速排除。
                三者解决不同问题：同一个网格可以被多个场景节点以不同变换重复使用；
                包围体可以先用廉价计算排除明显不可见物体，再进行昂贵的三角形处理。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="父子场景节点、局部点和世界坐标">
                <defs><marker id="scene-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#62727a"/></marker></defs>
                <rect x="245" y="25" width="150" height="58" rx="10" fill="#132832"/><text x="320" y="61" text-anchor="middle" fill="white">Parent · T</text>
                <path d="M320 83V128H205V175" fill="none" stroke="#62727a" stroke-width="2" marker-end="url(#scene-arrow)"/>
                <rect x="105" y="175" width="180" height="58" rx="10" fill="#087f83"/><text x="195" y="211" text-anchor="middle" fill="white">Child · R</text>
                <path d="M285 204L460 204" stroke="#c66b35" stroke-width="2" marker-end="url(#scene-arrow)"/>
                <rect x="470" y="168" width="140" height="72" rx="10" fill="#f2dfcf" stroke="#c66b35"/><text x="540" y="199" text-anchor="middle">p_local</text><text x="540" y="224" text-anchor="middle">→ p_world</text>
                </svg><p><strong>图 9.1</strong>　世界变换由父节点与子节点矩阵复合得到；包围体也随实例变换。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "模型矩阵、场景图、OBJ 与包围体",
                r"""<p>使用第 3 章的列向量约定，把局部点变到世界点的矩阵叫模型矩阵：</p>
                <div class="formula">\[\mathbf p_{\mathrm{world}}=M_{\mathrm{model}}\mathbf p_{\mathrm{local}}.\]</div>
                <p>若父变换为 \(M_{\mathrm{parent}}\)、子变换为 \(M_{\mathrm{child}}\)，则：</p>
                <div class="formula">\[M_{\mathrm{model}}=M_{\mathrm{parent}}M_{\mathrm{child}}.\]</div>
                <p>RenderTorch 当前采用行向量存储顶点，代码写为
                <code>vertices @ transform</code>。行向量习惯下乘法顺序看起来相反，
                即 <code>local @ child @ parent</code>；数学结果与上式相同。</p>
                <p>OBJ 文件用 <code>v x y z</code> 保存位置，用 <code>f i/j/k ...</code> 保存面。
                其中 <code>i</code> 是位置索引，<code>j</code>、<code>k</code> 可分别引用纹理坐标和法线；
                索引从 1 开始，导入内部数组时要减 1。</p>
                <p>轴对齐包围盒 AABB 由六个极值定义：</p>
                <div class="formula">\[\mathrm{AABB}=[\min_x,\max_x]\times[\min_y,\max_y]\times[\min_z,\max_z].\]</div>
                <p>包围球用中心 \(\mathbf c\) 和半径 \(r\) 表示，要求球包含物体的所有顶点：
                \(r=\max_i\lVert\mathbf p_i-\mathbf c\rVert\)。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "父子变换、包围盒与粗粒度视锥判断",
                r"""<p>父节点平移 \(\mathbf t=(2,0,0)\)，子节点绕 z 轴旋转 \(90^\circ\)。
                局部点 \(\mathbf p=(1,0,0)\) 先旋转到 \((0,1,0)\)，再平移到 \((2,1,0)\)：</p>
                <div class="formula">\[\mathbf p_{\mathrm{world}}=
                T(2,0,0)R_z(90^\circ)\mathbf p=(2,1,0).\]</div>
                <p>局部盒子旋转后通常不再轴对齐，因此必须在世界空间重新取所有变换顶点的坐标极值。
                不能只平移原来的 AABB，否则会高估宽高；也不能只旋转中心和半径，除非使用有足够保守余量的包围球。</p>
                <p>视锥可写成六个平面不等式 \(\mathbf n_k\cdot\mathbf x+d_k\ge0\)。
                对包围球，令 \(s_k=\mathbf n_k\cdot\mathbf c+d_k\)：</p>
                <div class="formula">\[\begin{aligned}
                \exists k:\ s_k&lt;-r&\Rightarrow \text{球在视锥外},\\
                \forall k:\ s_k\ge r&\Rightarrow \text{球完全在视锥内},\\
                \text{其他情况}&\Rightarrow \text{可能相交}.
                \end{aligned}\]</div>
                <p>这是保守判断：可能相交就要继续处理三角形，可能相交不等于一定可见。
                对 AABB，可用投影半径
                \(\rho_k=|n_{k,x}|h_x+|n_{k,y}|h_y+|n_{k,z}|h_z\)
                做类似判断，其中 \(h_x,h_y,h_z\) 是 AABB 半尺寸。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "旋转盒子后的世界 AABB",
                r"""<p>取局部正方形盒子 \([0,1]\times[0,1]\)，四个角点为
                \((0,0),(1,0),(1,1),(0,1)\)。先绕原点旋转 \(90^\circ\)：</p>
                <div class="formula">\[R_z(90^\circ)(x,y)=(-y,x).\]</div>
                <p>旋转后的角点为 \((0,0),(0,1),(-1,1),(-1,0)\)，范围是
                \([-1,0]\times[0,1]\)。再施加父节点平移 \((2,0,0)\)，世界 AABB 为：</p>
                <div class="formula">\[[1,2]\times[0,1].\]</div>
                <p>如果误把“先旋转再平移”写成“先平移再旋转”，会得到不同位置。
                场景图的意义正是让这类顺序显式可见，而不是把多个数字手工合并到模型文件里。</p>
                <p>局部盒子中心是 \((0.5,0.5)\)，包围球半径
                \(r=\sqrt{0.5^2+0.5^2}=\sqrt{0.5}\approx0.7071\)。
                包围球比 AABB 更保守，但求交判断通常更便宜。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "从 OBJ 索引到场景节点",
                r"""<div class="code-card"><div class="code-toolbar"><span>PYTHON · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>for line in obj_lines:
    if line.startswith("v "):
        vertices.append([float(value) for value in line.split()[1:]] + [1.0])
    elif line.startswith("f "):
        face = tuple(int(token.split("/")[0]) - 1 for token in line.split()[1:])
        faces.append(face)

world_matrix = parent_matrix @ child_matrix
world_vertices = local_vertices @ world_matrix.T
minimum = world_vertices.min(axis=0)
maximum = world_vertices.max(axis=0)</code></pre></div>
                <p>当前 OBJ 导入只读取位置和面，不解析法线或 UV。场景层也没有通用父子节点树，
                因此本章实验展示的是后续场景图设计，而不是对现有功能的重复。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "坐标空间与包围体",
                r"""<details><summary>问题 1：父平移和子旋转的先后顺序为什么重要？</summary>
                <p><strong>答案：</strong>旋转和平移通常不交换。矩阵乘法顺序决定局部点先经历哪一步。</p></details>
                <details><summary>问题 2：模型旋转后，原来的 AABB 一定仍是最小 AABB 吗？</summary>
                <p><strong>答案：</strong>不一定。应重新计算变换后所有顶点的坐标极值。</p></details>
                <details><summary>问题 3：包围球与视锥“可能相交”代表物体一定可见吗？</summary>
                <p><strong>答案：</strong>不代表。包围体是保守代理，仍可能被其他物体遮挡或完全没有三角形覆盖像素。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "场景坐标与可见性问题",
                r"""<ol class="debug-list"><li><strong>物体位置偏移：</strong>先打印局部点、模型矩阵和世界点，逐级检查父子顺序。</li>
                <li><strong>OBJ 导入崩溃：</strong>检查 1-based 索引、负索引和 <code>v/vt/vn</code> 三种索引语义。</li>
                <li><strong>包围体漏掉物体：</strong>确认包围体由变换后的顶点重新计算，且动画后更新。</li>
                <li><strong>该裁剪的没裁剪：</strong>检查平面法线与点积符号是否采用同一内外约定。</li>
                <li><strong>该显示的被误剔除：</strong>调试时先禁用粗粒度视锥判断，确认是包围体还是三角形管线的问题。</li></ol>""",
            ),
            section(
                "status",
                "实现状态",
                "场景图概念实验与当前项目边界",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验显示父平移、子旋转、局部点到世界点的映射，
                并切换 AABB 与包围球进行二维视锥粗检。它是场景图概念演示，不是三维模型编辑器。</p>
                <p><span class="status-pill">现有 RenderTorch</span>
                <code>core/scene.py</code> 为每个对象保存一个模型矩阵；
                <code>app/controller.py</code> 把基本网格加入场景；
                <code>solid/custom.py</code> 可读取 OBJ 的位置和面；
                <code>renderer/pipeline.py</code> 使用 <code>world_vertices()</code> 得到世界坐标。</p>
                <p><span class="status-pill">后续实现路线</span> 当前场景没有通用父子节点树、AABB 或视锥粗检。
                下一步可以增加场景节点父子关系、变换缓存和世界空间包围体，再考虑对象级剔除。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "scene-demo",
                "父子变换与世界空间包围体",
                "demos/chapter09.js",
                "调整父平移和子旋转，切换 AABB/包围球，查看局部点、世界点和视锥粗检结果。",
            ),
        ),
        "sources": (
            source("core/scene.py", "现有实现：SceneObject 模型矩阵与世界顶点"),
            source("app/controller.py", "现有实现：把网格实例加入场景"),
            source("solid/custom.py", "现有实现：读取 OBJ 位置与面"),
            source("renderer/pipeline.py", "现有实现：使用世界顶点进入投影管线"),
        ),
    },
    "chapter10": {
        "sections": (
            section(
                "intuition",
                "把连续三角形交给离散像素",
                "光栅化就是判断哪些像素中心被覆盖",
                r"""<p>经过投影和视口变换，三角形已经成为屏幕上的三个二维点。
                但屏幕并不是连续画布，而是由有限像素组成。渲染器必须决定：
                某个像素要不要写入这个三角形，以及写入时使用哪些顶点属性。</p>
                <p>齐次裁剪负责先去掉视锥外的部分，透视除法负责把裁剪空间变成 NDC。
                只有完成这两步，边函数才能在正常的二维屏幕坐标中工作。
                若相机背后的顶点直接除以 \(w\)，投影位置会被错误反射或发散，三角形可能连接错误的屏幕区域。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="三角形覆盖像素中心与包围盒">
                <defs><pattern id="raster-grid" width="42" height="42" patternUnits="userSpaceOnUse"><rect width="42" height="42" fill="#f6f3eb" stroke="#d9ddd5"/></pattern></defs>
                <rect x="55" y="30" width="420" height="230" fill="url(#raster-grid)"/>
                <path d="M97 219L347 219L138 73Z" fill="#087f83" fill-opacity=".22" stroke="#087f83" stroke-width="3"/>
                <path d="M76 241H387V52" fill="none" stroke="#c66b35" stroke-width="2" stroke-dasharray="6 5"/>
                <g fill="#c66b35"><circle cx="118" cy="198" r="4"/><circle cx="160" cy="198" r="4"/><circle cx="202" cy="198" r="4"/><circle cx="244" cy="198" r="4"/><circle cx="286" cy="198" r="4"/><circle cx="328" cy="198" r="4"/><circle cx="160" cy="156" r="4"/><circle cx="202" cy="156" r="4"/><circle cx="244" cy="156" r="4"/></g>
                <text x="392" y="245" fill="#854721">包围盒</text><text x="365" y="70" fill="#854721">像素中心</text>
                </svg><p><strong>图 10.1</strong>　只在三角形包围盒内测试像素中心；橙色虚线是包围盒，不是覆盖边界。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "齐次裁剪平面与边函数",
                r"""<p>设裁剪空间点为 \(\mathbf q=(x_c,y_c,z_c,w_c)\)。
                在除以 \(w_c\) 之前，六个裁剪平面要求：</p>
                <div class="formula">\[\begin{aligned}
                -w_c&\le x_c\le w_c,&-w_c&\le y_c\le w_c,\\
                -w_c&\le z_c\le w_c,&w_c&>0.
                \end{aligned}\]</div>
                <p>最后一条保证透视除数有效。near 平面对应 \(z_c+w_c\ge0\)，
                因此相机后方或穿过近平面的几何必须先被裁掉。</p>
                <p>对屏幕二维点，定义边函数：</p>
                <div class="formula">\[E(a,b,p)=(p_x-a_x)(b_y-a_y)-(p_y-a_y)(b_x-a_x).\]</div>
                <p>它等于有向面积的两倍，也等于二维叉积的相反数。
                对同一个三角形，三条边函数使用同一个面积作分母时，符号会相互抵消，
                因此不必强行让某一种顶点顺序产生正面积。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "从裁剪平面到重心坐标",
                r"""<p><strong>第一步：齐次裁剪。</strong> 对每条线段或每个多边形边，
                计算两端到裁剪平面的有符号距离。若两端都在外部，整段丢弃；
                若一内一外，在齐次坐标中求交点：</p>
                <div class="formula">\[t=\frac{d_{\mathrm{prev}}}{d_{\mathrm{prev}}-d_{\mathrm{cur}}},\qquad
                \mathbf q_{\mathrm{hit}}=\mathbf q_{\mathrm{prev}}+
                t(\mathbf q_{\mathrm{cur}}-\mathbf q_{\mathrm{prev}}).\]</div>
                <p>不能先在 NDC 中裁剪，因为除以 \(w\) 会丢失齐次空间中的直线关系。
                裁剪完成后，每个顶点的 \(w\) 都应为正，再执行
                \((x_{\mathrm{ndc}},y_{\mathrm{ndc}},z_{\mathrm{ndc}})=(x_c/w_c,y_c/w_c,z_c/w_c)\)。</p>
                <p><strong>第二步：有符号面积。</strong> 令 \(A=E(a,b,c)\)。
                大三角形面积是 \(|A|/2\)。点 \(p\) 与三条边分别形成三个子三角形，
                它们的面积比就是重心坐标：</p>
                <div class="formula">\[\lambda_0=\frac{E(b,c,p)}{A},\qquad
                \lambda_1=\frac{E(c,a,p)}{A},\qquad
                \lambda_2=1-\lambda_0-\lambda_1.\]</div>
                <p><strong>第三步：覆盖判断。</strong> 对非退化三角形，
                点在里面当且仅当三个 \(\lambda\) 都非负。屏幕像素的中心是
                \((x+0.5,y+0.5)\)，所以遍历整数 \(x,y\) 后要加 0.5。</p>
                <p><strong>第四步：包围盒。</strong> 最小候选范围由顶点极值给出：</p>
                <div class="formula">\[\begin{aligned}
                x_{\min}&=\max(0,\lfloor\min(x_a,x_b,x_c)\rfloor),&
                x_{\max}&=\min(W,\lceil\max(x_a,x_b,x_c)\rceil),\\
                y_{\min}&=\max(0,\lfloor\min(y_a,y_b,y_c)\rfloor),&
                y_{\max}&=\min(H,\lceil\max(y_a,y_b,y_c)\rceil).
                \end{aligned}\]</div>
                <p>若 \(|A|&lt;\epsilon\)，三角形退化，直接跳过。否则除以极小面积会产生巨大坐标和 NaN，
                后续属性插值与深度写入都会失去意义。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "一个像素中心的三个重心坐标",
                r"""<p>取屏幕三角形 \(a=(2,2)\)、\(b=(8,2)\)、\(c=(2,7)\)，
                待测像素中心 \(p=(3.5,2.5)\)。边函数面积为：</p>
                <div class="formula">\[A=E(a,b,c)=(8-2)(2-2)-(7-2)(8-2)=-30.\]</div>
                <p>三个子面积比为：</p>
                <div class="formula">\[\begin{aligned}
                \lambda_0&=\frac{E(b,c,p)}{-30}=0.65,\\
                \lambda_1&=\frac{E(c,a,p)}{-30}=0.25,\\
                \lambda_2&=1-0.65-0.25=0.10.
                \end{aligned}\]</div>
                <p>三者都非负且和为 1，因此像素被覆盖。用它们重建 \(p\)：
                \(0.65a+0.25b+0.10c=(3.5,2.5)\)，与原始像素中心一致。</p>
                <div class="callout"><strong>边界规则：</strong>当 \(\lambda\) 恰好为 0，点在三角形边上。
                相邻三角形共享边时，需要统一采用 top-left 等规则，避免同一条边被重复光栅化。</div>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "最小三角形覆盖循环",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>const area = edge(a, b, c);
if (Math.abs(area) &lt; 1e-12) return;

for (let y = minY; y &lt; maxY; y += 1) {
  for (let x = minX; x &lt; maxX; x += 1) {
    const p = { x: x + 0.5, y: y + 0.5 };
    const l0 = edge(b, c, p) / area;
    const l1 = edge(c, a, p) / area;
    const l2 = 1 - l0 - l1;
    const covered = Number.isFinite(l0) &amp;&amp; l0 &gt;= 0 &amp;&amp; l1 &gt;= 0 &amp;&amp; l2 &gt;= 0;
    if (covered) writePixel(x, y);
  }
}</code></pre></div>
                <p>RenderTorch 的 <code>rasterize_triangle</code> 使用 NumPy 批量计算整个包围盒，
                并用 <code>-1e-6</code> 容差处理共享边。这个容差很实用，但不等同于严格的 top-left 规则。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "覆盖、裁剪与退化",
                r"""<details><summary>问题 1：为什么不能先除以 w 再裁剪？</summary>
                <p><strong>答案：</strong>相机背后或近平面外的点可能得到错误符号，且除法会破坏齐次空间中用于求交的线性关系。</p></details>
                <details><summary>问题 2：包围盒内的像素一定被三角形覆盖吗？</summary>
                <p><strong>答案：</strong>不一定。包围盒只是候选范围，三个角上的像素常常在三角形外。</p></details>
                <details><summary>问题 3：面积接近零时为什么应直接跳过？</summary>
                <p><strong>答案：</strong>面积比会放大数值误差，退化成线段或点的三角形没有稳定内部区域。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "三角形消失、裂缝与重复边",
                r"""<ol class="debug-list"><li><strong>整个三角形消失：</strong>检查裁剪平面的内外符号、\(w\) 的正负和视口 y 翻转。</li>
                <li><strong>三角形巨大：</strong>检查是否漏掉近裁剪，或把齐次坐标误当作 NDC。</li>
                <li><strong>边缘双缝：</strong>检查共享边是否被两个面同时写入，以及是否使用统一 top-left 规则。</li>
                <li><strong>像素偏移：</strong>检查是否使用像素中心 \(x+0.5,y+0.5\)，而不是整数角点。</li>
                <li><strong>NaN 扩散：</strong>在面积比、逆深度和属性除法前检查退化与零分母。</li></ol>
                <p>最小复现应当只有 1 个三角形和 1 个像素。先打印 \(A\)、三个 \(\lambda\) 和像素中心，再逐步加入裁剪与深度。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "当前裁剪与光栅化实现",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验允许拖动三个顶点和探针点，
                实时显示边函数面积、重心坐标和覆盖结果，并可把三角形拖成退化状态。</p>
                <p><span class="status-pill">现有 RenderTorch</span>
                <code>renderer/clipping.py</code> 在六个齐次平面上裁剪多边形；
                <code>renderer/pipeline.py</code> 把裁剪后的多边形扇形成三角形；
                <code>renderer/rasterizer.py</code> 使用包围盒、像素中心和边函数覆盖率测试。</p>
                <p><span class="status-pill">后续实现路线</span> 当前共享边使用 <code>-1e-6</code> 容差，
                尚未实现严格 top-left 规则、保守光栅化和 MSAA。增加这些功能前，应先固定共享边的覆盖契约。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "raster-demo",
                "边函数、像素中心与三角形覆盖",
                "demos/chapter10.js",
                "拖动顶点或探针点，查看面积、重心坐标、覆盖状态与退化保护。",
            ),
        ),
        "sources": (
            source("renderer/clipping.py", "现有实现：六个齐次裁剪平面与多边形交点"),
            source("renderer/rasterizer.py", "现有实现：包围盒、边函数与像素中心"),
            source("renderer/pipeline.py", "现有实现：裁剪多边形扇形成三角形"),
        ),
    },
    "chapter11": {
        "sections": (
            section(
                "intuition",
                "像素里的深度与属性",
                "屏幕线性不等于世界线性",
                r"""<p>一个三角形覆盖某个像素后，渲染器需要得到该像素处的颜色、法线、纹理坐标和深度。
                最直觉的做法是对三个顶点值直接做线性插值，但透视投影会破坏这种线性关系：
                屏幕中央的一小段，可能对应世界中很长的一段。</p>
                <p>正确做法分为两拍：先插值顶点的 \(1/w\) 与 \(q/w\)，再相除恢复 \(q\)。
                深度缓存则保存每个像素已经写入的最近距离，用一次比较解决遮挡问题。
                它让三角形可以任意顺序提交，而画家算法必须依靠“远的先画、近的后画”。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="线性深度与透视正确深度对比">
                <rect x="60" y="45" width="240" height="180" rx="10" fill="#f6f3eb" stroke="#d9ddd5"/>
                <rect x="340" y="45" width="240" height="180" rx="10" fill="#f6f3eb" stroke="#d9ddd5"/>
                <line x1="100" y1="210" x2="260" y2="210" stroke="#68757a"/>
                <path d="M100 210L120 100L250 70Z" fill="#087f83" fill-opacity=".18" stroke="#087f83" stroke-width="2"/>
                <line x1="100" y1="210" x2="180" y2="135" stroke="#c66b35" stroke-width="4"/>
                <line x1="340" y1="210" x2="500" y2="210" stroke="#68757a"/>
                <path d="M340 210L360 100L490 70Z" fill="#c66b35" fill-opacity=".18" stroke="#c66b35" stroke-width="2"/>
                <line x1="340" y1="210" x2="420" y2="135" stroke="#087f83" stroke-width="4"/>
                <text x="88" y="255">直接线性 w：深度偏大</text><text x="350" y="255">插值 1/w：透视正确</text>
                <circle cx="140" cy="172" r="6" fill="#c66b35"/><circle cx="380" cy="172" r="6" fill="#087f83"/>
                </svg><p><strong>图 11.1</strong>　同一个屏幕参数 \(t\)，两种插值会得到不同距离。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "齐次 w、逆深度与深度缓存",
                r"""<p>设三角形三个顶点的裁剪空间齐次坐标为
                \(\mathbf q_i=(x_i,y_i,z_i,w_i)\)，且 \(w_i&gt;0\)。
                透视除法后，屏幕重心坐标为 \(\lambda_i\)。</p>
                <p>定义像素处的逆深度与属性：</p>
                <div class="formula">\[\frac{1}{w_{\mathrm{pixel}}}
                =\sum_{i=0}^{2}\frac{\lambda_i}{w_i},\qquad
                \widetilde q_{\mathrm{pixel}}=\sum_{i=0}^{2}\lambda_i\frac{q_i}{w_i}.\]</div>
                <div class="formula">\[w_{\mathrm{pixel}}=\frac{1}{\sum_i\lambda_i/w_i},\qquad
                q_{\mathrm{pixel}}=\frac{\widetilde q_{\mathrm{pixel}}}{1/w_{\mathrm{pixel}}}.\]</div>
                <p>RenderTorch 当前把正的 \(w\) 当作深度值保存。
                相机空间采用前向 \(-\mathbf z\) 时，\(w=-z_{\mathrm{view}}\)，所以 \(w\) 越小越近。</p>
                <p>深度缓存 \(D(x,y)\) 保存该像素已写入的最近深度，初始为 \(+\infty\)。
                只有更近的片元才能覆盖已有颜色。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "为什么必须插值倒数",
                r"""<p><strong>第一步：世界线段参数与屏幕参数不同。</strong>
                透视投影把每条视线上的距离压缩到屏幕位置。屏幕参数 \(\lambda\) 不是世界空间的线性参数，
                因此 \(w\) 与颜色不能直接用 \(\sum_i\lambda_i w_i\)、\(\sum_i\lambda_i q_i\) 恢复。</p>
                <p><strong>第二步：构造线性量。</strong> 齐次裁剪坐标中，
                顶点处有 \(\widetilde q_i=q_i/w_i\) 和 \(r_i=1/w_i\)。
                这两个量在屏幕空间随重心坐标线性变化：</p>
                <div class="formula">\[r_{\mathrm{pixel}}=\sum_i\lambda_i r_i,\qquad
                \widetilde q_{\mathrm{pixel}}=\sum_i\lambda_i\widetilde q_i.\]</div>
                <p><strong>第三步：消除分母。</strong> 因为 \(q=\widetilde q/r\)，所以：</p>
                <div class="formula">\[q_{\mathrm{pixel}}=
                \frac{\sum_i\lambda_i(q_i/w_i)}{\sum_i\lambda_i(1/w_i)},\qquad
                w_{\mathrm{pixel}}=\frac{1}{\sum_i\lambda_i(1/w_i)}.\]</div>
                <p>深度使用同一组 \(\lambda_i\)。标准透视投影的 NDC 深度可写成
                \(z_{\mathrm{ndc}}=A+B/w\)，因此 \(z_{\mathrm{ndc}}\) 也随 \(1/w\) 线性变化。
                但当前 RenderTorch 保存的是 \(w\)，不是 NDC 深度。</p>
                <p><strong>第四步：可见性判断。</strong> 新片元只有满足：</p>
                <div class="formula">\[w_{\mathrm{pixel}} < \mathrm{depth}(x,y)\]</div>
                <p>才写入颜色、深度和物体编号。等号不写入，可以保持已经可见的对象。</p>
                <p><strong>第五步：为什么画家算法不可靠。</strong> 若两个三角形在屏幕空间相交，
                它们的深度关系随像素改变，不存在一个全图通用绘制顺序。
                即使当前例子能靠排序画对，循环三角形相交时也会失败。z-buffer 在每个像素局部决定胜负。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "逆深度与属性恢复",
                r"""<p>三个顶点的 \(w=(1,2,4)\)，某像素重心坐标
                \((\lambda_0,\lambda_1,\lambda_2)=(0.2,0.3,0.5)\)。先计算逆深度：</p>
                <div class="formula">\[r=\frac{0.2}{1}+\frac{0.3}{2}+\frac{0.5}{4}=0.475,
                \qquad w_{\mathrm{pixel}}=\frac{1}{0.475}\approx2.105.\]</div>
                <p>若直接线性插值 \(w\)，会得到 \(0.2(1)+0.3(2)+0.5(4)=3.000\)，明显偏远。</p>
                <p>再设顶点属性 \(q=(10,20,40)\)。透视正确的插值为：</p>
                <div class="formula">\[\widetilde q=0.2\frac{10}{1}+0.3\frac{20}{2}+0.5\frac{40}{4}=10,
                \qquad q_{\mathrm{pixel}}=\frac{10}{0.475}\approx21.053.\]</div>
                <p>直接线性插值会得到 \(28\)。颜色和纹理坐标若使用错误插值，会出现明显的透视拉伸。</p>
                <p>遮挡例子：同一像素上 A 的深度为 3，B 的深度为 5。
                无论先提交 A 还是 B，z-buffer 最终都保留 A。若先画 B 再画 A，两者都写；
                若先画 A 再画 B，B 的比较失败。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "属性插值后做深度测试",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>const inverseW = l0 / w0 + l1 / w1 + l2 / w2;
if (!Number.isFinite(inverseW) || inverseW &lt;= 0) return;
const pixelW = 1 / inverseW;

const attr = (q0 * l0 / w0 + q1 * l1 / w1 + q2 * l2 / w2) / inverseW;
if (pixelW &lt; framebuffer.depth[y, x]) {
  framebuffer.depth[y, x] = pixelW;
  framebuffer.color[y, x] = shade(attr);
  framebuffer.objectIds[y, x] = objectId;
}</code></pre></div>
                <p><code>rasterize_triangle</code> 将 <code>clipped[:,3]</code> 作为三个顶点的 \(w\)，
                批量计算逆深度，再取倒数。所有可见像素的写入都由同一个深度比较掩码控制。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "插值、缓存与共享边",
                r"""<details><summary>问题 1：为什么 1/w 可以线性插值，而 w 不可以？</summary>
                <p><strong>答案：</strong>透视映射使 \(1/w\) 成为屏幕空间线性量；\(w\) 是对该线性量取倒数得到的非线性量。</p></details>
                <details><summary>问题 2：z-buffer 的初值为什么是正无穷？</summary>
                <p><strong>答案：</strong>任何有限近深度都小于正无穷，第一个可见片元可以正常写入。</p></details>
                <details><summary>问题 3：先画 A 再画 B，B 更远，会发生什么？</summary>
                <p><strong>答案：</strong>B 的深度比较失败，已有 A 的颜色和深度保持不变。</p></details>
                <details><summary>问题 4：深度正确能解决边缘重复覆盖吗？</summary>
                <p><strong>答案：</strong>不能。共享边上的两个片元深度可能相同，需要 top-left 规则决定唯一归属。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "纹理滑动、遮挡跳变与深度反转",
                r"""<ol class="debug-list"><li><strong>纹理随相机滑动：</strong>检查是否直接插值 UV，而没有使用 \(q/w\) 与 \(1/w\)。</li>
                <li><strong>远处三角形突然消失：</strong>检查远平面裁剪、深度初始化和浮点精度。</li>
                <li><strong>两个三角形随提交顺序闪烁：</strong>检查是否使用 z-buffer，而不是画家排序。</li>
                <li><strong>近处变远处：</strong>统一深度约定。当前实现保存 \(w\)，越小越近，不能混入越大越近的 NDC 约定。</li>
                <li><strong>共享边发亮：</strong>深度相等不能解决所有权，需要严格 top-left 规则或一致的保守覆盖策略。</li></ol>
                <p>最小调试单元是 1 个像素：打印三个 \(\lambda\)、三个 \(w\)、逆深度、恢复深度和缓存旧值。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "当前遮挡与抗锯齿实现",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实验并排展示直接插值与逆深度插值，
                并允许交换 A/B 提交顺序，对照画家算法和 z-buffer 的最终像素。</p>
                <p><span class="status-pill">现有 RenderTorch</span>
                <code>renderer/rasterizer.py</code> 使用重心坐标插值 \(1/w\)，取倒数得到像素深度；
                <code>renderer/framebuffer.py</code> 提供 RGBA 颜色、浮点深度和物体编号缓冲；
                深度初始为 <code>np.inf</code>，写入条件为更小深度。</p>
                <p><span class="status-pill">后续实现路线</span> 当前管线主要做逐面着色，没有完整属性插值管线。
                背面剔除、严格 top-left 规则和 MSAA 尚未实现；增加它们时需要同时明确面绕序、覆盖契约和多样本解析。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "occlusion-demo",
                "逆深度插值与 z-buffer 遮挡",
                "demos/chapter11.js",
                "比较线性 w 与 1/w 插值，交换绘制顺序，观察画家算法和深度缓存的不同结果。",
            ),
        ),
        "sources": (
            source("renderer/rasterizer.py", "现有实现：重心坐标、逆深度插值与深度测试"),
            source("renderer/framebuffer.py", "现有实现：颜色、深度与物体编号缓冲"),
            source("renderer/pipeline.py", "现有实现：逐面投影、着色与光栅提交"),
        ),
    },
    "chapter12": {
        "sections": (
            section(
                "intuition",
                "光如何变成颜色",
                "从几何法线到材质响应",
                r"""<p>几何决定表面朝向，光源决定能量从哪个方向射来，材质决定有多少能量被吸收、散射或反射。
                渲染器把这些信息压缩成一个着色计算：给定表面点、法线、光源方向和观察方向，输出颜色。</p>
                <p>本章先学习两个可分解、可验证的模型：Lambert 漫反射与 Blinn–Phong 高光。
                它们足以解释“迎着光更亮、侧面更暗、光滑表面出现亮斑”，也适合作为理解真实 PBR 的入口。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="环境光、漫反射与高光的贡献分解">
                <rect x="36" y="45" width="150" height="150" rx="75" fill="#263943"/>
                <path d="M111 195A75 75 0 0 1 186 120A75 75 0 0 1 111 45A75 75 0 0 0 111 195Z" fill="#314e59"/>
                <rect x="240" y="45" width="150" height="150" rx="75" fill="#263943"/>
                <path d="M315 195A75 75 0 0 1 390 120A75 75 0 0 1 315 45A75 75 0 0 0 315 195Z" fill="#8b4b35"/>
                <rect x="444" y="45" width="150" height="150" rx="75" fill="#263943"/>
                <path d="M519 195A75 75 0 0 1 594 120A75 75 0 0 1 519 45A75 75 0 0 0 519 195Z" fill="#e5e5e5"/>
                <circle cx="548" cy="91" r="25" fill="#fff" opacity=".9"/>
                <path d="M36 120L235 120M407 120L615 120" stroke="#c66b35" stroke-width="2"/>
                <text x="65" y="230">环境光</text><text x="267" y="230">漫反射</text><text x="470" y="230">高光</text>
                <text x="228" y="274" fill="#9db6b8">最终颜色 = 各项贡献之和</text>
                </svg><p><strong>图 12.1</strong>　把总颜色拆成独立贡献，便于验证每一项的方向、强度和颜色。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "方向、材质与半向量",
                r"""<p>在世界空间或统一的着色空间中，定义：</p>
                <ul><li>\(\mathbf N\)：单位表面法线；</li><li>\(\mathbf L\)：从表面点指向光源的单位方向；</li>
                <li>\(\mathbf V\)：从表面点指向观察者的单位方向；</li><li>\(\mathbf H=\operatorname{normalize}(\mathbf L+\mathbf V)\)：半向量；</li>
                <li>\(\mathbf C_{\mathrm{base}}\)：基础颜色，也称 albedo；</li><li>\(k_a,k_d,k_s\)：环境光、漫反射、高光权重；</li>
                <li>\(s\)：高光指数，越大亮斑越集中。</li></ul>
                <p>点积 \(\mathbf N\cdot\mathbf L\) 是入射方向与法线夹角的余弦。
                当它为负，光源位于表面背侧，该光源不应贡献漫反射。
                所有进入点积的方向都应先归一化，否则结果会随向量长度改变。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "Lambert 与 Blinn–Phong",
                r"""<p><strong>第一步：Lambert 余弦定律。</strong> 单位面积接收到的光通量与入射角余弦成正比，
                因此漫反射项为：</p>
                <div class="formula">\[I_{\mathrm{diffuse}}=k_d\,\mathbf C_{\mathrm{base}}\,
                \max(0,\mathbf N\cdot\mathbf L).\]</div>
                <p>代码中对应 <code>diffuse = max(0, N · L)</code>。这里的 max 同时完成背面光源剔除，
                不应改成绝对值，否则背面会被错误照亮。</p>
                <p><strong>第二步：高光半向量。</strong> 观察方向 \(\mathbf V\) 与光源方向 \(\mathbf L\) 关于法线对称时，
                镜面反射光最可能进入眼睛。两者的平均方向为：</p>
                <div class="formula">\[\mathbf H=\operatorname{normalize}(\mathbf L+\mathbf V).\]</div>
                <p>代码中对应 <code>H = normalize(L + V)</code>。若 \(\mathbf L+\mathbf V\) 接近零，半向量没有稳定方向，应将高光置零。</p>
                <p><strong>第三步：经验高光。</strong> Blinn–Phong 用法线与半向量的点积控制亮斑：</p>
                <div class="formula">\[I_{\mathrm{specular}}=k_s\,\mathbf C_{\mathrm{spec}}\,
                [\max(0,\mathbf N\cdot\mathbf H)]^s.\]</div>
                <p><strong>第四步：合成。</strong> 在线性颜色空间中：</p>
                <div class="formula">\[\mathbf C_{\mathrm{linear}}=k_a\mathbf C_{\mathrm{base}}+
                k_d\mathbf C_{\mathrm{base}}\max(0,\mathbf N\cdot\mathbf L)+
                k_s\mathbf C_{\mathrm{spec}}[\max(0,\mathbf N\cdot\mathbf H)]^s.\]</div>
                <p><strong>第五步：走向 PBR。</strong> 真实微表面模型把高光拆成 Fresnel、微表面分布和几何遮挡，
                并用粗糙度控制分布宽度、用金属度控制基础反射颜色。Blinn–Phong 是经验模型，不等于 PBR；
                它的优势是简单、稳定、容易逐项调试。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "一个方向明确的着色样本",
                r"""<p>取 \(\mathbf N=(0,0,1)\)、\(\mathbf V=(0,0,1)\)、
                \(\mathbf L=(0.6,0,0.8)\)，基础颜色 \((0.6,0.2,0.1)\)，
                参数 \(k_a=0.1,k_d=0.8,k_s=0.5,s=16\)。</p>
                <p>漫反射因子为 \(\max(0,0.8)=0.8\)。半向量为
                \(\mathbf H=\operatorname{normalize}((0.6,0,1.8))\approx(0.316,0,0.949)\)，
                所以 \(\mathbf N\cdot\mathbf H\approx0.949\)，高光强度约为 \(0.5(0.949)^{16}\approx0.214\)。</p>
                <div class="formula">\[\begin{aligned}
                \mathbf C_{\mathrm{ambient}}&=(0.060,0.020,0.010),\\
                \mathbf C_{\mathrm{diffuse}}&=0.8(0.6,0.2,0.1)(0.8)=(0.384,0.128,0.064),\\
                \mathbf C_{\mathrm{specular}}&=(0.214,0.214,0.214),\\
                \mathbf C_{\mathrm{linear}}&\approx(0.658,0.363,0.288).
                \end{aligned}\]</div>
                <p>若暂时直接映射到 8 位显示，结果约为 \((168,92,74)\)。
                第 14 章会说明为什么正式管线应先在线性空间计算，再做 gamma 与色调映射。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与源码映射",
                "逐项计算再相加",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>const diffuse = Math.max(0, dot(N, L));
const halfSum = add(L, V);
const H = length(halfSum) &gt; 1e-12 ? scale(halfSum, 1 / length(halfSum)) : null;
const specular = H ? Math.pow(Math.max(0, dot(N, H)), shininess) : 0;
const ambient = scale(baseColor, ambientStrength);
const diffuseColor = scale(baseColor, diffuseStrength * diffuse);
const specularColor = scale(specularColorBase, specularStrength * specular);
const linear = clamp(add(add(ambient, diffuseColor), specularColor), 0, 1);</code></pre></div>
                <p>RenderTorch 的 <code>_shade</code> 只按整个三角形计算一次法线，
                使用 <code>abs(N·L)</code> 生成明暗，并把结果乘以材质颜色。
                它没有逐像素法线、观察方向、独立高光项或粗糙度参数。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "方向、贡献与模型边界",
                r"""<details><summary>问题 1：为什么漫反射使用 max(0,N·L)，而不是 abs(N·L)？</summary>
                <p><strong>答案：</strong>背侧光源不应照亮单面材质；abs 会把背面错误地变亮。</p></details>
                <details><summary>问题 2：高光指数增大时亮斑如何变化？</summary>
                <p><strong>答案：</strong>点积小于 1 时，幂次越高数值越小，因此亮斑更窄、更集中。</p></details>
                <details><summary>问题 3：Blinn–Phong 是 PBR 吗？</summary>
                <p><strong>答案：</strong>不是。它是经验着色模型；PBR 还需要微表面、Fresnel、能量守恒等更完整的描述。</p></details>
                <details><summary>问题 4：金属度升高时基础颜色通常怎样变化？</summary>
                <p><strong>答案：</strong>金属的漫反射很弱，彩色反射主要由基础反射颜色决定；非金属则保留较强漫反射。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "过亮球体与不稳定高光",
                r"""<ol class="debug-list"><li><strong>整个表面发白：</strong>检查各贡献是否在线性空间相加，以及最终是否过晚 clamp。</li>
                <li><strong>法线长度导致亮度变化：</strong>进入点积前归一化 N、L、V。</li>
                <li><strong>高光在球面上游走：</strong>确认 L 与 V 都指向表面外，且使用同一个坐标空间。</li>
                <li><strong>背面发亮：</strong>漫反射误用了 abs，或未统一面绕序。</li>
                <li><strong>亮斑突然消失：</strong>检查 L+V 是否接近零，以及 shininess 是否过大造成数值下溢。</li></ol>
                <p>调试时先关闭高光只验证漫反射，再单独验证半向量，最后合成三项。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "当前逐面着色与后续材质路线",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章绘制逐像素着色球体，
                显示样本点的 N、L、V 与环境光、漫反射、高光和最终 RGB。</p>
                <p><span class="status-pill">现有 RenderTorch</span> <code>renderer/pipeline.py</code>
                使用固定方向光与逐面叉积法线，强度为 <code>0.5 + 0.5 * abs(N·L)</code>。
                材质目前主要提供 RGBA 基础颜色。</p>
                <p><span class="status-pill">后续实现路线</span> 最小升级顺序是：
                世界空间点积与观察方向 → 顶点法线插值 → 漫反射/高光分解 → 线性颜色输出。
                完整 PBR 还需要材质系统、粗糙度、金属度、Fresnel 与微表面模型。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "lighting-demo",
                "Lambert 漫反射与 Blinn–Phong 高光",
                "demos/chapter12.js",
                "改变光源方向和材质参数，观察环境光、漫反射、高光与最终颜色的分解。",
            ),
        ),
        "sources": (
            source("renderer/pipeline.py", "现有实现：固定方向光与逐面 abs(N·L) 着色"),
            source("core/scene.py", "现有实现：材质 RGBA 基础颜色"),
            source("core/primitives.py", "现有实现：网格几何与共享顶点"),
        ),
    },
    "chapter13": {
        "sections": (
            section(
                "intuition",
                "给三角形贴上图案",
                "UV 是几何与图像之间的坐标契约",
                r"""<p>网格顶点只有位置还不够。若想让一个表面显示木纹、棋盘或图像，
                需要给每个顶点附加纹理坐标 \((u,v)\)。光栅化得到像素的覆盖位置后，
                先恢复该像素的 UV，再从纹理中采样颜色。</p>
                <p>最容易犯的错误是直接对 UV 做屏幕线性插值。透视投影前后，
                屏幕参数与世界距离并不成正比，因此必须复用第 11 章的逆深度方法，
                对 \(u/w\)、\(v/w\) 和 \(1/w\) 插值，再相除得到透视矫正 UV。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="棋盘纹理与三角形 UV 映射">
                <defs><pattern id="checker" width="40" height="40" patternUnits="userSpaceOnUse"><rect width="40" height="40" fill="#f6f3eb"/><rect width="20" height="20" fill="#087f83"/><rect x="20" y="20" width="20" height="20" fill="#087f83"/></pattern></defs>
                <rect x="45" y="45" width="200" height="200" fill="url(#checker)" stroke="#d9ddd5"/>
                <path d="M330 240L565 240L385 65Z" fill="#eef3ee" stroke="#087f83" stroke-width="3"/>
                <path d="M385 65L457 153L330 240Z" fill="url(#checker)" opacity=".7"/>
                <circle cx="330" cy="240" r="6" fill="#c66b35"/><circle cx="565" cy="240" r="6" fill="#c66b35"/><circle cx="385" cy="65" r="6" fill="#c66b35"/>
                <text x="322" y="267">(0,0)</text><text x="555" y="267">(1,0)</text><text x="375" y="50">(0,1)</text>
                <text x="57" y="278">纹理空间</text><text x="407" y="278">网格参数空间</text>
                </svg><p><strong>图 13.1</strong>　纹理坐标把二维图案附着到三角形参数域，采样时再恢复每个像素的 UV。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "UV、逆深度与采样位置",
                r"""<p>设三角形三个顶点的纹理坐标为 \((u_i,v_i)\)，裁剪空间齐次坐标为
                \((x_i,y_i,z_i,w_i)\)，像素重心坐标为 \(\lambda_i\)。
                纹理通常定义在 \([0,1]^2\)，超出范围时由 wrap 或 clamp 策略处理。</p>
                <p>定义像素处的逆深度与纹理分子：</p>
                <div class="formula">\[\frac{1}{w_{\mathrm{pixel}}}=\sum_i\frac{\lambda_i}{w_i},\qquad
                \widetilde u=\sum_i\lambda_i\frac{u_i}{w_i},\qquad
                \widetilde v=\sum_i\lambda_i\frac{v_i}{w_i}.\]</div>
                <div class="formula">\[u_{\mathrm{pixel}}=\frac{\widetilde u}{1/w_{\mathrm{pixel}}},\qquad
                v_{\mathrm{pixel}}=\frac{\widetilde v}{1/w_{\mathrm{pixel}}}.\]</div>
                <p>代码中常写成 <code>u_pixel</code>、<code>v_pixel</code> 与 <code>w_pixel</code>。
                它们不是三个独立近似，而是同一透视矫正公式的三个输出。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "透视矫正 UV 与纹理过滤",
                r"""<p><strong>第一步：识别错误插值。</strong> 屏幕线性 UV 为
                \(u=\sum_i\lambda_i u_i\)、\(v=\sum_i\lambda_i v_i\)。
                它只在一个与屏幕平行的表面上正确；一旦三角形前后顶点距离不同，纹理就会滑动或拉伸。</p>
                <p><strong>第二步：构造可线性插值的量。</strong> 与深度相同，
                在屏幕空间线性变化的是 \(1/w\) 与 \(u/w\)、\(v/w\)：</p>
                <div class="formula">\[u_{\mathrm{pixel}}=
                \frac{\sum_i\lambda_i(u_i/w_i)}{\sum_i\lambda_i(1/w_i)},\qquad
                v_{\mathrm{pixel}}=
                \frac{\sum_i\lambda_i(v_i/w_i)}{\sum_i\lambda_i(1/w_i)}.\]</div>
                <p><strong>第三步：映射到纹素。</strong> 对 \(W_t\times H_t\) 纹理，
                最近邻采样取：</p>
                <div class="formula">\[i=\operatorname{clamp}(\lfloor uW_t\rfloor,0,W_t-1),\qquad
                j=\operatorname{clamp}(\lfloor vH_t\rfloor,0,H_t-1).\]</div>
                <p><strong>第四步：双线性过滤。</strong> 在相邻四个纹素之间按小数部分混合：</p>
                <div class="formula">\[\mathbf T(u,v)=(1-\alpha)(1-\beta)\mathbf T_{00}+
                \alpha(1-\beta)\mathbf T_{10}+(1-\alpha)\beta\mathbf T_{01}+
                \alpha\beta\mathbf T_{11}.\]</div>
                <p>最近邻保留硬边但会产生锯齿；双线性更平滑，但远景缩小后仍可能摩尔纹。
                mipmap 为同一纹理预生成逐级缩小版本，根据屏幕导数选择层级；
                mipmap 链、三线性过滤和各向异性过滤属于 GPU 纹理管线，本章浏览器实验不实现它们。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "同一像素的线性和透视 UV",
                r"""<p>三个顶点的 \(w=(1,2,4)\)，重心坐标
                \((\lambda_0,\lambda_1,\lambda_2)=(0.2,0.3,0.5)\)，
                纹理坐标为 \((0,0),(1,0),(0,1)\)。</p>
                <p>屏幕线性结果为：</p>
                <div class="formula">\[u_{\mathrm{linear}}=0.3,\qquad v_{\mathrm{linear}}=0.5.\]</div>
                <p>透视矫正先求逆深度：</p>
                <div class="formula">\[\frac{1}{w_{\mathrm{pixel}}}=0.2+\frac{0.3}{2}+\frac{0.5}{4}=0.475.\]</div>
                <p>纹理分子为 \(\widetilde u=0.3/2=0.15\)、\(\widetilde v=0.5/4=0.125\)，因此：</p>
                <div class="formula">\[u_{\mathrm{pixel}}=\frac{0.15}{0.475}\approx0.316,\qquad
                v_{\mathrm{pixel}}=\frac{0.125}{0.475}\approx0.263.\]</div>
                <p>两种结果的差异会直接改变棋盘格采样位置。远端 \(w\) 越大，透视矫正越明显。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "先矫正 UV，再采样纹理",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>const inverse_w_pixel = l0 / w0 + l1 / w1 + l2 / w2;
if (!Number.isFinite(inverse_w_pixel) || inverse_w_pixel &lt;= 0) return;
const w_pixel = 1 / inverse_w_pixel;
const u_pixel = (l0 * u0 / w0 + l1 * u1 / w1 + l2 * u2 / w2) / inverse_w_pixel;
const v_pixel = (l0 * v0 / w0 + l1 * v1 / w1 + l2 * v2 / w2) / inverse_w_pixel;
const checker = ((Math.floor(u_pixel * 8) + Math.floor(v_pixel * 8)) % 2) === 0;
const color = filter === 'nearest' ? sampleNearest(u_pixel, v_pixel) : sampleBilinear(u_pixel, v_pixel);</code></pre></div>
                <p>当前 RenderTorch 的 OBJ 导入只读取 <code>v</code> 与 <code>f</code>，
                网格不保存 UV；光栅器也只插值深度。纹理链路需要从顶点属性契约开始增加。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "坐标、过滤与远景",
                r"""<details><summary>问题 1：什么时候屏幕线性 UV 恰好正确？</summary>
                <p><strong>答案：</strong>三个顶点的 w 相同，或三角形所在平面与屏幕平行时。</p></details>
                <details><summary>问题 2：为什么不能先算 w_pixel，再直接用它修正 UV？</summary>
                <p><strong>答案：</strong>必须保留三个顶点各自的 1/w_i 加权；单个像素深度不能替代顶点信息。</p></details>
                <details><summary>问题 3：双线性过滤能完全消除远景摩尔纹吗？</summary>
                <p><strong>答案：</strong>不能。缩小时一个纹素可能覆盖多个像素，需要 mipmap 或更完整的过滤策略。</p></details>
                <details><summary>问题 4：UV 超出 [0,1] 时必须报错吗？</summary>
                <p><strong>答案：</strong>不一定。可以由纹理的 wrap/clamp 策略决定重复、镜像或延伸边缘。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "纹理滑动、翻转与锯齿",
                r"""<ol class="debug-list"><li><strong>相机移动时纹理滑动：</strong>检查是否使用透视矫正 UV。</li>
                <li><strong>图像上下颠倒：</strong>统一图像原点与 UV 的 v 方向；很多图像文件从顶部开始，而参数曲面从底部开始。</li>
                <li><strong>接缝闪烁：</strong>检查共享顶点是否具有一致 UV，wrap 模式是否统一。</li>
                <li><strong>最近邻出现锯齿：</strong>可切到双线性；远景仍闪烁时考虑 mipmap。</li>
                <li><strong>边缘拉出条纹：</strong>采样前按约定 clamp 或 wrap，并检查纹素坐标是否少加/多加 0.5。</li></ol>
                <p>先用纯色 UV 渐变代替真实纹理，再验证坐标连续性，最后加入棋盘与双线性过滤。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "当前网格与纹理实现",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章用程序棋盘纹理对比屏幕线性 UV 与透视矫正 UV，
                支持 nearest/bilinear 过滤和远端深度调节。</p>
                <p><span class="status-pill">现有 RenderTorch</span> <code>core/primitives.py</code> 只生成位置与面索引；
                <code>solid/custom.py</code> 不解析 OBJ 的 UV/法线；<code>renderer/rasterizer.py</code>
                只对深度做 \(1/w\) 插值，没有通用顶点属性缓冲。</p>
                <p><span class="status-pill">后续实现路线</span> 增加顶点 UV → 透视矫正属性插值 →
                最近邻/双线性采样 → wrap/clamp → mipmap。完整 GPU 纹理链不在当前浏览器实验中实现。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "texture-demo",
                "透视矫正 UV 与棋盘过滤",
                "demos/chapter13.js",
                "对比屏幕线性 UV 与透视矫正 UV，切换最近邻和双线性过滤，并改变远端顶点深度。",
            ),
        ),
        "sources": (
            source("renderer/rasterizer.py", "现有实现：可复用的重心坐标与逆深度插值"),
            source("core/primitives.py", "现有实现：位置与面，不含 UV"),
            source("solid/custom.py", "现有实现：OBJ 只导入 v 与 f"),
        ),
    },
    "chapter14": {
        "sections": (
            section(
                "intuition",
                "阴影与显示变换",
                "一个判断可见性，一个决定如何上屏",
                r"""<p>阴影不是把表面整体调暗，而是询问：从当前表面点走向光源的路径上，是否有更近的物体？
                 shadow mapping 把这个问题转换成两次渲染：先从光源视角记录最近深度，再从相机视角比较深度。</p>
                <p>着色器计算出的颜色通常在线性空间，可能大于 1，也不等于人眼感知的显示亮度。
                上屏前需要控制高光不过曝，并做 gamma 显示变换。阴影与显示变换属于两个独立问题，
                本章分别推导，最后组合成完整输出。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="阴影图两次渲染与颜色变换">
                <rect x="30" y="50" width="175" height="175" rx="12" fill="#f6f3eb" stroke="#d9ddd5"/>
                <circle cx="105" cy="92" r="18" fill="#ef9a67"/><path d="M105 110L183 211H27Z" fill="#9db6b8"/>
                <text x="62" y="252">光源视图：记录深度</text>
                <path d="M215 138H270" stroke="#c66b35" stroke-width="3"/><path d="M260 128L270 138L260 148Z" fill="#c66b35"/>
                <rect x="285" y="50" width="175" height="175" rx="12" fill="#f6f3eb" stroke="#d9ddd5"/>
                <circle cx="340" cy="91" r="18" fill="#ef9a67"/><ellipse cx="385" cy="177" rx="52" ry="17" fill="#31515b"/>
                <text x="324" y="252">相机视图：比较深度</text>
                <rect x="500" y="55" width="105" height="34" fill="#263943"/><rect x="500" y="101" width="105" height="34" fill="#8aa1a5"/><rect x="500" y="147" width="105" height="34" fill="#f6f3eb"/>
                <text x="498" y="214">线性 → 色调映射 → gamma</text>
                </svg><p><strong>图 14.1</strong>　阴影图先回答“是否被挡住”，显示变换再回答“这个亮度如何呈现”。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "光源深度、bias 与线性颜色",
                r"""<p>设光源相机为观察者，\(d_{\mathrm{map}}(u_s,v_s)\) 是 shadow map 中记录的最小光空间距离，
                \(d_{\mathrm{frag}}\) 是当前表面点的光空间距离。两者必须使用同一投影、同一坐标约定和同一单位。</p>
                <p>深度偏移 \(b\ge0\) 用来抵消浮点误差与离散阴影图的深度量化。
                设阴影可见因子 \(\xi\in[0,1]\)，\(\xi=1\) 表示受光，\(\xi=0\) 表示完全遮挡。</p>
                <p>设曝光为 \(E\)，色调映射函数为 \(T\)，gamma 近似指数为 \(\gamma=2.2\)。
                线性颜色可以超过 1，显示颜色必须最终落在 \([0,1]\)。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "shadow mapping、Reinhard 与 gamma",
                r"""<p><strong>第一步：光源深度Pass。</strong> 把相机放在光源处，
                渲染场景深度但不写颜色。每个阴影图像素保留最近物体距离：</p>
                <div class="formula">\[d_{\mathrm{map}}=\min_{\text{覆盖样本}} d_{\mathrm{light}}.\]</div>
                <p><strong>第二步：相机颜色Pass。</strong> 把当前表面点变换到同一光源空间，
                采样 shadow map 并比较：</p>
                <div class="formula">\[\xi=
                \begin{cases}
                1,&d_{\mathrm{frag}}\le d_{\mathrm{map}}+b,\\
                0,&d_{\mathrm{frag}}&gt;d_{\mathrm{map}}+b.
                \end{cases}\]</div>
                <p>bias 太小会产生 shadow acne（表面自己遮挡自己）；太大则产生 peter panning（阴影与物体脱离）。
                实际系统还常用斜率相关 bias 和 PCF 多样本平均。</p>
                <p><strong>第三步：应用曝光与 Reinhard。</strong> 曝光先把场景亮度缩放到可观察范围，
                Reinhard 将无限正数单调压缩到 \([0,1)\)：</p>
                <div class="formula">\[\mathbf C_{\mathrm{exposed}}=E\mathbf C_{\mathrm{linear}},\qquad
                \mathbf C_{\mathrm{reinhard}}=\frac{\mathbf C_{\mathrm{exposed}}}{1+\mathbf C_{\mathrm{exposed}}}.\]</div>
                <p><strong>第四步：gamma 显示变换。</strong> 近似显示编码为：</p>
                <div class="formula">\[\mathbf C_{\mathrm{display}}\approx
                \operatorname{clamp}(\mathbf C_{\mathrm{reinhard}}^{1/2.2},0,1).\]</div>
                <p>代码中可写成 <code>C_linear^(1/2.2)</code> 与 <code>C_reinhard</code>。
                必须先色调映射再 gamma；顺序交换会改变高光滚动与中间调亮度。
                 gamma 2.2 是教学近似，严格 sRGB 使用分段传递函数。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "高光压缩与阴影比较",
                r"""<p>某像素未遮挡时线性颜色为 \((0.5,0.5,0.5)\)，被遮挡后 Direct 项乘以 \(\xi=0\)。
                若仍有环境光 \((0.1,0.1,0.1)\)，两像素会分别保留受光与阴影差异。</p>
                <p>取线性值 \(C=2\)，曝光 \(E=1\)。直接 gamma 会得到 \(2^{1/2.2}\approx1.366\)，随后被 clamp 成纯白；
                先 Reinhard 则得到：</p>
                <div class="formula">\[\mathbf C_{\mathrm{reinhard}}=\frac{2}{1+2}=\frac23,\qquad
                \mathbf C_{\mathrm{display}}\approx\left(\frac23\right)^{1/2.2}\approx0.821.\]</div>
                <p>这样高于 1 的高光仍保留亮度层次，而不是全部截断为白色。</p>
                <p>阴影比较例：\(d_{\mathrm{map}}=4.0\)、\(d_{\mathrm{frag}}=4.02\)、\(b=0.01\)。
                因为 \(4.02&gt;4.01\)，该样本被遮挡；把 bias 调到 0.03 后变为受光，可清楚观察 bias 的作用。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与源码映射",
                "阴影可见性与显示输出",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>const lightDepth = sampleShadowMap(lightUv);
const visible = fragDepth &lt;= lightDepth + bias ? 1 : 0;
const litLinear = ambient + direct * visible;
const exposed = litLinear * exposure;
const reinhard = exposed / (1 + exposed);
const display = reinhard.map((channel) =&gt; Math.pow(Math.max(0, channel), 1 / 2.2));
framebuffer.color[pixel] = clamp(display, 0, 1);</code></pre></div>
                <p>RenderTorch 的 <code>FrameBuffer</code> 只有一份颜色、一份相机深度和物体编号，
                没有光源深度 Pass，也没有在线性 HDR 缓冲上执行色调映射。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "深度比较与输出顺序",
                r"""<details><summary>问题 1：为什么 shadow map 不能直接用相机深度缓冲？</summary>
                <p><strong>答案：</strong>阴影需要从光源视角记录遮挡关系，相机视角无法回答“光源是否看得见该点”。</p></details>
                <details><summary>问题 2：bias 过小和过大分别有什么症状？</summary>
                <p><strong>答案：</strong>过小出现 shadow acne；过大出现阴影脱离物体的 peter panning。</p></details>
                <details><summary>问题 3：Reinhard 的作用是什么？</summary>
                <p><strong>答案：</strong>把无上界的线性 HDR 颜色平滑压缩到有限显示范围，同时保留高光层次。</p></details>
                <details><summary>问题 4：应该先 gamma 还是先 Reinhard？</summary>
                <p><strong>答案：</strong>先在线性空间曝光和色调映射，最后做 gamma 显示编码。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "自阴影、漏光与灰蒙蒙",
                r"""<ol class="debug-list"><li><strong>表面全是条纹：</strong>检查 shadow map 分辨率、斜率 bias 和光源空间变换。</li>
                <li><strong>阴影悬空：</strong>bias 或法线偏移过大。</li>
                <li><strong>移动光源时阴影不动：</strong>检查光源视图矩阵与 shadow map 是否同步更新。</li>
                <li><strong>高光全白：</strong>在线性 HDR 中保留大于 1 的值，先 Reinhard 再 gamma。</li>
                <li><strong>中间调太暗：</strong>检查是否在线性空间提前 clamp，或遗漏 gamma 显示变换。</li></ol>
                <p>调试阴影时先把所有材质设为无高光纯色，再检查 PCF，最后恢复光照细节。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "当前阴影与颜色管理状态",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章在 CPU 上生成小型 shadow map，
                用 depth bias 比较方块与球的遮挡，并显示线性、Reinhard 和 gamma 后颜色。</p>
                <p><span class="status-pill">现有 RenderTorch</span> <code>renderer/framebuffer.py</code>
                直接保存 8 位 RGBA 与相机深度；<code>renderer/pipeline.py</code> 每帧只做相机投影和逐面着色。
                当前没有 shadow map、HDR 颜色缓冲、曝光、色调映射或 gamma 输出。</p>
                <p><span class="status-pill">后续实现路线</span> 光源视图矩阵 → 光源深度 Pass →
                bias/PCF → 线性 HDR framebuffer → 曝光与 Reinhard → gamma/sRGB 编码。</p>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "shadow-demo",
                "CPU 阴影图与显示变换",
                "demos/chapter14.js",
                "移动光源观察 shadow map 与遮挡同步变化，比较线性、Reinhard 和 gamma 后颜色。",
            ),
        ),
        "sources": (
            source("renderer/framebuffer.py", "现有实现：RGBA、相机深度和物体编号，无 HDR"),
            source("renderer/pipeline.py", "现有实现：相机空间深度与直接着色，无阴影 Pass"),
            source("compute/projection.py", "现有实现：透视与视口投影，可复用到光源视图"),
        ),
    },
    "chapter15": {
        "sections": (
            section(
                "intuition",
                "从像素发出光线",
                "用求交代替三角形覆盖",
                r"""<p>光栅化从物体出发，把三角形投影到屏幕；光线追踪从像素出发，沿视线反向发射光线，
                询问它首先碰到哪个物体。命中后可以计算法线、光照、反射、折射与阴影。</p>
                <p>两种方法得到的是同一个成像问题：从相机穿过像素的射线，与场景相互作用后到达相机。
                    光栅化用硬件友好的三角形流程解决它，光线追踪用通用求交提供更直接的全局效果表达。</p>
                <div class="visual-card"><svg viewBox="0 0 640 300" role="img" aria-label="射线与球体平面求交">
                <path d="M40 230H590" stroke="#9db6b8" stroke-width="3"/>
                <circle cx="350" cy="157" r="70" fill="#087f83" fill-opacity=".2" stroke="#58c7bd" stroke-width="3"/>
                <path d="M45 80L350 157" stroke="#ef9a67" stroke-width="3"/>
                <path d="M350 157L490 73" stroke="#ffd48a" stroke-width="3"/>
                <circle cx="45" cy="80" r="7" fill="#ef9a67"/><circle cx="350" cy="157" r="6" fill="#fff"/>
                <path d="M320 136L350 157L320 177" fill="none" stroke="#fff" stroke-width="2"/>
                <text x="45" y="58">相机 o</text><text x="420" y="55">反射光线</text><text x="215" y="103">p(t)=o+td</text>
                </svg><p><strong>图 15.1</strong>　一条主射线命中球面，法线决定后续反射或折射方向。</p></div>""",
            ),
            section(
                "definitions",
                "符号与定义",
                "射线、球心与命中参数",
                r"""<p>射线定义为 \(\mathbf p(t)=\mathbf o+t\mathbf d\)，其中
                \(\mathbf o\) 是原点，\(\mathbf d\) 是方向，\(t\ge0\) 是沿射线的距离参数。
                方向归一化时，\(t\) 也可直接解释为世界距离。</p>
                <p>球由球心 \(\mathbf c\) 与半径 \(r\) 定义。命中的要求是：
                \(t\) 为实数、\(t\ge0\)，并且在所有候选物体中取最小值。</p>
                <p>若射线起点位于物体内部，二次方程可能给出一个负根和一个正根。
                只接受正根；若需要透明物体出口，可把负根转换为沿反方向的出口距离。</p>""",
            ),
            section(
                "derivation",
                "逐步推导",
                "ray-sphere 二次方程与反射",
                r"""<p><strong>第一步：代入射线。</strong> 球面方程为
                \(\lVert\mathbf p-\mathbf c\rVert^2=r^2\)。代入 <code>p(t) = o + t d</code>：</p>
                <div class="formula">\[\lVert\mathbf o+t\mathbf d-\mathbf c\rVert^2=r^2.\]</div>
                <p>令 \(\mathbf m=\mathbf o-\mathbf c\)，展开得到：</p>
                <div class="formula">\[(\mathbf d\cdot\mathbf d)t^2+
                2(\mathbf m\cdot\mathbf d)t+(\mathbf m\cdot\mathbf m-r^2)=0.\]</div>
                <p>即 \(at^2+bt+c=0\)，其中 \(a=\mathbf d\cdot\mathbf d\)、
                \(b=2\mathbf m\cdot\mathbf d\)、\(c=\mathbf m\cdot\mathbf m-r^2\)。</p>
                <p><strong>第二步：判别式与根。</strong></p>
                <div class="formula">\[\Delta=b^2-4ac,\qquad
                t_{\pm}=\frac{-b\pm\sqrt{\Delta}}{2a}.\]</div>
                <p>若 \(\Delta&lt;0\)，没有交点；否则检查两个根，保留有效的最小非负 \(t\)。
                当 \(a\) 接近零时，射线退化，应跳过而不是除以零。</p>
                <p><strong>第三步：命中点与法线。</strong></p>
                <div class="formula">\[\mathbf p=\mathbf o+t\mathbf d,\qquad
                \mathbf n=\operatorname{normalize}(\mathbf p-\mathbf c).\]</div>
                <p><strong>第四步：镜面反射。</strong> 将方向沿法线分解，反向法线分量：</p>
                <div class="formula">\[\mathbf r=\mathbf d-2(\mathbf d\cdot\mathbf n)\mathbf n.\]</div>
                <p>代码中对应 <code>r = d − 2(d · n)n</code>。反射方向应单位化，
                并以微小偏移从表面发出，避免自交。</p>
                <p><strong>第五步：概念扩展。</strong> 折射使用 Snell 定律。设入射方向指向表面，
                法线朝向入射侧，\(\eta=n_1/n_2\)、\(\cos i=-\mathbf N\cdot\mathbf I\)：</p>
                <div class="formula">\[\mathbf T=\eta\mathbf I+
                \left(\eta\cos i-\sqrt{1-\eta^2(1-\cos^2i)}\right)\mathbf N.\]</div>
                <p>根号内为负时发生全内反射。Fresnel 可用 Schlick 近似
                \(F=F_0+(1-F_0)(1-\cos\theta)^5\)。
                路径追踪进一步用渲染方程递归积分多条随机路径；本章浏览器实验不实现完整 PBR 或路径追踪。</p>""",
            ),
            section(
                "example",
                "数值算例",
                "一个球的两个交点",
                r"""<p>取 \(\mathbf o=(0,0,-5)\)、\(\mathbf d=(0,0,1)\)，
                球心 \(\mathbf c=(0,0,0)\)、半径 \(r=2\)。</p>
                <div class="formula">\[a=1,\qquad b=2(-5)=-10,\qquad c=25-4=21,\qquad
                \Delta=100-84=16.\]</div>
                <div class="formula">\[t_{\pm}=\frac{10\pm4}{2}=3,\ 7.\]</div>
                <p>最近交点为 \(t=3\)，位置 \(\mathbf p=(0,0,-2)\)，法线 \(\mathbf n=(0,0,-1)\)。
                因为 \(\mathbf d\cdot\mathbf n=-1\)，反射方向为：</p>
                <div class="formula">\[\mathbf r=(0,0,1)-2(-1)(0,0,-1)=(0,0,-1).\]</div>
                <p>增大最大反射次数后， secondary ray 会继续寻找其他球或平面，形成可见的反射贡献。</p>""",
            ),
            section(
                "pseudocode",
                "伪代码与实现映射",
                "最近命中和受限反射",
                r"""<div class="code-card"><div class="code-toolbar"><span>JAVASCRIPT · 概念伪代码</span><button class="copy-button" type="button" data-copy>复制</button></div>
                <pre><code>const oc = subtract(origin, center);
const a = dot(direction, direction);
const b = 2 * dot(oc, direction);
const c = dot(oc, oc) - radius * radius;
const discriminant = b * b - 4 * a * c;
let nearest = Infinity;
if (a &gt; 1e-12 &amp;&amp; discriminant &gt;= 0) {
  for (const sign of [-1, 1]) {
    const t = (-b + sign * Math.sqrt(discriminant)) / (2 * a);
    if (t &gt;= 0 &amp;&amp; t &lt; nearest) nearest = t;
  }
}
if (nearest !== Infinity) {
  const p = add(origin, scale(direction, nearest));
  const n = normalize(subtract(p, center));
  const r = subtract(direction, scale(n, 2 * dot(direction, n)));
}</code></pre></div>
                <p>RenderTorch 当前没有光线追踪器。本章实验是独立浏览器概念实现，
                通过球与平面的求交展示射线模型，不替代现有光栅管线。</p>""",
            ),
            section(
                "self-test",
                "自测",
                "求交、反射与算法边界",
                r"""<details><summary>问题 1：为什么两个根中要取最小非负值？</summary>
                <p><strong>答案：</strong>射线首先遇到的是沿方向最近的表面；负根位于射线反方向。</p></details>
                <details><summary>问题 2：判别式为负表示什么？</summary>
                <p><strong>答案：</strong>射线不与球面相交。</p></details>
                <details><summary>问题 3：反射公式中为什么要减 2(d·n)n？</summary>
                <p><strong>答案：</strong>它把方向沿法线的分量反向，同时保留切向分量。</p></details>
                <details><summary>问题 4：最大反射次数为 0 时会看到什么？</summary>
                <p><strong>答案：</strong>只计算主射线命中的直接颜色，不继续追踪 secondary ray。</p></details>""",
            ),
            section(
                "debugging",
                "常见错误与调试",
                "漏交、自交与反射噪声",
                r"""<ol class="debug-list"><li><strong>球体出现空洞：</strong>检查方向是否归一化、判别式符号与根的有效性。</li>
                <li><strong>表面出现黑点：</strong> secondary ray 起点与表面重合导致自交，应沿新方向做微小偏移。</li>
                <li><strong>法线方向错误：</strong>球面法线由命中点减球心得到；若起点在内部，需按入射侧决定朝向。</li>
                <li><strong>反射越来越暗：</strong>每次反弹可按材质能量衰减，但不能在线性颜色中提前 clamp。</li>
                <li><strong>多样本边缘闪烁：</strong>使用固定、可复现的样本偏移，并避免无约束随机数造成测试不可重复。</li></ol>
                <p>先用 1 条光线、0 次反射验证球体与平面，再增加多样本，最后开启 secondary ray。</p>""",
            ),
            section(
                "status",
                "实现状态",
                "光栅与光线追踪的定位",
                r"""<p><span class="status-pill">浏览器概念实验</span> 本章实现 CPU ray-sphere、平面求交、镜面反射，
                最大反射次数 0–3，每像素 1–4 条光线，只在用户操作后重绘。</p>
                <p><span class="status-pill">现有 RenderTorch</span> <code>renderer/pipeline.py</code>
                走裁剪、透视除法、三角形光栅和 z-buffer；没有 BVH、光线求交、反射、折射或路径追踪。</p>
                <p><span class="status-pill">后续实现路线</span> 先添加射线与球/三角形求交，
                再增加简单 BVH 和阴影射线；材质完整后再加入 Fresnel、折射、HDR 与随机路径积分。</p>
                <table><thead><tr><th>维度</th><th>光栅化</th><th>光线追踪</th></tr></thead><tbody>
                <tr><td>主要循环</td><td>物体 → 三角形 → 像素</td><td>像素 → 射线 → 物体</td></tr>
                <tr><td>可见性</td><td>z-buffer</td><td>最近射线交点</td></tr>
                <tr><td>优势</td><td>硬件成熟、实时友好</td><td>反射、折射与全局效果表达直接</td></tr>
                <tr><td>主要成本</td><td>三角形吞吐与属性带宽</td><td>求交次数、内存访问与样本数</td></tr>
                </tbody></table>""",
            ),
        ),
        "demos": (
            DemoSpec(
                "raytrace-demo",
                "球体求交与受限镜面反射",
                "demos/chapter15.js",
                "从像素发射射线，求球和平面的最近交点，并逐级增加反射次数与每像素光线数。",
            ),
        ),
        "sources": (
            source("renderer/pipeline.py", "对比实现：三角形光栅与 z-buffer，不是光线追踪"),
            source("renderer/rasterizer.py", "对比实现：边函数覆盖与逐像素可见性"),
            source("renderer/framebuffer.py", "可复用概念：最终像素颜色与深度输出"),
        ),
    },
}


def chapters() -> list[Chapter]:
    rows = []
    for row in CHAPTER_METADATA:
        details = CHAPTER_DETAILS.get(row.slug)
        if details is None:
            rows.append(row)
        else:
            rows.append(
                replace(
                    row,
                    sections=tuple(details["sections"]),
                    demos=tuple(details["demos"]),
                    sources=tuple(details["sources"]),
                )
            )
    return rows


def references() -> list[dict]:
    path = Path(__file__).with_name("references.json")
    return json.loads(path.read_text(encoding="utf-8"))
