BRANCH = "feat/m1-qt-minimal-loop"
REPOSITORY = "https://github.com/Zhenghong-Liu/RenderTorch"


def source_url(path):
    return f"{REPOSITORY}/blob/{BRANCH}/{path}"


CHAPTERS = [
    {
        "number": 1,
        "title": "为什么从零搭建 3D 与物理教学项目",
        "summary": "先确定学习目标与边界，再决定需要自己实现什么。",
        "intuition": """
<p>物理引擎通常把渲染、场景、碰撞和求解器藏在黑盒里。对于教学，这恰好隐藏了最值得观察的部分：一个粒子如何前进、一条约束如何把位置拉回、一个像素为什么被更近的三角形覆盖。</p>
<p>RenderTorch 把这条链路拆开。你既能修改 PyTorch 中的数值计算，也能在 Qt 视口里立即看到结果。项目不追求商业引擎的完整性，而是追求每一步都可读、可运行、可验证。</p>
<div class="callout"><strong>生活类比：</strong>这像先拆开一台机械钟理解齿轮传动，再决定是否重新组装。拆开的目的不是否定钟表，而是掌握因果关系。</div>
""",
        "formula": """
<div class="formula">学习闭环 = 直觉 → 最小实现 → 数值测试 → 可视化 → 误差解释</div>
<p>这里没有新的物理公式。重要的是验收顺序：先定义可观察的不变量，例如固定粒子不动、近处表面遮挡远处表面，再写实现。</p>
""",
        "pseudocode": """
for milestone in [最小闭环, 公共物理, 三种布料, 刚体与软体, 教学书]:
    阅读对应参考资料与现有代码
    写一个能失败的验收测试
    实现最小但完整的纵向切片
    运行测试并观察真实画面
    记录指标、限制和下一步
""",
        "sources": ["docs/development-roadmap.md", "docs/references.md"],
        "visual": """
<svg viewBox="0 0 720 260" role="img" aria-label="RenderTorch 学习闭环">
  <defs><marker id="arrow1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="currentColor"/></marker></defs>
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow1)">
    <path d="M120 80 H250"/><path d="M330 80 H460"/><path d="M540 80 H620"/>
    <path d="M620 110 V190"/><path d="M620 190 H120"/><path d="M120 190 V115"/>
  </g>
  <g class="svg-node"><rect x="25" y="48" width="95" height="64" rx="12"/><text x="72" y="86">问题直觉</text>
    <rect x="250" y="48" width="80" height="64" rx="12"/><text x="290" y="86">实现</text>
    <rect x="460" y="48" width="80" height="64" rx="12"/><text x="500" y="86">测试</text>
    <rect x="620" y="48" width="80" height="64" rx="12"/><text x="660" y="86">画面</text>
    <rect x="350" y="158" width="110" height="64" rx="12"/><text x="405" y="196">解释指标</text></g>
</svg>
""",
        "metrics": [
            ("可运行闭环", "main_qt.py 启动后显示场景"),
            ("核心可测", "core / compute / renderer 不依赖 Qt"),
            ("当前基线", "102 项自动化测试通过"),
        ],
        "debugging": [
            "如果不知道先做什么，回到路线图的当前里程碑，只补一条完整链路。",
            "如果测试通过但画面不对，检查相机方向、矩阵约定和屏幕 Y 轴翻转。",
            "如果引入新库后难以解释，先问它是否解决了当前验收中的具体问题。",
        ],
    },
    {
        "number": 2,
        "title": "总体架构与模块边界",
        "summary": "一个进程、清晰分层，让 UI、场景、计算与求解器各自保持简单。",
        "intuition": """
<p>项目采用模块化单体。Qt 负责窗口与输入，<code>Scene</code> 保存唯一场景状态，软件渲染器读取场景并输出帧缓冲，物理求解器只修改粒子状态。</p>
<p>控制层把界面上的重力、算法和迭代次数转换成求解器参数。这样同一套物理代码既能由 Qt 驱动，也能在测试中直接运行。</p>
<div class="callout"><strong>关键不变量：</strong>核心逻辑不能读取 QSpinBox，也不能创建 QWidget。UI 状态与模拟状态通过 Controller 单向连接。</div>
""",
        "formula": """
<div class="formula">FrameBuffer = Render(Scene, Camera, display_mode)<br>State(t + h) = Solver(State(t), parameters, colliders)</div>
<p>渲染与模拟是两个独立阶段。模拟改变位置和速度；渲染只读取当前状态，不反向修改物理量。</p>
""",
        "pseudocode": """
PhysicsPanel --参数--> PhysicsController --构建--> Simulation
Simulation --更新位置/速度--> Scene.mesh.vertices
Viewport --请求帧--> SoftwareRenderer --> FrameBuffer
FrameBuffer --复制--> QImage --绘制--> QPainter
""",
        "sources": ["app/physics_controller.py", "app/controller.py", "core/scene.py", "renderer/pipeline.py"],
        "visual": """
<svg viewBox="0 0 720 300" role="img" aria-label="RenderTorch 模块架构">
  <g class="svg-node"><rect x="30" y="20" width="150" height="54" rx="10"/><text x="105" y="53">Qt 面板与视口</text>
    <rect x="235" y="20" width="150" height="54" rx="10"/><text x="310" y="53">Controller</text>
    <rect x="440" y="20" width="150" height="54" rx="10"/><text x="515" y="53">Scene</text>
    <rect x="30" y="125" width="150" height="54" rx="10"/><text x="105" y="158">Physics</text>
    <rect x="235" y="125" width="150" height="54" rx="10"/><text x="310" y="158">Compute</text>
    <rect x="440" y="125" width="150" height="54" rx="10"/><text x="515" y="158">Renderer</text>
    <rect x="235" y="230" width="150" height="54" rx="10"/><text x="310" y="263">FrameBuffer</text></g>
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow1)"><path d="M180 47 H235"/><path d="M385 47 H440"/><path d="M515 74 V125"/><path d="M310 74 V125"/><path d="M105 74 V200 H235"/><path d="M515 179 V200 H385"/><path d="M310 179 V230"/></g>
</svg>
""",
        "metrics": [
            ("UI 依赖", "仅 app 层导入 PySide6"),
            ("模拟步长", "1/120 s，2 个子步"),
            ("场景同步", "每步复制粒子位置到网格"),
        ],
        "debugging": [
            "如果核心测试必须创建 QApplication，说明依赖方向反了。",
            "如果切换算法后场景重置但相机不重置，检查 Controller 的构建边界。",
            "如果球体位置更新而网格没动，检查 _sync_meshes 是否访问了正确的 body_simulation。",
        ],
    },
    {
        "number": 3,
        "title": "最小闭环：相机、投影和 FrameBuffer",
        "summary": "把三维坐标经过视图、投影和视口变换送到像素平面。",
        "intuition": """
<p>轨道相机围绕目标点旋转。视图矩阵把世界坐标变成相机坐标，透视矩阵把视锥映射到齐次裁剪空间，视口矩阵再把 NDC 映射到屏幕像素。</p>
<p>项目使用行向量约定：<code>world × View × Projection × Viewport</code>。这与不少 OpenGL 教材的列向量顺序不同，混用会导致看似随机的旋转和裁剪错误。</p>
<div class="callout"><strong>生活类比：</strong>相机位置决定你站在哪里，视锥决定你能看多宽，FrameBuffer 则相当于一张等待曝光的数字底片。</div>
""",
        "formula": """
<div class="formula">p<sub>clip</sub> = p<sub>world</sub> M<sub>view</sub> M<sub>projection</sub><br>p<sub>ndc</sub> = p<sub>clip</sub> / w<sub>clip</sub><br>x<sub>screen</sub> = (x<sub>ndc</sub> + 1) width / 2<br>y<sub>screen</sub> = (1 − y<sub>ndc</sub>) height / 2</div>
<p>透视除法必须在近裁剪面之后进行。像素 Y 向下，而 NDC Y 向上，所以需要翻转。</p>
""",
        "pseudocode": """
view = look_at(camera.position, camera.target, world_up)
projection = perspective(fov_y, aspect, near, far)
clip = world_vertices @ view @ projection
inside = clip.z >= -clip.w and clip.z <= clip.w
ndc = inside_clip / inside_clip.w
pixel = viewport(ndc, width, height)
""",
        "sources": ["compute/camera.py", "compute/projection.py", "renderer/framebuffer.py"],
        "visual": """
<svg viewBox="0 0 720 280" role="img" aria-label="世界坐标到屏幕像素的变换链">
  <g class="svg-node"><rect x="20" y="100" width="115" height="70" rx="10"/><text x="77" y="141">世界坐标</text>
    <rect x="175" y="100" width="115" height="70" rx="10"/><text x="232" y="141">视图矩阵</text>
    <rect x="330" y="100" width="115" height="70" rx="10"/><text x="387" y="141">投影矩阵</text>
    <rect x="485" y="100" width="95" height="70" rx="10"/><text x="532" y="141">NDC</text>
    <rect x="620" y="100" width="80" height="70" rx="10"/><text x="660" y="141">像素</text></g>
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow1)"><path d="M135 135 H175"/><path d="M290 135 H330"/><path d="M445 135 H485"/><path d="M580 135 H620"/></g>
  <path d="M77 100 L45 35 L110 35 Z" fill="none" stroke="var(--accent)" stroke-width="2"/><text x="47" y="22">相机视锥</text>
</svg>
""",
        "metrics": [
            ("相机距离", "3.0，范围 0.5–50"),
            ("默认 FOV", "55° 垂直视场角"),
            ("深度缓冲", "float32，初始 +∞"),
        ],
        "debugging": [
            "所有面都消失：检查 forward 方向与近远平面符号。",
            "图像上下颠倒：检查视口矩阵是否翻转 Y。",
            "窗口缩放后物体被拉长：确认 projection_aspect 使用最新宽高比。",
        ],
    },
    {
        "number": 4,
        "title": "软件光栅化与 Qt 图像显示",
        "summary": "裁剪三角形、判断像素覆盖、用深度测试决定谁可见。",
        "intuition": """
<p>软件渲染器逐个三角形工作。齐次裁剪先删除视锥外几何并生成新交点，透视除法后计算屏幕包围盒，最后用重心坐标判断像素是否落在三角形内。</p>
<p>每个候选像素都比较深度。更近的像素写入颜色、深度和物体 ID；更远的像素被拒绝。ID 缓冲让“这一像素属于哪个对象”可以独立于颜色被检查。</p>
<div class="callout"><strong>线框的特殊处理：</strong>一条线段穿过近裁剪面时可能变成两个端点。参数化裁剪保证返回的仍是有限线段，而不是重复顶点。</div>
""",
        "formula": """
<div class="formula">λ<sub>1</sub> + λ<sub>2</sub> + λ<sub>3</sub> = 1，λ<sub>i</sub> ≥ 0<br>z<sub>pixel</sub> = λ<sub>1</sub>z<sub>1</sub> + λ<sub>2</sub>z<sub>2</sub> + λ<sub>3</sub>z<sub>3</sub><br>写入条件：z<sub>pixel</sub> &lt; depth[x, y]</div>
<p>深度测试不是“最后画的盖住先画的”，因此对象添加顺序不应改变遮挡结果。</p>
""",
        "pseudocode": """
for triangle in clipped_triangles:
    divide_by_w_and_map_to_screen()
    for pixel in triangle.bounding_box:
        barycentric = compute_barycentric(pixel, triangle)
        if barycentric.inside():
            depth = interpolate_perspective_correct(barycentric)
            if depth < framebuffer.depth[pixel]:
                write_color_depth_and_object_id()
""",
        "sources": ["renderer/clipping.py", "renderer/rasterizer.py", "renderer/pipeline.py", "app/viewport.py"],
        "visual": """
<svg viewBox="0 0 720 250" role="img" aria-label="三角形光栅化与深度缓冲">
  <rect x="40" y="25" width="280" height="190" class="svg-grid"/><polygon points="80,190 210,45 295,190" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="2"/>
  <g fill="var(--accent)"><circle cx="120" cy="150" r="4"/><circle cx="160" cy="110" r="4"/><circle cx="200" cy="70" r="4"/><circle cx="240" cy="120" r="4"/></g>
  <g class="svg-node"><rect x="385" y="40" width="270" height="48" rx="8"/><text x="520" y="70">颜色 RGBA · uint8</text>
    <rect x="385" y="102" width="270" height="48" rx="8"/><text x="520" y="132">深度 · float32</text>
    <rect x="385" y="164" width="270" height="48" rx="8"/><text x="520" y="194">物体 ID · int32</text></g>
</svg>
""",
        "metrics": [
            ("显示链路", "FrameBuffer → QImage → QPainter"),
            ("显示模式", "实体 / 线框 / 粒子"),
            ("数值保护", "裁剪后再做透视除法"),
        ],
        "debugging": [
            "三角形有裂缝：检查共享边像素规则与浮点误差。",
            "线框出现 NaN：确认使用 clip_line 而不是对两个顶点调用多边形裁剪。",
            "Qt 画面闪烁或变黑：确认 QImage 引用的 NumPy 数据在 paintEvent 期间仍有效。",
        ],
    },
    {
        "number": 5,
        "title": "物理模拟基础：粒子、时间步和积分",
        "summary": "用半隐式欧拉推进状态，用子步和累加器保持时间一致。",
        "intuition": """
<p>一个动态粒子由位置、速度和逆质量描述。逆质量为 0 的粒子被固定；其余粒子接受重力、阻尼、碰撞与约束产生的速度变化。</p>
<p>固定时间步让模拟不依赖帧率。累加器保存“还没模拟完的真实时间”，每次取出一个 <code>h</code>；渲染帧可以快也可以慢，但单个物理步的长度保持一致。</p>
<div class="callout"><strong>为什么先更新速度：</strong>半隐式欧拉使用更新后的速度计算新位置，比显式欧拉更早看到重力影响，对简单摆和弹簧通常更稳定。</div>
""",
        "formula": """
<div class="formula">v ← v + (g + F / m) h<br>x ← x + v h<br>h = fixed_dt / substeps</div>
<p>阻尼实现为 <code>v ← v · exp(−c h)</code>。指数形式在较大时间步下仍保持非负衰减。</p>
""",
        "pseudocode": """
accumulator += min(frame_dt, 0.25)
while accumulator >= fixed_dt:
    accumulator -= fixed_dt
    for substep in range(substeps):
        h = fixed_dt / substeps
        apply_gravity_and_damping(h)
        integrate_positions(h)
        solve_constraints(h)
        collide(h)
""",
        "sources": ["physics/common/state.py", "physics/common/simulation.py", "physics/common/colliders.py"],
        "visual": """
<svg viewBox="0 0 720 240" role="img" aria-label="固定时间步累加器">
  <rect x="45" y="35" width="630" height="75" rx="12" class="svg-grid"/>
  <g stroke="currentColor" stroke-width="2"><path d="M85 72 H630"/><path d="M150 55 V90"/><path d="M255 55 V90"/><path d="M360 55 V90"/><path d="M465 55 V90"/><path d="M570 55 V90"/></g>
  <g fill="var(--accent)"><rect x="85" y="63" width="65" height="18" rx="5"/><rect x="255" y="63" width="105" height="18" rx="5"/><rect x="465" y="63" width="105" height="18" rx="5"/></g>
  <text x="82" y="155">每格 = 1/120 s；渲染帧只需把时间倒入累加器</text><text x="82" y="190">单步、暂停和重置共用同一条 _substep 路径</text>
</svg>
""",
        "metrics": [
            ("固定步长", "1/120 s"),
            ("子步", "2"),
            ("帧时间上限", "0.25 s，防止卡顿后追帧爆炸"),
        ],
        "debugging": [
            "不同帧率结果差异大：检查是否直接用 frame_dt 积分。",
            "重置后第一步不一致：检查初始状态是否被原地修改。",
            "固定粒子漂移：确认 inverse_mass 为 0，且碰撞投影遵守权重。",
        ],
    },
    {
        "number": 6,
        "title": "质点弹簧布料模型",
        "summary": "把布看作粒子网络，用结构、剪切和弯曲弹簧产生恢复力。",
        "intuition": """
<p>布料网格的每条边都可以携带弹簧。结构弹簧抵抗拉伸，剪切弹簧抵抗网格扭斜，弯曲弹簧连接隔一个粒子的顶点，让布不会像纸一样任意折曲。</p>
<p>弹簧力是速度相关的。沿弹簧方向的相对速度产生阻尼，抑制振动；垂直于弹簧方向的相对速度不贡献这条弹簧的阻尼力。</p>
<div class="callout"><strong>主要风险：</strong>显式力积分存在稳定时间步上限。刚度过大或步长过大时，能量会迅速增加，出现 NaN 或网格爆炸。</div>
""",
        "formula": """
<div class="formula">F<sub>spring</sub> = k (L − L<sub>0</sub>) + c v<sub>rel,along</sub><br>v<sub>rel,along</sub> = (v<sub>b</sub> − v<sub>a</sub>) · n<br>a = F / m</div>
<p>当前实现把成对弹簧力写入零力数组，并使用 <code>index_add_</code> 累积到每个粒子。</p>
""",
        "pseudocode": """
for spring in geometry.constraints:
    delta = position[b] - position[a]
    direction = normalize(delta)
    stretch = length(delta) - rest_length
    relative_v = velocity[b] - velocity[a]
    magnitude = stiffness * stretch + damping * dot(relative_v, direction)
    force = direction * magnitude
    force[a] += force
    force[b] -= force
velocity += force * inverse_mass * h
""",
        "sources": ["physics/cloth/mass_spring.py", "physics/cloth/geometry.py"],
        "visual": """
<svg viewBox="0 0 720 250" role="img" aria-label="布料结构剪切与弯曲约束">
  <g fill="var(--accent)"><circle cx="100" cy="70" r="6"/><circle cx="180" cy="70" r="6"/><circle cx="260" cy="70" r="6"/><circle cx="100" cy="150" r="6"/><circle cx="180" cy="150" r="6"/><circle cx="260" cy="150" r="6"/><circle cx="100" cy="230" r="6"/><circle cx="180" cy="230" r="6"/><circle cx="260" cy="230" r="6"/></g>
  <g stroke="var(--accent)" stroke-width="3"><path d="M100 70 H260 M100 150 H260 M100 230 H260 M100 70 V230 M180 70 V230 M260 70 V230"/></g>
  <g stroke="var(--warm)" stroke-width="2" stroke-dasharray="7 5"><path d="M100 70 L180 150 L260 230 M260 70 L180 150 L100 230 M100 70 L260 150 M260 70 L100 150"/></g>
  <text x="350" y="80">实线：结构弹簧</text><text x="350" y="125">虚线：剪切弹簧</text><text x="350" y="170">跨两格：弯曲弹簧</text><text x="350" y="215">同一拓扑，三种算法共享</text>
</svg>
""",
        "metrics": [
            ("默认结构刚度", "800"),
            ("剪切 / 弯曲", "200 / 20"),
            ("弹簧阻尼", "2.0"),
        ],
        "debugging": [
            "轻微拉伸就爆炸：减小 fixed_dt、增大 substeps 或降低刚度。",
            "布像橡胶膜一样抖动：提高阻尼，但过高会拖慢运动。",
            "约束误差仍很大：确认 last_constraint_error 是相对误差，不是绝对长度。",
        ],
    },
    {
        "number": 7,
        "title": "PBD：位置约束与迭代求解",
        "summary": "不直接求弹簧力，而是反复把粒子投影回距离约束允许的位置。",
        "intuition": """
<p>PBD 把“粒子应该在哪里”当作主要问题。每条距离约束产生一个沿连线方向的修正，按逆质量分配给两个粒子，然后进入下一条约束。</p>
<p>求解采用顺序 Gauss–Seidel：修正立即生效，后面的约束看到更新后的位置。与之相对，同时计算所有修正再一次性累加，会在共享粒子多的高刚度网格中过度修正。</p>
<div class="callout"><strong>为什么稳定：</strong>位置投影不会像过硬的显式弹簧那样放大速度。但 PBD 的 stiffness 更像每轮投影比例，不是材料模量。</div>
""",
        "formula": """
<div class="formula">C = |x<sub>b</sub> − x<sub>a</sub>| − L<sub>0</sub><br>Δx<sub>a</sub> = w<sub>a</sub> (C / (w<sub>a</sub> + w<sub>b</sub>)) n<br>Δx<sub>b</sub> = −w<sub>b</sub> (C / (w<sub>a</sub> + w<sub>b</sub>)) n</div>
<p>当前实现把修正乘以 stiffness。固定粒子权重为 0，因此不会被投影移动。</p>
""",
        "pseudocode": """
for iteration in range(iterations):
    for constraint in constraints:
        weight = inverse_mass[a] + inverse_mass[b]
        if weight == 0: continue
        delta = position[b] - position[a]
        correction = normalize(delta) * (length(delta) - rest_length)
        position[a] += stiffness * correction * inverse_mass[a] / weight
        position[b] -= stiffness * correction * inverse_mass[b] / weight
reconstruct_velocities_from_position_change(h)
""",
        "sources": ["physics/cloth/pbd.py", "physics/cloth/geometry.py", "physics/common/simulation.py"],
        "visual": """
<svg viewBox="0 0 720 250" role="img" aria-label="PBD 顺序位置投影">
  <g stroke="var(--muted)" stroke-width="3"><path d="M90 75 H280"/><path d="M90 175 H280"/></g>
  <g stroke="var(--accent)" stroke-width="3"><path d="M90 75 H315"/><path d="M90 175 H245"/></g>
  <g fill="var(--accent)"><circle cx="90" cy="75" r="7"/><circle cx="280" cy="75" r="7"/><circle cx="315" cy="75" r="7"/><circle cx="90" cy="175" r="7"/><circle cx="280" cy="175" r="7"/><circle cx="245" cy="175" r="7"/></g>
  <text x="70" y="35">被拉伸</text><text x="335" y="115">第一条约束先修正</text><text x="335" y="150">后续约束立即读取新位置</text><text x="335" y="185">重复 iterations 轮</text>
</svg>
""",
        "metrics": [
            ("默认结构刚度", "0.95"),
            ("默认迭代", "8"),
            ("共享拓扑", "8 × 6 网格，固定顶行"),
        ],
        "debugging": [
            "迭代增加但误差不降：检查约束是否按顺序求解，以及固定粒子权重。",
            "网格爆炸：不要同时累加所有约束修正，改为逐条更新。",
            "运动没有惯性：确认求解后根据位置变化重建速度。",
        ],
    },
    {
        "number": 8,
        "title": "XPBD：柔度与拉格朗日乘子",
        "summary": "在 PBD 投影中加入 compliance，让“软硬”具备时间步一致的物理含义。",
        "intuition": """
<p>XPBD 为每条约束维护一个拉格朗日乘子 <code>λ</code>。它不仅看当前误差，也看这个子步中已经施加过多少修正，因此能表达有限的材料柔度。</p>
<p>compliance 越小，约束越硬。为了在不同子步下保持意义，代码使用 <code>α̃ = α / h²</code>。乘子在一个子步的所有迭代中保留，进入下一个子步时清零。</p>
<div class="callout"><strong>与 PBD 的关系：</strong>compliance 为 0 时，XPBD 的距离约束投影退化为 PBD 的硬约束形式。</div>
""",
        "formula": """
<div class="formula">α̃ = α / h²<br>Δλ = (−C − α̃λ) / (w<sub>a</sub> + w<sub>b</sub> + α̃)<br>Δx<sub>a</sub> = −w<sub>a</sub> Δλ n<br>Δx<sub>b</sub> = w<sub>b</sub> Δλ n</div>
<p>这里的 h 是子步长，而不是一次渲染帧的时间。</p>
""",
        "pseudocode": """
effective_compliance = compliance / (step * step)
lambdas.zero_()
for iteration in range(iterations):
    for constraint in constraints:
        error = distance - rest_length
        delta_lambda = (-error - alpha * lambdas[i]) / (weight + alpha)
        lambdas[i] += delta_lambda
        position[a] -= inverse_mass[a] * delta_lambda * normal
        position[b] += inverse_mass[b] * delta_lambda * normal
""",
        "sources": ["physics/cloth/xpbd.py", "physics/cloth/geometry.py"],
        "visual": """
<svg viewBox="0 0 720 260" role="img" aria-label="XPBD 乘子在迭代中累积">
  <g class="svg-node"><rect x="35" y="95" width="145" height="65" rx="10"/><text x="107" y="133">当前误差 C</text>
    <rect x="245" y="95" width="145" height="65" rx="10"/><text x="317" y="133">上轮乘子 λ</text>
    <rect x="455" y="95" width="145" height="65" rx="10"/><text x="527" y="133">柔度 α̃</text></g>
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow1)"><path d="M180 127 H245"/><path d="M390 127 H455"/></g>
  <path d="M527 160 V215 H107 V160" fill="none" stroke="var(--warm)" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow1)"/>
  <text x="210" y="242">同一子步内迭代循环；下一子步 λ 清零</text>
</svg>
""",
        "metrics": [
            ("默认结构柔度", "1e−6"),
            ("剪切 / 弯曲", "5e−6 / 1e−5"),
            ("默认迭代", "8"),
        ],
        "debugging": [
            "改子步后软硬变化：检查是否使用 α / h²。",
            "约束越来越硬或振荡：确认 λ 没有在每次迭代前被清零。",
            "仍无法收敛：减小 compliance 或增加迭代，先分别观察。",
        ],
    },
    {
        "number": 9,
        "title": "三种布料算法的比较",
        "summary": "在相同网格、固定条件和碰撞体下观察稳定性、误差与计算成本。",
        "intuition": """
<p>三种算法不是同一公式的三种写法。质点弹簧直接积分力，PBD 直接修正位置，XPBD 用乘子表达有限柔度。比较时应固定初始位置、时间步、重力、拓扑和显示模式。</p>
<p>下面的交互 Demo 是浏览器中的简化二维类比，用于观察趋势，不是 Qt 求解器的逐像素复刻。桌面程序使用第 6–8 章链接的真实 Python 实现。</p>
""",
        "formula": """
<div class="formula">质量弹簧：更新力 → 更新速度 → 更新位置<br>PBD：预测位置 → 迭代投影 → 重建速度<br>XPBD：预测位置 → 带 λ 与 α 的投影 → 重建速度</div>
<p>公平比较需要同时报告误差和耗时。更低的误差可能来自更多迭代，而不一定来自更好的算法。</p>
""",
        "pseudocode": """
for algorithm in [mass_spring, pbd, xpbd]:
    使用相同 cloth_grid 与 colliders
    运行相同步数
    记录 finite、max_speed、constraint_error、step_ms
    在 solid / wireframe / particles 三种视图下观察
""",
        "sources": ["app/physics_controller.py", "tests/test_cloth_mass_spring.py", "tests/test_cloth_pbd.py", "tests/test_cloth_xpbd.py"],
        "visual": """
<div id="comparison-demo" class="comparison-demo">
  <div class="demo-toolbar">
    <button id="play-comparison" type="button">暂停</button>
    <button id="reset-comparison" type="button">重置</button>
    <label>迭代 <output id="iterations-output">8</output><input id="comparison-iterations" type="range" min="1" max="20" value="8"></label>
    <label>柔度 <output id="compliance-output">1.0e-6</output><input id="comparison-compliance" type="range" min="-8" max="-3" step="0.5" value="-6"></label>
  </div>
  <div class="comparison-grid">
    <figure><canvas data-comparison-canvas data-algorithm="mass_spring" width="320" height="280"></canvas><figcaption>质量弹簧 <output data-error>0.0000</output></figcaption></figure>
    <figure><canvas data-comparison-canvas data-algorithm="pbd" width="320" height="280"></canvas><figcaption>PBD <output data-error>0.0000</output></figcaption></figure>
    <figure><canvas data-comparison-canvas data-algorithm="xpbd" width="320" height="280"></canvas><figcaption>XPBD <output data-error>0.0000</output></figcaption></figure>
  </div>
  <p class="demo-caption">显示值为平均相对约束误差。拖动参数后重置，可重复比较相同初始状态。</p>
</div>
""",
        "metrics": [
            ("质点弹簧", "受力直观；高刚度需小步长"),
            ("PBD", "稳定易实现；stiffness 不够物理"),
            ("XPBD", "柔度可解释；需要乘子状态"),
        ],
        "debugging": [
            "比较结果不可重复：确认每个算法从完全相同的初始状态开始。",
            "把 Qt 参数直接搬进浏览器会得到不同数值：浏览器 Demo 故意简化了维度。",
            "只比较截图不比较误差：增加线框视图并记录数值指标。",
        ],
    },
    {
        "number": 10,
        "title": "刚体与软体 Demo",
        "summary": "刚体保持形状，软体用距离和四面体体积约束抵抗变形。",
        "intuition": """
<p>刚体 Demo 使用一个有半径的动态球体。它没有旋转自由度，重点观察重力、地面与静态球碰撞、恢复系数、摩擦，以及球体对布料粒子的双向冲量。</p>
<p>软体 Demo 使用 3×3×3 粒子格点，其中包含内部中心粒子。距离约束维持形状，六个四面体体积约束共同维持立方体体积。体积梯度来自四面体有符号体积对各顶点位置的偏导。</p>
<div class="callout"><strong>边界：</strong>这是教学级实现，不包含完整接触流形、连续碰撞检测、真实刚体旋转或通用有限元。</div>
""",
        "formula": """
<div class="formula">刚体法向冲量：j = −(1 + e) v<sub>n</sub> / (w<sub>body</sub> + w<sub>particle</sub>)<br>V<sub>tet</sub> = ((b − a) × (c − a)) · (d − a) / 6<br>XPBD 体积：Δλ = (−C − α̃λ) / (Σw<sub>i</sub>|∇C<sub>i</sub>|² + α̃)</div>
<p>体积约束使用解析梯度，不需要 autograd。梯度方向决定每个顶点应该移动多少。</p>
""",
        "pseudocode": """
# 刚体
velocity += gravity * h
position += velocity * h
project_out_of_ground_and_static_sphere()
apply_restitution_and_coulomb_friction()

# 软体
for iteration:
    project_distance_constraints_with_xpbd()
    for tetrahedron:
        compute_volume_and_analytic_gradients()
        update_lambda_and_particle_positions()
""",
        "sources": ["physics/rigidbody/sphere.py", "physics/softbody/geometry.py", "physics/softbody/xpbd.py"],
        "visual": """
<svg viewBox="0 0 720 280" role="img" aria-label="刚体球与软体体积约束">
  <circle cx="140" cy="95" r="58" fill="var(--warm-soft)" stroke="var(--warm)" stroke-width="3"/><path d="M35 210 H245" stroke="var(--muted)" stroke-width="3"/><path d="M140 153 V199" stroke="var(--accent)" stroke-width="2" marker-end="url(#arrow1)"/><text x="92" y="250">刚体：形状不变</text>
  <g fill="var(--accent)"><circle cx="430" cy="50" r="5"/><circle cx="530" cy="50" r="5"/><circle cx="430" cy="150" r="5"/><circle cx="530" cy="150" r="5"/><circle cx="480" cy="100" r="5"/></g>
  <g stroke="var(--accent)" stroke-width="2"><path d="M430 50 L530 50 L530 150 L430 150 Z M430 50 L480 100 L530 150 M530 50 L480 100 L430 150"/></g>
  <text x="405" y="250">软体：体积恢复</text>
</svg>
""",
        "metrics": [
            ("软体格点", "27 粒子，含 1 个内部粒子"),
            ("体积单元", "6 个四面体"),
            ("刚体摩擦", "库仑上限截断"),
        ],
        "debugging": [
            "球体穿过地面：碰撞判断要使用球心到平面的距离减去 radius。",
            "软体体积恢复但整体偏斜：顺序投影会破坏对称，增加迭代或检查拓扑。",
            "冲量交换不守恒：确认粒子与球体使用相同法线方向，并按总逆质量分配。",
        ],
    },
    {
        "number": 11,
        "title": "性能、稳定性与局限性",
        "summary": "先测量，再优化；把 NaN、能量增长和约束误差当作诊断信号。",
        "intuition": """
<p>教学实现的性能瓶颈通常不是矩阵库，而是 Python 中逐条顺序约束循环。它能清楚表达算法，但不适合直接扩展到大规模生产网格。</p>
<p>稳定性诊断应从时间步、刚度、柔度、阻尼和迭代次数入手。不要用钳制位置或忽略 NaN 掩盖问题，那会破坏动量与约束语义。</p>
<div class="callout"><strong>测量原则：</strong>分别记录单步耗时、最大速度、相对约束误差和接触数。只报告一个“FPS”无法区分渲染成本与求解成本。</div>
""",
        "formula": """
<div class="formula">相对约束误差 = max |L − L<sub>0</sub>| / L<sub>0</sub><br>体积误差 = |V − V<sub>0</sub>| / V<sub>0</sub><br>step_ms = elapsed / completed_steps</div>
<p>这些指标回答不同问题：误差看正确性，速度看爆炸风险，耗时看瓶颈，接触数看碰撞是否真的发生。</p>
""",
        "pseudocode": """
before = initial_positions.clone()
run_fixed_number_of_steps()
assert all_states_are_finite()
measure_max_speed_and_relative_constraint_error()
measure_elapsed_time_per_step()
change_one_parameter_and_repeat()
""",
        "sources": ["physics/common/simulation.py", "physics/softbody/xpbd.py", "tests/test_physics_controller.py"],
        "visual": """
<svg viewBox="0 0 720 270" role="img" aria-label="稳定性诊断流程">
  <g class="svg-node"><rect x="30" y="95" width="130" height="65" rx="10"/><text x="95" y="133">出现异常</text>
    <rect x="205" y="95" width="130" height="65" rx="10"/><text x="270" y="133">检查 finite</text>
    <rect x="380" y="95" width="130" height="65" rx="10"/><text x="445" y="133">减小时间步</text>
    <rect x="555" y="95" width="130" height="65" rx="10"/><text x="620" y="133">检查参数</text></g>
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow1)"><path d="M160 127 H205"/><path d="M335 127 H380"/><path d="M510 127 H555"/></g>
  <path d="M620 160 V220 H95 V160" fill="none" stroke="var(--warm)" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow1)"/><text x="245" y="247">一次只改一个变量，保留可重复测试</text>
</svg>
""",
        "metrics": [
            ("完整测试", "102 项通过"),
            ("默认布料", "约 0.59 ms/step（本机测量）"),
            ("XPBD / 软体", "顺序循环成本更高，需单独测量"),
        ],
        "debugging": [
            "先检查是否 finite，再判断误差大小；NaN 时后续指标没有意义。",
            "不要通过放大 clamp 隐藏位置爆炸，应回溯稳定时间步。",
            "优化顺序求解前先保留参考输出，避免性能提升改变算法行为。",
        ],
    },
    {
        "number": 12,
        "title": "后续扩展方向",
        "summary": "沿现有边界扩展，而不是重写整个系统。",
        "intuition": """
<p>当前版本已经覆盖从相机到三类物理 Demo 的纵向链路。下一步应优先补教学体验和工程基础，而不是立即加入通用 ECS 或大型碰撞引擎。</p>
<p>推荐的扩展顺序是：书籍交互与指标回放、场景保存加载、基础对象操作、更丰富的碰撞体，最后再考虑自碰撞与性能优化。</p>
<div class="callout"><strong>可维护性：</strong>每次扩展仍应满足“核心不依赖 UI、测试覆盖数值行为、文档链接真实源码”这三条约束。</div>
""",
        "formula": """
<div class="formula">扩展价值 = 学习收益 × 可验证性 ÷ 维护成本</div>
<p>一个候选功能如果无法定义输入、输出和不变量，就应先写设计，而不是直接改求解器。</p>
""",
        "pseudocode": """
candidates = [指标回放, 项目保存加载, 对象选择与变换, 自碰撞, GPU 求解]
for feature in candidates:
    定义独立接口与失败测试
    保持现有 Demo 可运行
    更新书籍章节与路线图
    测量收益与成本后再进入下一项
""",
        "sources": ["docs/development-roadmap.md", "app/physics_panel.py", "app/main_window.py"],
        "visual": """
<svg viewBox="0 0 720 240" role="img" aria-label="项目扩展路线">
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow1)"><path d="M70 120 H650"/></g>
  <g class="svg-node"><rect x="25" y="85" width="120" height="70" rx="10"/><text x="85" y="125">教学书</text>
    <rect x="190" y="85" width="120" height="70" rx="10"/><text x="250" y="125">保存加载</text>
    <rect x="355" y="85" width="120" height="70" rx="10"/><text x="415" y="125">对象编辑</text>
    <rect x="520" y="85" width="120" height="70" rx="10"/><text x="580" y="125">高级物理</text></g>
</svg>
""",
        "metrics": [
            ("近期", "完善交互与指标展示"),
            ("中期", "io 模块与基础编辑操作"),
            ("长期", "自碰撞、性能与更多材料模型"),
        ],
        "debugging": [
            "扩展前运行全量测试，建立稳定基线。",
            "新功能先进入独立模块，避免继续扩大 main_window.py。",
            "文档与代码同批次更新，防止书籍描述旧架构。",
        ],
    },
]
