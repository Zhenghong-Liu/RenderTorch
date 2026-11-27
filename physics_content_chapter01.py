"""第一章：从真实运动到数字状态。正文、图示与实验均为本教材原创。"""

CHAPTER = (
    1,
    "physics01",
    "单元一：运动会怎样发生",
    "从真实运动到数字状态",
    "把一颗小球的位置、速度与时间步写成有限个数字，理解离散帧、向量初值和可复现实验。",
    (
        "用位置、速度、加速度说明质点的数字状态，并区分状态量与绘图量",
        "完成二维向量的分量运算、单位检查和一步位置更新",
        "解释离散帧与线性插值的关系，诊断初值、时间步和坐标方向错误",
        "重置并复现两个浏览器概念实验，说明它们与当前 Python 实现的边界",
    ),
    ("基础 Python 算术", "数轴与正负数", "直角三角形的边与角", "简单函数图象"),
    (
        (
            "digitize-ball",
            "从现象到数字",
            "屏幕上的小球也是数字",
            r"""
<p><strong>生活直觉。</strong>视频里的小球看似连续移动，其实每个显示时刻只收到一幅新图像。若每秒显示六十幅，相邻两幅之间仍可能藏着许多位置；电脑并没有储存小球经过的每一个点。</p>
<p>我们要做的第一件事，不是立刻加入重力，而是给“这一时刻的小球在哪里”一个可写入、可计算、可检查的表示。设二维位置为 \(\mathbf p=(x,y)\)，速度向量为 \(\mathbf v=(v_x,v_y)\)，时间步长为 \(dt\)。这四个位置/速度分量与一个时间步，已经足以描述本章的匀速直线运动。</p>
<ul><li>先把真实小球<strong>抽象</strong>为坐标平面上的一个点；</li><li>再约定坐标轴、正方向和单位；</li><li>最后写出一条能从旧位置算出新位置的规则。</li></ul>
<p>所以“数字化”不是把小球画得更逼真，而是把观察结果变成有限个<strong>数</strong>。数后面还必须带单位，否则 \(3\) 可能表示三米、三秒或三个像素。</p>
<p><strong>本节小结：</strong>下一节先统一单位与坐标，避免不同含义的数混在一张状态表里。</p>
""",
        ),
        (
            "units-coordinates",
            "统一语言",
            "单位、二维坐标与正方向",
            r"""
<p><strong>符号与定义。</strong>本章物理量使用国际单位制：位置用米，符号 \(\mathrm m\)；速度用米每秒，符号 \(\mathrm{m/s}\)；时间步用秒，符号 \(\mathrm s\)。Canvas 绘制使用像素，像素只是显示单位，不能直接代替米。</p>
<p>坐标原点可任选，但选定后不能中途改变。本章约定 \(x\) 向右、\(y\) 向上。因此 \(\mathbf v=(2,-1)\,\mathrm{m/s}\) 表示向右每秒两米、向下每秒一米；它的速度大小不是两个分量的简单相加，而是</p>
\[
\|\mathbf v\|=\sqrt{v_x^2+v_y^2}.
\]
<p><strong>手算。</strong>代入 \(v_x=2\)、\(v_y=-1\)，得到 \(\|\mathbf v\|=\sqrt{2^2+(-1)^2}=\sqrt5\,\mathrm{m/s}\approx2.236\,\mathrm{m/s}\)。负号只说明方向，不说明“负速度大小”。</p>
<p><strong>坐标转换。</strong>若绘图区左上角为 \((L,T)\)，宽为 \(W\)、高为 \(H\)，世界范围为 \(x\in[-1,6]\)、\(y\in[-1,5]\)，则</p>
\[
x_s=L+\frac{x+1}{7}W,\qquad y_s=T+\frac{5-y}{6}H.
\]
<p>Canvas 的 \(y_s\) 向下增加，所以公式中有 \(5-y\)。把米直接当像素，会让“移动两米”在不同窗口宽度下看起来速度不同。</p>
<p><strong>常见误区。</strong>症状是设 \(v_y>0\) 小球却向下走。根因常常是忘记屏幕纵轴反向。最小调试动作是只改 \(v_y\)，同时对照坐标轴箭头，而不是同时改两个分量。</p>
<p><strong>本节小结：</strong>统一单位与方向后，才能讨论哪些真实物体可以简化为一个点。</p>
""",
        ),
        (
            "particle-model",
            "建立模型",
            "质点假设：什么时候可以只看一个点",
            r"""
<p><strong>生活直觉。</strong>研究地球公转时，地球半径远比轨道尺度小；研究一颗小球在桌面上平移时，若暂不关心它怎样滚动，也可以先只看球心的位置与速度。这就是质点模型。</p>
<p>质点是一个<strong>理想化对象</strong>：它有位置、速度等状态，却没有需要单独记录的大小、朝向和角速度。用球心代表小球，并不等于真实小球真的没有一个点那么大，而是说明当前问题忽略这些差异。</p>
<ul><li><strong>适用：</strong>物体尺度远小于运动尺度，或只研究平移。</li><li><strong>不适用：</strong>要解释篮球旋转、齿轮啮合、长方体倾倒，或碰撞点影响力矩。</li><li><strong>本章边界：</strong>两个实验都是二维质点概念实验，不是三维刚体模拟。</li></ul>
<p>模型是否合理，要看它能否回答当前问题，而不是看它是否“完全真实”。如果忽略大小后预测与观察明显不同，就应增加形状、旋转或碰撞几何，而不是随意调大速度掩盖问题。</p>
<p><strong>可操作观察。</strong>在 <code>particle-lab</code> 中，圆点只是质点的可见标记；拖拽改变的是球心坐标，圆点半径不进入位置更新公式。</p>
<p><strong>本节小结：</strong>确定研究范围后，我们把最关心的量组织成状态。</p>
""",
        ),
        (
            "state-pva",
            "组织信息",
            "状态是一组会变化的数：p、v、a",
            r"""
<p><strong>定义。</strong>状态是“为了预测下一刻，现在必须知道的信息”。对本章的二维匀加速质点，位置 \(\mathbf p\)、速度 \(\mathbf v\)、加速度 \(\mathbf a\) 都是状态量；帧号与时间步是帮助定位和更新的元数据。</p>
<table><thead><tr><th>符号</th><th>意义</th><th>单位</th></tr></thead><tbody>
<tr><td>\(\mathbf p\)</td><td>质点在参考系中的位置</td><td>\(\mathrm m\)</td></tr>
<tr><td>\(\mathbf v\)</td><td>位置随时间变化的快慢与方向</td><td>\(\mathrm{m/s}\)</td></tr>
<tr><td>\(\mathbf a\)</td><td>速度随时间变化的快慢与方向</td><td>\(\mathrm{m/s^2}\)</td></tr>
<tr><td>\(dt\)</td><td>相邻两个离散状态之间的时间间隔</td><td>\(\mathrm s\)</td></tr>
</tbody></table>
<p>只给一张小球图片，通常无法知道下一刻往哪里走；只给速度，不知道起点也算不出位置。反之，若没有摩擦力、没有外力且初值与时间步确定，那么当前状态就足以决定后续状态。</p>
<p>本章实验令 \(\mathbf a=(0,0)\)，集中检查位置与速度。加速度的含义是“速度每秒改变多少”：</p>
\[
\mathbf a=\frac{\Delta\mathbf v}{\Delta t}.
\]
<p>若 \(dt=0.5\,\mathrm s\)，速度在一步内改变 \((1,-2)\,\mathrm{m/s}\)，则该步平均加速度为 \((2,-4)\,\mathrm{m/s^2}\)。这与“位置每秒改变多少”是两种不同的变化率。</p>
<p><strong>本节小结：</strong>先把位置单独写成向量，再讨论速度怎样由位置变化得到。</p>
""",
        ),
        (
            "position-vector",
            "读懂坐标",
            "位置向量：从原点指向质点",
            r"""
<p><strong>几何意义。</strong>位置向量是从坐标原点指向质点的一条有向线段。二维位置 \(\mathbf p=(x,y)\) 既能写成两个分量，也能写成两个基向量的和：</p>
\[
\mathbf p=x(1,0)+y(0,1).
\]
<p>向量加法按分量进行。若旧位置为 \(\mathbf p_0=(1,2)\,\mathrm m\)，位移为 \(\Delta\mathbf p=(1,-0.5)\,\mathrm m\)，则</p>
\[
\mathbf p_1=\mathbf p_0+\Delta\mathbf p=(1+1,\,2-0.5)=(2,1.5)\,\mathrm m.
\]
<p>这与先更新 \(x\) 再更新 \(y\) 完全一致。向量记号的好处，是让“同时处理两个方向”变得清楚，并不要求你先学更高深的数学。</p>
<p><strong>伪代码。</strong></p>
<pre><code>p.x = p.x + dx
p.y = p.y + dy</code></pre>
<p><strong>自测。</strong>若 \(dx=0\)、\(dy=1\)，位置应只向上变一米；若只把屏幕纵坐标当物理纵坐标，则会得到相反的可视结果。</p>
<p><strong>本节小结：</strong>位移是位置之差；下一步把“每秒位移”写成速度。</p>
""",
        ),
        (
            "velocity-vector",
            "核心概念",
            "速度是位置的变化率，也是向量",
            r"""
<p><strong>逐步推导。</strong>观察一小段时间 \(\Delta t\)，位置从 \(\mathbf p_0\) 变为 \(\mathbf p_1\)。这段时间里的平均速度为</p>
\[
\mathbf v_{\mathrm{avg}}=\frac{\Delta\mathbf p}{\Delta t}
=\frac{\mathbf p_1-\mathbf p_0}{\Delta t}.
\]
<p>分子是向量，分母是正的时间，所以速度仍有方向。把 \(\Delta t\) 取得足够小，就得到瞬时速度的直觉；本章不要求用微积分，先用有限时间步计算平均速度。</p>
<p>反过来，若一个时间步内速度近似不变，则</p>
\[
\Delta\mathbf p=\mathbf v\,dt,\qquad \mathbf p_{n+1}=\mathbf p_n+\mathbf v\,dt.
\]
<p><strong>手算。</strong>从 \(\mathbf p_0=(1,2)\,\mathrm m\) 出发，取 \(\mathbf v=(2,-1)\,\mathrm{m/s}\)、\(dt=0.5\,\mathrm s\)：</p>
\[
\mathbf v\,dt=(2\times0.5,\,-1\times0.5)=(1,-0.5)\,\mathrm m,
\]
\[
\mathbf p_1=(1,2)+(1,-0.5)=(2,1.5)\,\mathrm m.
\]
<p><strong>浏览器概念实验：particle-lab。</strong>蓝球表示当前质点，黄色箭头表示速度向量，虚线圆表示按当前 \(dt\) 预测的下一帧位置。暂停后拖篮球可改变位置；速度 \(x\)、速度 \(y\)、时间步三个范围控件会立即更新预测。动画每次推进一个选定的 \(dt\)，播放速度不是真实秒表。</p>
<p><strong>观察任务。</strong>先把 \(v_y=0\)，再看 \(v_x\) 的正负；然后把 \(v_x=0\)，检查 \(v_y\)。最后只增大 \(dt\)，确认预测位移按比例增大，而速度大小不变。</p>
<p><strong>常见误区。</strong>把速度大小乘时间后仍当作向量，会丢失方向。必须先分别乘两个分量，再相加。</p>
<p><strong>本节小结：</strong>上式把连续观察变成了可以逐帧执行的离散规则。</p>
""",
        ),
        (
            "discrete-frames",
            "理解时间",
            "离散帧：五帧状态与线性插值",
            r"""
<p><strong>生活直觉。</strong>手翻动画时，你看到的是按固定时间间隔出现的一幅幅画面。设第 \(n\) 帧时刻为 \(t_n=n/f\)，其中 \(f\) 是采样帧率，单位为帧每秒。相邻帧间隔为</p>
\[
\Delta t=\frac{1}{f}.
\]
<p>只保存帧位置时，两帧之间最自然的猜测是线性插值。若当前时刻 \(t\) 位于 \(t_n\) 与 \(t_{n+1}\) 之间，令</p>
\[
\alpha=\frac{t-t_n}{t_{n+1}-t_n},\qquad 0\leq\alpha\leq1,
\]
\[
\mathbf p_{\mathrm{lerp}}(t)=\mathbf p_n+\alpha(\mathbf p_{n+1}-\mathbf p_n).
\]
<p><strong>浏览器概念实验：frame-states。</strong>表格显示围绕当前帧的五帧位置，Canvas 同时画出实际保存的帧位置与线性插值位置。帧率决定“多久保存一个新状态”，时间缩放决定模拟时间相对共享更新时钟走得多快；暂停与单步使用公共传输控件，不重复添加。</p>
<p>本章把插值误差定义为两个显示位置之间的距离：</p>
\[
e_{\mathrm{lerp}}=\|\mathbf p_{\mathrm{lerp}}-\mathbf p_n\|.
\]
<p>对匀速直线运动，连续真实位置本来就满足线性关系，因此这个误差来自“当前显示帧位置”与“两帧之间插值位置”的差别，<strong>不是</strong>积分误差。若运动弯曲，线性插值还可能与真实轨迹有偏差。</p>
<p><strong>观察任务。</strong>把帧率调到 \(60\)、时间缩放调到 \(2\)，单步后模拟时间增加 \(1/30\,\mathrm s\)，帧号增加 \(2\)，恰好落在帧边界时误差为零。再改为帧率 \(10\)、缩放 \(0.5\)，同一个单步只推进 \(1/120\,\mathrm s\)，仍在第 \(0\) 帧内，插值误差变为正。</p>
<p><strong>本节小结：</strong>离散状态需要明确初值，也明确“哪一次重置”。</p>
""",
        ),
        (
            "drag-initial-conditions",
            "动手设置",
            "拖拽与初始条件：每一次实验从哪里开始",
            r"""
<p><strong>初始条件。</strong>初始位置 \(\mathbf p_0\) 与初始速度 \(\mathbf v_0\) 决定本章匀速运动的起点和走向。只改其中一个，后面的轨迹就会改变。拖拽是一种直观的初值输入方式，但屏幕坐标必须先转换成物理坐标。</p>
<p>在 <code>particle-lab</code> 中，pointer 事件的坐标先由公共函数转换为 Canvas 的 CSS 像素坐标，再使用逆变换得到世界坐标。拖拽范围限制在绘图区内，避免得到看不见的初值。</p>
<pre><code>on_pointer_down:
    if pointer 靠近小球:
        pause()
        dragging = true

on_pointer_move while dragging:
    p = screen_to_world(pointer)
    redraw_and_report()

on_pointer_up:
    dragging = false</code></pre>
<p><strong>为什么暂停？</strong>若拖拽时小球仍在推进，手移动的是“旧状态”，下一帧又被积分更新，实验很难复现。暂停后设置初值，再运行，才能把“我做了什么”与“模拟做了什么”分开。</p>
<p><strong>可复现记录。</strong>一次完整实验至少记录 \(\mathbf p_0\)、\(\mathbf v_0\)、\(\mathbf a\)、\(dt\)、帧率和时间缩放。口头说“把速度调大一点”不足以复现。</p>
<p><strong>本节小结：</strong>有了明确初值，还需要固定的更新顺序和重置规则。</p>
""",
        ),
        (
            "reproducible-experiments",
            "实验方法",
            "可复现：相同输入应得到相同状态",
            r"""
<p><strong>实验步骤。</strong>先点击重置，再记录参数；每次只改变一个量；运行相同步数；比较末状态。若结果不同，先检查是否有未记录的初值、随机数、实时墙钟时间或拖拽残留。</p>
<p><strong>确定性。</strong>本章更新中没有随机数。给定 \(\mathbf p_0\)、\(\mathbf v\)、\(\mathbf a=0\) 与 \(dt\)，执行 \(N\) 步后：</p>
\[
\mathbf p_N=\mathbf p_0+N\mathbf v\,dt.
\]
<p><strong>自测。</strong>取 \(\mathbf p_0=(1,2)\)、\(\mathbf v=(2,-1)\)、\(dt=0.5\)，执行两步应得到 \((3,1)\)。这个期望来自手算，不应再由待测公式反推。</p>
<p><strong>当前 Python 实现。</strong><code>physics/common/simulation.py</code> 的默认外层步长为 \(fixed\_dt=1/120\,\mathrm s\)，每个外层步含 \(substeps=2\) 个积分子步，因此实际积分子步为 \(h=(1/120)/2=1/240\,\mathrm s\)。它使用半隐式 Euler，并在约束求解后按投影前后位移重建速度；本章两个浏览器实验没有实现这套完整流程。</p>
<p>Python 状态默认采用 <code>float32</code>，浏览器 JavaScript 数值采用双精度。因此复现时比较趋势与合理容限，不要求两者逐位相同。当前 Python 类也没有本章 Canvas 的二维拖拽面板。</p>
<p><strong>本节小结：</strong>把更新规则、算例与伪代码放在一起检查。</p>
""",
        ),
        (
            "worked-example-pseudocode",
            "完整算例",
            "数值算例与匀速更新伪代码",
            r"""
<p><strong>题目。</strong>一个小球从 \((1,2)\,\mathrm m\) 出发，速度为 \((2,-1)\,\mathrm{m/s}\)，时间步为 \(0.5\,\mathrm s\)，加速度为零。求下一帧位置与速度大小。</p>
<p><strong>第一步，乘时间步。</strong></p>
\[
\mathbf v\,dt=(2,-1)\times0.5=(1,-0.5)\,\mathrm m.
\]
<p><strong>第二步，加旧位置。</strong></p>
\[
\mathbf p_1=\mathbf p_0+\mathbf v\,dt=(1,2)+(1,-0.5)=(2,1.5)\,\mathrm m.
\]
<p><strong>第三步，检查单位与方向。</strong>速度单位是 \(\mathrm{m/s}\)，乘秒后得到米；\(x\) 增加一米，\(y\) 减少半米，符合速度分量方向。速度本身不变，仍为 \((2,-1)\,\mathrm{m/s}\)。</p>
<p>算例原始数据：<code>p_0=(1,2), v=(2,-1), dt=0.5 → p=(2,1.5)</code>。位置用米，速度用米每秒，时间步用秒。</p>
<p><strong>与实验一致的伪代码。</strong></p>
<pre><code>state = {p: (1, 2), v: (2, -1), a: (0, 0)}
dt = 0.5

function update(state, dt):
    v = state.v
    p_next = state.p + v * dt   # 两个分量分别相乘、相加
    return {p: p_next, v: v, a: state.a}</code></pre>
<p><strong>逐步核对。</strong>若写成 <code>p + v + dt</code>，单位不相容；若只更新 \(x\)，会丢失 \(y=-0.5\,\mathrm m\) 的位移；若把旧位置覆盖后才读取速度，虽然本例速度不变，但会成为后续加速度更新的隐患。</p>
<p><strong>本节小结：</strong>会用算例核对规则后，再给未来章节建立可重复的自测。</p>
""",
        ),
        (
            "self-tests",
            "自己验证",
            "自测：用手算期望发现实现错误",
            r"""
<p><strong>为什么先写期望？</strong>如果测试直接复制待测公式，公式写错时测试也可能一起错。应像改作业一样，先独立算出答案，再让程序执行。</p>
<table><thead><tr><th>测试</th><th>输入</th><th>手算期望</th></tr></thead><tbody>
<tr><td>零速度</td><td>\(\mathbf p=(1,2)\)、\(\mathbf v=(0,0)\)、\(dt=0.5\)</td><td>位置仍为 \((1,2)\)</td></tr>
<tr><td>正方向</td><td>\(\mathbf p=(0,0)\)、\(\mathbf v=(3,0)\)、\(dt=2\)</td><td>位置变为 \((6,0)\)</td></tr>
<tr><td>负方向</td><td>\(\mathbf p=(0,0)\)、\(\mathbf v=(0,-2)\)、\(dt=0.5\)</td><td>位置变为 \((0,-1)\)</td></tr>
<tr><td>两步推进</td><td>\(\mathbf p=(1,2)\)、\(\mathbf v=(2,-1)\)、\(dt=0.5\)</td><td>第二步位置为 \((3,1)\)</td></tr>
<tr><td>帧边界</td><td>\(f=60\)、缩放 \(2\)、共享单步 \(1/60\,\mathrm s\)</td><td>模拟时间 \(1/30\,\mathrm s\)，帧号 \(2\)</td></tr>
</tbody></table>
<p><strong>重置测试。</strong>先改控件、拖拽并推进，再点击重置。位置应恢复 \((1,2)\)，速度恢复 \((2,-1)\)，时间步恢复 \(0.5\,\mathrm s\)，帧号和时间恢复零。不能只看小球“差不多回到原处”。</p>
<p><strong>有限值测试。</strong>检查状态与文字读数中不出现 <code>NaN</code> 或 <code>Infinity</code>。公共组件发现非有限状态会暂停并提示重置；这只阻止错误继续扩散，不代替定位错误来源。</p>
<p><strong>本节小结：</strong>最后用“症状—根因—动作”组织调试清单。</p>
""",
        ),
        (
            "debugging",
            "定位问题",
            "调试清单与实现边界",
            r"""
<table><thead><tr><th>症状</th><th>优先检查的根因</th><th>最小调试动作</th></tr></thead><tbody>
<tr><td>小球方向反了</td><td>世界 \(y\) 与屏幕 \(y\) 反向</td><td>只设一个速度分量，对照轴箭头</td></tr>
<tr><td>预测点不动</td><td>速度为零，或拖拽时未暂停</td><td>暂停、重置，再只改 \(v_x\)</td></tr>
<tr><td>位移随窗口变化</td><td>把像素当成米</td><td>统一使用秒与米，最后才转屏幕坐标</td></tr>
<tr><td>结果无法复现</td><td>初值、步数或时间缩放未记录</td><td>重置后逐项记录，再单步核对</td></tr>
<tr><td>插值误差突然很大</td><td>跨越帧边界或帧率设置过稀</td><td>查看帧号、模拟时间与 \(\alpha\)</td></tr>
<tr><td>读数出现非有限值提示</td><td>除零或无效输入进入计算</td><td>停止运行，重置后逐个恢复参数</td></tr>
</tbody></table>
<p><strong>分层结论。</strong><code>particle-lab</code> 与 <code>frame-states</code> 是浏览器二维概念实验，验证向量更新、拖拽、离散采样与插值；它们没有碰撞、约束、旋转或真实三维渲染。共享 <code>physics-core.js</code> 的 <code>vec2</code> 提供本章使用的纯向量运算。</p>
<p><strong>当前 Python 实现。</strong><code>physics/common/state.py</code> 保存位置、速度与逆质量；<code>physics/common/simulation.py</code> 提供固定步长推进与重置。它们不是这两个 Canvas 面板的 Python 版本，也不意味着当前应用已经提供相同交互。</p>
<p><strong>后续路线。</strong>第二章将加入加速度与多种时间积分；之后才讨论力、约束和碰撞。遇到轨迹偏差时，先分清模型假设、离散误差与实现错误，不要直接把所有差异都归因于“帧率不够”。</p>
<p><strong>本章小结：</strong>状态是一组有单位的数；速度连接位置与时间；离散帧需要明确初值和更新规则；可复现实验让每次错误都能被定位。</p>
""",
        ),
    ),
    (
        (
            "particle-lab",
            "质点状态实验室",
            "demos/physics01.js",
            "暂停后拖拽小球，改变速度分量与时间步，核对下一帧位置和速度大小。",
            "velocity-vector",
        ),
        (
            "frame-states",
            "五帧状态与插值",
            "demos/physics02.js",
            "调节采样帧率与时间缩放，比较五帧表、离散位置和线性插值位置。",
            "discrete-frames",
        ),
    ),
    (
        (
            "physics/common/state.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/state.py",
            "当前 Python 实现：ParticleState 保存并克隆位置、速度与逆质量。",
        ),
        (
            "physics/common/simulation.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/simulation.py",
            "当前 Python 实现：固定步长、子步、状态重置与诊断指标。",
        ),
        (
            "physics/rigidbody/sphere.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/rigidbody/sphere.py",
            "当前 Python 实现：平移球体速度与位置更新；不是本章拖拽实验。",
        ),
        (
            "book/assets/physics-core.js",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/book/assets/physics-core.js",
            "浏览器共享数值核心：本章使用 vec2 的纯向量运算。",
        ),
    ),
)
