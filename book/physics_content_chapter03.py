"""第三章：力、能量与阻尼。正文、图示与实验均为本教材原创。"""

CHAPTER = (
    3,
    "physics03",
    "单元一：运动会怎样发生",
    "力、能量与阻尼",
    "从重力、线性阻力与弹簧力出发，用功和能量核对运动，再用终端速度与能量账本解释阻尼。",
    (
        "区分重力、阻力、浮力与合力，并用牛顿第二定律计算加速度",
        "用功和能的关系推导动能，用弹簧势能核对一维振子的能量账本",
        "解释线性阻力与阻尼的适用边界，判断何时能使用终端速度公式",
        "复现终端速度算例和相对能量损失，并检查读数始终有限",
        "区分浏览器双精度概念实验与当前 Python float32 实现",
    ),
    ("第二章的速度、加速度与时间积分", "牛顿第二定律", "基本函数图象", "百分比与比值"),
    (
        (
            "gravity",
            "第一种外力",
            "重力：质量越大，重力越大",
            r"""
<p><strong>生活直觉。</strong>同样一颗小球，在地面附近总被拉向地面；质量更大的石块受到的重力也更大。力不是速度，而是改变速度的原因。先求合力，再用牛顿第二定律求加速度。</p>
<p>取竖直向上为正，地面附近的重力加速度为 \(\mathbf g=(0,-9.81)\,\mathrm{m/s^2}\)。质量 \(m\) 的重力向量为</p>
\[
\mathbf F_g=m\mathbf g.
\]
<p>因此重力的大小是 \(mg\)，方向竖直向下。质量为 \(2\,\mathrm{kg}\) 时，\(F_{g,y}=2\times(-9.81)=-19.62\,\mathrm N\)。负号表示方向，不表示重力“比较小”。</p>
<p><strong>合力优先。</strong>若小球还受到阻力与浮力，不能直接把 \(mg\) 除以质量当作当前加速度。应先计算</p>
\[
\mathbf F_{\mathrm{net}}=\mathbf F_g+\mathbf F_d+\mathbf F_b,
\qquad
\mathbf a=\frac{\mathbf F_{\mathrm{net}}}{m}.
\]
<p>忽略其他力时，\(m\) 在 \(m\mathbf g/m\) 中约去，所以不同质量的自由落体小球有相同加速度；加入阻力后，这个结论不再自动成立。</p>
<p><strong>常见误区。</strong>“小球正在下落”只说明速度方向向下，不说明合力一定向下。若向上的阻力暂时大于重力，合力向上，向下运动的小球就会减速。</p>
<p><strong>本节小结：</strong>重力只是外力之一；下一节加入与速度方向相反的阻力。</p>
""",
        ),
        (
            "air-drag",
            "速度相关的力",
            "空气阻力：线性模型与合力平衡",
            r"""
<p><strong>模型近似。</strong>线性阻力是模型近似，不是所有空气阻力都满足 \(F_d\propto v\)。在低速、黏性占主导或经过刻意简化的教学模型中，常把阻力写成</p>
\[
\mathbf F_d=-c\mathbf v,
\]
<p>其中 \(c\) 是阻力系数，单位为 \(\mathrm{N\,s/m}\)。负号保证阻力方向与速度相反；小球向下速度为 \(v>0\) 时，采用向下为正的标量写法，阻力大小就是 \(cv\)，方向向上。真实高速物体的阻力还可能近似与 \(v^2\) 成正比，并受形状、空气密度和湍流影响。</p>
<p><strong>浏览器概念实验：force-balance。</strong>实验固定 \(g=9.81\,\mathrm{m/s^2}\)，从静止释放小球，用固定 \(1/60\,\mathrm s\) 步长绘制重力、阻力、浮力与合力向量。质量范围是 \(0.5\) 到 \(2\,\mathrm{kg}\)，阻力系数范围是 \(0.5\) 到 \(5\,\mathrm{N\,s/m}\)。关闭浮力时 \(F_b=0\)；打开后固定 \(F_b=4\,\mathrm N\)，因此重力与浮力都进入合力。</p>
<p><strong>两种终端模型必须区分。</strong>关闭浮力时，向下的终端速度大小为</p>
\[
v_t=\frac{mg}{c}.
\]
<p>打开浮力且 \(mg>F_b\) 时，终端速度大小为</p>
\[
v_t=\frac{mg-F_b}{c}.
\]
<p>本实验的最小质量为 \(0.5\,\mathrm{kg}\)，此时 \(mg=4.905\,\mathrm N>F_b\)，所以始终处于“合力仍能向下驱动”的有效范围。若 \(mg\le F_b\)，物体可能上浮或悬浮，不能继续把 \(mg/c\) 当作向下终端速度。</p>
<p><strong>观察任务。</strong>先保持 \(m=1,c=2\)，只切换浮力，比较终端速度是否由 \(4.905\) 变为 \(2.905\,\mathrm{m/s}\)。再只增大质量，观察重力箭头和终端速度如何变化。每次改变控件都会暂停并从初始状态重建，避免把不同参数的轨迹混在一起。</p>
<p><strong>本节小结：</strong>阻力随速度变化；当向下驱动力与向上阻力平衡时，加速度为零。</p>
""",
        ),
        (
            "spring-force",
            "恢复力回顾",
            "弹簧力：位移产生恢复，速度产生阻尼",
            r"""
<p><strong>生活直觉。</strong>把弹簧从原长位置拉开，它趋向恢复；松手后振子来回运动。若振子还经过减震器，运动通常会越来越小。这里用一个一维质量—弹簧—阻尼器表达这两种效果。</p>
<p>设 \(x\) 是相对于平衡位置的伸长量，\(v\) 是沿弹簧方向的速度。理想弹性力满足胡克定律，阻尼项与速度成正比：</p>
\[
F=-kx-cv.
\]
<p>\(k\) 是刚度，单位 \(\mathrm{N/m}\)；\(c\) 是阻尼系数。弹簧项总把物体拉向平衡位置，阻尼项总反对当前相对运动。二者合成后，合力不一定始终指向平衡位置，但阻尼不会主动向振子注入能量。</p>
<p><strong>共享核心。</strong>浏览器 <code>RTPhysics.springForce(extension, relativeVelocity, k, c)</code> 直接返回带符号的标量 \( -kx-cv \)。代入 \(k=10\,\mathrm{N/m}\)、\(x=0.2\,\mathrm m\)、\(v=0.5\,\mathrm{m/s}\)、\(c=0.5\,\mathrm{N\,s/m}\)，得到</p>
\[
F=-10\times0.2-0.5\times0.5=-2.25\,\mathrm N.
\]
<p><strong>稳定步长提示。</strong><code>RTPhysics.stableSpringStep(k,m)</code> 返回 \(2/\sqrt{k/m}\)。这个量来自无阻尼简谐运动的离散稳定性提示，只用于教学诊断，不保证任意步长稳定。阻尼、非弹簧力、约束、积分顺序和过大位移都可能改变实际稳定性。</p>
<p><strong>本节小结：</strong>知道了力怎样随状态变化，下一步问力对物体做了多少功。</p>
""",
        ),
        (
            "work-kinetic",
            "力与能量",
            "功和动能：力改变运动的状态账本",
            r"""
<p><strong>生活直觉。</strong>推购物车时，你的力沿运动方向做正功；拉力与运动方向相反时做负功。只有力在位移方向上的分量参与做功。</p>
<p>恒力 \(\mathbf F\) 使物体发生位移 \(\Delta\mathbf x\)，功的标量为</p>
\[
W=\mathbf F\cdot\Delta\mathbf x
=F\Delta x\cos\theta,
\]
<p>其中 \(\theta\) 是力与位移方向的夹角。功的单位是焦耳：\(1\,\mathrm J=1\,\mathrm N\,m\)。</p>
<p><strong>动能。</strong>质量为 \(m\)、速度大小为 \(v\) 的质点具有动能</p>
\[
E=\frac12mv^2.
\]
<p>合外力对质点做的净功等于动能变化：</p>
\[
W_{\mathrm{net}}=\Delta E
=\frac12mv_1^2-\frac12mv_0^2.
\]
<p>速度变为两倍时，动能变为四倍，因为速度在公式中平方。负功可以使动能减小，但不能简单理解为“力越小，物体一定越慢”；还要看力的方向与运动方向。</p>
<p><strong>算例。</strong>质量 \(2\,\mathrm{kg}\) 的小球以 \(3\,\mathrm{m/s}\) 运动，动能是 \(\frac12\times2\times3^2=9\,\mathrm J\)。若沿运动方向的 \(10\,\mathrm N\) 合力作用 \(2\,\mathrm m\)，净功为 \(20\,\mathrm J\)，新动能为 \(29\,\mathrm J\)，新速度满足 \(v_1^2=29\)，即约 \(5.385\,\mathrm{m/s}\)。</p>
<p><strong>本节小结：</strong>动能记录“现在运动得多快”；势能记录“位置储存了多少可释放的能量”。</p>
""",
        ),
        (
            "potential-energy",
            "位置的能量",
            "势能：选择零点，计算储存的能量",
            r"""
<p><strong>生活直觉。</strong>被拉长的弹簧和举高的石块都能在释放后推动其他物体运动。这类与位置有关的能量称为势能。势能数值依赖零点的选择，但同一过程中势能的变化不依赖零点。</p>
<p>对一维理想弹簧，若平衡位置 \(x=0\) 处势能为零，则</p>
\[
U_{\mathrm{spring}}=\frac12kx^2.
\]
<p>因为 \(x^2\) 非负，拉长和压缩同样储存势能。取 \(k=10\,\mathrm{N/m}\)、\(|x|=0.4\,\mathrm m\)，得到 \(U=0.5\times10\times0.4^2=0.8\,\mathrm J\)。</p>
<p>在恒定重力场中，取竖直向上为正、选定某一高度为 \(y=0\)，重力势能可写成</p>
\[
U_g=mgy.
\]
<p>把零点从地面移到桌面，会改变 \(U_g\) 的读数，却不会改变物体下落时 \(U_g\) 的减少量。因此比较能量时必须使用同一个零点。</p>
<p><strong>共享核心。</strong><code>RTPhysics.energy1d(m,v,k,x)</code> 返回 <code>{kinetic, potential}</code>，分别计算 \(\frac12mv^2\) 与 \(\frac12kx^2\)。它是弹簧振子的能量账本，不包含重力势能、浮力势能或热力学内能。</p>
<p><strong>本节小结：</strong>把动能与势能相加，就能检查理想弹簧振子的总能量。</p>
""",
        ),
        (
            "total-energy",
            "能量账本",
            "总能量：动能与势能相加",
            r"""
<p><strong>理想模型。</strong>对没有阻尼、没有外力做功的一维弹簧振子，总机械能为</p>
\[
E=K+U=\frac12mv^2+\frac12kx^2.
\]
<p>弹簧拉长时，势能较大；经过平衡位置时，动能较大。两种能量可以互相转化，总和应保持不变。数值积分会有离散误差，所以应看趋势与容差，而不是要求每一位小数都完全相同。</p>
<p><strong>浏览器概念实验：energy-ledger。</strong>默认 \(m=1\,\mathrm{kg}\)、\(k=10\,\mathrm{N/m}\)、\(c=0.5\,\mathrm{N\,s/m}\)，初始 \(x=1\,\mathrm m\)、\(v=0\)。初始总能量为</p>
\[
E_{\mathrm{initial}}=\frac12\times10\times1^2=5\,\mathrm J,
\]
<p>三段能量条分别显示动能、势能和总能量。振子、公式更新与读数都调用共享核心，不使用随机数或墙钟决定状态。</p>
<p><strong>相对能量损失定义。</strong>为了比较不同初始能量，本章使用有限比值</p>
\[
\frac{E_{\mathrm{initial}}-E_{\mathrm{current}}}{E_{\mathrm{initial}}}.
\]
<p>默认初始能量非零；控件把质量限制为正、刚度限制为至少 \(1\,\mathrm{N/m}\)，初始伸长保持 \(1\,\mathrm m\)，所以 \(E_{\mathrm{initial}}\ge0.5\,\mathrm J\)，分母不会接近零。该比值为正表示总能量下降，为负表示当前总能量高于初值；无阻尼数值振荡时后者可能出现，不应掩盖它。</p>
<p><strong>观察任务。</strong>先把阻尼调到零，短时间观察能量是否只在有限范围内波动；再增大阻尼，比较总能量条变短的速度和相对能量损失。</p>
<p><strong>本节小结：</strong>阻尼出现后，机械能不再守恒；下一节追踪减少的能量去了哪里。</p>
""",
        ),
        (
            "damping",
            "能量耗散",
            "阻尼：机械能转化为其他形式",
            r"""
<p><strong>阻尼不是“删除能量”。</strong>黏性阻尼把宏观机械能转化为内能、声或流体运动。对 \(F_d=-cv\)，连续模型的机械能变化率为</p>
\[
\frac{dE}{dt}=F_dv=-cv^2\le0.
\]
<p>因此阻尼越大、速度越大，机械能减少得越快。物体静止时 \(v=0\)，阻尼功率也为零；阻尼不会让物体继续自行加速。</p>
<p><strong>离散观察。</strong>浏览器使用半隐式欧拉：先用当前 \(x,v\) 求力，再更新速度，最后用新速度更新位置。固定步长下它是近似，不是连续方程的无误差实现。无阻尼时总能量可能出现小幅数值波动；有阻尼时通常总体下降，但不应把“每一帧必须严格单调”当成通用测试条件。</p>
<p><strong>两种阻尼不要混淆。</strong>本章线性阻力使用与速度成正比的 \(cv\)，弹簧阻尼项也使用相对速度。当前 Python <code>physics/common/simulation.py</code> 的公共阻尼却是速度乘性衰减 <code>exp(-damping*h)</code>；只有在小步长下，它才可近似看作与速度成正比的连续阻尼。布料 <code>MassSpringCloth</code> 另有沿质点相对速度的弹簧阻尼，作用对象也与本章单振子不同。</p>
<p><strong>当前边界。</strong>本章实验不模拟空气湍流、浮力流体求解或真实热传导，只计算机械能账本与相对损失。</p>
<p><strong>本节小结：</strong>当阻尼与恒定驱动力平衡时，速度会趋向一个终端值。</p>
""",
        ),
        (
            "terminal-speed",
            "稳定速度",
            "终端速度：合力为零，不是速度为零",
            r"""
<p><strong>终端速度的含义。</strong>小球下落时，向下驱动不变，向上阻力随速度增大。当阻力恰好平衡向下驱动，合力为零，加速度为零；此时速度达到终端值，但通常不为零。</p>
<p>采用向下为正的标量，忽略浮力时</p>
\[
F_{\mathrm{net}}=mg-cv.
\]
<p>令 \(F_{\mathrm{net}}=0\)，得到</p>
\[
v_t=\frac{mg}{c}.
\]
<p>打开浮力后，重力仍向下，浮力向上，二者都必须保留：</p>
\[
F_{\mathrm{net}}=mg-F_b-cv,
\qquad
v_t=\frac{mg-F_b}{c}\quad(mg>F_b).
\]
<p><strong>不会瞬间到达。</strong>从静止开始，初始阻力为零，加速度仍向下。随着速度增大，阻力增大，合力与加速度减小，速度渐近终端值。质量越大，\(mg\) 通常越大；阻力系数越大，终端速度通常越小，但瞬态过程还取决于 \(m/c\)。</p>
<p><strong>模型检查。</strong>若 \(mg<F_b\)，初始合力向上，物体可能向上加速；若 \(mg=F_b\)，重力与浮力平衡。在这两种情况下，都不能把 \(mg/c\) 继续标为向下终端速度。</p>
<p><strong>本节小结：</strong>下一节用默认参数独立算一次，核对实验读数。</p>
""",
        ),
        (
            "worked-example",
            "完整算例",
            "算例：m=1、g=9.81、c=2",
            r"""
<p><strong>题目。</strong>质量 \(m=1\,\mathrm{kg}\) 的小球从静止下落，重力加速度大小 \(g=9.81\,\mathrm{m/s^2}\)，线性阻力系数 \(c=2\,\mathrm{N\,s/m}\)，先关闭浮力。求终端速度。</p>
<p><strong>第一步，列合力。</strong>取向下为正：</p>
\[
F_{\mathrm{net}}=mg-cv=1\times9.81-2v.
\]
<p><strong>第二步，令终端状态合力为零。</strong></p>
\[
9.81-2v_t=0,
\qquad
v_t=\frac{9.81}{2}=4.905\,\mathrm{m/s}.
\]
<p>所以默认实验的终端速度必须是 \(4.905\,\mathrm{m/s}\)，而不是 \(9.81\,\mathrm{m/s}\)，也不是零。</p>
<p><strong>第三步，核对首帧。</strong>共享传输步长为 \(h=1/60\,\mathrm s\)。实验在旧状态求力，初始 \(v_0=0\)，故 \(F_{\mathrm{net},0}=9.81\,\mathrm N\)：</p>
\[
v_1=v_0+\frac{F_{\mathrm{net},0}}{m}h
=\frac{9.81}{60}
\approx0.1635\,\mathrm{m/s}
\]
<p>这是用旧状态合力做的显式力更新。更新后显示的阻力改为 \(2\times9.81/60=0.327\,\mathrm N\)，对应新速度；不要把显示用的新阻力又代回本步已经完成的速度更新。</p>
<p><strong>第四步，打开浮力。</strong>本实验固定 \(F_b=4\,\mathrm N\)，且 \(mg>F_b\)，于是</p>
\[
v_t=\frac{mg-F_b}{c}
=\frac{9.81-4}{2}
=2.905\,\mathrm{m/s}.
\]
<p><strong>单位检查。</strong>\(mg\) 的单位是 \(\mathrm N\)，\(c\) 的单位是 \(\mathrm{N\,s/m}\)，相除得到 \(\mathrm{m/s}\)。</p>
<p><strong>本节小结：</strong>终端速度来自合力平衡；下一步把公式写成可检查的伪代码。</p>
""",
        ),
        (
            "energy-diagnostics",
            "落地检查",
            "伪代码、能量诊断与调试清单",
            r"""
<p><strong>受力与下落伪代码。</strong>向下为正，阻力在本步开始时用旧速度计算：</p>
<pre><code>gravity = m * g
buoyancy = enabled ? F_b : 0
drag = c * v
net = gravity - buoyancy - drag
a = net / m
v = v + a * h
distance = distance + v * h
terminal = (gravity - buoyancy) / c</code></pre>
<p>只有 \(mg>F_b\) 时，最后一行才表示向下终端速度。无浮力时它自然退化为 \(mg/c\)。</p>
<p><strong>弹簧与能量伪代码。</strong></p>
<pre><code>force = springForce(x, v, k, c)
v = v + force / m * h
x = x + v * h
energy = energy1d(m, v, k, x)
total = energy.kinetic + energy.potential
relative_loss = (initial_energy - total) / initial_energy
step_hint = stableSpringStep(k, m)</code></pre>
<p><code>step_hint</code> 只显示为诊断，不用来覆盖共享传输的 \(1/60\,\mathrm s\)，也不宣称满足它就一定稳定。初始能量必须非零；本实验用 \(x_0=1\)、\(k\ge1\) 保证这一点。</p>
<table><thead><tr><th>症状</th><th>优先检查</th><th>最小动作</th></tr></thead><tbody>
<tr><td>速度无限增大</td><td>遗漏阻力或阻力符号错误</td><td>打印 \(mg\)、\(cv\) 与合力</td></tr>
<tr><td>打开浮力后终端速度未变</td><td>仍误用 \(mg/c\)</td><td>核对 \(F_b\) 是否同时进入合力与终端公式</td></tr>
<tr><td>无阻尼时能量明显漂移</td><td>步长过大、积分顺序或状态更新错误</td><td>减小时间步并核对首帧 \(x,v\)</td></tr>
<tr><td>相对能量损失出现 NaN</td><td>初始能量为零或状态非有限</td><td>检查 \(E_{\mathrm{initial}}&gt;0\)，暂停并重置</td></tr>
<tr><td>把稳定步长提示当保证</td><td>忽略阻尼、约束与其他力</td><td>逐步增大步长做有界性观察，不跨过提示就宣称稳定</td></tr>
</tbody></table>
<p><strong>浏览器概念实验。</strong><code>force-balance</code> 与 <code>energy-ledger</code> 使用原生 Canvas 和 JavaScript 双精度浮点，共享固定 \(1/60\,\mathrm s\) 传输；它们展示一维力学模型，不包含 RenderTorch 的三维形状、碰撞或完整布料求解。两个实验都在 \(12\,\mathrm s\) 后暂停，读数保持有限。</p>
<p><strong>当前 Python 实现。</strong><code>physics/common/simulation.py</code> 使用固定步长、半隐式位置更新和全局乘性阻尼；<code>physics/common/state.py</code> 提供质点动能诊断；<code>physics/cloth/mass_spring.py</code> 对布料质点施加弹簧力和沿相对速度的阻尼。相关调用方的状态与力使用 <code>float32</code> 张量。浏览器双精度概念实验与 Python <code>float32</code> 实现属于不同层次，只比较趋势、量级和有限性，不要求逐位相同；当前 Python 公共路径也没有本章的通用线性空气阻力与浮力开关。</p>
<p><strong>本章小结：</strong>力决定加速度，功连接力与动能，势能记录位置，阻尼改变机械能；用合力、能量和有限读数三类证据，才能同时检查物理模型与数值实现。</p>
""",
        ),
    ),
    (
        (
            "force-balance",
            "重力、阻力与浮力的合力",
            "demos/physics05.js",
            "默认暂停；调节质量、阻力与浮力开关，观察合力向量、加速度、当前速度和终端速度。",
            "air-drag",
        ),
        (
            "energy-ledger",
            "阻尼弹簧振子能量账本",
            "demos/physics06.js",
            "默认暂停；调节质量、刚度与阻尼，比较动能、势能、总能量和相对能量损失。",
            "total-energy",
        ),
    ),
    (
        (
            "physics/common/simulation.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/simulation.py",
            "当前 Python 实现：固定步长、半隐式位置更新、全局乘性阻尼与动能诊断；调用方状态使用 float32。",
        ),
        (
            "physics/common/state.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/state.py",
            "当前 Python 实现：质点位置、速度、逆质量与动能计算，张量精度由调用方决定。",
        ),
        (
            "physics/cloth/mass_spring.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/cloth/mass_spring.py",
            "当前 Python 实现：布料质点间弹簧力与沿相对速度的阻尼，使用 float32 张量；不是本章单振子的逐位复刻。",
        ),
        (
            "book/assets/physics-core.js",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/book/assets/physics-core.js",
            "浏览器共享数值核心：springForce、energy1d 与 stableSpringStep；本章概念实验使用 JavaScript 双精度。",
        ),
    ),
)
