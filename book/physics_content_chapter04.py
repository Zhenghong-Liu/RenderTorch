"""第四章：质量弹簧系统。正文、图示与实验均为本教材原创。"""

CHAPTER = (
    4,
    "physics04",
    "单元二：物体为什么保持形状",
    "质量弹簧系统",
    "从静止长度和胡克定律出发，用单位方向与等大反向力连接质点，再研究单振子、显式积分稳定性和结构弹簧布片。",
    (
        "区分静止长度、当前长度与伸长量，并计算一维弹簧力",
        "把弹簧力分解为单位方向上的标量，解释等大反向力与逆质量加权",
        "写出单振子方程，区分重力静平衡位移与相对静平衡的振动",
        "使用稳定步长诊断和有限值检查判断显式积分是否可信",
        "说明结构、剪切、弯曲弹簧以及固定点、碰撞和当前 Python 实现的边界",
    ),
    ("第三章的弹簧力、动能与势能", "牛顿第二定律", "向量与单位方向", "比值和平方根"),
    (
        (
            "rest-length",
            "弹簧的起点",
            "静止长度：先知道没有形变时有多长",
            r"""
<p><strong>先分清三个量。</strong>弹簧没有被拉长也没有被压缩时的长度叫<strong>静止长度</strong>，记作 \(L_0\)。某一时刻实际测得的长度叫当前长度 \(L\)。沿弹簧方向从静止长度算出的变化叫伸长量：</p>
\[
x=L-L_0.
\]
<p>\(x>0\) 表示拉长，\(x&lt;0\) 表示压缩，\(x=0\) 表示处于静止长度。伸长量是长度差，单位是米，不能把当前长度本身直接叫作伸长量。</p>
<p><strong>算一笔小账。</strong>一根弹簧的 \(L_0=0.8\,\mathrm m\)，被拉到 \(L=1.1\,\mathrm m\)，则 \(x=0.3\,\mathrm m\)；若被压到 \(0.6\,\mathrm m\)，则 \(x=-0.2\,\mathrm m\)。拉长和压缩都可能储存弹性势能。</p>
<p><strong>模型边界。</strong>真实弹簧有粗细、质量和螺旋结构，过度压缩还可能并圈。本章用无质量、理想线性的弹簧解释基本原理，不宣称任意伸长范围内都满足同一公式。</p>
<p><strong>本节小结：</strong>先测 \(L_0\)，再算 \(x=L-L_0\)，最后才讨论恢复力。</p>
""",
        ),
        (
            "hookes-law",
            "恢复力的大小",
            "胡克定律：形变越大，恢复力越强",
            r"""
<p><strong>生活直觉。</strong>轻拉弹簧时它产生的拉力较小，继续拉长通常需要更大的力。在线弹性范围内，恢复力的方向指向静止长度，大小与伸长量成正比，这就是胡克定律。</p>
<p>在一维问题中，取伸长为正，写成</p>
\[
F=-kx.
\]
<p>\(k\) 是刚度，单位是 \(\mathrm{N/m}\)。负号表示恢复力总与伸长量符号相反：拉长时 \(x>0\)，力指向缩短方向；压缩时 \(x&lt;0\)，力指向伸长方向。</p>
<p><strong>共享核心。</strong><code>RTPhysics.springForce(x,v,k,c)</code> 返回带符号标量 \(F=-kx-cv\)。先只看弹性项，若 \(k=20\,\mathrm{N/m}\)、\(x=0.5\,\mathrm m\)，则 \(F=-20\times0.5=-10\,\mathrm N\)。这里的 \(-10\,\mathrm N\) 不是“力的大小是负十牛顿”，而是说方向与正方向相反。</p>
<p><strong>刚度不是质量。</strong>质量回答“物体有多难被加速”，刚度回答“弹簧形变后恢复得多强”。同一根弹簧上，更大的 \(k\) 通常意味着更硬，而不是更重。</p>
<p><strong>本节小结：</strong>一维公式给出沿弹簧方向的带符号力；进入二维后还要补上方向。</p>
""",
        ),
        (
            "force-pairs",
            "方向与成对的力",
            "单位方向与等大反向力：一条弹簧连接两个质点",
            r"""
<p><strong>从标量到向量。</strong>设质点 \(a,b\) 的位置为 \(\mathbf p_a,\mathbf p_b\)，定义从 \(a\) 指向 \(b\) 的差向量、当前长度和单位方向：</p>
\[
\Delta\mathbf p=\mathbf p_b-\mathbf p_a,\qquad
L=\lVert\Delta\mathbf p\rVert,\qquad
\mathbf n=\frac{\Delta\mathbf p}{L}.
\]
<p>弹簧只能沿两端连线方向拉或推。令 \(s=-k(L-L_0)-c\,\dot L\)，其中 \(\dot L\) 是两端沿连线方向的相对速度，则两端受到的力为</p>
\[
\mathbf F_a=-s\mathbf n,\qquad
\mathbf F_b=s\mathbf n.
\]
<p>两力大小相等、方向相反，并作用在同一条直线上。若弹簧被拉长，\(s<0\)，质点 \(a\) 被拉向 \(b\)，质点 \(b\) 被拉向 \(a\)。若 \(L=0\)，单位方向没有定义，必须停止并检查状态，不能随便选一个方向继续算。</p>
<p><strong>手算两质点。</strong>\(L_0=1\,\mathrm m\)、\(k=20\,\mathrm{N/m}\)、当前 \(L=1.2\,\mathrm m\)，先令阻尼为零。伸长量为 \(0.2\,\mathrm m\)，所以 \(s=-4\,\mathrm N\)。若 \(\mathbf n=(1,0)\)，则 \(\mathbf F_a=(4,0)\,\mathrm N\)，\(\mathbf F_b=(-4,0)\,\mathrm N\)。</p>
<p><strong>等大反向不等于等加速度。</strong>加速度还要除以各自质量。若 \(m_a=1\,\mathrm{kg}\)、\(m_b=2\,\mathrm{kg}\)，加速度大小分别为 \(4\) 与 \(2\,\mathrm{m/s^2}\)。固定点的逆质量为零，因此它可以被施加力而位置不被积分更新。</p>
<p><strong>本节小结：</strong>用单位方向安排力的方向，用一对相反力保证弹簧内部作用满足牛顿第三定律。</p>
""",
        ),
        (
            "relative-damping",
            "沿弹簧的运动",
            "相对速度阻尼：反对两端彼此远离或靠近",
            r"""
<p><strong>为什么用相对速度。</strong>弹簧连接的是两个端点，所以阻尼应取决于两端沿弹簧方向的相对运动，而不是只看一个质点的绝对速度。令 \(\mathbf v_{rel}=\mathbf v_b-\mathbf v_a\)，则沿连线方向的相对速度为</p>
\[
\dot L=\mathbf v_{rel}\cdot\mathbf n.
\]
<p>把阻尼加入沿弹簧方向的标量力：</p>
\[
s=-k(L-L_0)-c\dot L.
\]
<p>两端仍然受到 \(-s\mathbf n\) 与 \(s\mathbf n\)，大小相等、方向相反。相对速度为正表示两端正在远离，阻尼项反对远离；相对速度为负表示正在靠近，阻尼项反对靠近。它消耗机械能，但不会主动向系统注入能量。</p>
<p><strong>观察量。</strong>弹簧刚度由 \(k(L-L_0)\) 决定，阻尼由 \(c\dot L\) 决定。二者可以同时存在：弹簧形变很大但两端暂时静止时，阻尼为零；弹簧恰好回到静止长度但两端仍在运动时，弹性项为零而阻尼不为零。</p>
<p><strong>当前实现对照。</strong>Python <code>physics/cloth/mass_spring.py</code> 也计算两端距离、单位方向与沿方向相对速度，再对两端施加相反弹簧力。调用方状态和力使用 <code>float32</code> 张量；本章浏览器概念实验使用 JavaScript 双精度，二者只比较趋势与有限性，不要求逐位相同。</p>
<p><strong>本节小结：</strong>弹性项看形变，阻尼项看相对速度，两者都沿同一条弹簧连线作用。</p>
""",
        ),
        (
            "oscillator",
            "一个质点的振动",
            "单振子：先找到静平衡，再描述上下振动",
            r"""
<p><strong>竖直弹簧要多做一步。</strong>设弹簧静止长度为 \(L_0\)，质点质量为 \(m\)。挂上质点后，重力 \(mg\) 与弹簧拉力平衡，静止时弹簧已经比自然长度多伸长了</p>
\[
x_{\mathrm{static}}=\frac{mg}{k}.
\]
<p>本章 Demo 的 \(x\) 不是从弹簧自然长度量起，而是从这个<strong>静平衡位置</strong>量起的位移，向下为正。重力已经用于求出静平衡位置，因此振动方程中<strong>不再重复加重力</strong>：</p>
\[
m\ddot x+c\dot x+kx=0.
\]
<p>若错误地继续减去 \(mg\)，相当于把重力算了两遍，平衡位置会发生偏移。改变质量或刚度时，应先重新计算 \(x_{\mathrm{static}}=mg/k\)，再从新的静平衡位置开始观察。</p>
<p><strong>能量账本。</strong>以静平衡为弹性势能零点，质点相对静平衡的动能与弹性势能之和为</p>
\[
E=K+U=\frac12mv^2+\frac12kx^2.
\]
<p>无阻尼时它应近似守恒；有阻尼时机械能通常逐渐减少。重力势能的变化已经与平衡点的弹性势能变化相互抵消，所以这个表达式不会遗漏竖直重力的影响。</p>
<p><strong>周期近似。</strong>\(2\pi\sqrt{m/k}\) 是<strong>小振幅无阻尼周期近似</strong>，不是所有阻尼下都精确的周期。阻尼会改变振动包络，过阻尼时甚至不再出现周期性来回振荡。</p>
<p><strong>浏览器概念实验：spring-oscillator。</strong>控件包括质量、刚度、阻尼和初始伸长 \(x_0\)。默认 \(m=1\,\mathrm{kg}\)、\(k=20\,\mathrm{N/m}\)、\(c=0\)、\(x_0=0.5\,\mathrm m\)。读数给出周期近似、当前伸长、速度、总能量和稳定步长上限。共享传输仍为每次 \(1/60\,\mathrm s\)，实验内部把它分成四个 \(1/240\,\mathrm s\) 子步。</p>
<p><strong>本节小结：</strong>先处理重力静平衡，再用 \(x,v\) 描述相对平衡的运动。</p>
""",
        ),
        (
            "explicit-stability",
            "显式积分的边界",
            "显式刚度与稳定步长：为什么不能无限增大 dt",
            r"""
<p><strong>显式更新会使用旧状态。</strong>本章采用半隐式欧拉：先由当前 \(x,v\) 求力，更新速度，再用新速度更新位置。无阻尼单振子的连续圆频率为 \(\omega=\sqrt{k/m}\)。对这种显式格式，一个常用稳定性诊断是</p>
\[
dt&lt;\frac{2}{\sqrt{k/m}}=\frac{2}{\omega}.
\]
<p>换成直接可输入的写法，就是 <code>dt &lt; 2/sqrt(k/m)</code>。<code>RTPhysics.stableSpringStep(k,m)</code> 返回 \(2/\sqrt{k/m}\)，在 Demo 中标注为“稳定步长上限”。它来自无阻尼简谐运动的线性稳定性分析，<strong>只是诊断，不是任意系统稳定的保证</strong>。阻尼、风力、网格连接、约束、积分顺序和过大位移都可能改变实际稳定性。</p>
<p><strong>两种时间步不要混淆。</strong>共享播放传输每次调用 <code>update</code> 的固定步长是 \(1/60\,\mathrm s\)。为了在控件范围内获得更细的积分，spring-oscillator 与 spring-cloth 都在每次传输更新内部再做四个子步，所以内部积分步长为</p>
\[
h=\frac{1/60}{4}=\frac1{240}\,\mathrm s.
\]
<p>子步不会改变共享播放、暂停、单步和重置按钮，也不会把一次“单步”变成四次界面帧。它只减少每个 \(1/60\,\mathrm s\) 内的显式积分误差。</p>
<p><strong>有界控件。</strong>质量、刚度和阻尼都限制在有限范围内；布料风力与刚度也有限。若出现非有限值或零长度弹簧，应暂停并提示重置，不能把速度裁成某个上限后假装物理状态仍然正确。</p>
<p><strong>本节小结：</strong>稳定步长帮助选积分精度，但不能替代能量、有限值和最大伸长检查。</p>
""",
        ),
        (
            "spring-mesh",
            "从一根弹簧到网格",
            "弹簧网格：用结构弹簧连接 6×4 个质点",
            r"""
<p><strong>把两质点弹簧铺开。</strong>spring-cloth 使用 6 列、4 行，共 \(6\times4=24\) 个质点。顶行六个质点全部固定，其余十八个质点可以运动。每个自由质点有质量，固定质点的逆质量为零。</p>
<p><strong>结构弹簧。</strong>相邻水平质点之间连接 20 根弹簧，相邻竖直质点之间连接 18 根弹簧，共 38 根。水平静止长度为 \(0.35\,\mathrm m\)，竖直静止长度为 \(0.30\,\mathrm m\)。这些连接抵抗拉长与压缩，维持布片的基本边长。</p>
<p><strong>当前 Demo 范围。</strong>本 Demo 只启用结构弹簧，并加入向下的重力与可调水平风力。风力是施加在每个自由质点上的简化外力，不是完整空气动力学。Canvas 用线的颜色和箭头显示当前伸长与拉力方向，并列出固定点、最大伸长、最大速度和有限值状态。视图会自适应缩放，但不会裁剪或改写物理坐标。</p>
<p><strong>为什么仍会变形。</strong>只有结构弹簧时，方格可能变成菱形，因为阻止相邻横列错动的对角连接没有启用。这正是下一节要区分的剪切作用。布片也没有自碰撞，因此可能穿过自身。</p>
<p><strong>观察任务。</strong>先把风力设为零，观察重力使竖向结构弹簧伸长；再增大风力，观察顶行固定点如何把载荷传给弹簧。刚度增大通常减小伸长，但显式积分仍需满足稳定条件。</p>
<p><strong>本节小结：</strong>网格把多条成对弹簧力组合起来；结构弹簧只负责一部分形状保持能力。</p>
""",
        ),
        (
            "spring-types",
            "三类连接",
            "结构、剪切与弯曲：三种弹簧抵抗不同形变",
            r"""
<p><strong>结构弹簧。</strong>连接同一行或同一列相邻质点，像布片的经纬线，主要抵抗边长变化。spring-cloth 的 38 根弹簧全部属于这一类。</p>
<p><strong>剪切弹簧。</strong>连接小方格对角顶点，抵抗方格被推成菱形。例如同一方格中的左上与右下、右上与左下各连一根对角弹簧。它们提高抗剪切能力，但不会自动保证所有三角形面积恒定。</p>
<p><strong>弯曲弹簧。</strong>常连接隔一个质点的位置，例如第 \(i\) 与第 \(i+2\) 个质点，抵抗过度弯折。它可以提供简单的抗弯效果，但一根距离弹簧并不等于完整的连续薄板弯曲理论，也不能描述真实织物的全部弯曲行为。</p>
<table><thead><tr><th>类型</th><th>常见连接</th><th>主要抵抗</th><th>本 Demo</th></tr></thead><tbody>
<tr><td>结构</td><td>水平、竖直相邻点</td><td>拉长与压缩</td><td>启用</td></tr>
<tr><td>剪切</td><td>方格对角点</td><td>菱形错动</td><td>未启用，但正文解释</td></tr>
<tr><td>弯曲</td><td>隔一个质点的点</td><td>局部过度弯折</td><td>未启用，但正文解释</td></tr>
</tbody></table>
<p><strong>Python 对照。</strong><code>physics/cloth/geometry.py</code> 会按 structural、shear、bend 三类建立距离约束，<code>MassSpringCloth</code> 为三类使用不同刚度。当前 Python 路径不是完整的弯曲薄板或自碰撞求解器；本章浏览器 Demo 也没有照搬全部三类约束，而是只演示结构弹簧。</p>
<p><strong>本节小结：</strong>看到“弹簧网格”要先问连接了哪些点，不能默认它包含全部剪切和弯曲行为。</p>
""",
        ),
        (
            "pins-collisions",
            "边界与接触",
            "固定点与碰撞：约束位置不等于弹簧力",
            r"""
<p><strong>固定点。</strong>固定质点像被图钉钉住，位置与速度始终不变。实现上常把它的逆质量设为零，使它不参与由力产生的加速度积分，并在每个子步后恢复初始位置。spring-cloth 顶行正好有六个固定点；若固定数变成五个或七个，就不再符合本章规格。</p>
<p><strong>固定与弹簧是不同机制。</strong>弹簧力由形变和相对速度计算；固定点直接限制某个质点能去哪里。固定点也会通过结构弹簧承受其余质点的载荷，但它本身不会因这些力而移动。</p>
<p><strong>碰撞。</strong>布片与地面、桌子或球接触时，需要额外的碰撞检测与响应。碰撞可能改变位置和速度，不能仅靠一根弹簧力替代。自碰撞还要检测布片不同部位何时相遇，计算成本更高，也更容易出现数值问题。</p>
<p><strong>当前边界。</strong>spring-cloth 不模拟地面碰撞或自碰撞，顶行固定点只是防止整片布自由下落。因此不能根据本 Demo 判断真实衣物的折叠、穿透或碰撞响应。RenderTorch 其他 Python 模块提供碰撞器，也不意味着这个浏览器 Demo 自动获得同样能力。</p>
<p><strong>本节小结：</strong>固定点处理位置边界，弹簧处理弹性连接，碰撞处理接触，三者不能混为一谈。</p>
""",
        ),
        (
            "worked-example",
            "把公式算到底",
            "算例：k=20、m=1、x0=0.5",
            r"""
<p><strong>题目。</strong>竖直弹簧—质点系统中，\(k=20\,\mathrm{N/m}\)、\(m=1\,\mathrm{kg}\)、\(c=0\)，从静平衡向下 \(x_0=0.5\,\mathrm m\) 处由静止释放。求周期近似、稳定步长提示、初始能量和第一次 \(1/60\,\mathrm s\) 传输更新后的状态。</p>
<p><strong>先算诊断量。</strong>小振幅无阻尼周期近似为</p>
\[
T\approx2\pi\sqrt{\frac{m}{k}}
=2\pi\sqrt{\frac1{20}}
\approx1.405\,\mathrm s.
\]
<p>稳定步长提示为 \(2/\sqrt{20}\approx0.4472\,\mathrm s\)，只作诊断。重力造成的静止伸长为 \(mg/k=9.81/20=0.4905\,\mathrm m\)，它不参与相对静平衡的 \(x_0\) 方程。初始总机械能为</p>
\[
E_0=\frac12mv_0^2+\frac12kx_0^2
=0+\frac12\times20\times0.5^2=2.5\,\mathrm J.
\]
<p><strong>再按 Demo 的四个子步计算。</strong>共享传输步长为 \(1/60\,\mathrm s\)，内部子步 \(h=1/240\,\mathrm s\)。每个子步先算 \(F=-20x\)，再执行 \(v\leftarrow v+Fh/m\)、\(x\leftarrow x+vh\)。</p>
<p>第一个子步可以手算：\(F_0=-10\,\mathrm N\)，所以 \(v_1=-10h=-1/24\,\mathrm{m/s}\)，位置更新为 \(x_1=0.5+v_1h=2879/5760\,\mathrm m\)。后续子步必须用已经更新的 \(x_j\) 重新求力，不能继续套用同一个 \(F_0\)。</p>
<table><thead><tr><th>子步 \(j\)</th><th>位置 \(x_j\)（m）</th><th>速度 \(v_j\)（m/s）</th></tr></thead><tbody>
<tr><td>0</td><td>0.500000000</td><td>0.000000000</td></tr>
<tr><td>1</td><td>0.499826389</td><td>-0.041666667</td></tr>
<tr><td>2</td><td>0.499479227</td><td>-0.083318866</td></tr>
<tr><td>3</td><td>0.498958635</td><td>-0.124942135</td></tr>
<tr><td>4</td><td>0.498264793</td><td>-0.166522021</td></tr>
</tbody></table>
<p>表中显示值四舍五入，计算时保留双精度。第四次更新后的当前合力为 \(F_4=-20x_4\approx-9.965295859\,\mathrm N\)，总能量为</p>
\[
E_4=\frac12mv_4^2+\frac12kx_4^2
\approx2.496542831\,\mathrm J.
\]
<p>无阻尼连续模型的总能量应保持 \(2.5\,\mathrm J\)；离散半隐式欧拉的这次更新略有能量漂移，所以不能把数值结果误当作新的物理定律。</p>
<p><strong>本节小结：</strong>周期、静平衡、能量和子步更新各有自己的公式，计算前先确认坐标原点和时间步。</p>
""",
        ),
        (
            "debugging",
            "可检查的实现",
            "伪代码与调试：让每一步都能追溯",
            r"""
<p><strong>单振子伪代码。</strong>一次界面更新包含四个内部子步，重力只用于创建状态时计算静平衡：</p>
<pre><code>static_extension = m * g / k
x = x0
v = 0
repeat substeps = 4:
    h = (1 / 60) / 4
    force = springForce(x, v, k, c)
    v = v + force / m * h
    x = x + v * h
energy = energy1d(m, v, k, x)
total = energy.kinetic + energy.potential
period_approx = 2 * pi * sqrt(m / k)
step_hint = stableSpringStep(k, m)</code></pre>
<p><strong>布片伪代码。</strong>每个子步先清零力累加器，再遍历所有结构弹簧；两端获得相等且相反的力：</p>
<pre><code>for each free particle:
    force = (wind, gravity * mass)
for each structural spring (a, b):
    delta = position[b] - position[a]
    distance = length(delta)
    if distance is zero or non-finite: stop and request reset
    n = delta / distance
    relative = dot(velocity[b] - velocity[a], n)
    s = springForce(distance - rest_length, relative, k, c)
    force[a] -= s * n
    force[b] += s * n
for each free particle:
    velocity += force / mass * h
    position += velocity * h
restore all six pinned top-row particles
measure maximum stretch, speed and finite state</code></pre>
<table><thead><tr><th>症状</th><th>优先检查</th><th>最小动作</th></tr></thead><tbody>
<tr><td>竖直振子平衡位置偏移</td><td>在相对静平衡的方程中重复加入重力</td><td>核对 \(x_{\mathrm{static}}=mg/k\) 与振动方程</td></tr>
<tr><td>两端加速度相同</td><td>遗漏逆质量加权</td><td>分别核对 \(F_a/m_a\) 与 \(F_b/m_b\)</td></tr>
<tr><td>布片呈菱形</td><td>只有结构弹簧，没有剪切弹簧</td><td>查看弹簧类型，不把它误判为积分错误</td></tr>
<tr><td>固定点移动</td><td>逆质量不为零或子步后未恢复</td><td>检查顶行六点的位置和速度</td></tr>
<tr><td>读数出现 NaN 或 Infinity</td><td>零长度方向、非有限力或错误步长</td><td>暂停、检查状态并重置，不裁剪成假解</td></tr>
<tr><td>把稳定提示当保证</td><td>忽略阻尼、风力和网格耦合</td><td>减小内部子步并检查能量、伸长与速度</td></tr>
</tbody></table>
<p><strong>浏览器与 Python 边界。</strong>本章两个浏览器概念实验使用 JavaScript 双精度浮点和共享固定 \(1/60\,\mathrm s\) 传输，内部四个子步为 \(1/240\,\mathrm s\)。当前 Python <code>physics/common/simulation.py</code> 默认 <code>fixed_dt=1/120</code>、<code>substeps=2</code>，实际积分子步也是 \(1/240\,\mathrm s\)，但状态、速度与力通常由调用方保存为 <code>float32</code> 张量。<code>MassSpringCloth.solve_constraints</code> 只计算最大相对约束误差，并不把位置投影回约束长度。</p>
<p>因此，浏览器适合观察公式、方向和有限性，Python 源码用于了解 RenderTorch 当前实现；两者层次、精度和功能范围不同，不应要求逐位一致，也不能把浏览器结构弹簧 Demo 称为完整弯曲或自碰撞求解器。</p>
<p><strong>本章小结：</strong>从静止长度出发，用单位方向和成对力连接质点，用静平衡处理重力，再用稳定步长、能量和有限读数检查数值结果。</p>
""",
        ),
    ),
    (
        (
            "spring-oscillator",
            "竖直弹簧—质点振子",
            "demos/physics07.js",
            "默认暂停；从重力静平衡量起位移，调节质量、刚度、阻尼和初始伸长，观察周期近似、能量与稳定步长。",
            "oscillator",
        ),
        (
            "spring-cloth",
            "6×4 结构弹簧布片",
            "demos/physics08.js",
            "默认暂停；顶行六点固定，只启用结构弹簧，调节风力和刚度并检查最大伸长、最大速度与有限值。",
            "spring-mesh",
        ),
    ),
    (
        (
            "physics/cloth/geometry.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/cloth/geometry.py",
            "当前 Python 实现：按结构、剪切、弯曲三类建立布料网格距离约束，顶行固定点的逆质量为零。",
        ),
        (
            "physics/cloth/mass_spring.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/cloth/mass_spring.py",
            "当前 Python 实现：对距离约束计算单位方向、相对速度阻尼和等大反向弹簧力；float32 张量，不是完整弯曲或自碰撞求解器。",
        ),
        (
            "physics/common/simulation.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/simulation.py",
            "当前 Python 实现：默认 fixed_dt=1/120、substeps=2，内部步长 1/240；半隐式更新、固定点恢复与速度重建。",
        ),
        (
            "book/assets/physics-core.js",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/book/assets/physics-core.js",
            "浏览器共享数值核心：springForce、energy1d 与 stableSpringStep；本章概念实验使用 JavaScript 双精度。",
        ),
    ),
)
