"""第二章：速度、加速度与时间积分。正文、图示与实验均为本教材原创。"""

CHAPTER = (
    2,
    "physics02",
    "单元一：运动会怎样发生",
    "速度、加速度与时间积分",
    "从速度的变化率出发，比较前向欧拉、半隐式欧拉与精确自由落体，再用固定时间步和累加器解释稳定、可复现的模拟时间。",
    (
        "用速度差除以时间差计算加速度，并用牛顿第二定律连接力、质量与加速度",
        "逐步执行前向欧拉与半隐式欧拉，独立核对首帧速度、位置和离散误差",
        "说明精确自由落体条件，比较三种方法在同一终点的位置误差",
        "解释固定时间步、累加器、追帧上限，以及 RenderTorch 的子步与速度重建",
        "复现两个浏览器概念实验，并说明暂停单步与当前 Python 实现的边界",
    ),
    ("第一章的位置、速度与离散帧", "负数的四则运算", "一次函数与抛物线", "基本 Python 循环"),
    (
        (
            "acceleration-rate",
            "变化率",
            "加速度是速度的变化率",
            r"""
<p><strong>生活直觉。</strong>汽车速度表从每秒十米变为每秒二十米，与从每秒三十米变为每秒四十米，速度增量相同；若一个用了一秒、另一个用了两秒，前者加速度的大小就更大。加速度描述的不是“现在多快”，而是“速度改变得多快”。</p>
<p>设一段有限时间内，一个方向的速度分量从 \(v_0\) 变为 \(v_1\)，时间间隔为 \(\Delta t>0\)。这段时间的平均加速度分量定义为</p>
\[
a=\frac{\Delta v}{\Delta t}
=\frac{v_1-v_0}{\Delta t}.
\]
<p>写成向量形式时，同样有 \(\mathbf a=(\mathbf v_1-\mathbf v_0)/\Delta t\)。下面先按一个方向手算，再检查两个方向是否都符合。</p>
<p>分子是速度向量，单位是 \(\mathrm{m/s}\)；分母是秒，所以加速度单位是 \(\mathrm{m/s^2}\)。例如速度在 \(0.5\,\mathrm s\) 内由 \((2,4)\,\mathrm{m/s}\) 变为 \((1,2)\,\mathrm{m/s}\)，则</p>
\[
\Delta\mathbf v=(-1,-2)\,\mathrm{m/s},\qquad
\mathbf a=\frac{(-1,-2)}{0.5}=(-2,-4)\,\mathrm{m/s^2}.
\]
<p>这表示水平速度每秒减少 \(2\,\mathrm{m/s}\)，竖直速度每秒减少 \(4\,\mathrm{m/s}\)。加速度为负不代表物体一定向下运动；它只说明速度正在沿负方向改变。速度为负且加速度也为负时，物体仍可能越来越快地向下运动。</p>
<p><strong>分量计算。</strong>二维计算可以按方向分开：\(a_x=(v_{x,1}-v_{x,0})/\Delta t\)，\(a_y=(v_{y,1}-v_{y,0})/\Delta t\)。这与先算向量差再除以时间是同一件事。注意不能把两个速度的大小直接相减来代替向量差，否则会丢失方向。</p>
<p><strong>本节小结：</strong>加速度连接“速度差”和“时间差”；下一步用牛顿第二定律说明速度为什么改变。</p>
""",
        ),
        (
            "newton-second-law",
            "因果模型",
            "牛顿第二定律：合力决定加速度",
            r"""
<p><strong>核心关系。</strong>物体的质量 \(m\) 越大，同样的力越难改变它的速度。牛顿第二定律写成向量形式是</p>
\[
\mathbf F_{\mathrm{net}}=m\mathbf a,\qquad
\mathbf a=\frac{\mathbf F_{\mathrm{net}}}{m}.
\]
<p>这里 \(\mathbf F_{\mathrm{net}}\) 是所有外力的合力，不是某一个力。若水平拉力和摩擦力同时存在，应先求合力，再除以质量。若合力为零且速度原来不为零，物体保持匀速直线运动，而不是自动停下。</p>
<p><strong>自由落体。</strong>忽略空气阻力后，小球只受重力。取竖直向上为正，重力加速度为 \(\mathbf g=(0,g)\)，其中 \(g=-9.81\,\mathrm{m/s^2}\)。由 \(\mathbf F_{\mathrm{net}}=m\mathbf g\) 可得</p>
\[
\mathbf a=\frac{m\mathbf g}{m}=\mathbf g.
\]
<p>质量在推导中约去，因此本章不同质量的小球有相同自由落体轨迹。这个结论依赖“忽略空气阻力”的模型；真实羽毛与铁球在空气中的运动不能这样简单比较。</p>
<p><strong>手算。</strong>质量 \(m=2\,\mathrm{kg}\) 的小球受到竖直向下的合力 \(19.62\,\mathrm N\)，则 \(a_y=-19.62/2=-9.81\,\mathrm{m/s^2}\)。若合力加倍而质量不变，加速度也加倍；若质量加倍而合力不变，加速度减半。</p>
<p><strong>常见误区。</strong>“向下运动”不是“受到向下合力”的同义词。向上抛出的小球仍在上升时，重力已经向下，所以它的加速度始终向下，速度则先减小到零，再反向增大。</p>
<p><strong>本节小结：</strong>模型先给出加速度，积分器才能把加速度逐步变成速度和位置。</p>
""",
        ),
        (
            "velocity-update",
            "离散更新",
            "从加速度更新速度，再更新位置",
            r"""
<p>连续运动中，速度在每一步里也会变化。有限时间步 \(h>0\) 内，若用加速度近似这一小段的变化率，则</p>
\[
\mathbf v_{n+1}=\mathbf v_n+\mathbf a\,h,\qquad
\Delta\mathbf p\approx\mathbf v\,h.
\]
<p>速度单位 \(\mathrm{m/s}\) 乘秒得到米，量纲正确。若 \(\mathbf a=(0,-9.81)\,\mathrm{m/s^2}\)、\(h=0.1\,\mathrm s\)，速度竖直分量的改变量为 \(-0.981\,\mathrm{m/s}\)。无论旧速度是正是负，重力造成的速度改变量都相同。</p>
<p><strong>为什么必须保存旧状态？</strong>更新位置时，如果先覆盖速度，就可能把“这一秒的速度”误当成“下一秒的速度”。因此常把旧状态记为 \(\mathbf p_n,\mathbf v_n\)，先计算新状态，最后一次性替换：</p>
<pre><code>v_next = v + a * h
p_next = p + v_chosen * h
state.p = p_next
state.v = v_next</code></pre>
<p>其中 \(v_{\mathrm{chosen}}\) 到底是旧速度还是新速度，取决于积分方法。两个选择都合法，但误差与稳定性不同。</p>
<p><strong>时间符号。</strong>本章用 \(h\) 表示一次数值积分的实际时长，用 \(dt\) 表示概念实验的积分步长或显示帧时长。RenderTorch 中还有外层固定步长与积分子步的区别，不能把这些时间混成一个数。</p>
<p><strong>本节小结：</strong>速度更新规则确定后，先看最简单的前向欧拉。</p>
""",
        ),
        (
            "forward-euler",
            "第一种方法",
            "前向欧拉：先用旧速度走完这一步",
            r"""
<p><strong>更新顺序。</strong>前向欧拉用本步开始时的速度计算整步位移，同时计算下一时刻的速度：</p>
\[
\mathbf p_{n+1}=\mathbf p_n+\mathbf v_n h,\qquad
\mathbf v_{n+1}=\mathbf v_n+\mathbf a h.
\]
<p>“前向”可以理解为用当前斜率向时间前方推测。位置公式只显式使用旧速度，因此它也称显式欧拉。若 \(\mathbf a\) 随时间或位置变化，必须先在本步开始时求出它，再执行这两行。</p>
<p><strong>算例。</strong>取 \(\mathbf p_0=(0,0)\,\mathrm m\)、\(\mathbf v_0=(5,20)\,\mathrm{m/s}\)、\(\mathbf g=(0,-9.81)\,\mathrm{m/s^2}\)、\(h=0.1\,\mathrm s\)。第一步先用旧竖直速度 \(20\,\mathrm{m/s}\)：</p>
\[
\mathbf v_1=(5,20-9.81\times0.1)=(5,19.019)\,\mathrm{m/s},
\]
\[
\mathbf p_1=(0,0)+(5,20)\times0.1=(0.5,2)\,\mathrm m.
\]
<p><strong>为什么偏高？</strong>匀加速运动的速度图象是一条斜线，前向欧拉相当于用左端点的速度近似整段。重力向下时，左端速度通常比这一段的平均速度大，所以第一步位置会高于精确值。</p>
<p><strong>观察任务。</strong>在 <code>integration-race</code> 中保持初速度和重力不变，只减小 \(dt\)。橙色前向欧拉轨迹应逐渐靠近黄色精确参考线。不要同时改三个控件，否则难以判断是哪一项造成差异。</p>
<p><strong>本节小结：</strong>把速度更新提前，就得到另一种常用的半隐式欧拉。</p>
""",
        ),
        (
            "semi-implicit-euler",
            "第二种方法",
            "半隐式欧拉：先用新速度走这一步",
            r"""
<p><strong>更新顺序。</strong>半隐式欧拉先更新速度，再用刚得到的新速度更新位置：</p>
\[
\mathbf v_{n+1}=\mathbf v_n+\mathbf a h,\qquad
\mathbf p_{n+1}=\mathbf p_n+\mathbf v_{n+1}h.
\]
<p>它与前向欧拉只有一行位置公式不同，却改变了离散误差的方向。沿用上一节数据，第一步新竖直速度为 \(19.019\,\mathrm{m/s}\)，于是</p>
\[
\mathbf p_1=(0,0)+(5,19.019)\times0.1=(0.5,1.9019)\,\mathrm m.
\]
<p><strong>对称性。</strong>对本章恒定加速度，若只比较同长的完整步，前向欧拉第一步多算了 \(0.5|g|h^2\)，半隐式欧拉第一步少算了同样的量。两者平均恰好等于精确位置。这个漂亮关系来自匀加速度，不能直接推广到任意非线性力。</p>
<p><strong>RenderTorch 的选择。</strong>当前 <code>physics/common/simulation.py</code> 在每个积分子步中先施加力并更新速度，再更新位置，因此核心位置积分属于半隐式欧拉。之后它还会处理阻尼、约束与碰撞，所以完整模拟不再只是这两行公式。</p>
<p><strong>不等于无条件稳定。</strong>半隐式欧拉通常比前向欧拉更适合许多力学问题，但“半隐式”不是稳定保证。 刚度很大的弹簧、过大的 \(h\)、错误的坐标或遗漏约束仍可能产生发散结果。</p>
<p><strong>本节小结：</strong>第三种方法不逐步近似，而是在满足模型条件时直接计算精确轨迹。</p>
""",
        ),
        (
            "exact-projectile",
            "精确对照",
            "精确自由落体：同一抛物线作为标尺",
            r"""
<p><strong>适用条件。</strong>若加速度始终是恒定重力，忽略空气阻力、碰撞和旋转，并且水平速度不变，就有闭式解。设起点为 \((x_0,y_0)\)，初速度为 \((v_{x0},v_{y0})\)，经过时间 \(t\) 后：</p>
\[
x(t)=x_0+v_{x0}t,\qquad
y(t)=y_0+v_{y0}t+\frac12gt^2,
\]
\[
v_x(t)=v_{x0},\qquad v_y(t)=v_{y0}+gt.
\]
<p>水平运动与竖直运动可以分开计算。回到地面 \(y=0\) 的上升段总时间，在 \(y_0=0\)、\(v_{y0}>0\)、\(g<0\) 时为</p>
\[
T=\frac{2v_{y0}}{|g|}.
\]
<p><strong>浏览器概念实验：integration-race。</strong>实验固定 \(v_x=5\,\mathrm{m/s}\)，允许调节积分步长 \(dt\)、初速度 \(v_0\) 与重力 \(g\)。三条轨迹都从同一起点出发，在同一个 \(T\) 结束：橙色实线是前向欧拉，青色实线是半隐式欧拉，黄色虚线是精确参考。起点、整段轨迹与终点读数一开始就可见，避免小球迅速飞出画面。</p>
<p>实验默认暂停。共享“单步”每次推进一个选定的 \(dt\)，最后一步按剩余时间缩短，以保证三种方法在共同终点比较；到终点后自动暂停并提示重置。轨迹数组在改变控件时确定性重建，不使用随机数或真实墙钟。</p>
<p><strong>共享核心实现。</strong>实验直接调用 <code>RTPhysics.integrateProjectile</code>；其 <code>exact</code> 分支对水平与竖直两个分量使用恒定加速度闭式：</p>
\[
\mathbf v_{n+1}=\mathbf v_n+\mathbf a h,\qquad
\mathbf p_{n+1}=\mathbf p_n+\mathbf v_n h+\frac12\mathbf a h^2.
\]
<p>位置使用本步开始时的速度，每一步也同时更新速度，因此连续调用会正确累积恒定加速度的运动。浏览器 JavaScript 使用双精度表示这些量；概念实验的时间传输规则与当前 Python <code>float32</code> 实现属于不同层次，不能把本面板当作 Python 求解器的逐位复刻。</p>
<p><strong>本节小结：</strong>有了同一时刻的精确位置，才能定量定义数值误差。</p>
""",
        ),
        (
            "discrete-error",
            "比较方法",
            "离散误差：同一时刻相差多少",
            r"""
<p><strong>先统一比较时刻。</strong>不能拿前向欧拉的第十步与半隐式欧拉的第九步比较，因为时间不同。设在同一模拟时刻 \(t\)，精确位置为 \(\mathbf p_{\mathrm{exact}}(t)\)，数值位置为 \(\mathbf p_{\mathrm{num}}(t)\)，定义位置误差大小</p>
\[
e(t)=\left\|\mathbf p_{\mathrm{num}}(t)-\mathbf p_{\mathrm{exact}}(t)\right\|.
\]
<p>在第一步算例中，精确竖直位置为 \(20\times0.1+\frac12(-9.81)\times0.1^2=1.95095\,\mathrm m\)。因此</p>
\[
e_{\mathrm{forward}}=|2-1.95095|=0.04905\,\mathrm m,
\]
\[
e_{\mathrm{semi}}=|1.9019-1.95095|=0.04905\,\mathrm m.
\]
<p>大小相同，但方向相反：前向欧拉在精确点上方，半隐式欧拉在下方。只记录绝对值会丢失这个有用信息。</p>
<p><strong>步长影响。</strong>对恒定加速度和同长完整步，单步位置误差分别写成</p>
\[
e_{\mathrm{forward}}=\frac12|g|h^2,\qquad
e_{\mathrm{semi}}=\frac12|g|h^2.
\]
<p>把 \(h\) 减半，这部分误差约变为四分之一。例如 \(h=0.1\,\mathrm s\) 时为 \(0.04905\,\mathrm m\)，\(h=0.05\,\mathrm s\) 时为 \(0.0122625\,\mathrm m\)。误差随步长减半而减半是一次收敛，随平方减小是二次收敛。</p>
<p><strong>误差来源。</strong>用有限步近似连续变化产生离散误差；二进制浮点舍入还会产生舍入误差。浏览器 JavaScript 使用双精度，RenderTorch 状态默认使用 <code>float32</code>，所以不要追求跨平台逐位相同，应检查趋势、量级和有限性。</p>
<p><strong>本节小结：</strong>减小积分步长能减小误差，但还需要稳定的时间调度。</p>
""",
        ),
        (
            "fixed-timestep",
            "稳定时钟",
            "固定时间步：把显示帧与积分解耦",
            r"""
<p><strong>两个时钟。</strong>显示帧率 \(f\) 决定多久画一次画面，理想帧间隔为 \(dt_{\mathrm{frame}}=1/f\)。它不直接等于物理积分步长。若每帧都把真实经过时间原样送入积分器，卡顿时一次积分会跨得很大，轻则轨迹跳变，重则数值发散。</p>
<p>固定时间步让积分始终使用同一个 \(\Delta\)：</p>
\[
\Delta=\texttt{fixed\_dt},\qquad
t_{\mathrm{sim}}=N\Delta,
\]
<p>其中 \(N\) 是已经完成的外层固定步数。画面可以晚到，但每次积分不会因为显示卡顿而突然变大。</p>
<p><strong>RenderTorch 默认值。</strong>当前 Python <code>Simulation</code> 默认 <code>fixed_dt=1/120</code>，也就是外层步长为 \(1/120\,\mathrm s\)。它还默认 <code>substeps=2</code>，把每个外层步分成两个实际积分子步：</p>
\[
h=\frac{\texttt{fixed\_dt}}{\texttt{substeps}}
=\frac{1/120}{2}=\frac1{240}\,\mathrm s.
\]
<p>所以“一秒推进一百二十个外层步”与“实际积分两百四十次”同时成立。每个子步依次完成力、速度、位置、约束和速度重建，不能把 \(1/240\,\mathrm s\) 误写成新的外层固定步长。</p>
<p><strong>速度重建。</strong>每个子步先保存投影前位置 \(\mathbf p^{-}\)，约束求解得到投影后位置 \(\mathbf p^{+}\)，再按</p>
\[
\mathbf v_{\mathrm{rebuilt}}=\frac{\mathbf p^{+}-\mathbf p^{-}}{h}
\]
<p>重建自由粒子速度。这样约束修正也能反映到速度中，避免位置被拉回而速度仍指向旧方向。固定粒子位置保持在初始锚点，速度置零。</p>
<p><strong>浏览器概念实验：fixed-step。</strong>左右并排比较“每帧直接 dt”与“累加器固定步长”。默认固定步长是 \(1/120\,\mathrm s\)；每五个显示帧按固定规则加入一次 \(0.12\,\mathrm s\) 卡顿，十二帧后停止。共享“单步”在此表示推进一个显示帧，而不是 RenderTorch 的一个外层步。面板默认暂停，所有轨迹都在各自绘图范围内。</p>
<p><strong>暂停仍可单步。</strong>当前 <code>PhysicsController.advance</code> 在暂停时不会自动追帧，但 <code>single_step</code> 仍直接调用 <code>Simulation.single_step</code>。因此调试时可以暂停、单步、检查一个外层步，再继续；浏览器两个实验也保留同样的公共单步入口。</p>
<p><strong>本节小结：</strong>固定步长规定一次走多远，累加器决定一帧到来时该走几步。</p>
""",
        ),
        (
            "accumulator-catchup",
            "追帧调度",
            "累加器：欠下的模拟时间怎样补",
            r"""
<p><strong>核心思想。</strong>累加器 \(A\) 保存“已经到达、但尚未被固定步消费”的模拟时间。设本帧实际用时为 \(dt_{\mathrm{frame}}\)，RenderTorch 先把它限制在 \([0,0.25]\,\mathrm s\)，再执行：</p>
\[
A\leftarrow A+\max(0,\min(dt_{\mathrm{frame}},0.25)),
\]
\[
\text{while }A+10^{-9}\geq\Delta:\quad
\text{single\_step};\quad A\leftarrow A-\Delta.
\]
<p>若一帧只积累了 \(0.005\,\mathrm s\)，而 \(\Delta=1/120\,\mathrm s\)，这次不做物理步，余量留给下一帧。若积累超过一个 \(\Delta\)，就连续执行多个固定步，直到不够一步。\(10^{-9}\) 是边界容差，避免浮点误差让恰好等于固定步长的余量被遗漏。</p>
<p><strong>为什么限制追帧？</strong>程序切到后台再回来时，真实经过时间可能非常大。把全部时间都补上会让一次更新运行成千上万步，进一步加重卡顿。RenderTorch 最多接受 \(0.25\,\mathrm s\) 的帧时间；超出部分被丢弃，模拟因此可能落后于墙钟，但不会瞬间“补到天荒地老”。</p>
<p><strong>卡顿算例。</strong>在 \(60\) 帧每秒下，一个普通帧为 \(1/60\,\mathrm s\)。第 \(k\) 帧若额外卡顿 \(0.12\,\mathrm s\)，总帧时间就是</p>
\[
dt_{\mathrm{frame}}=\frac1{60}+0.12\approx0.136667\,\mathrm s.
\]
<p>以 \(\Delta=1/120\,\mathrm s\) 计算，本次执行 \(16\) 个固定外层步，模拟推进 \(16/120=0.133333\,\mathrm s\)，累加器剩余约 \(0.003333\,\mathrm s\)。RenderTorch 默认每个外层步还有两个子步，因此这一次外层追帧实际包含 \(32\) 个积分子步。</p>
<p><strong>读数定义。</strong><code>fixed-step</code> 的目标时间 \(T_{\mathrm{target}}\) 是各显示帧时间之和，固定支路模拟时间为 \(T_{\mathrm{sim}}=N\Delta\)。最大追帧误差记录每帧追帧完成后 \(|T_{\mathrm{target}}-T_{\mathrm{sim}}|\) 的最大值；直接支路用完整帧时间积分一次，所以这项为零。浏览器面板的“固定子步数”计的是外层循环次数，不包含 Python 默认的 <code>substeps=2</code>。</p>
<p><strong>观察任务。</strong>先保持固定步长 \(1/120\,\mathrm s\)，只增大卡顿帧。固定支路会在一次显示帧内执行更多外层步，但每步时长不变；直接支路的一步则明显变长。再改变固定步长频率，比较余量与最大追帧误差。</p>
<p><strong>本节小结：</strong>把累加器伪代码与下一节手算结合，检查每一帧到底执行了几步。</p>
""",
        ),
        (
            "worked-example",
            "完整算例",
            "首帧算例：g=-9.81、v0=20、dt=0.1",
            r"""
<p><strong>题目。</strong>原始数据为 <code>g=-9.81, v0=20, dt=0.1</code>。小球从 \((0,0)\,\mathrm m\) 出发，初速度 \(\mathbf v_0=(5,20)\,\mathrm{m/s}\)，重力加速度 \(\mathbf g=(0,-9.81)\,\mathrm{m/s^2}\)，忽略空气阻力。求三种积分方法第一帧的竖直速度与位置。</p>
<p><strong>第一步，共同速度更新。</strong></p>
\[
\Delta v_y=g\,dt=(-9.81)\times0.1=-0.981\,\mathrm{m/s},
\]
\[
v_{y,1}=20-0.981=19.019\,\mathrm{m/s},\qquad v_{x,1}=5\,\mathrm{m/s}.
\]
<p><strong>第二步，前向欧拉位置。</strong>用旧速度 \(v_{y,0}=20\,\mathrm{m/s}\)：</p>
\[
x_1=5\times0.1=0.5\,\mathrm m,\qquad
y_1=20\times0.1=2\,\mathrm m.
\]
<p><strong>第三步，半隐式欧拉位置。</strong>用新速度 \(v_{y,1}=19.019\,\mathrm{m/s}\)：</p>
\[
x_1=0.5\,\mathrm m,\qquad
y_1=19.019\times0.1=1.9019\,\mathrm m.
\]
<p><strong>第四步，精确位置与速度。</strong></p>
\[
x_1=0.5\,\mathrm m,\qquad
y_1=20\times0.1+\frac12(-9.81)\times0.1^2=1.95095\,\mathrm m,
\]
\[
v_{y,1}=19.019\,\mathrm{m/s}.
\]
<table><thead><tr><th>方法</th><th>第一步后 \(v_y\)（m/s）</th><th>第一步后 \((x,y)\)（m）</th><th>竖直位置误差（m）</th></tr></thead><tbody>
<tr><td>前向欧拉</td><td>\(19.019\)</td><td>\((0.5,2)\)</td><td>\(+0.04905\)</td></tr>
<tr><td>半隐式欧拉</td><td>\(19.019\)</td><td>\((0.5,1.9019)\)</td><td>\(-0.04905\)</td></tr>
<tr><td>精确解</td><td>\(19.019\)</td><td>\((0.5,1.95095)\)</td><td>\(0\)</td></tr>
</tbody></table>
<p><strong>检查。</strong>速度是米每秒，位置是米，重力是米每二次方秒；三种方法水平位移都为 \(0.5\,\mathrm m\)，因为水平加速度为零。若把重力误写成 \(+9.81\)，第一帧竖直速度会变为 \(20.981\,\mathrm{m/s}\)，与向上为正的约定矛盾。</p>
<p><strong>本节小结：</strong>独立手算出的期望，应成为实现自动化测试的答案。</p>
""",
        ),
        (
            "pseudocode-debugging",
            "落地检查",
            "伪代码、自测与调试清单",
            r"""
<p><strong>半隐式子步伪代码。</strong>下面只展示自由粒子的核心顺序；RenderTorch 在约束求解后还会重建速度，完整实现请阅读源码映射。</p>
<pre><code>function integrate(state, h, gravity):
    previous_position = state.position
    apply_forces(state, h)
    state.velocity += gravity * h
    state.velocity *= exp(-damping * h)
    state.position += state.velocity * h
    solve_constraints(state, h)
    free_velocity = (state.position - previous_position) / h
    state.velocity[free] = free_velocity</code></pre>
<p><strong>外层固定步伪代码。</strong></p>
<pre><code>function advance(frame_dt):
    if paused:
        return
    accumulator += clamp(frame_dt, 0, 0.25)
    while accumulator + 1e-9 &gt;= fixed_dt:
        single_step()
        accumulator -= fixed_dt

function single_step():
    h = fixed_dt / substeps
    repeat substeps times:
        integrate(state, h, gravity)
    step_count += 1
    time += fixed_dt</code></pre>
<p>暂停时 <code>advance</code> 直接返回；若显式调用 <code>single_step</code>，仍执行一个外层步。默认 <code>fixed_dt=1/120</code>、<code>substeps=2</code>，对应实际 \(h=1/240\,\mathrm s\)。</p>
<table><thead><tr><th>症状</th><th>优先检查</th><th>最小动作</th></tr></thead><tbody>
<tr><td>上抛却一直向下加速</td><td>重力符号、世界坐标方向</td><td>只打印 \(v_y\) 与 \(g\) 一个子步</td></tr>
<tr><td>轨迹随帧率明显改变</td><td>把显示帧时间直接当固定步长</td><td>记录外层步数与累加器余量</td></tr>
<tr><td>前向、半隐式结果相同</td><td>位置更新意外使用同一速度</td><td>核对第一步 \(2\) 与 \(1.9019\)</td></tr>
<tr><td>约束后位置正确但速度穿过物体</td><td>遗漏速度重建</td><td>比较投影前后位置差除以 \(h\)</td></tr>
<tr><td>一次卡顿后模拟时间暴涨</td><td>缺少追帧上限</td><td>把单帧时间限制到 \(0.25\,\mathrm s\)</td></tr>
<tr><td>读数出现 NaN 或 Infinity</td><td>除零、无效输入或非有限状态</td><td>暂停、重置，再逐个恢复控件</td></tr>
</tbody></table>
<p><strong>分层结论。</strong><code>integration-race</code> 与 <code>fixed-step</code> 是浏览器二维概念实验：前者比较积分公式，后者比较时间调度。它们没有 RenderTorch 的刚体形状、碰撞库、阻尼实现或三维渲染。<code>fixed-step</code> 的十二帧边界与每五帧一次卡顿是为可复现教学设计的规则，不是生产环境的真实帧率采集。</p>
<p><strong>当前 Python 实现。</strong><code>physics/common/simulation.py</code> 包含固定步长、子步、约束后的速度重建与诊断；<code>app/physics_controller.py</code> 连接暂停、恢复、单步和网格同步；<code>book/assets/physics-core.js</code> 提供浏览器实验使用的 <code>integrateProjectile</code>。这些文件处于同一概念的不同层次，不能说浏览器面板就是 Python 求解器的完整复刻。</p>
<p><strong>本章小结：</strong>加速度改变速度，积分器把改变累积成位置；固定时间步隔离显示波动，累加器决定追帧次数；用共同时刻、手算期望与有限读数，才能区分模型、离散误差和实现错误。</p>
""",
        ),
    ),
    (
        (
            "integration-race",
            "三种积分轨迹对比",
            "demos/physics03.js",
            "默认暂停并显示完整轨迹；调节 dt、初速度与重力，比较前向欧拉、半隐式欧拉和精确解的终点位置与误差。",
            "exact-projectile",
        ),
        (
            "fixed-step",
            "固定步长与追帧",
            "demos/physics04.js",
            "并排比较每帧直接 dt 与累加器固定步长；调节帧率、固定步长频率和卡顿帧，观察模拟时间、固定子步数与最大追帧误差。",
            "fixed-timestep",
        ),
    ),
    (
        (
            "physics/common/simulation.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/simulation.py",
            "当前 Python 实现：固定步长、子步、半隐式位置更新、约束后速度重建与诊断。",
        ),
        (
            "app/physics_controller.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/app/physics_controller.py",
            "当前 Python 实现：暂停时停止 advance，但仍可显式 single_step，并同步渲染网格。",
        ),
        (
            "physics/common/state.py",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/physics/common/state.py",
            "当前 Python 实现：ParticleState 的位置、速度、逆质量与确定性重置。",
        ),
        (
            "book/assets/physics-core.js",
            "https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/book/assets/physics-core.js",
            "浏览器共享数值核心：本章积分竞赛直接调用 integrateProjectile，使用两个分量的恒定加速度闭式。",
        ),
    ),
)
