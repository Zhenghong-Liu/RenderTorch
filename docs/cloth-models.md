# 三种布料模型讲解

> 对应源码：[公共模拟](../physics/common/simulation.py)、[网格拓扑](../physics/cloth/geometry.py)、[质点弹簧](../physics/cloth/mass_spring.py)、[PBD](../physics/cloth/pbd.py)、[XPBD](../physics/cloth/xpbd.py)。

## 1. 共同实验条件

三种算法都从 `build_cloth_grid()` 生成相同的矩形网格。粒子位置位于 XZ 平面，顶行粒子通过零逆质量固定。约束分为：

- **结构约束**：连接水平、垂直相邻粒子，抵抗拉伸；
- **剪切约束**：连接对角相邻粒子，抵抗网格错动；
- **弯曲约束**：连接隔一个粒子的水平、垂直粒子，近似抵抗折叠。

公共 `Simulation` 使用半隐式欧拉积分、固定时间步和子步。每个子步先预测位置，再执行算法特有的力或约束阶段，最后由碰撞体投影穿透粒子。固定粒子在子步结束时恢复到初始位置。

公平比较时必须保持网格、初始位置、逆质量、重力、时间步、子步、碰撞体和迭代/力参数一致。`tests/test_cloth_mass_spring.py`、`tests/test_cloth_pbd.py` 和 `tests/test_cloth_xpbd.py` 都包含相同初始条件的检查。

## 2. 质点弹簧模型

### 2.1 模型与公式

一根弹簧连接粒子 \(a,b\)，当前长度为 \(L\)，静止长度为 \(L_0\)，沿连线方向的相对速度为 \(v_{rel}\)。教学实现采用：

\[
F = k(L-L_0) + c v_{rel}
\]

力沿单位方向 \(\mathbf n=(\mathbf p_b-\mathbf p_a)/L\) 作用，两个粒子受到等大反向力。结构、剪切、弯曲可以使用不同的 \(k\)。

积分顺序是“先计算弹簧力并更新速度，再加入重力、阻尼和位置预测”。显式积分的时间步必须足够小；刚度过大时，稳定条件大致要求：

\[
\Delta t < \frac{2}{\sqrt{k/m}}
\]

这只是直观判据，网格耦合后的实际稳定范围还受质量、阻尼和约束拓扑影响。

### 2.2 伪代码

```text
对每个距离约束：
    计算当前方向、长度和沿方向相对速度
    计算弹簧力
    在粒子 a 上加力，在粒子 b 上加反向力

速度 += 力 * 逆质量 * 子步
速度 += 重力 * 子步
速度 *= exp(-阻尼 * 子步)
位置 += 速度 * 子步
执行碰撞投影
```

### 2.3 参数与指标

- `structural_stiffness`、`shear_stiffness`、`bend_stiffness`：三类弹簧刚度；
- `spring_damping`：沿弹簧方向的相对速度阻尼；
- `last_constraint_error`：最大相对长度误差，只作为诊断，不用于强制修正位置。

如果网格突然变硬或出现 NaN，优先降低刚度、增大子步数或减小固定时间步，不要把误差指标归零当作修复。

## 3. PBD：位置约束

### 3.1 模型与公式

距离约束为：

\[
C(\mathbf p_a,\mathbf p_b)=|\mathbf p_b-\mathbf p_a|-L_0
\]

梯度分别为 \(\nabla_a C=-\mathbf n\)、\(\nabla_b C=\mathbf n\)。PBD 用位置投影代替显式弹簧力。两个自由粒子、逆质量为 \(w_a,w_b\) 时，一次完整投影为：

\[
\Delta\mathbf p_a=w_a\mathbf n C,\qquad
\Delta\mathbf p_b=-w_b\mathbf n C
\]

更一般地，以刚度因子 \(s\in[0,1]\) 缩放校正量：

\[
\Delta\mathbf p_a=
\frac{w_a}{w_a+w_b}s\,\mathbf n C,\qquad
\Delta\mathbf p_b=
-\frac{w_b}{w_a+w_b}s\,\mathbf n C
\]

固定粒子的 \(w=0\)，因此不会被推动。

### 3.2 为什么逐约束更新

本项目按约束顺序执行 Gauss–Seidel 投影：处理完一根弹簧后，下一根弹簧读取更新后的位置。若用一次 `index_add_` 同时更新所有约束，共享粒子会把多个大校正量直接相加；在高刚度网格中容易过校正并迅速放大。

因此 [pbd.py](../physics/cloth/pbd.py) 的外层是迭代次数，内层是约束顺序。教学网格规模下，这种直接实现比复杂的图着色批处理更容易读懂。

### 3.3 伪代码

```text
重复 iterations 次：
    对每根距离约束：
        如果两个粒子都固定：跳过
        计算方向 n、当前长度 L 和误差 C=L-L0
        计算按逆质量分配的校正量
        立即更新两个粒子
记录最大相对约束误差
```

### 3.4 参数与调试

- `iterations`：投影迭代次数，通常比继续增大“刚度”更有效；
- 三类 `*_stiffness`：每次投影的校正比例，不是材料模量；
- 位置投影会改变速度，因为公共积分器用投影前后的位移重新估计速度；
- 误差下降缓慢时增加迭代；出现抖动时检查子步、碰撞顺序和速度重建。

## 4. XPBD：柔度与拉格朗日乘子

### 4.1 模型与公式

XPBD 为每根约束引入拉格朗日乘子 \(\lambda\)。柔度 \(\alpha\) 定义约束势能：

\[
\frac{1}{2}\frac{1}{\alpha}C^2
\]

\(\alpha=0\) 表示硬约束，\(\alpha\) 越大表示越柔软。子步 \(h\) 内使用有效柔度：

\[
\tilde\alpha=\frac{\alpha}{h^2}
\]

每次投影更新：

\[
\Delta\lambda=
\frac{-C-\tilde\alpha\lambda}{w_a+w_b+\tilde\alpha}
\]

\[
\lambda\leftarrow\lambda+\Delta\lambda
\]

\[
\Delta\mathbf p_a=-w_a\Delta\lambda\mathbf n,\qquad
\Delta\mathbf p_b=w_b\Delta\lambda\mathbf n
\]

关键生命周期是：**同一子步内跨迭代保留 \(\lambda\)，进入下一个子步时归零**。这样乘子能够累计校正，同时避免把上一子步的冲量重复带入下一子步。

### 4.2 伪代码

```text
进入子步：lambda 全部归零
alpha_tilde = alpha / h²

重复 iterations 次：
    对每根距离约束：
        计算 C、法线 n 和总逆质量
        delta_lambda = (-C - alpha_tilde*lambda) / (总逆质量 + alpha_tilde)
        lambda += delta_lambda
        按法线和逆质量更新粒子位置
```

### 4.3 参数与对比

- `structural_compliance`、`shear_compliance`、`bend_compliance`：具有时间步平方单位的柔度；
- `iterations`：同一子步内的投影次数；
- `lambdas`：教学可观察的约束乘子，重启模拟时归零；
- `last_constraint_error`：最大相对距离误差。

相同材料柔度下，XPBD 的结果不应仅因为改变子步而完全改变；但比较时仍需保持足够的约束迭代。PBD 的 `0.95` 与 XPBD 的 compliance 不是可以按同一数值直接替换的参数。

## 5. 自动化验证

| 测试文件 | 核心覆盖 |
|---|---|
| `tests/test_cloth_mass_spring.py` | 拓扑、静止长度、等大反向力、默认稳定性、固定粒子、重置 |
| `tests/test_cloth_pbd.py` | 单次投影、固定粒子、迭代收敛、相同初始条件、地面碰撞、重置 |
| `tests/test_cloth_xpbd.py` | 硬约束、柔度、乘子累积与重置、相同初始条件、地面碰撞、复现 |

运行全部物理测试：

```bash
/opt/anaconda3/envs/pythonTry/bin/python -m unittest tests.test_physics_common tests.test_cloth_mass_spring tests.test_cloth_pbd tests.test_cloth_xpbd -v
```

## 6. 参考资料

- Müller 等，*Position Based Dynamics*，2007，[DOI](https://doi.org/10.1016/j.jvcir.2007.01.005)。
- Macklin 等，*XPBD: Position-Based Simulation of Compliant Constrained Dynamics*，2016，[作者 PDF](https://matthias-research.github.io/pages/publications/XPBD.pdf)。
- 参数组织与演示交互仅参考 Matter.js 和 Blender 的公开文档，未复制其求解器或界面素材；详见 [参考资料索引](references.md)。
