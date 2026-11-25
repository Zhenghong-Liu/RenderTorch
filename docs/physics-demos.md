# 刚体与软体 Demo 说明

> 对应里程碑：M6
> 实现范围：教学级球体刚体、球体/布料粒子耦合、XPBD 软体立方体。
> 不作为生产级碰撞引擎或通用有限元求解器。

## 1. 刚体：球形运动与冲量

`physics/rigidbody/sphere.py` 使用 `SphereBody` 描述球心位置、速度、半径和逆质量。第一版不包含角速度和旋转惯量，因此“刚体”体现在碰撞过程中形状与半径不变。

每个子步执行：

1. 更新重力与阻尼速度。
2. 半隐式欧拉更新球心位置。
3. 将球心投影出地面和静态球体。
4. 应用恢复系数与切向摩擦。

地面碰撞使用球心到平面的有符号距离减去球半径。静态球碰撞使用两个半径之和。切向速度按摩擦系数衰减，是简化摩擦模型，不宣称覆盖完整库仑接触求解。

### 球体与布料粒子

`collide_with_particles()` 对每个侵入球体的粒子计算法线、穿透量和相对速度。位置修正按球体与粒子的逆质量比例分配；法向冲量为：

```text
j = -(1 + restitution) * normal_speed / (body_weight + particle_weight)
```

粒子与球体受到大小相等、方向相反的冲量。固定粒子逆质量为 0，因此不移动，但仍可向动态球体传递冲量。切向冲量限制在 `±friction * j` 内。

## 2. 软体：格点、距离与体积

`physics/softbody/geometry.py` 构建 3×3×3 立方体格点，共 27 个粒子，其中包含内部中心粒子。距离约束连接结构、剪切与弯曲邻域，防止格点散架。

立方体沿一条体对角线分解为六个四面体。每个四面体保存四个顶点索引和静止有符号体积：

```text
V = dot(cross(b - a, c - a), d - a) / 6
```

构建拓扑时统一顶点绕序，使静止体积为正。总体积是六个单元体积之和，而不是硬编码为 1。

`physics/softbody/xpbd.py` 每轮先求解距离约束，再求解六个体积约束。体积梯度为：

```text
grad_b = cross(edge_ac, edge_ad) / 6
grad_c = cross(edge_ad, edge_ab) / 6
grad_d = cross(edge_ab, edge_ac) / 6
grad_a = -(grad_b + grad_c + grad_d)
```

体积约束的 XPBD 更新使用：

```text
denominator = sum(w_i * ||grad_i||²)
delta_lambda = (-C - alpha_tilde * lambda) / (denominator + alpha_tilde)
position_i += w_i * delta_lambda * grad_i
```

其中 `alpha_tilde = compliance / h²`。距离与体积乘子在同一子步的迭代之间保留，进入下一子步时清零。

## 3. 控制与显示

`app/physics_controller.py` 负责构建场景、推进模拟和同步网格：

- 刚体：动态球体、静态障碍球和地面。
- 软体：可变形立方体、静态碰撞球和地面。
- 布料：动态碰撞球与布料粒子双向耦合。
- 显示：实体、线框、粒子。
- 控制：开始、暂停、单步、重置、重力、迭代、恢复系数或距离柔度。

`step_ms`、`contacts`、最大速度、约束误差和软体体积误差用于诊断。性能数字依赖运行设备，应现场测量，不作为固定验收阈值。

## 4. 自动化验证

- `tests/test_rigidbody.py`：重力、半径感知碰撞、恢复、摩擦、静态球、动量交换、固定粒子和重置。
- `tests/test_softbody.py`：内部粒子、体积单元、静止长度、压缩恢复、体积柔度、外部球碰撞和重置。
- `tests/test_physics_controller.py`：三类场景、网格同步、控制行为和软体单步。
- `tests/test_app.py`：Qt 面板与真实模拟控制连接。

## 5. 已知限制

- 刚体没有旋转、角动量和完整接触流形。
- 碰撞是离散检测，快速运动可能穿透。
- 软体使用六个粗粒度体积单元，不模拟真实材料参数。
- 顺序 Gauss–Seidel 投影可能打破初始对称；判断恢复效果时应同时看体积、约束误差和整体形变。
- 当前实现优先清晰与可测试，不进行大规模网格性能优化。

## 6. 参考资料边界

- PBD/XPBD 论文用于位置约束、compliance 与拉格朗日乘子公式。
- Blender 刚体、Cloth 与 Soft Body 文档用于参数组织和行为对照。
- Matter.js 用于暂停、单步、碰撞反馈和交互组织参考。

本项目使用原创 Python 教学实现，不复制论文插图、Blender 求解器或 Matter.js 源码。
