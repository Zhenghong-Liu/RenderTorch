# RenderTorch 参考资料索引

> 整理日期：2026-11-26
> 用途：约束阶段一 Qt 最小闭环与后续物理 Demo 的实现边界。只记录思想与接口经验，不复制受版权保护的正文、插图或实现代码。

## 1. 当前实现依据

| 阶段 | 主依据 | 本项目中的用途 | 使用边界 |
|---|---|---|---|
| M1：Qt 最小闭环 | [Qt for Python QWidget](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QWidget.html)、[QImage](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QImage.html)、[QPainter](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QPainter.html) | 自定义视口、`paintEvent`、`resizeEvent`、鼠标事件、FrameBuffer 到图像的显示 | 使用 Qt 官方 API；像素格式采用 `Format_RGBA8888`，显示时让 QImage 持有帧缓冲数据生命周期 |
| M1：PyTorch 计算 | [PyTorch Tensor 属性](https://github.com/pytorch/pytorch/blob/main/docs/source/tensor_attributes.md)、[MPS 说明](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/mps.md) | `torch.float32` 矩阵、设备选择、CPU 数值测试 | 新软件光栅化链路默认使用 CPU，避免把像素数组与加速器设备混用；旧原型仍保留设备选择逻辑 |
| M2-M5：公共物理与布料 | [Matter.js Body](https://brm.io/matter-js/docs/classes/Body.html)、[Constraint](https://brm.io/matter-js/docs/classes/Constraint.html)、[Collision](https://brm.io/matter-js/docs/classes/Collision.html)、[Engine.update](https://brm.io/matter-js/docs/files/src_core_Engine.js.html)、[Runner](https://brm.io/matter-js/docs/classes/Runner.html) | 固定时间步、力/冲量、约束刚度与阻尼、碰撞法线、暂停/单步/重置的交互组织 | Matter.js 是二维引擎，只参考流程与参数直觉，不作为三维布料算法依据 |
| M3-M6：参数与行为对照 | [Blender Cloth 简介](https://docs.blender.org/manual/en/5.3/physics/cloth/introduction.html)、[布料物理属性](https://docs.blender.org/manual/zh-hans/5.0/physics/cloth/settings/physical_properties.html)、[Soft Body](https://docs.blender.org/manual/zh-hans/4.1/physics/soft_body/introduction.html)、[刚体约束](https://docs.blender.org/manual/zh-hans/5.1/physics/rigid_body/constraints/introduction.html) | 拉伸/剪切/弯曲参数、软体目标、碰撞与约束调试面板 | 只做行为验收对照；不照搬 Blender 求解器，不把商业引擎参数直接映射为 PBD/XPBD 参数 |
| M4：PBD | [Position Based Dynamics (2007)](https://doi.org/10.1016/j.jvcir.2007.01.005) | 位置投影、约束迭代、碰撞投影 | 算法主依据；引用论文，不转载图表或大段原文 |
| M5：XPBD | [XPBD 作者 PDF](https://matthias-research.github.io/pages/publications/XPBD.pdf)、[XPBD DOI](https://doi.org/10.1145/1399504.1360662) | compliance、拉格朗日乘子、子步与刚度控制 | 算法主依据；用公式与原创伪代码讲解，保留作者、年份与 DOI |
| M6：刚体与软体 | [Blender 刚体简介](https://docs.blender.org/manual/zh-hans/5.1/physics/rigid_body/introduction.html)、[Blender Soft Body](https://docs.blender.org/manual/zh-hans/4.1/physics/soft_body/introduction.html)、PBD/XPBD 论文 | 恢复系数、摩擦、软体目标、距离与体积约束、碰撞调试 | 只参考行为与算法直觉；本项目刚体不含旋转，软体使用六个四面体单元，不照搬商业求解器 |
| M7：HTML 讲解书 | [Hands-On Modern RL 指标页](https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/metrics) | 信息组织、源码与图示结合、指标展示、交互式讲解 | 只参考结构，不复制页面素材；书籍内容必须链接当前真实源码 |

## 2. 许可证与署名

- **Matter.js**：MIT。若后续复用代码，保留 Liam Brummitt 的版权与许可声明；许可证来源：[matter-js](https://github.com/liabru/matter-js)。
- **Blender 文档**：CC BY-SA 4.0。若复用文字或图像，署名 Blender Documentation Team、附来源链接，并按相同方式共享衍生内容；许可证来源：[Blender Copyright](https://docs.blender.org/manual/zh-hans/3.6/copyright.html)。
- **PBD / XPBD 论文**：未发现开放再许可，按默认版权处理。课程中只给出原创概述、公式引用与伪代码，不复制论文插图或大段文字。
- **Qt / PyTorch**：使用官方公开 API，不把文档示例视为可自由再分发的项目素材。

## 3. 从资料到项目的设计约束

1. 核心场景与相机数学不导入 Qt、Pygame 或窗口 API。
2. FrameBuffer 独立保存 RGBA、深度和物体 ID；Qt 只负责显示与输入。
3. 相机采用目标点、偏航、俯仰和距离的轨道模型，窗口变化只更新投影宽高比与视口。
4. 三角形先做齐次裁剪，再做透视除法、屏幕映射和深度测试，避免近裁剪面产生 NaN。
5. 后续三种布料算法共用粒子、约束、碰撞体和固定步长，但每个算法独立成文件、独立测试。
6. 每个教学 Demo 都应提供可重复初始状态、重置、暂停、单步和误差指标。

## 4. 当前环境核验

- Python：`pythonTry` / Python 3.11.9。
- PyTorch：2.3.1；MPS 是否可用由运行时检查决定。
- NumPy：1.26.4。
- PySide6：6.12.0（阶段一安装）。
- Pygame：2.6.1，仅保留给现有原型与回归测试，不作为新架构运行依赖。
