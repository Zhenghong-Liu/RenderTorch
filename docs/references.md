# RenderTorch 参考资料索引

> 整理日期：2026-11-27
> 用途：记录 Qt 最小闭环、图形学子书与物理子书的实现边界。只核对知识与接口经验，不复制受版权保护的正文、插图或实现代码。

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

## 2. v2 图形学子书参考资料

完整机器可读索引见 `book/references.json`。以下资料只用于知识点核对、教学顺序与许可边界，不复制受版权保护的正文、插图或实现代码。

| 资料 | 作者 | 许可 | 本书用途与边界 |
|---|---|---|---|
| [Introduction to Computer Graphics, Version 1.4](https://math.hws.edu/graphicsbook/) | David J. Eck | CC BY-NC-SA 4.0 | 核对图形学知识点与教学顺序；不复制正文或图片。 |
| [LearnOpenGL](https://learnopengl.com/About) | Joey de Vries | 图片 CC BY 4.0；代码示例 CC BY-NC 4.0 | 核对 OpenGL 管线、纹理和阴影概念；图解全部原创，不下载或复用站点图片。 |
| [Physically Based Rendering, 4th edition](https://pbr-book.org/4ed/contents) | Matt Pharr、Wenzel Jakob、Greg Humphreys | CC BY-NC-ND 4.0 | 引用材质、光照与渲染知识；不复制图片、正文或改编内容。 |
| [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) | Peter Shirley、Trevor David Black、Steve Hollasch | All rights reserved | 仅作学习参考；不复制正文、代码示例或图片。 |
| [University Physics Volume 1](https://openstax.org/books/university-physics-volume-1/pages/preface) | OpenStax | CC BY 4.0 | 用于物理子书，核对牛顿力学、能量、动量与振动。 |
| [Physically Based Modeling, SIGGRAPH 2001 Course Notes](https://www.cs.cmu.edu/~baraff/sigcourse/) | David Baraff、Andrew Witkin | Course notes | 用于物理子书，核对约束、碰撞与数值积分。 |
| [Foundations of Physically Based Modeling and Animation](https://books.google.com/books?vid=ISBN9781568818770) | John House、John Keyser | Commercial publication | 仅作商业教材参考；不复制正文、图片或代码。 |
| [Position Based Dynamics](https://doi.org/10.1016/j.jvcir.2007.01.005) | Matthias Müller et al. | Paper citation | 用于物理子书，只引用算法、公式出处与 DOI。 |
| [XPBD](https://doi.org/10.1145/1399504.1360662) | Miles Macklin et al. | Paper citation | 用于物理子书，只引用算法、公式出处与 DOI。 |

图形学子书另使用本地 KaTeX 0.16.47 渲染公式；其 MIT 许可证随静态资源保存在 `book/assets/vendor/katex/LICENSE`。

## 3. 物理子书参考资料

物理子书包含 10 章、20 个 Demo。机器可读索引见 [book/physics_references.json](../book/physics_references.json)，构建后生成 `book/physics/references.html` 参考资料页。四类分组依次为基础力学、约束与碰撞、PBD/XPBD、工程参考。

| 分组 | 资料 | 作者 | 许可 | 本书用途与边界 |
|---|---|---|---|---|
| 基础力学 | [University Physics Volume 1](https://openstax.org/books/university-physics-volume-1/pages/preface) | OpenStax | CC BY 4.0 | 核对牛顿力学、能量、动量与振动。 |
| 约束与碰撞 | [Physically Based Modeling, SIGGRAPH 2001 Course Notes](https://www.cs.cmu.edu/~baraff/sigcourse/) | David Baraff、Andrew Witkin | Course notes | 核对约束、碰撞与数值积分。 |
| 约束与碰撞 | [Foundations of Physically Based Modeling and Animation](https://books.google.com/books?vid=ISBN9781568818770) | John House、John Keyser | Commercial publication | 仅作商业教材的建模与动画知识参考。 |
| PBD/XPBD | [Position Based Dynamics](https://doi.org/10.1016/j.jvcir.2007.01.005) | Matthias Müller et al. | Paper citation | 核对位置投影、约束迭代与碰撞投影，引用算法与 DOI。 |
| PBD/XPBD | [XPBD: Position-Based Simulation of Compliant Constrained Dynamics](https://doi.org/10.1145/1399504.1360662) | Miles Macklin et al. | Paper citation | 核对柔度、拉格朗日乘子、子步与刚度控制，引用算法与 DOI。 |
| 工程参考 | [Matter.js Engine](https://brm.io/matter-js/docs/files/src_core_Engine.js.html) | Liam Brummitt | MIT | 核对固定时间步、力/冲量与交互流程；Matter.js 是二维引擎，不作为三维布料算法依据。 |
| 工程参考 | [Blender 物理文档](https://docs.blender.org/manual/en/5.3/physics/cloth/introduction.html) | Blender Documentation Team | CC BY-SA 4.0 | 核对布料、软体、刚体与约束行为；不照搬求解器或直接把商业引擎参数映射为 PBD/XPBD 参数。 |

### 原创内容与使用边界

- 所有上述资料只用于**只核对知识，不复制正文/图片/代码**。
- 中文讲解、公式推导、数值算例、伪代码与 Canvas/SVG 图示由本教材重新组织；不转载商业教材或论文插图，也不复制大段文字。
- 开放许可不等于可以省略署名。后续若复用开放许可素材，必须保留作者、来源和许可声明，并遵守相应许可条件。
- 物理子书另使用本地 KaTeX 0.16.47 渲染公式，采用 MIT 许可；完整许可证见 [book/assets/vendor/katex/LICENSE](../book/assets/vendor/katex/LICENSE)。

### 当前 Python 能力边界

- 公共质点路径提供状态、固定时间步、半隐式积分、静态平面/球碰撞和诊断；Python 调用方状态使用 `float32`，浏览器 Demo 使用 JavaScript 双精度，不承诺逐位相同。
- 布料已有质量弹簧以及结构、剪切、弯曲距离约束的 PBD/XPBD；当前没有独立弯曲约束、自碰撞或生产级接触流形。
- 软体使用含内部粒子的 3×3×3 格点与六个四面体体积约束；浏览器中的二维面积 XPBD 是教学类比，不表示 Python 已有独立面积约束。
- 当前 Python 刚体是只含平移的球体，没有角速度、转动惯量、力矩、滚动、动态刚体对或 CCD；浏览器旋转实验用于解释概念。
- 本书浏览器实验用于可推导、可操作和可验证的教学，不是生产物理引擎，也不把概念 Demo 描述为已有 Python 实现。

## 4. 许可证与署名

- **Matter.js**：MIT。若后续复用代码，保留 Liam Brummitt 的版权与许可声明；许可证来源：[matter-js](https://github.com/liabru/matter-js)。
- **Blender 文档**：CC BY-SA 4.0。若复用文字或图像，署名 Blender Documentation Team、附来源链接，并按相同方式共享衍生内容；许可证来源：[Blender Copyright](https://docs.blender.org/manual/zh-hans/3.6/copyright.html)。
- **PBD / XPBD 论文**：未发现开放再许可，按默认版权处理。课程中只给出原创概述、公式引用与伪代码，不复制论文插图或大段文字。
- **Qt / PyTorch**：使用官方公开 API，不把文档示例视为可自由再分发的项目素材。

## 5. 从资料到项目的设计约束

1. 核心场景与相机数学不导入 Qt、Pygame 或窗口 API。
2. FrameBuffer 独立保存 RGBA、深度和物体 ID；Qt 只负责显示与输入。
3. 相机采用目标点、偏航、俯仰和距离的轨道模型，窗口变化只更新投影宽高比与视口。
4. 三角形先做齐次裁剪，再做透视除法、屏幕映射和深度测试，避免近裁剪面产生 NaN。
5. 后续三种布料算法共用粒子、约束、碰撞体和固定步长，但每个算法独立成文件、独立测试。
6. 每个教学 Demo 都应提供可重复初始状态、重置、暂停、单步和误差指标。

## 6. 当前环境核验

- Python：`pythonTry` / Python 3.11.9。
- PyTorch：2.3.1；MPS 是否可用由运行时检查决定。
- NumPy：1.26.4。
- PySide6：6.12.0（阶段一安装）。
- Pygame：2.6.1，仅保留给现有原型与回归测试，不作为新架构运行依赖。
