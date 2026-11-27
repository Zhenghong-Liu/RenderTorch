# RenderTorch

基于 Python、PyTorch 与 PySide6 的自定义 3D 软件渲染器和教学级物理实验平台。

当前进度：**M1–M7 已完成**。程序使用 `FrameBuffer → QImage → QPainter` 显示场景，可在同一 Qt 窗口切换刚体、三种布料算法和 XPBD 软体。中文 HTML 教学书包含 **15 章 3D 图形学子书与 10 章物理模拟子书**，提供 **20 个物理实验**，并统一发布到 [GitHub Pages](https://zhenghong-liu.github.io/RenderTorch/)。

## 运行

```bash
conda activate pythonTry
python -m pip install -r requirements.txt
python main_qt.py
```

操作：

- 左键拖拽：旋转相机；滚轮：缩放。
- 右侧物理面板：切换模拟类型、布料算法与实体/线框/粒子显示。
- 开始、暂停、单步、重置：控制模拟。
- 重力、刚度/柔度、迭代次数：修改当前算法参数。
- 底部指标：步数、模拟时间、单步耗时、最大速度、约束误差与接触数。

三种布料算法共享 8×6 网格、固定顶行、地面和动态碰撞球，便于直接比较。软体使用含内部粒子的 3×3×3 格点与六个四面体体积约束。

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -t . -v
```

自动化测试覆盖场景、相机投影、裁剪、光栅化、Qt 显示与交互、公共物理、三种布料、刚体、软体、控制器、图形学与物理教材、全站链接和本地资源；本次提交范围对应 **258 项测试**，当前工作区完整套件（含 14 项未纳入提交的本地回归测试）实测 **272 项通过**。本地旧 Pygame 回归测试未纳入本次提交。

## 教学书

```bash
python book/build.py
python book/physics_build.py
python -m http.server 8000 --directory book
```

打开 `http://localhost:8000`。所有页面、脚本、样式、字体与公式资源均在本地，不依赖 CDN；也可以直接用浏览器打开 `book/index.html`，通过 `file://` 阅读。

全书包含 15 章中文 3D 图形学教材，分为像素与数学、几何建模、光栅管线、光照与完整渲染四个单元；另包含 10 章中文物理模拟教材，分为运动与积分、形状与约束、材料与时间、刚体与软体四个单元，共 20 个可操作物理 Demo。每章包含原理推导、数值算例、伪代码、真实源码映射和实验；公式统一由本地 KaTeX 0.16.47 渲染，不依赖 CDN 或重型前端框架。物理子书的参考资料按基础力学、约束与碰撞、PBD/XPBD、工程参考分组，并明确知识核对、原创内容与当前 Python 能力边界。

构建产物位于 `book/`。发布前依次运行上述两个构建命令，再通过 `git subtree push --prefix=book origin gh-pages` 正常发布到独立 `gh-pages` 分支根目录，使图形学、物理子书与本地资源一次发布。

- 路线图与验收：[docs/development-roadmap.md](docs/development-roadmap.md)
- 参考资料与许可边界：[docs/references.md](docs/references.md)
- 三种布料模型：[docs/cloth-models.md](docs/cloth-models.md)
- 刚体与软体说明：[docs/physics-demos.md](docs/physics-demos.md)

## 架构

```text
app/          Qt 主窗口、物理面板、视口、Controller
core/         Scene、SceneObject、MeshResource、内置图元
compute/      PyTorch 轨道相机与投影矩阵
renderer/     FrameBuffer、裁剪、光栅化与软件渲染管线
physics/      公共模拟、刚体、三种布料与软体
book/         中文 HTML 图形学与物理教材源文件与静态产物
docs/         路线图、参考资料和算法说明
tests/        数值、物理、Qt 与书籍验收测试
```

现有 `main.py`、`render.py`、`solid/` 和 `viewing/` 保留为 Pygame 原型与迁移参考；新功能不向这些旧文件堆叠。`renderer/` 是过渡期渲染包名，用于避免与现有 `render.py` 冲突，旧原型退役后可统一为路线图中的 `render/`。
