# RenderTorch

基于 Python、PyTorch 与 PySide6 的自定义 3D 软件渲染器和教学级物理实验平台。

当前进度：**M1–M7 已完成**。程序使用 `FrameBuffer → QImage → QPainter` 显示场景，可在同一 Qt 窗口切换刚体、三种布料算法和 XPBD 软体。M7 中文静态教学书已发布到 [GitHub Pages](https://zhenghong-liu.github.io/RenderTorch/)。

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

仓库提交包含 **88 项自动化测试**，覆盖场景、相机投影、裁剪、光栅化、Qt 显示与交互、公共物理、三种布料、刚体、软体、控制器和静态书籍结构；本地旧 Pygame 回归测试未纳入本次提交。

## 教学书

```bash
python book/build.py
python -m http.server 8000 --directory book
```

打开 `http://localhost:8000`。书籍包含 12 个独立章节、真实源码链接和三种布料算法的浏览器交互对比；不使用重型前端框架。构建产物位于 `book/`，通过 `git subtree split --prefix book -b gh-pages` 发布到独立 `gh-pages` 分支根目录。

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
book/         中文 HTML 教学书源文件与静态产物
docs/         路线图、参考资料和算法说明
tests/        数值、物理、Qt 与书籍验收测试
```

现有 `main.py`、`render.py`、`solid/` 和 `viewing/` 保留为 Pygame 原型与迁移参考；新功能不向这些旧文件堆叠。`renderer/` 是过渡期渲染包名，用于避免与现有 `render.py` 冲突，旧原型退役后可统一为路线图中的 `render/`。
