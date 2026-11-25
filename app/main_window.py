from PySide6.QtCore import QElapsedTimer, QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDockWidget, QMainWindow

from app.controller import AppController
from app.physics_panel import PhysicsPanel
from app.viewport import Viewport


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setFont(QFont("Helvetica"))
        self.controller = AppController()
        self.controller.enable_physics("cloth")
        self.viewport = Viewport(self.controller)
        self.setCentralWidget(self.viewport)
        self.physics_panel = PhysicsPanel(
            self.controller,
            refresh=self.viewport.render_frame,
        )
        self.physics_dock = QDockWidget("物理控制", self)
        self.physics_dock.setWidget(self.physics_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.physics_dock)
        self.setWindowTitle("RenderTorch - 3D 与物理实验平台")
        self.resize(1280, 800)
        self.statusBar().showMessage(
            "左键拖拽：旋转视角    滚轮：缩放    右侧面板：切换物理 Demo"
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._last_frame = QElapsedTimer()
        self._last_frame.start()
        self.timer.start(16)

    def _tick(self) -> None:
        frame_dt = self._last_frame.elapsed() / 1000.0
        self._last_frame.restart()
        self.controller.advance(frame_dt)
        self.viewport.render_frame()
        self.physics_panel.update_metrics()
