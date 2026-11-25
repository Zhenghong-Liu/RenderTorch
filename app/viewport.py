from typing import Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QWidget

from app.controller import AppController


class Viewport(QWidget):
    def __init__(self, controller: AppController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self.image = QImage()
        self._last_position: Optional[QPoint] = None
        self._dragging = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)

    def render_frame(self) -> None:
        width = max(1, self.width())
        height = max(1, self.height())
        framebuffer = self.controller.render(width, height)
        self.image = QImage(
            framebuffer.color.data,
            width,
            height,
            framebuffer.color.strides[0],
            QImage.Format_RGBA8888,
        ).copy()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self.image.isNull():
            painter.fillRect(self.rect(), self.palette().window())
        else:
            painter.drawImage(0, 0, self.image)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.render_frame()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_position = event.position().toPoint()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging or self._last_position is None:
            return
        position = event.position().toPoint()
        delta = position - self._last_position
        self._last_position = position
        self.controller.orbit(delta.x(), delta.y())
        self.render_frame()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_position = None
            event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.controller.zoom(event.angleDelta().y())
        self.render_frame()
        event.accept()
