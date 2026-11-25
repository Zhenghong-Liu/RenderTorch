import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QImage, QResizeEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.controller import AppController
from app.main_window import MainWindow
from app.physics_panel import PhysicsPanel
from app.viewport import Viewport


class ViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = AppController()
        self.viewport = Viewport(self.controller)
        self.viewport.resize(400, 300)
        self.viewport.render_frame()

    def tearDown(self):
        self.viewport.close()

    def test_frame_buffer_becomes_qimage_at_widget_size(self):
        self.assertIsInstance(self.viewport.image, QImage)
        self.assertEqual(self.viewport.image.size().width(), 400)
        self.assertEqual(self.viewport.image.size().height(), 300)
        self.assertEqual(self.viewport.image.format(), QImage.Format_RGBA8888)

    def test_left_drag_orbits_camera(self):
        yaw = self.controller.camera.yaw
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
        QTest.mouseMove(self.viewport, QPoint(60, 40), 10)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=QPoint(60, 40))

        self.assertNotEqual(self.controller.camera.yaw, yaw)

    def test_wheel_zooms_camera(self):
        distance = self.controller.camera.distance
        QApplication.sendEvent(
            self.viewport,
            QWheelEvent(
                QPointF(20, 20),
                QPointF(20, 20),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.NoButton,
                Qt.NoModifier,
                Qt.ScrollUpdate,
                False,
            ),
        )

        self.assertGreater(self.controller.camera.distance, distance)

    def test_resize_rerenders_with_new_aspect(self):
        self.viewport.resize(640, 300)
        QApplication.sendEvent(
            self.viewport,
            QResizeEvent(self.viewport.size(), QSize(400, 300)),
        )

        self.assertEqual(self.viewport.image.size().width(), 640)
        self.assertEqual(self.viewport.image.size().height(), 300)


class ControllerTests(unittest.TestCase):
    def test_enable_physics_switches_renderer_and_camera(self):
        controller = AppController()

        controller.enable_physics("softbody")

        self.assertIsNotNone(controller.physics)
        self.assertEqual(controller.physics.simulation_type, "softbody")
        self.assertGreater(int((controller.render(160, 100).object_ids >= 0).sum()), 0)
        yaw = controller.camera.yaw
        controller.orbit(10.0, 5.0)
        self.assertNotEqual(controller.camera.yaw, yaw)

    def test_demo_scene_renders_visible_cube_pixels(self):
        controller = AppController()

        framebuffer = controller.render(160, 100)

        visible = framebuffer.object_ids >= 0
        self.assertGreater(int(visible.sum()), 100)
        self.assertEqual(len(controller.scene.objects), 1)


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_hosts_rendering_viewport(self):
        window = MainWindow()
        window.show()
        self.app.processEvents()

        self.assertIsInstance(window.centralWidget(), Viewport)
        self.assertFalse(window.viewport.image.isNull())
        self.assertIn("RenderTorch", window.windowTitle())
        self.assertIsInstance(window.physics_panel, PhysicsPanel)
        self.assertEqual(window.controller.physics.simulation_type, "cloth")
        window.close()

    def test_physics_panel_controls_connected_simulation(self):
        window = MainWindow()
        window.timer.stop()
        panel = window.physics_panel

        panel.type_combo.setCurrentIndex(panel.type_combo.findData("rigidbody"))
        self.assertEqual(window.controller.physics.simulation_type, "rigidbody")
        panel.type_combo.setCurrentIndex(panel.type_combo.findData("cloth"))
        panel.algorithm_combo.setCurrentIndex(panel.algorithm_combo.findData("xpbd"))
        self.assertEqual(window.controller.physics.cloth_algorithm, "xpbd")
        panel.display_combo.setCurrentIndex(panel.display_combo.findData("particles"))
        self.assertEqual(window.controller.physics.display_mode, "particles")
        panel.gravity_input.setValue(-5.0)
        self.assertAlmostEqual(window.controller.physics.gravity_value, -5.0)

        window.controller.physics.pause()
        before = window.controller.physics.simulation.step_count
        panel.single_button.click()
        self.assertEqual(window.controller.physics.simulation.step_count, before + 1)
        panel.start_button.click()
        self.assertFalse(window.controller.physics.simulation.paused)
        panel.pause_button.click()
        self.assertTrue(window.controller.physics.simulation.paused)
        panel.reset_button.click()
        self.assertEqual(window.controller.physics.simulation.step_count, 0)
        self.assertIn("步数", panel.metrics_label.text())
        window.close()


if __name__ == "__main__":
    unittest.main()
