import math
import unittest

import torch

from compute.camera import OrbitCamera
from compute.projection import perspective, viewport


class CameraTests(unittest.TestCase):
    def test_default_orbit_camera_looks_along_positive_z(self):
        camera = OrbitCamera()

        torch.testing.assert_close(
            camera.position(),
            torch.tensor([0.0, 0.0, -3.0]),
        )
        view_point = torch.tensor([[0.0, 0.0, 0.0, 1.0]]) @ camera.view_matrix()
        torch.testing.assert_close(
            view_point,
            torch.tensor([[0.0, 0.0, 3.0, 1.0]]),
            atol=1e-6,
            rtol=0,
        )

    def test_orbit_clamps_pitch_and_zoom_clamps_distance(self):
        camera = OrbitCamera()
        camera.orbit(100.0, -100.0)
        self.assertLessEqual(camera.pitch, math.pi / 2 - 0.01)
        self.assertGreaterEqual(camera.pitch, -math.pi / 2 + 0.01)

        camera.zoom(-1000.0)
        self.assertAlmostEqual(camera.distance, 0.5)
        camera.zoom(1000.0)
        self.assertAlmostEqual(camera.distance, 50.0)


class ProjectionTests(unittest.TestCase):
    def test_perspective_maps_near_and_far_planes_to_ndc_limits(self):
        matrix = perspective(fov_y=math.pi / 2, aspect=2.0, near=1.0, far=10.0)
        near = torch.tensor([[0.0, 0.0, 1.0, 1.0]]) @ matrix
        far = torch.tensor([[0.0, 0.0, 10.0, 1.0]]) @ matrix

        self.assertAlmostEqual((near[0, 2] / near[0, 3]).item(), -1.0, places=6)
        self.assertAlmostEqual((far[0, 2] / far[0, 3]).item(), 1.0, places=6)

    def test_viewport_flips_y_for_image_coordinates(self):
        matrix = viewport(width=200, height=100)
        top_left = torch.tensor([[-1.0, 1.0, 0.0, 1.0]]) @ matrix
        bottom_right = torch.tensor([[1.0, -1.0, 0.0, 1.0]]) @ matrix

        torch.testing.assert_close(top_left[0, :2], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(
            bottom_right[0, :2], torch.tensor([200.0, 100.0])
        )


if __name__ == "__main__":
    unittest.main()
