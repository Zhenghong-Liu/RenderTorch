import unittest

import numpy as np
import torch

from compute.camera import OrbitCamera
from core.scene import Material, MeshResource, Scene, SceneObject
from renderer.clipping import clip_line, clip_polygon
from renderer.framebuffer import FrameBuffer
from renderer.pipeline import SoftwareRenderer
from renderer.rasterizer import rasterize_line, rasterize_triangle


def make_triangle(name, z, color):
    mesh = MeshResource(
        vertices=torch.tensor(
            [
                [-1.0, -1.0, z, 1.0],
                [1.0, -1.0, z, 1.0],
                [0.0, 1.0, z, 1.0],
            ],
            dtype=torch.float32,
        ),
        faces=((0, 1, 2),),
    )
    return SceneObject(name=name, mesh=mesh, material=Material(color=color))


class FrameBufferTests(unittest.TestCase):
    def test_clear_resets_pixels_depth_and_object_ids(self):
        framebuffer = FrameBuffer(4, 3)
        framebuffer.color[:] = 10
        framebuffer.depth[:] = 1.0
        framebuffer.object_ids[:] = 7

        framebuffer.clear((20, 30, 40, 255))

        self.assertTrue(np.all(framebuffer.color == [20, 30, 40, 255]))
        self.assertTrue(np.all(np.isinf(framebuffer.depth)))
        self.assertTrue(np.all(framebuffer.object_ids == -1))


class ClippingTests(unittest.TestCase):
    def test_line_crossing_near_plane_returns_two_finite_endpoints(self):
        vertices = np.array(
            [[0.0, 0.0, -1.1, 1.0], [0.2, 0.0, 0.2, 1.0]],
            dtype=np.float32,
        )

        clipped = clip_line(vertices)

        self.assertEqual(clipped.shape, (2, 4))
        self.assertTrue(np.isfinite(clipped).all())
        self.assertTrue(np.all(clipped[:, 2] + clipped[:, 3] >= -1e-6))

    def test_polygon_behind_near_plane_is_rejected(self):
        vertices = np.array(
            [[0.0, 0.0, -1.1, 1.0], [0.1, 0.0, -1.1, 1.0], [0.0, 0.1, -1.1, 1.0]],
            dtype=np.float32,
        )

        self.assertEqual(clip_polygon(vertices).shape, (0, 4))

    def test_polygon_crossing_near_plane_is_clipped_without_nan(self):
        vertices = np.array(
            [[0.0, 0.0, -1.1, 1.0], [0.2, 0.0, 0.2, 1.0], [0.0, 0.2, 0.2, 1.0]],
            dtype=np.float32,
        )

        clipped = clip_polygon(vertices)

        self.assertEqual(len(clipped), 4)
        self.assertTrue(np.isfinite(clipped).all())
        self.assertTrue(np.all(clipped[:, 2] + clipped[:, 3] >= -1e-6))


class RasterizerLineTests(unittest.TestCase):
    def test_line_writes_color_depth_and_object_id(self):
        framebuffer = FrameBuffer(5, 5)
        framebuffer.clear()

        rasterize_line(
            framebuffer,
            np.array([[0.5, 0.5, 2.0], [4.5, 4.5, 2.0]]),
            (10, 20, 30, 255),
            7,
        )

        self.assertEqual(int(framebuffer.object_ids[2, 2]), 7)
        np.testing.assert_array_equal(framebuffer.color[2, 2], [10, 20, 30, 255])
        self.assertAlmostEqual(float(framebuffer.depth[2, 2]), 2.0, places=6)


class PipelineTests(unittest.TestCase):
    def test_nearer_triangle_wins_depth_and_id_buffer(self):
        scene = Scene()
        far = scene.add(make_triangle("far", 0.5, (0, 0, 255, 255)))
        near = scene.add(make_triangle("near", -1.0, (255, 0, 0, 255)))
        renderer = SoftwareRenderer(OrbitCamera())

        framebuffer = renderer.render(scene, width=120, height=80)
        pixel = framebuffer.color[40, 60]

        self.assertGreater(int(pixel[0]), int(pixel[2]) * 2)
        self.assertEqual(int(framebuffer.object_ids[40, 60]), near.object_id)
        self.assertAlmostEqual(float(framebuffer.depth[40, 60]), 2.0, places=5)
        self.assertNotEqual(near.object_id, far.object_id)

    def test_wireframe_and_particle_modes_do_not_fill_triangle(self):
        mesh = MeshResource(
            vertices=torch.tensor(
                [
                    [-1.0, -1.0, 0.0, 1.0],
                    [1.0, -1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            faces=((0, 1, 2),),
            lines=((0, 1), (1, 2), (2, 0)),
        )
        scene = Scene()
        scene.add(SceneObject(name="triangle", mesh=mesh))
        renderer = SoftwareRenderer(OrbitCamera())

        solid = renderer.render(scene, width=120, height=80)
        wireframe = renderer.render(scene, width=120, height=80, display_mode="wireframe")
        particles = renderer.render(scene, width=120, height=80, display_mode="particles")

        self.assertGreaterEqual(
            int((solid.object_ids >= 0).sum()),
            int((wireframe.object_ids >= 0).sum()),
        )
        self.assertGreater(
            int((wireframe.object_ids >= 0).sum()),
            int((particles.object_ids >= 0).sum()),
        )
        self.assertEqual(int(solid.object_ids[40, 60]), 0)
        self.assertEqual(int(wireframe.object_ids[40, 60]), -1)
        self.assertEqual(int(particles.object_ids[40, 60]), -1)

    def test_renderer_resize_preserves_framebuffer_shape(self):
        scene = Scene()
        scene.add(make_triangle("triangle", 0.0, (255, 0, 0, 255)))
        renderer = SoftwareRenderer(OrbitCamera())

        small = renderer.render(scene, width=120, height=80)
        wide = renderer.render(scene, width=240, height=80)

        self.assertEqual(small.color.shape, (80, 120, 4))
        self.assertEqual(wide.color.shape, (80, 240, 4))
        self.assertTrue(np.isfinite(wide.depth).any())

    def test_geometry_crossing_near_plane_renders_without_nan(self):
        scene = Scene()
        mesh = MeshResource(
            vertices=torch.tensor(
                [
                    [-0.2, -0.2, -0.2, 1.0],
                    [0.2, -0.2, -0.2, 1.0],
                    [0.0, 0.2, 0.2, 1.0],
                ],
                dtype=torch.float32,
            ),
            faces=((0, 1, 2),),
        )
        scene.add(SceneObject(name="crossing", mesh=mesh))
        renderer = SoftwareRenderer(OrbitCamera(near=3.0, far=10.0))

        framebuffer = renderer.render(scene, width=120, height=80)

        self.assertTrue(np.isfinite(framebuffer.depth).any())
        self.assertTrue(np.isfinite(framebuffer.color).all())


if __name__ == "__main__":
    unittest.main()
