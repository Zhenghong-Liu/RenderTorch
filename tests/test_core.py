import unittest

import torch

from core.primitives import cube_mesh, plane_mesh, sphere_mesh
from core.scene import MeshResource, Scene, SceneObject


class SceneTests(unittest.TestCase):
    def test_world_vertices_apply_object_transform(self):
        mesh = MeshResource(
            vertices=torch.tensor([[1.0, 2.0, 3.0, 1.0]]),
            faces=((0, 0, 0),),
        )
        transform = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [10.0, 20.0, 30.0, 1.0],
            ]
        )
        obj = SceneObject(name="scaled", mesh=mesh, transform=transform)

        torch.testing.assert_close(
            obj.world_vertices(),
            torch.tensor([[12.0, 22.0, 33.0, 1.0]]),
        )

    def test_scene_assigns_unique_ids_and_tracks_selection(self):
        mesh = MeshResource(
            vertices=torch.zeros((3, 4)),
            faces=((0, 1, 2),),
        )
        scene = Scene()
        first = scene.add(SceneObject(name="first", mesh=mesh))
        second = scene.add(SceneObject(name="second", mesh=mesh))

        self.assertNotEqual(first.object_id, second.object_id)
        self.assertGreaterEqual(first.object_id, 0)
        scene.select(second.object_id)
        self.assertEqual(scene.selected_id, second.object_id)
        self.assertEqual(scene.selected, second)


class PrimitiveTests(unittest.TestCase):
    def test_cube_mesh_has_eight_vertices_and_six_valid_faces(self):
        mesh = cube_mesh(size=2.0)

        self.assertEqual(mesh.vertices.shape, (8, 4))
        self.assertEqual(len(mesh.faces), 6)
        self.assertTrue(all(len(face) == 4 for face in mesh.faces))
        self.assertTrue(all(0 <= index < 8 for face in mesh.faces for index in face))
        torch.testing.assert_close(
            mesh.vertices[:, :3].abs().max(dim=0).values,
            torch.tensor([1.0, 1.0, 1.0]),
        )
    def test_sphere_mesh_has_valid_triangles_on_requested_radius(self):
        mesh = sphere_mesh(radius=0.75, segments=12, rings=6)

        self.assertEqual(mesh.vertices.shape[1], 4)
        self.assertGreater(len(mesh.faces), 12)
        self.assertTrue(all(len(face) == 3 for face in mesh.faces))
        self.assertTrue(
            all(0 <= index < len(mesh.vertices) for face in mesh.faces for index in face)
        )
        radii = torch.linalg.vector_norm(mesh.vertices[:, :3], dim=1)
        torch.testing.assert_close(radii, torch.full_like(radii, 0.75))

    def test_plane_mesh_lies_on_ground_and_spans_requested_size(self):
        mesh = plane_mesh(size=8.0)

        self.assertEqual(mesh.vertices.shape, (4, 4))
        self.assertTrue(torch.all(mesh.vertices[:, 1] == 0))
        torch.testing.assert_close(
            mesh.vertices[:, [0, 2]].abs().max(dim=0).values,
            torch.tensor([4.0, 4.0]),
        )


if __name__ == "__main__":
    unittest.main()
