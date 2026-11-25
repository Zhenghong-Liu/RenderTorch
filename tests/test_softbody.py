import unittest

import torch

from physics.common.colliders import PlaneCollider, SphereCollider
from physics.softbody.geometry import build_softbody_cube
from physics.softbody.xpbd import SoftBodyXPBD


def compressed_geometry():
    geometry = build_softbody_cube(resolution=3, size=1.0)
    geometry.state.positions *= 0.6
    return geometry


class SoftBodyGeometryTests(unittest.TestCase):
    def test_cube_contains_surface_internal_particles_and_volume_cells(self):
        geometry = build_softbody_cube(resolution=3, size=1.0)

        self.assertEqual(geometry.state.positions.shape, (27, 3))
        coordinates = geometry.state.positions + 0.5
        interior = torch.all(
            (coordinates > 0.0) & (coordinates < 1.0),
            dim=1,
        )
        self.assertEqual(int(interior.sum()), 1)
        self.assertEqual(len(geometry.triangles), 48)
        self.assertEqual(len(geometry.volume_constraints), 6)
        self.assertAlmostEqual(
            sum(constraint.rest_volume for constraint in geometry.volume_constraints),
            1.0,
            places=6,
        )

    def test_distance_rest_lengths_match_initial_lattice(self):
        geometry = build_softbody_cube(resolution=3, size=1.0)

        for constraint in geometry.distance_constraints:
            a, b = constraint.a, constraint.b
            distance = torch.linalg.vector_norm(
                geometry.state.positions[a] - geometry.state.positions[b]
            )
            self.assertAlmostEqual(distance.item(), constraint.rest_length, places=6)


class SoftBodySimulationTests(unittest.TestCase):
    def test_compressed_cube_recovers_volume_and_reduces_deformation(self):
        geometry = compressed_geometry()
        initial = build_softbody_cube(resolution=3, size=1.0).state.positions
        softbody = SoftBodyXPBD(
            geometry,
            iterations=20,
            volume_compliance=0.0,
        )

        softbody.solve_constraints(1.0 / 120.0)

        self.assertTrue(torch.isfinite(geometry.state.positions).all())
        self.assertGreater(softbody.volume(), 0.9)
        self.assertLess(softbody.last_constraint_error, 0.02)
        self.assertLess(
            torch.max(torch.linalg.vector_norm(geometry.state.positions - initial, dim=1)).item(),
            0.2,
        )

    def test_volume_compliance_controls_correction_strength(self):
        hard_geometry = compressed_geometry()
        soft_geometry = compressed_geometry()
        hard = SoftBodyXPBD(
            hard_geometry,
            iterations=1,
            distance_compliance=1e6,
            volume_compliance=0.0,
        )
        soft = SoftBodyXPBD(
            soft_geometry,
            iterations=1,
            distance_compliance=1e6,
            volume_compliance=1.0,
        )

        hard.solve_constraints(0.1)
        soft.solve_constraints(0.1)

        self.assertLess(hard.last_volume_error, soft.last_volume_error)

    def test_default_cube_is_stable_on_ground(self):
        geometry = build_softbody_cube(
            resolution=3,
            size=1.0,
            center=(0.0, 1.0, 0.0),
        )
        softbody = SoftBodyXPBD(
            geometry,
            fixed_dt=1.0 / 120.0,
            substeps=2,
            colliders=[PlaneCollider(restitution=0.0, friction=0.5)],
        )

        for _ in range(240):
            softbody.single_step()

        self.assertTrue(torch.isfinite(geometry.state.positions).all())
        self.assertTrue(torch.isfinite(geometry.state.velocities).all())
        self.assertGreaterEqual(geometry.state.positions[:, 1].min().item(), -1e-4)
        self.assertLess(softbody.metrics()["max_displacement"], 5.0)
        self.assertGreater(softbody.volume(), 0.5)

    def test_external_sphere_projects_softbody_particles(self):
        geometry = build_softbody_cube(resolution=3, size=1.0)
        collider = SphereCollider(
            center=(0.6, 0.0, 0.0),
            radius=0.2,
            restitution=0.0,
            friction=0.2,
        )
        softbody = SoftBodyXPBD(
            geometry,
            fixed_dt=0.1,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
            colliders=[collider],
        )

        softbody.single_step()

        distances = torch.linalg.vector_norm(
            geometry.state.positions - collider.center,
            dim=1,
        )
        self.assertTrue(torch.all(distances >= collider.radius - 1e-4))

    def test_reset_reproduces_positions_volume_and_multipliers(self):
        geometry = build_softbody_cube(
            resolution=3,
            size=1.0,
            center=(0.0, 1.0, 0.0),
        )
        softbody = SoftBodyXPBD(geometry, colliders=[PlaneCollider()])
        for _ in range(20):
            softbody.single_step()
        first = geometry.state.positions.clone()
        first_volume = softbody.volume()
        first_lambdas = softbody.lambdas.clone()
        first_volume_lambdas = softbody.volume_lambdas.clone()

        softbody.reset()

        self.assertEqual(softbody.step_count, 0)
        torch.testing.assert_close(softbody.lambdas, torch.zeros_like(softbody.lambdas))
        torch.testing.assert_close(
            softbody.volume_lambdas,
            torch.zeros_like(softbody.volume_lambdas),
        )
        for _ in range(20):
            softbody.single_step()

        torch.testing.assert_close(geometry.state.positions, first)
        self.assertAlmostEqual(softbody.volume(), first_volume, places=6)
        torch.testing.assert_close(softbody.lambdas, first_lambdas)
        torch.testing.assert_close(softbody.volume_lambdas, first_volume_lambdas)


if __name__ == "__main__":
    unittest.main()
