import unittest

import torch

from physics.cloth.geometry import ClothGeometry, DistanceConstraint, build_cloth_grid
from physics.cloth.mass_spring import MassSpringCloth
from physics.common.colliders import PlaneCollider
from physics.common.state import ParticleState


def two_particle_geometry():
    state = ParticleState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        velocities=torch.zeros((2, 3)),
        inverse_mass=torch.ones(2),
    )
    constraints = (DistanceConstraint(0, 1, 1.0, "structural"),)
    return ClothGeometry(
        state=state,
        constraints=constraints,
        triangles=((0, 1, 1),),
        rows=1,
        cols=2,
    )


class ClothGeometryTests(unittest.TestCase):
    def test_grid_builds_expected_topology_and_pins(self):
        geometry = build_cloth_grid(
            rows=4,
            cols=3,
            width=2.0,
            depth=1.5,
            height=2.0,
        )

        self.assertEqual(geometry.state.positions.shape, (12, 3))
        self.assertEqual(len(geometry.triangles), 12)
        self.assertEqual(len(geometry.constraints), 39)
        self.assertEqual(
            sum(constraint.kind == "structural" for constraint in geometry.constraints),
            17,
        )
        self.assertEqual(
            sum(constraint.kind == "shear" for constraint in geometry.constraints),
            12,
        )
        self.assertEqual(
            sum(constraint.kind == "bend" for constraint in geometry.constraints),
            10,
        )
        self.assertTrue(torch.all(geometry.state.inverse_mass[:3] == 0))
        self.assertTrue(torch.all(geometry.state.inverse_mass[3:] > 0))
        self.assertTrue(torch.allclose(geometry.state.positions[:, 1], torch.full((12,), 2.0)))

    def test_all_rest_lengths_match_initial_grid(self):
        geometry = build_cloth_grid(rows=5, cols=4, width=3.0, depth=2.0)

        for constraint in geometry.constraints:
            a, b = constraint.a, constraint.b
            distance = torch.linalg.vector_norm(
                geometry.state.positions[a] - geometry.state.positions[b]
            )
            self.assertAlmostEqual(distance.item(), constraint.rest_length, places=6)


class MassSpringTests(unittest.TestCase):
    def test_stretched_spring_applies_equal_and_opposite_forces(self):
        geometry = two_particle_geometry()
        cloth = MassSpringCloth(
            geometry,
            structural_stiffness=10.0,
            spring_damping=0.0,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
        )

        cloth.apply_forces(0.1)

        torch.testing.assert_close(
            geometry.state.velocities,
            torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        )

    def test_constraint_error_reports_relative_stretch(self):
        geometry = two_particle_geometry()
        cloth = MassSpringCloth(geometry)

        cloth.solve_constraints(0.1)

        self.assertAlmostEqual(cloth.last_constraint_error, 1.0, places=6)
        self.assertAlmostEqual(cloth.metrics()["constraint_error"], 1.0, places=6)

    def test_default_grid_is_stable_and_pins_do_not_move(self):
        geometry = build_cloth_grid(rows=8, cols=6, width=3.0, depth=2.0, height=2.0)
        pinned = geometry.state.positions[:6].clone()
        cloth = MassSpringCloth(
            geometry,
            fixed_dt=1.0 / 120.0,
            substeps=2,
            colliders=[PlaneCollider(restitution=0.0, friction=0.5)],
        )

        for _ in range(240):
            cloth.single_step()

        self.assertTrue(torch.isfinite(geometry.state.positions).all())
        self.assertTrue(torch.isfinite(geometry.state.velocities).all())
        self.assertGreaterEqual(geometry.state.positions[:, 1].min().item(), -1e-4)
        self.assertLess(cloth.metrics()["max_displacement"], 10.0)
        torch.testing.assert_close(geometry.state.positions[:6], pinned)

    def test_reset_reproduces_the_same_simulation(self):
        geometry = build_cloth_grid(rows=5, cols=4, width=2.0, depth=1.5, height=2.0)
        cloth = MassSpringCloth(geometry, colliders=[PlaneCollider()])
        for _ in range(20):
            cloth.single_step()
        first = geometry.state.positions.clone()

        cloth.reset()
        for _ in range(20):
            cloth.single_step()

        torch.testing.assert_close(geometry.state.positions, first)


if __name__ == "__main__":
    unittest.main()
