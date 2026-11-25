import unittest

import torch

from physics.cloth.geometry import ClothGeometry, DistanceConstraint, build_cloth_grid
from physics.cloth.mass_spring import MassSpringCloth
from physics.cloth.pbd import PBDCloth
from physics.common.colliders import PlaneCollider
from physics.common.state import ParticleState


def stretched_geometry(inverse_mass=(1.0, 1.0)):
    state = ParticleState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        velocities=torch.zeros((2, 3)),
        inverse_mass=torch.tensor(inverse_mass, dtype=torch.float32),
    )
    return ClothGeometry(
        state=state,
        constraints=(DistanceConstraint(0, 1, 1.0, "structural"),),
        triangles=((0, 1, 1),),
        rows=1,
        cols=2,
    )


class PBDClothTests(unittest.TestCase):
    def test_single_projection_satisfies_distance_constraint(self):
        geometry = stretched_geometry()
        cloth = PBDCloth(
            geometry,
            iterations=1,
            structural_stiffness=1.0,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
        )

        cloth.solve_constraints(0.1)

        torch.testing.assert_close(
            geometry.state.positions,
            torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        )
        self.assertAlmostEqual(cloth.last_constraint_error, 0.0, places=6)

    def test_pinned_particle_does_not_move_during_projection(self):
        geometry = stretched_geometry(inverse_mass=(0.0, 1.0))
        cloth = PBDCloth(geometry, iterations=1, structural_stiffness=1.0)

        cloth.solve_constraints(0.1)

        torch.testing.assert_close(
            geometry.state.positions,
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )

    def test_more_iterations_reduce_constraint_error(self):
        shallow = stretched_geometry()
        deep = stretched_geometry()
        pbd_shallow = PBDCloth(shallow, iterations=1, structural_stiffness=0.5)
        pbd_deep = PBDCloth(deep, iterations=12, structural_stiffness=0.5)

        pbd_shallow.solve_constraints(0.1)
        pbd_deep.solve_constraints(0.1)

        self.assertLess(pbd_deep.last_constraint_error, pbd_shallow.last_constraint_error)

    def test_algorithms_share_identical_initial_conditions(self):
        mass_geometry = build_cloth_grid(rows=6, cols=5)
        pbd_geometry = build_cloth_grid(rows=6, cols=5)
        MassSpringCloth(mass_geometry)
        PBDCloth(pbd_geometry)

        torch.testing.assert_close(
            mass_geometry.state.positions,
            pbd_geometry.state.positions,
        )
        torch.testing.assert_close(
            mass_geometry.state.inverse_mass,
            pbd_geometry.state.inverse_mass,
        )
        self.assertEqual(mass_geometry.constraints, pbd_geometry.constraints)
        self.assertEqual(mass_geometry.triangles, pbd_geometry.triangles)

    def test_default_grid_is_stable_with_ground_collision(self):
        geometry = build_cloth_grid(rows=8, cols=6, width=3.0, depth=2.0, height=2.0)
        pinned = geometry.state.positions[:6].clone()
        cloth = PBDCloth(
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

    def test_reset_reproduces_pbd_simulation(self):
        geometry = build_cloth_grid(rows=5, cols=4, width=2.0, depth=1.5, height=2.0)
        cloth = PBDCloth(geometry, colliders=[PlaneCollider()])
        for _ in range(20):
            cloth.single_step()
        first = geometry.state.positions.clone()

        cloth.reset()
        for _ in range(20):
            cloth.single_step()

        torch.testing.assert_close(geometry.state.positions, first)


if __name__ == "__main__":
    unittest.main()
