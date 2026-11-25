import unittest

import torch

from physics.cloth.geometry import ClothGeometry, DistanceConstraint, build_cloth_grid
from physics.cloth.mass_spring import MassSpringCloth
from physics.cloth.pbd import PBDCloth
from physics.cloth.xpbd import XPBDCloth
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


def chain_geometry():
    state = ParticleState(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        ),
        velocities=torch.zeros((3, 3)),
        inverse_mass=torch.ones(3),
    )
    return ClothGeometry(
        state=state,
        constraints=(
            DistanceConstraint(0, 1, 1.0, "structural"),
            DistanceConstraint(1, 2, 1.0, "structural"),
        ),
        triangles=((0, 1, 1),),
        rows=1,
        cols=3,
    )


class XPBDClothTests(unittest.TestCase):
    def test_hard_constraint_matches_pbd_projection(self):
        geometry = stretched_geometry()
        cloth = XPBDCloth(
            geometry,
            iterations=1,
            structural_compliance=0.0,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
        )

        cloth.solve_constraints(0.1)

        torch.testing.assert_close(
            geometry.state.positions,
            torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        )
        self.assertAlmostEqual(cloth.lambdas[0].item(), -0.5, places=6)
        self.assertAlmostEqual(cloth.last_constraint_error, 0.0, places=6)

    def test_pinned_particle_does_not_move_during_projection(self):
        geometry = stretched_geometry(inverse_mass=(0.0, 1.0))
        cloth = XPBDCloth(geometry, iterations=1, structural_compliance=0.0)

        cloth.solve_constraints(0.1)

        torch.testing.assert_close(
            geometry.state.positions,
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )

    def test_compliance_softens_constraint_using_substep(self):
        geometry = stretched_geometry()
        cloth = XPBDCloth(
            geometry,
            iterations=1,
            structural_compliance=1.0,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
        )

        cloth.solve_constraints(0.1)

        correction = 1.0 / 102.0
        torch.testing.assert_close(
            geometry.state.positions,
            torch.tensor([[correction, 0.0, 0.0], [2.0 - correction, 0.0, 0.0]]),
        )
        self.assertAlmostEqual(cloth.lambdas[0].item(), -correction, places=6)
        self.assertAlmostEqual(cloth.last_constraint_error, 0.980392, places=6)

    def test_lagrange_multipliers_persist_across_iterations(self):
        geometry = chain_geometry()
        cloth = XPBDCloth(
            geometry,
            iterations=2,
            structural_compliance=1.0,
        )

        cloth.solve_constraints(0.1)

        self.assertAlmostEqual(cloth.lambdas[0].item(), -0.009900981, places=6)
        self.assertAlmostEqual(cloth.lambdas[1].item(), -0.009900990, places=6)

    def test_lagrange_multipliers_reset_between_solves(self):
        geometry = chain_geometry()
        cloth = XPBDCloth(
            geometry,
            iterations=1,
            structural_compliance=1.0,
        )
        geometry.state.positions[:] = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        cloth.solve_constraints(0.1)
        geometry.state.positions[:] = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        )

        cloth.solve_constraints(0.1)

        self.assertAlmostEqual(cloth.lambdas[0].item(), -2.0 / 102.0, places=6)

    def test_algorithms_share_identical_initial_conditions(self):
        mass_geometry = build_cloth_grid(rows=6, cols=5)
        pbd_geometry = build_cloth_grid(rows=6, cols=5)
        xpbd_geometry = build_cloth_grid(rows=6, cols=5)
        MassSpringCloth(mass_geometry)
        PBDCloth(pbd_geometry)
        XPBDCloth(xpbd_geometry)

        torch.testing.assert_close(
            mass_geometry.state.positions,
            xpbd_geometry.state.positions,
        )
        torch.testing.assert_close(
            mass_geometry.state.inverse_mass,
            xpbd_geometry.state.inverse_mass,
        )
        self.assertEqual(mass_geometry.constraints, xpbd_geometry.constraints)
        self.assertEqual(mass_geometry.triangles, xpbd_geometry.triangles)

    def test_default_grid_is_stable_with_ground_collision(self):
        geometry = build_cloth_grid(rows=8, cols=6, width=3.0, depth=2.0, height=2.0)
        pinned = geometry.state.positions[:6].clone()
        cloth = XPBDCloth(
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

    def test_reset_reproduces_xpbd_simulation(self):
        geometry = build_cloth_grid(rows=5, cols=4, width=2.0, depth=1.5, height=2.0)
        cloth = XPBDCloth(geometry, colliders=[PlaneCollider()])
        for _ in range(20):
            cloth.single_step()
        first = geometry.state.positions.clone()
        first_lambdas = cloth.lambdas.clone()

        cloth.reset()
        torch.testing.assert_close(cloth.lambdas, torch.zeros_like(cloth.lambdas))
        for _ in range(20):
            cloth.single_step()

        torch.testing.assert_close(geometry.state.positions, first)
        torch.testing.assert_close(cloth.lambdas, first_lambdas)


if __name__ == "__main__":
    unittest.main()
