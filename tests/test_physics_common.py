import math
import unittest

import torch

from physics.common.colliders import PlaneCollider, SphereCollider
from physics.common.simulation import Simulation
from physics.common.state import ParticleState


def make_state(positions, velocities=None, inverse_mass=None):
    positions = torch.tensor(positions, dtype=torch.float32)
    if velocities is None:
        velocities = torch.zeros_like(positions)
    else:
        velocities = torch.tensor(velocities, dtype=torch.float32)
    if inverse_mass is None:
        inverse_mass = torch.ones(len(positions), dtype=torch.float32)
    else:
        inverse_mass = torch.tensor(inverse_mass, dtype=torch.float32)
    return ParticleState(positions, velocities, inverse_mass)


class ParticleStateTests(unittest.TestCase):
    def test_clone_and_reset_restore_initial_values(self):
        state = make_state([[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]], [0.0])
        initial = state.clone()

        state.positions[0, 1] = 5.0
        state.velocities[0, 0] = 9.0
        state.reset(initial)

        torch.testing.assert_close(state.positions, torch.tensor([[0.0, 1.0, 0.0]]))
        torch.testing.assert_close(state.velocities, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(state.inverse_mass, torch.zeros(1))

    def test_kinetic_energy_and_displacement_use_inverse_mass(self):
        state = make_state(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [0.5, 0.0],
        )
        reference = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        self.assertAlmostEqual(state.kinetic_energy().item(), 4.0)
        self.assertAlmostEqual(state.max_displacement(reference).item(), 1.0)


class SimulationTests(unittest.TestCase):
    def test_semi_implicit_integration_applies_gravity(self):
        state = make_state([[0.0, 1.0, 0.0]])
        simulation = Simulation(
            state,
            fixed_dt=1.0 / 60.0,
            substeps=1,
            damping=0.0,
            gravity=(0.0, -10.0, 0.0),
        )

        simulation.single_step()

        self.assertAlmostEqual(state.velocities[0, 1].item(), -10.0 / 60.0, places=5)
        self.assertAlmostEqual(
            state.positions[0, 1].item(),
            1.0 - 10.0 / 60.0**2,
            places=6,
        )
        self.assertEqual(simulation.step_count, 1)

    def test_fixed_step_accumulator_does_not_advance_early(self):
        simulation = Simulation(make_state([[0.0, 1.0, 0.0]]), fixed_dt=1.0 / 60.0)

        simulation.advance(1.0 / 30.0)
        self.assertEqual(simulation.step_count, 2)
        simulation.advance(1.0 / 120.0)
        self.assertEqual(simulation.step_count, 2)
        simulation.advance(1.0 / 120.0)
        self.assertEqual(simulation.step_count, 3)

    def test_substeps_use_smaller_constraint_step(self):
        class CountingSimulation(Simulation):
            def __init__(self, state):
                super().__init__(state, fixed_dt=1.0 / 60.0, substeps=4)
                self.steps = []

            def solve_constraints(self, step):
                self.steps.append(step)

        simulation = CountingSimulation(make_state([[0.0, 1.0, 0.0]]))

        simulation.single_step()

        self.assertEqual(len(simulation.steps), 4)
        for step in simulation.steps:
            self.assertAlmostEqual(step, 1.0 / 240.0, places=7)

    def test_pause_single_step_resume_and_reset_are_consistent(self):
        state = make_state([[0.0, 1.0, 0.0]])
        simulation = Simulation(state, fixed_dt=1.0 / 60.0, gravity=(0.0, -10.0, 0.0))
        simulation.pause()
        simulation.advance(1.0)
        self.assertEqual(simulation.step_count, 0)

        simulation.single_step()
        self.assertEqual(simulation.step_count, 1)
        simulation.resume()
        simulation.advance(1.0 / 60.0)
        self.assertEqual(simulation.step_count, 2)

        simulation.reset()
        self.assertEqual(simulation.step_count, 0)
        self.assertEqual(simulation.time, 0.0)
        self.assertFalse(simulation.paused)
        torch.testing.assert_close(state.positions, torch.tensor([[0.0, 1.0, 0.0]]))
        torch.testing.assert_close(state.velocities, torch.zeros((1, 3)))

    def test_zero_inverse_mass_particle_stays_pinned(self):
        state = make_state([[0.0, 1.0, 0.0]], inverse_mass=[0.0])
        simulation = Simulation(state, fixed_dt=0.1, gravity=(0.0, -10.0, 0.0))

        simulation.single_step()

        torch.testing.assert_close(state.positions, torch.tensor([[0.0, 1.0, 0.0]]))
        torch.testing.assert_close(state.velocities, torch.zeros((1, 3)))

    def test_metrics_report_time_energy_and_displacement(self):
        state = make_state([[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [0.5])
        simulation = Simulation(state, substeps=1, damping=0.0)
        simulation.single_step()

        metrics = simulation.metrics()

        self.assertEqual(metrics["steps"], 1)
        self.assertAlmostEqual(metrics["time"], simulation.fixed_dt, places=7)
        self.assertAlmostEqual(metrics["kinetic_energy"], 4.0, delta=0.02)
        self.assertAlmostEqual(
            metrics["max_displacement"],
            simulation.fixed_dt * 2.0,
            delta=2e-5,
        )
        self.assertTrue(math.isfinite(metrics["max_speed"]))


class ColliderTests(unittest.TestCase):
    def test_ground_projects_penetration_and_applies_restitution(self):
        state = make_state([[0.0, 0.01, 0.0]], [[0.0, -1.0, 0.0]])
        simulation = Simulation(
            state,
            fixed_dt=0.02,
            gravity=(0.0, 0.0, 0.0),
            colliders=[PlaneCollider(restitution=0.5)],
        )

        simulation.single_step()

        self.assertAlmostEqual(state.positions[0, 1].item(), 0.0, places=6)
        self.assertGreater(state.velocities[0, 1].item(), 0.0)

    def test_sphere_projects_particle_to_surface(self):
        state = make_state([[0.5, 0.0, 0.0]])
        simulation = Simulation(
            state,
            gravity=(0.0, 0.0, 0.0),
            colliders=[SphereCollider(center=(0.0, 0.0, 0.0), radius=1.0)],
        )

        simulation.single_step()

        self.assertAlmostEqual(torch.linalg.vector_norm(state.positions[0]).item(), 1.0, places=6)

    def test_sphere_handles_particle_at_center_without_nan(self):
        state = make_state([[0.0, 0.0, 0.0]])
        simulation = Simulation(
            state,
            gravity=(0.0, 0.0, 0.0),
            colliders=[SphereCollider(center=(0.0, 0.0, 0.0), radius=1.0)],
        )

        simulation.single_step()

        self.assertTrue(torch.isfinite(state.positions).all())
        self.assertAlmostEqual(torch.linalg.vector_norm(state.positions[0]).item(), 1.0, places=6)

    def test_collider_does_not_move_pinned_particle(self):
        state = make_state([[0.5, 0.0, 0.0]], inverse_mass=[0.0])
        simulation = Simulation(
            state,
            gravity=(0.0, 0.0, 0.0),
            colliders=[SphereCollider(center=(0.0, 0.0, 0.0), radius=1.0)],
        )

        simulation.single_step()

        torch.testing.assert_close(state.positions, torch.tensor([[0.5, 0.0, 0.0]]))
        torch.testing.assert_close(state.velocities, torch.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
