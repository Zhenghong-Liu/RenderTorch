import unittest

import torch

from physics.common.colliders import PlaneCollider, SphereCollider
from physics.rigidbody.sphere import RigidBodySimulation, SphereBody


def make_body(position=(0.0, 1.0, 0.0), velocity=(0.0, 0.0, 0.0), radius=0.5):
    return SphereBody(
        position=torch.tensor(position, dtype=torch.float32),
        velocity=torch.tensor(velocity, dtype=torch.float32),
        radius=radius,
        inverse_mass=1.0,
    )


class RigidBodyTests(unittest.TestCase):
    def test_gravity_integrates_sphere_center(self):
        body = make_body()
        simulation = RigidBodySimulation(
            body,
            fixed_dt=0.1,
            substeps=1,
            gravity=(0.0, -10.0, 0.0),
            damping=0.0,
        )

        simulation.single_step()

        self.assertAlmostEqual(body.velocity[1].item(), -1.0, places=6)
        self.assertAlmostEqual(body.position[1].item(), 0.9, places=6)

    def test_ground_collision_uses_sphere_radius_and_restitution(self):
        body = make_body(position=(0.0, 0.51, 0.0), velocity=(0.0, -1.0, 0.0))
        simulation = RigidBodySimulation(
            body,
            fixed_dt=0.1,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
            colliders=[PlaneCollider(restitution=0.5, friction=0.0)],
        )

        simulation.single_step()

        self.assertAlmostEqual(body.position[1].item(), 0.5, places=6)
        self.assertAlmostEqual(body.velocity[1].item(), 0.5, places=6)

    def test_ground_friction_reduces_tangential_velocity(self):
        body = make_body(
            position=(0.0, 0.51, 0.0),
            velocity=(2.0, -1.0, 0.0),
        )
        simulation = RigidBodySimulation(
            body,
            fixed_dt=0.1,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
            colliders=[PlaneCollider(restitution=0.0, friction=0.5)],
        )

        simulation.single_step()

        self.assertAlmostEqual(body.position[1].item(), 0.5, places=6)
        self.assertAlmostEqual(body.velocity[0].item(), 1.0, places=6)
        self.assertAlmostEqual(body.velocity[1].item(), 0.0, places=6)

    def test_static_sphere_projects_to_combined_radius(self):
        body = make_body(position=(0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))
        simulation = RigidBodySimulation(
            body,
            fixed_dt=0.1,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
            damping=0.0,
            colliders=[
                SphereCollider(
                    center=(1.0, 0.0, 0.0),
                    radius=0.5,
                    restitution=0.5,
                    friction=0.0,
                )
            ],
        )

        simulation.single_step()

        self.assertAlmostEqual(body.position[0].item(), 0.0, places=6)
        self.assertAlmostEqual(body.velocity[0].item(), -0.5, places=6)

    def test_sphere_particle_collision_exchanges_momentum(self):
        body = make_body(position=(0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))
        simulation = RigidBodySimulation(body, gravity=(0.0, 0.0, 0.0))
        positions = torch.tensor([[0.4, 0.0, 0.0]])
        velocities = torch.zeros((1, 3))
        inverse_mass = torch.ones(1)

        contacts = simulation.collide_with_particles(
            positions,
            velocities,
            inverse_mass,
            restitution=0.0,
            friction=0.0,
        )

        self.assertEqual(contacts, 1)
        torch.testing.assert_close(body.position, torch.tensor([-0.05, 0.0, 0.0]))
        torch.testing.assert_close(positions, torch.tensor([[0.45, 0.0, 0.0]]))
        torch.testing.assert_close(body.velocity, torch.tensor([0.5, 0.0, 0.0]))
        torch.testing.assert_close(velocities, torch.tensor([[0.5, 0.0, 0.0]]))

    def test_pinned_particle_transfers_impulse_without_moving(self):
        body = make_body(position=(0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))
        simulation = RigidBodySimulation(body, gravity=(0.0, 0.0, 0.0))
        positions = torch.tensor([[0.4, 0.0, 0.0]])
        velocities = torch.zeros((1, 3))
        inverse_mass = torch.zeros(1)

        simulation.collide_with_particles(
            positions,
            velocities,
            inverse_mass,
            restitution=0.0,
            friction=0.0,
        )

        torch.testing.assert_close(positions, torch.tensor([[0.4, 0.0, 0.0]]))
        torch.testing.assert_close(velocities, torch.zeros((1, 3)))
        torch.testing.assert_close(body.position, torch.tensor([-0.1, 0.0, 0.0]))
        torch.testing.assert_close(body.velocity, torch.zeros(3))

    def test_reset_restores_body_and_simulation_controls(self):
        body = make_body()
        simulation = RigidBodySimulation(body, fixed_dt=0.1, substeps=1)
        for _ in range(5):
            simulation.single_step()
        moved = body.position.clone()

        simulation.pause()
        simulation.advance(1.0)
        self.assertEqual(simulation.step_count, 5)
        simulation.reset()

        self.assertEqual(simulation.step_count, 0)
        self.assertEqual(simulation.time, 0.0)
        self.assertFalse(simulation.paused)
        torch.testing.assert_close(body.position, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(body.velocity, torch.zeros(3))
        self.assertFalse(torch.equal(body.position, moved))


if __name__ == "__main__":
    unittest.main()
