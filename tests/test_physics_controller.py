import unittest

import torch

from app.physics_controller import PhysicsController
from physics.cloth.mass_spring import MassSpringCloth
from physics.cloth.pbd import PBDCloth
from physics.cloth.xpbd import XPBDCloth
from physics.rigidbody.sphere import RigidBodySimulation
from physics.softbody.xpbd import SoftBodyXPBD


class PhysicsControllerTests(unittest.TestCase):
    def test_default_demo_is_mass_spring_cloth(self):
        controller = PhysicsController()

        self.assertEqual(controller.simulation_type, "cloth")
        self.assertEqual(controller.cloth_algorithm, "mass_spring")
        self.assertIsInstance(controller.simulation, MassSpringCloth)
        self.assertIn("Cloth", [obj.name for obj in controller.scene.objects])
        self.assertIn("Ground", [obj.name for obj in controller.scene.objects])

    def test_cloth_algorithms_share_initial_state_and_can_switch(self):
        controller = PhysicsController()
        initial = controller.simulation.state.positions.clone()
        constraints = controller.simulation.geometry.constraints

        controller.set_cloth_algorithm("pbd")
        pbd_positions = controller.simulation.state.positions.clone()
        self.assertIsInstance(controller.simulation, PBDCloth)
        controller.set_cloth_algorithm("xpbd")
        xpbd_positions = controller.simulation.state.positions.clone()

        self.assertEqual(controller.simulation.step_count, 0)
        torch.testing.assert_close(pbd_positions, initial)
        torch.testing.assert_close(xpbd_positions, initial)
        self.assertEqual(controller.simulation.geometry.constraints, constraints)

    def test_single_step_updates_physics_and_rendered_mesh(self):
        controller = PhysicsController()
        initial = controller.simulation.state.positions.clone()

        controller.single_step()

        self.assertEqual(controller.simulation.step_count, 1)
        self.assertFalse(torch.equal(controller.simulation.state.positions, initial))
        torch.testing.assert_close(
            controller.dynamic_object.mesh.vertices[:, :3],
            controller.simulation.state.positions,
        )
        self.assertGreaterEqual(controller.metrics()["step_ms"], 0.0)

    def test_pause_resume_single_step_and_reset_are_consistent(self):
        controller = PhysicsController()
        controller.pause()
        controller.advance(1.0)
        self.assertEqual(controller.simulation.step_count, 0)

        controller.single_step()
        self.assertEqual(controller.simulation.step_count, 1)
        controller.resume()
        controller.advance(1.0 / 120.0)
        self.assertEqual(controller.simulation.step_count, 2)

        controller.reset()
        self.assertEqual(controller.simulation.step_count, 0)
        self.assertEqual(controller.simulation.time, 0.0)
        self.assertFalse(controller.simulation.paused)

    def test_gravity_and_iterations_update_active_simulation(self):
        controller = PhysicsController()

        controller.set_gravity(-5.0)
        controller.set_iterations(12)

        self.assertAlmostEqual(controller.simulation.gravity[1].item(), -5.0)
        self.assertEqual(controller.iterations, 12)
        self.assertAlmostEqual(controller.body_simulation.gravity[1].item(), -5.0)
        controller.set_cloth_algorithm("pbd")
        self.assertEqual(controller.simulation.iterations, 12)

    def test_primary_parameter_maps_to_cloth_algorithm(self):
        controller = PhysicsController()
        controller.set_primary_parameter(400.0)
        self.assertAlmostEqual(controller.simulation.structural_stiffness, 400.0)

        controller.set_cloth_algorithm("pbd")
        controller.set_primary_parameter(0.5)
        self.assertAlmostEqual(controller.simulation.structural_stiffness, 0.5)

        controller.set_cloth_algorithm("xpbd")
        controller.set_primary_parameter(0.002)
        self.assertAlmostEqual(controller.simulation.structural_compliance, 0.002)

    def test_all_simulation_types_build_scenes(self):
        controller = PhysicsController()
        controller.set_simulation_type("rigidbody")
        self.assertIsInstance(controller.simulation, RigidBodySimulation)
        self.assertIn("Rigid sphere", [obj.name for obj in controller.scene.objects])

        controller.set_simulation_type("softbody")
        self.assertIsInstance(controller.simulation, SoftBodyXPBD)
        self.assertIn("Soft cube", [obj.name for obj in controller.scene.objects])
        self.assertEqual(len(controller.simulation.geometry.state.positions), 27)

    def test_each_display_mode_renders_visible_finite_pixels(self):
        controller = PhysicsController()
        counts = []
        for mode in ("solid", "wireframe", "particles"):
            framebuffer = controller.render(160, 100, display_mode=mode)
            counts.append(int((framebuffer.object_ids >= 0).sum()))
            visible = framebuffer.object_ids >= 0
            self.assertTrue(torch.isfinite(torch.from_numpy(framebuffer.depth[visible])).all())

        self.assertTrue(all(count > 0 for count in counts))
        self.assertGreater(counts[0], counts[2])

    def test_softbody_single_step_updates_mesh_without_dynamic_body(self):
        controller = PhysicsController()
        controller.set_simulation_type("softbody")
        initial = controller.simulation.state.positions.clone()

        controller.single_step()

        self.assertEqual(controller.simulation.step_count, 1)
        self.assertTrue(torch.isfinite(controller.simulation.state.positions).all())
        self.assertFalse(torch.equal(controller.simulation.state.positions, initial))
        torch.testing.assert_close(
            controller.dynamic_object.mesh.vertices[:, :3],
            controller.simulation.state.positions,
        )

    def test_rigid_body_mesh_tracks_sphere_position(self):
        controller = PhysicsController()
        controller.set_simulation_type("rigidbody")
        initial = controller.body_simulation.body.position.clone()

        controller.single_step()

        self.assertFalse(torch.equal(controller.body_simulation.body.position, initial))
        torch.testing.assert_close(
            controller.body_object.transform[:3, 3],
            controller.body_simulation.body.position,
        )


if __name__ == "__main__":
    unittest.main()
