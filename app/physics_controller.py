import time

import torch

from compute.camera import OrbitCamera
from core.primitives import plane_mesh, sphere_mesh
from core.scene import Material, MeshResource, Scene, SceneObject
from physics.cloth.geometry import build_cloth_grid
from physics.cloth.mass_spring import MassSpringCloth
from physics.cloth.pbd import PBDCloth
from physics.cloth.xpbd import XPBDCloth
from physics.common.colliders import PlaneCollider, SphereCollider
from physics.rigidbody.sphere import RigidBodySimulation, SphereBody
from physics.softbody.geometry import build_softbody_cube
from physics.softbody.xpbd import SoftBodyXPBD
from renderer.pipeline import SoftwareRenderer


class PhysicsController:
    def __init__(self) -> None:
        self.simulation_type = "cloth"
        self.cloth_algorithm = "mass_spring"
        self.display_mode = "solid"
        self.gravity_value = -9.81
        self.iterations = 8
        self.primary_parameter = 800.0
        self.camera = OrbitCamera(
            target=(0.0, 1.2, 0.0),
            yaw=0.5,
            pitch=0.3,
            distance=6.0,
        )
        self.renderer = SoftwareRenderer(self.camera)
        self.simulation = None
        self.body_simulation = None
        self.dynamic_object = None
        self.body_object = None
        self.last_step_ms = 0.0
        self.last_contacts = 0
        self._build()

    def set_simulation_type(self, simulation_type: str) -> None:
        if simulation_type not in {"rigidbody", "cloth", "softbody"}:
            raise ValueError(f"unsupported simulation type: {simulation_type}")
        self.simulation_type = simulation_type
        if simulation_type == "cloth":
            defaults = {
                "mass_spring": 800.0,
                "pbd": 0.95,
                "xpbd": 1e-6,
            }
            self.primary_parameter = defaults[self.cloth_algorithm]
        elif simulation_type == "softbody":
            self.primary_parameter = 1e-7
        else:
            self.primary_parameter = 0.5
        self._build()

    def set_cloth_algorithm(self, algorithm: str) -> None:
        if algorithm not in {"mass_spring", "pbd", "xpbd"}:
            raise ValueError(f"unsupported cloth algorithm: {algorithm}")
        self.simulation_type = "cloth"
        self.cloth_algorithm = algorithm
        self.primary_parameter = {
            "mass_spring": 800.0,
            "pbd": 0.95,
            "xpbd": 1e-6,
        }[algorithm]
        self._build()

    def set_gravity(self, gravity: float) -> None:
        self.gravity_value = float(gravity)
        gravity_vector = (0.0, self.gravity_value, 0.0)
        self.simulation.gravity = torch.tensor(gravity_vector, dtype=torch.float32)
        if self.body_simulation is not None:
            self.body_simulation.gravity = torch.tensor(
                gravity_vector,
                dtype=torch.float32,
            )

    def set_iterations(self, iterations: int) -> None:
        self.iterations = max(1, int(iterations))
        if hasattr(self.simulation, "iterations"):
            self.simulation.iterations = self.iterations

    def set_primary_parameter(self, value: float) -> None:
        value = float(value)
        if value < 0:
            raise ValueError("primary parameter must be non-negative")
        self.primary_parameter = value
        self._build()

    def set_display_mode(self, display_mode: str) -> None:
        if display_mode not in {"solid", "wireframe", "particles"}:
            raise ValueError(f"unsupported display mode: {display_mode}")
        self.display_mode = display_mode

    def _build(self) -> None:
        self.scene = Scene()
        self.body_simulation = None
        self.body_object = None
        self.dynamic_object = None
        self.last_contacts = 0
        self.last_step_ms = 0.0
        self._add_ground()
        if self.simulation_type == "cloth":
            self._build_cloth()
        elif self.simulation_type == "rigidbody":
            self._build_rigidbody()
        else:
            self._build_softbody()
        self._sync_meshes()

    def _add_ground(self) -> None:
        self.scene.add(
            SceneObject(
                name="Ground",
                mesh=plane_mesh(size=12.0),
                material=Material(color=(62, 70, 84, 255)),
            )
        )

    def _build_cloth(self) -> None:
        geometry = build_cloth_grid(
            rows=8,
            cols=6,
            width=3.0,
            depth=2.0,
            height=2.5,
        )
        kwargs = {
            "fixed_dt": 1.0 / 120.0,
            "substeps": 2,
            "gravity": (0.0, self.gravity_value, 0.0),
            "colliders": [PlaneCollider(restitution=0.0, friction=0.5)],
        }
        value = self.primary_parameter
        if self.cloth_algorithm == "mass_spring":
            self.simulation = MassSpringCloth(
                geometry,
                structural_stiffness=value,
                shear_stiffness=value * 0.25,
                bend_stiffness=value * 0.025,
                **kwargs,
            )
        elif self.cloth_algorithm == "pbd":
            self.simulation = PBDCloth(
                geometry,
                iterations=self.iterations,
                structural_stiffness=value,
                shear_stiffness=value * (0.8 / 0.95),
                bend_stiffness=value * (0.5 / 0.95),
                **kwargs,
            )
        else:
            self.simulation = XPBDCloth(
                geometry,
                iterations=self.iterations,
                structural_compliance=value,
                shear_compliance=value * 5.0,
                bend_compliance=value * 10.0,
                **kwargs,
            )
        self.dynamic_object = self.scene.add(
            SceneObject(
                name="Cloth",
                mesh=self._mesh_from_positions(
                    geometry.state.positions,
                    geometry.triangles,
                    tuple(
                        (constraint.a, constraint.b)
                        for constraint in geometry.constraints
                    ),
                ),
                material=Material(color=(92, 164, 235, 255)),
            )
        )
        body = SphereBody(
            position=torch.tensor([0.0, 0.45, 0.0]),
            velocity=torch.zeros(3),
            radius=0.45,
            inverse_mass=1.0,
        )
        self.body_simulation = RigidBodySimulation(
            body,
            fixed_dt=1.0 / 120.0,
            substeps=2,
            gravity=(0.0, self.gravity_value, 0.0),
            colliders=[PlaneCollider(restitution=0.0, friction=0.5)],
        )
        self.body_object = self.scene.add(
            SceneObject(
                name="Cloth collider",
                mesh=sphere_mesh(radius=body.radius),
                material=Material(color=(235, 160, 92, 255)),
            )
        )

    def _build_rigidbody(self) -> None:
        obstacle = SphereCollider(
            center=(1.3, 0.5, 0.0),
            radius=0.5,
            restitution=self.primary_parameter,
            friction=0.2,
        )
        self.scene.add(
            SceneObject(
                name="Static sphere",
                mesh=sphere_mesh(radius=obstacle.radius),
                material=Material(color=(110, 118, 135, 255)),
                transform=self._translation(obstacle.center),
            )
        )
        body = SphereBody(
            position=torch.tensor([0.0, 3.0, 0.0]),
            velocity=torch.zeros(3),
            radius=0.5,
            inverse_mass=1.0,
        )
        self.simulation = RigidBodySimulation(
            body,
            fixed_dt=1.0 / 120.0,
            substeps=2,
            gravity=(0.0, self.gravity_value, 0.0),
            colliders=[
                PlaneCollider(
                    restitution=self.primary_parameter,
                    friction=0.2,
                ),
                obstacle,
            ],
        )
        self.body_simulation = self.simulation
        self.body_object = self.scene.add(
            SceneObject(
                name="Rigid sphere",
                mesh=sphere_mesh(radius=body.radius),
                material=Material(color=(235, 160, 92, 255)),
            )
        )

    def _build_softbody(self) -> None:
        geometry = build_softbody_cube(
            resolution=3,
            size=1.0,
            center=(0.0, 2.0, 0.0),
        )
        obstacle = SphereCollider(
            center=(0.9, 0.5, 0.0),
            radius=0.5,
            restitution=0.0,
            friction=0.3,
        )
        self.scene.add(
            SceneObject(
                name="Soft body collider",
                mesh=sphere_mesh(radius=obstacle.radius),
                material=Material(color=(110, 118, 135, 255)),
                transform=self._translation(obstacle.center),
            )
        )
        self.simulation = SoftBodyXPBD(
            geometry,
            iterations=self.iterations,
            distance_compliance=self.primary_parameter,
            volume_compliance=self.primary_parameter * 10.0,
            fixed_dt=1.0 / 120.0,
            substeps=2,
            gravity=(0.0, self.gravity_value, 0.0),
            colliders=[
                PlaneCollider(restitution=0.0, friction=0.5),
                obstacle,
            ],
        )
        self.dynamic_object = self.scene.add(
            SceneObject(
                name="Soft cube",
                mesh=self._mesh_from_positions(
                    geometry.state.positions,
                    geometry.triangles,
                    tuple(
                        (constraint.a, constraint.b)
                        for constraint in geometry.distance_constraints
                    ),
                ),
                material=Material(color=(104, 205, 150, 255)),
            )
        )

    @staticmethod
    def _mesh_from_positions(positions, faces, lines=()):
        homogeneous = torch.cat((positions, torch.ones((len(positions), 1))), dim=1)
        return MeshResource(
            vertices=homogeneous,
            faces=tuple(faces),
            lines=tuple(lines),
        )

    @staticmethod
    def _translation(position):
        transform = torch.eye(4)
        transform[:3, 3] = position
        return transform

    def _sync_meshes(self) -> None:
        if self.dynamic_object is not None:
            self.dynamic_object.mesh.vertices[:, :3] = self.simulation.state.positions
        if self.body_object is not None:
            self.body_object.transform = self._translation(
                self.body_simulation.body.position
            )

    def _couple_cloth_body(self) -> None:
        if self.simulation_type != "cloth":
            return
        state = self.simulation.state
        self.last_contacts = self.body_simulation.collide_with_particles(
            state.positions,
            state.velocities,
            state.inverse_mass,
            restitution=0.0,
            friction=0.2,
        )

    def single_step(self) -> None:
        start = time.perf_counter()
        self.simulation.single_step()
        if self.body_simulation is not None and self.body_simulation is not self.simulation:
            self.body_simulation.single_step()
            self._couple_cloth_body()
        self._sync_meshes()
        self.last_step_ms = (time.perf_counter() - start) * 1000.0

    def advance(self, frame_dt: float) -> None:
        if self.simulation.paused:
            return
        start = time.perf_counter()
        simulation = self.simulation
        simulation.accumulator += max(0.0, min(frame_dt, 0.25))
        steps = 0
        while simulation.accumulator + 1e-9 >= simulation.fixed_dt:
            simulation.accumulator -= simulation.fixed_dt
            self.single_step()
            steps += 1
        if steps:
            self.last_step_ms = (time.perf_counter() - start) * 1000.0 / steps

    def pause(self) -> None:
        self.simulation.pause()

    def resume(self) -> None:
        self.simulation.resume()

    def reset(self) -> None:
        self.simulation.reset()
        if self.body_simulation is not None:
            self.body_simulation.reset()
        self.last_contacts = 0
        self._sync_meshes()

    def render(self, width: int, height: int, display_mode: str | None = None):
        mode = self.display_mode if display_mode is None else display_mode
        return self.renderer.render(self.scene, width, height, display_mode=mode)

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.camera.orbit(delta_x, delta_y)

    def zoom(self, delta: float) -> None:
        self.camera.zoom(delta)

    def metrics(self) -> dict[str, float | int]:
        metrics = self.simulation.metrics()
        metrics["step_ms"] = self.last_step_ms
        metrics["contacts"] = self.last_contacts
        return metrics
