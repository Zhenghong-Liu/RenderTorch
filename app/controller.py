import math
from typing import Optional

import torch

from app.physics_controller import PhysicsController
from compute.camera import OrbitCamera
from core.primitives import cube_mesh
from core.scene import Material, Scene, SceneObject
from renderer.pipeline import SoftwareRenderer


class AppController:
    def __init__(self) -> None:
        self._demo_scene = self._create_demo_scene()
        self._camera = OrbitCamera(distance=4.2)
        self.renderer = SoftwareRenderer(self._camera)
        self.physics: Optional[PhysicsController] = None

    @property
    def scene(self) -> Scene:
        return self.physics.scene if self.physics is not None else self._demo_scene

    @property
    def camera(self) -> OrbitCamera:
        return self.physics.camera if self.physics is not None else self._camera

    @staticmethod
    def _create_demo_scene() -> Scene:
        angle_x = -0.35
        angle_y = 0.65
        cosine_x, sine_x = math.cos(angle_x), math.sin(angle_x)
        cosine_y, sine_y = math.cos(angle_y), math.sin(angle_y)
        rotate_x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cosine_x, sine_x, 0.0],
                [0.0, -sine_x, cosine_x, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        rotate_y = torch.tensor(
            [
                [cosine_y, 0.0, -sine_y, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sine_y, 0.0, cosine_y, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        scene = Scene()
        scene.add(
            SceneObject(
                name="Cube",
                mesh=cube_mesh(size=1.8),
                material=Material(color=(92, 164, 235, 255)),
                transform=rotate_x @ rotate_y,
            )
        )
        return scene

    def enable_physics(self, simulation_type: str = "cloth") -> None:
        if self.physics is None:
            self.physics = PhysicsController()
        self.physics.set_simulation_type(simulation_type)

    def set_simulation_type(self, simulation_type: str) -> None:
        if self.physics is None:
            self.enable_physics(simulation_type)
        else:
            self.physics.set_simulation_type(simulation_type)

    def set_cloth_algorithm(self, algorithm: str) -> None:
        if self.physics is None:
            self.enable_physics("cloth")
        self.physics.set_cloth_algorithm(algorithm)

    def set_display_mode(self, display_mode: str) -> None:
        self.physics.set_display_mode(display_mode)

    def set_gravity(self, gravity: float) -> None:
        self.physics.set_gravity(gravity)

    def set_iterations(self, iterations: int) -> None:
        self.physics.set_iterations(iterations)

    def set_primary_parameter(self, value: float) -> None:
        self.physics.set_primary_parameter(value)

    def start(self) -> None:
        self.physics.resume()

    def pause(self) -> None:
        self.physics.pause()

    def reset(self) -> None:
        self.physics.reset()

    def single_step(self) -> None:
        self.physics.single_step()

    def advance(self, frame_dt: float) -> None:
        if self.physics is not None:
            self.physics.advance(frame_dt)

    def metrics(self) -> dict[str, float | int]:
        return self.physics.metrics() if self.physics is not None else {}

    def render(self, width: int, height: int):
        if self.physics is not None:
            return self.physics.render(width, height)
        return self.renderer.render(self.scene, width, height)

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.camera.orbit(delta_x, delta_y)

    def zoom(self, delta: float) -> None:
        self.camera.zoom(delta)
