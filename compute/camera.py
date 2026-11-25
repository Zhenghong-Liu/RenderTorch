import math

import torch


class OrbitCamera:
    def __init__(
        self,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        pitch: float = 0.0,
        distance: float = 3.0,
        fov_y: float = math.radians(55.0),
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        self.target = torch.tensor(target, dtype=torch.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.distance = distance
        self.fov_y = fov_y
        self.near = near
        self.far = far

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.yaw -= delta_x * 0.008
        self.pitch += delta_y * 0.008
        limit = math.pi / 2 - 0.01
        self.pitch = max(-limit, min(limit, self.pitch))

    def zoom(self, delta: float) -> None:
        self.distance += delta * 0.05
        self.distance = max(0.5, min(50.0, self.distance))

    def position(self) -> torch.Tensor:
        horizontal = self.distance * math.cos(self.pitch)
        return self.target + torch.tensor(
            [
                horizontal * math.sin(self.yaw),
                self.distance * math.sin(self.pitch),
                -horizontal * math.cos(self.yaw),
            ],
            dtype=torch.float32,
        )

    def view_matrix(self) -> torch.Tensor:
        position = self.position()
        forward = self.target - position
        forward = forward / torch.linalg.vector_norm(forward)
        world_up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.linalg.cross(world_up, forward)
        right = right / torch.linalg.vector_norm(right)
        up = torch.linalg.cross(forward, right)
        return torch.tensor(
            [
                [right[0], up[0], forward[0], 0.0],
                [right[1], up[1], forward[1], 0.0],
                [right[2], up[2], forward[2], 0.0],
                [
                    -torch.dot(position, right),
                    -torch.dot(position, up),
                    -torch.dot(position, forward),
                    1.0,
                ],
            ],
            dtype=torch.float32,
        )
