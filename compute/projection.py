import math

import torch


def perspective(
    fov_y: float,
    aspect: float,
    near: float,
    far: float,
) -> torch.Tensor:
    scale_y = 1.0 / math.tan(fov_y / 2.0)
    scale_x = scale_y / aspect
    return torch.tensor(
        [
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, scale_y, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (far - near), 1.0],
            [0.0, 0.0, -2.0 * far * near / (far - near), 0.0],
        ],
        dtype=torch.float32,
    )


def viewport(width: int, height: int) -> torch.Tensor:
    return torch.tensor(
        [
            [width / 2.0, 0.0, 0.0, 0.0],
            [0.0, -height / 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [width / 2.0, height / 2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
