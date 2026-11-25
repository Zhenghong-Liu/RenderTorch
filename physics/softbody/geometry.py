from dataclasses import dataclass
from itertools import product

import torch

from physics.cloth.geometry import DistanceConstraint
from physics.common.state import ParticleState


@dataclass(frozen=True)
class VolumeConstraint:
    indices: tuple[int, int, int, int]
    rest_volume: float


@dataclass
class SoftBodyGeometry:
    state: ParticleState
    distance_constraints: tuple[DistanceConstraint, ...]
    volume_constraints: tuple[VolumeConstraint, ...]
    triangles: tuple[tuple[int, int, int], ...]


def signed_tetrahedron_volume(positions, indices) -> torch.Tensor:
    a, b, c, d = indices
    return torch.dot(
        positions[b] - positions[a],
        torch.linalg.cross(positions[c] - positions[a], positions[d] - positions[a]),
    ) / 6.0


def build_softbody_cube(
    resolution: int = 3,
    size: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SoftBodyGeometry:
    if resolution < 3 or resolution % 2 == 0:
        raise ValueError("resolution must be an odd integer of at least 3")
    if size <= 0:
        raise ValueError("size must be positive")

    coordinates = list(product(range(resolution), repeat=3))
    index_by_coordinate = {
        coordinate: index for index, coordinate in enumerate(coordinates)
    }
    positions = torch.tensor(
        [
            [
                center[0] + (coordinate[0] / (resolution - 1) - 0.5) * size,
                center[1] + (coordinate[1] / (resolution - 1) - 0.5) * size,
                center[2] + (coordinate[2] / (resolution - 1) - 0.5) * size,
            ]
            for coordinate in coordinates
        ],
        dtype=torch.float32,
    )
    state = ParticleState(
        positions=positions,
        velocities=torch.zeros_like(positions),
        inverse_mass=torch.ones(len(positions), dtype=torch.float32),
    )

    distance_constraints = []
    offsets = [offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0)]
    for index, coordinate in enumerate(coordinates):
        for offset in offsets:
            neighbor = tuple(value + delta for value, delta in zip(coordinate, offset))
            if any(value < 0 or value >= resolution for value in neighbor):
                continue
            other = index_by_coordinate[neighbor]
            if other <= index:
                continue
            kind = "structural" if sum(abs(value) for value in offset) == 1 else "shear"
            distance_constraints.append(
                DistanceConstraint(
                    index,
                    other,
                    torch.linalg.vector_norm(positions[index] - positions[other]).item(),
                    kind,
                )
            )

    triangles = []
    for axis in range(3):
        other_axes = [value for value in range(3) if value != axis]
        for side in (0, resolution - 1):
            for u in range(resolution - 1):
                for v in range(resolution - 1):
                    corners = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        coordinate = [0, 0, 0]
                        coordinate[axis] = side
                        coordinate[other_axes[0]] = u + du
                        coordinate[other_axes[1]] = v + dv
                        corners.append(index_by_coordinate[tuple(coordinate)])
                    a, b, c, d = corners
                    triangles.extend(((a, b, c), (a, c, d)))

    maximum = resolution - 1
    corner = lambda x, y, z: index_by_coordinate[(x, y, z)]
    tetrahedra = (
        (corner(0, 0, 0), corner(maximum, 0, 0), corner(0, 0, maximum), corner(maximum, maximum, maximum)),
        (corner(0, 0, 0), corner(0, 0, maximum), corner(0, maximum, 0), corner(maximum, maximum, maximum)),
        (corner(0, 0, 0), corner(0, maximum, 0), corner(0, maximum, maximum), corner(maximum, maximum, maximum)),
        (corner(0, 0, 0), corner(0, maximum, maximum), corner(maximum, maximum, 0), corner(maximum, maximum, maximum)),
        (corner(0, 0, 0), corner(maximum, 0, 0), corner(maximum, 0, maximum), corner(maximum, maximum, maximum)),
        (corner(0, 0, 0), corner(maximum, 0, maximum), corner(0, 0, maximum), corner(maximum, maximum, maximum)),
    )
    volume_constraints = []
    for indices in tetrahedra:
        if signed_tetrahedron_volume(positions, indices).item() < 0:
            indices = (indices[0], indices[2], indices[1], indices[3])
        volume_constraints.append(
            VolumeConstraint(indices, signed_tetrahedron_volume(positions, indices).item())
        )

    return SoftBodyGeometry(
        state=state,
        distance_constraints=tuple(distance_constraints),
        volume_constraints=tuple(volume_constraints),
        triangles=tuple(triangles),
    )
