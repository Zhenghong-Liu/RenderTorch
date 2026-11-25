from dataclasses import dataclass

import torch

from physics.common.state import ParticleState


@dataclass(frozen=True)
class DistanceConstraint:
    a: int
    b: int
    rest_length: float
    kind: str


@dataclass
class ClothGeometry:
    state: ParticleState
    constraints: tuple[DistanceConstraint, ...]
    triangles: tuple[tuple[int, int, int], ...]
    rows: int
    cols: int


def build_cloth_grid(
    rows: int = 12,
    cols: int = 10,
    width: float = 4.0,
    depth: float = 3.0,
    height: float = 2.5,
    pin_top: bool = True,
) -> ClothGeometry:
    xs = torch.linspace(-width / 2.0, width / 2.0, cols)
    zs = torch.linspace(-depth / 2.0, depth / 2.0, rows)
    positions = torch.tensor(
        [[x, height, z] for z in zs for x in xs],
        dtype=torch.float32,
    )
    inverse_mass = torch.ones(rows * cols, dtype=torch.float32)
    if pin_top:
        inverse_mass[:cols] = 0.0
    state = ParticleState(
        positions=positions,
        velocities=torch.zeros_like(positions),
        inverse_mass=inverse_mass,
    )

    constraints = []
    seen = set()

    def add_edge(a: int, b: int, kind: str) -> None:
        edge = tuple(sorted((a, b)))
        if edge in seen:
            return
        seen.add(edge)
        rest_length = torch.linalg.vector_norm(positions[a] - positions[b]).item()
        constraints.append(DistanceConstraint(a, b, rest_length, kind))

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if col + 1 < cols:
                add_edge(index, index + 1, "structural")
            if row + 1 < rows:
                add_edge(index, index + cols, "structural")
            if row + 1 < rows and col + 1 < cols:
                add_edge(index, index + cols + 1, "shear")
                add_edge(index + 1, index + cols, "shear")
            if col + 2 < cols:
                add_edge(index, index + 2, "bend")
            if row + 2 < rows:
                add_edge(index, index + 2 * cols, "bend")

    triangles = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = a + cols
            d = c + 1
            triangles.extend(((a, c, b), (b, c, d)))

    return ClothGeometry(
        state=state,
        constraints=tuple(constraints),
        triangles=tuple(triangles),
        rows=rows,
        cols=cols,
    )
