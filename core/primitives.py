import math

import torch

from core.scene import MeshResource


def cube_mesh(size: float = 1.0) -> MeshResource:
    half = size / 2.0
    vertices = torch.tensor(
        [
            [-half, -half, -half, 1.0],
            [half, -half, -half, 1.0],
            [half, half, -half, 1.0],
            [-half, half, -half, 1.0],
            [-half, -half, half, 1.0],
            [half, -half, half, 1.0],
            [half, half, half, 1.0],
            [-half, half, half, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 2, 6, 7),
    )
    return MeshResource(vertices=vertices, faces=faces)


def plane_mesh(size: float = 10.0) -> MeshResource:
    half = size / 2.0
    vertices = torch.tensor(
        [
            [-half, 0.0, -half, 1.0],
            [half, 0.0, -half, 1.0],
            [half, 0.0, half, 1.0],
            [-half, 0.0, half, 1.0],
        ],
        dtype=torch.float32,
    )
    return MeshResource(vertices=vertices, faces=((0, 1, 2, 3),))


def sphere_mesh(radius: float = 0.5, segments: int = 16, rings: int = 8) -> MeshResource:
    if segments < 3 or rings < 2:
        raise ValueError("segments must be at least 3 and rings at least 2")
    vertices = [[0.0, radius, 0.0, 1.0]]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(
                [
                    radius * math.sin(phi) * math.cos(theta),
                    radius * math.cos(phi),
                    radius * math.sin(phi) * math.sin(theta),
                    1.0,
                ]
            )
    bottom = len(vertices)
    vertices.append([0.0, -radius, 0.0, 1.0])
    faces = []
    for segment in range(segments):
        current = 1 + segment
        following = 1 + (segment + 1) % segments
        faces.append((0, current, following))
    for ring in range(rings - 2):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            following = (segment + 1) % segments
            a = first + segment
            b = first + following
            c = second + segment
            d = second + following
            faces.extend(((a, c, b), (b, c, d)))
    last_ring = 1 + (rings - 2) * segments
    for segment in range(segments):
        current = last_ring + segment
        following = last_ring + (segment + 1) % segments
        faces.append((bottom, following, current))
    return MeshResource(
        vertices=torch.tensor(vertices, dtype=torch.float32),
        faces=tuple(faces),
    )
