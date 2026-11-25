import torch


class PlaneCollider:
    def __init__(
        self,
        normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
        offset: float = 0.0,
        restitution: float = 0.5,
        friction: float = 0.2,
    ) -> None:
        self.normal = torch.tensor(normal, dtype=torch.float32)
        self.normal = self.normal / torch.linalg.vector_norm(self.normal)
        self.offset = offset
        self.restitution = restitution
        self.friction = friction

    def project(self, positions, velocities, inverse_mass) -> int:
        distances = positions @ self.normal - self.offset
        contacts = (distances < 0.0) & (inverse_mass > 0)
        if not contacts.any():
            return 0
        positions[contacts] -= distances[contacts, None] * self.normal
        normal_velocity = velocities[contacts] @ self.normal
        velocities[contacts] -= (
            (1.0 + self.restitution) * normal_velocity[:, None] * self.normal
        )
        tangent = velocities[contacts] - (
            (velocities[contacts] @ self.normal)[:, None] * self.normal
        )
        velocities[contacts] -= self.friction * tangent
        return int(contacts.sum())


class SphereCollider:
    def __init__(
        self,
        center: tuple[float, float, float],
        radius: float,
        restitution: float = 0.5,
        friction: float = 0.2,
    ) -> None:
        self.center = torch.tensor(center, dtype=torch.float32)
        self.radius = radius
        self.restitution = restitution
        self.friction = friction

    def project(self, positions, velocities, inverse_mass) -> int:
        delta = positions - self.center
        distances = torch.linalg.vector_norm(delta, dim=1)
        contacts = (distances < self.radius) & (inverse_mass > 0)
        if not contacts.any():
            return 0
        normals = delta[contacts] / distances[contacts, None].clamp_min(1e-6)
        center_mask = distances[contacts] < 1e-6
        normals[center_mask] = torch.tensor([1.0, 0.0, 0.0])
        positions[contacts] = self.center + normals * self.radius
        normal_velocity = torch.sum(velocities[contacts] * normals, dim=1)
        velocities[contacts] -= (
            (1.0 + self.restitution) * normal_velocity[:, None] * normals
        )
        tangent = velocities[contacts] - normal_velocity[:, None] * normals
        velocities[contacts] -= self.friction * tangent
        return int(contacts.sum())
