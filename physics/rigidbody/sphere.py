import math
from dataclasses import dataclass

import torch

from physics.common.colliders import PlaneCollider, SphereCollider
from physics.common.simulation import Simulation
from physics.common.state import ParticleState


@dataclass
class SphereBody:
    position: torch.Tensor
    velocity: torch.Tensor
    radius: float
    inverse_mass: float


class RigidBodySimulation(Simulation):
    def __init__(
        self,
        body: SphereBody,
        fixed_dt: float = 1.0 / 120.0,
        substeps: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        damping: float = 0.0,
        colliders=(),
    ) -> None:
        state = ParticleState(
            positions=body.position.unsqueeze(0).clone(),
            velocities=body.velocity.unsqueeze(0).clone(),
            inverse_mass=torch.tensor([body.inverse_mass], dtype=torch.float32),
        )
        super().__init__(
            state,
            fixed_dt=fixed_dt,
            substeps=substeps,
            gravity=gravity,
            damping=damping,
        )
        self.body = body
        self.body.position = state.positions[0]
        self.body.velocity = state.velocities[0]
        self.static_colliders = list(colliders)

    def _substep(self, step: float) -> None:
        if self.body.inverse_mass <= 0:
            return
        self.body.velocity += self.gravity * step
        self.body.velocity *= math.exp(-self.damping * step)
        self.body.position += self.body.velocity * step
        for collider in self.static_colliders:
            if isinstance(collider, PlaneCollider):
                self._collide_plane(collider)
            elif isinstance(collider, SphereCollider):
                self._collide_static_sphere(collider)

    def _collide_plane(self, collider: PlaneCollider) -> None:
        normal = collider.normal
        distance = self.body.position @ normal - collider.offset - self.body.radius
        if distance >= 0:
            return
        self.body.position -= distance * normal
        normal_velocity = self.body.velocity @ normal
        if normal_velocity < 0:
            self.body.velocity -= (
                (1.0 + collider.restitution) * normal_velocity * normal
            )
        tangent = self.body.velocity - (self.body.velocity @ normal) * normal
        self.body.velocity -= collider.friction * tangent

    def _collide_static_sphere(self, collider: SphereCollider) -> None:
        delta = self.body.position - collider.center
        distance = torch.linalg.vector_norm(delta)
        minimum_distance = self.body.radius + collider.radius
        if distance >= minimum_distance:
            return
        normal = delta / distance.clamp_min(1e-6)
        if distance < 1e-6:
            normal = torch.tensor([1.0, 0.0, 0.0])
        self.body.position += normal * (minimum_distance - distance)
        normal_velocity = self.body.velocity @ normal
        if normal_velocity < 0:
            self.body.velocity -= (
                (1.0 + collider.restitution) * normal_velocity * normal
            )
        tangent = self.body.velocity - (self.body.velocity @ normal) * normal
        self.body.velocity -= collider.friction * tangent

    def collide_with_particles(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        inverse_mass: torch.Tensor,
        restitution: float = 0.0,
        friction: float = 0.2,
    ) -> int:
        contacts = 0
        body_weight = self.body.inverse_mass
        for index in range(len(positions)):
            particle_weight = inverse_mass[index].item()
            if body_weight + particle_weight <= 0:
                continue
            delta = positions[index] - self.body.position
            distance = torch.linalg.vector_norm(delta)
            if distance >= self.body.radius:
                continue
            contacts += 1
            normal = delta / distance.clamp_min(1e-6)
            if distance < 1e-6:
                normal = torch.tensor([1.0, 0.0, 0.0])
            total_weight = body_weight + particle_weight
            penetration = self.body.radius - distance
            positions[index] += normal * penetration * particle_weight / total_weight
            self.body.position -= normal * penetration * body_weight / total_weight

            relative_velocity = velocities[index] - self.body.velocity
            normal_speed = relative_velocity @ normal
            if normal_speed >= 0:
                continue
            impulse = -(1.0 + restitution) * normal_speed / total_weight
            velocities[index] += impulse * particle_weight * normal
            self.body.velocity -= impulse * body_weight * normal

            relative_velocity = velocities[index] - self.body.velocity
            tangent = relative_velocity - (relative_velocity @ normal) * normal
            tangent_speed = torch.linalg.vector_norm(tangent)
            if tangent_speed <= 1e-6:
                continue
            tangent /= tangent_speed
            friction_impulse = -(relative_velocity @ tangent) / total_weight
            friction_impulse = torch.clamp(
                friction_impulse,
                -friction * impulse,
                friction * impulse,
            )
            velocities[index] += friction_impulse * particle_weight * tangent
            self.body.velocity -= friction_impulse * body_weight * tangent
        return contacts
