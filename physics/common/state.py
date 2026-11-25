from dataclasses import dataclass

import torch


@dataclass
class ParticleState:
    positions: torch.Tensor
    velocities: torch.Tensor
    inverse_mass: torch.Tensor

    def clone(self) -> "ParticleState":
        return ParticleState(
            positions=self.positions.detach().clone(),
            velocities=self.velocities.detach().clone(),
            inverse_mass=self.inverse_mass.detach().clone(),
        )

    def reset(self, initial: "ParticleState") -> None:
        self.positions.copy_(initial.positions)
        self.velocities.copy_(initial.velocities)
        self.inverse_mass.copy_(initial.inverse_mass)

    def kinetic_energy(self) -> torch.Tensor:
        mass = torch.where(
            self.inverse_mass > 0,
            self.inverse_mass.reciprocal(),
            torch.zeros_like(self.inverse_mass),
        )
        return 0.5 * torch.sum(mass * torch.sum(self.velocities**2, dim=1))

    def max_displacement(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self.positions - reference, dim=1).max()
