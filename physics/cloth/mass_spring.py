import torch

from physics.cloth.geometry import ClothGeometry
from physics.common.simulation import Simulation


class MassSpringCloth(Simulation):
    def __init__(
        self,
        geometry: ClothGeometry,
        structural_stiffness: float = 800.0,
        shear_stiffness: float = 200.0,
        bend_stiffness: float = 20.0,
        spring_damping: float = 2.0,
        **simulation_kwargs,
    ) -> None:
        super().__init__(geometry.state, **simulation_kwargs)
        self.geometry = geometry
        self.structural_stiffness = structural_stiffness
        self.shear_stiffness = shear_stiffness
        self.bend_stiffness = bend_stiffness
        self.spring_damping = spring_damping
        self.indices = torch.tensor(
            [[constraint.a, constraint.b] for constraint in geometry.constraints],
            dtype=torch.long,
        )
        self.rest_lengths = torch.tensor(
            [constraint.rest_length for constraint in geometry.constraints],
            dtype=torch.float32,
        )
        stiffness_by_kind = {
            "structural": structural_stiffness,
            "shear": shear_stiffness,
            "bend": bend_stiffness,
        }
        self.stiffnesses = torch.tensor(
            [stiffness_by_kind[constraint.kind] for constraint in geometry.constraints],
            dtype=torch.float32,
        )

    def apply_forces(self, step: float) -> None:
        a = self.indices[:, 0]
        b = self.indices[:, 1]
        delta = self.state.positions[b] - self.state.positions[a]
        distances = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-6)
        directions = delta / distances[:, None]
        relative_velocity = self.state.velocities[b] - self.state.velocities[a]
        along_velocity = torch.sum(relative_velocity * directions, dim=1)
        magnitude = (
            self.stiffnesses * (distances - self.rest_lengths)
            + self.spring_damping * along_velocity
        )
        spring_forces = directions * magnitude[:, None]
        forces = torch.zeros_like(self.state.positions)
        forces.index_add_(0, a, spring_forces)
        forces.index_add_(0, b, -spring_forces)
        self.state.velocities += (
            forces * self.state.inverse_mass[:, None] * step
        )

    def solve_constraints(self, step: float) -> None:
        a = self.indices[:, 0]
        b = self.indices[:, 1]
        distances = torch.linalg.vector_norm(
            self.state.positions[b] - self.state.positions[a],
            dim=1,
        )
        relative_error = torch.abs(distances - self.rest_lengths) / self.rest_lengths
        self.last_constraint_error = relative_error.max().item()
