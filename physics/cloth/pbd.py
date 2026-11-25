import torch

from physics.cloth.geometry import ClothGeometry
from physics.common.simulation import Simulation


class PBDCloth(Simulation):
    def __init__(
        self,
        geometry: ClothGeometry,
        iterations: int = 8,
        structural_stiffness: float = 0.95,
        shear_stiffness: float = 0.8,
        bend_stiffness: float = 0.5,
        **simulation_kwargs,
    ) -> None:
        super().__init__(geometry.state, **simulation_kwargs)
        self.geometry = geometry
        self.iterations = iterations
        self.structural_stiffness = structural_stiffness
        self.shear_stiffness = shear_stiffness
        self.bend_stiffness = bend_stiffness
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

    def solve_constraints(self, step: float) -> None:
        positions = self.state.positions
        inverse_mass = self.state.inverse_mass
        for _ in range(self.iterations):
            for index, constraint in enumerate(self.geometry.constraints):
                a, b = constraint.a, constraint.b
                weight = inverse_mass[a] + inverse_mass[b]
                if weight <= 0:
                    continue
                delta = positions[b] - positions[a]
                distance = torch.linalg.vector_norm(delta).clamp_min(1e-6)
                correction = delta * (
                    (distance - self.rest_lengths[index]) / distance
                )
                correction = correction * (self.stiffnesses[index] / weight)
                positions[a] += correction * inverse_mass[a]
                positions[b] -= correction * inverse_mass[b]

        a = self.indices[:, 0]
        b = self.indices[:, 1]
        distances = torch.linalg.vector_norm(positions[b] - positions[a], dim=1)
        relative_error = torch.abs(distances - self.rest_lengths) / self.rest_lengths
        self.last_constraint_error = relative_error.max().item()
