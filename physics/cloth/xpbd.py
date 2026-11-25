import torch

from physics.cloth.geometry import ClothGeometry
from physics.common.simulation import Simulation


class XPBDCloth(Simulation):
    def __init__(
        self,
        geometry: ClothGeometry,
        iterations: int = 8,
        structural_compliance: float = 1e-6,
        shear_compliance: float = 5e-6,
        bend_compliance: float = 1e-5,
        **simulation_kwargs,
    ) -> None:
        super().__init__(geometry.state, **simulation_kwargs)
        self.geometry = geometry
        self.iterations = iterations
        self.structural_compliance = structural_compliance
        self.shear_compliance = shear_compliance
        self.bend_compliance = bend_compliance
        self.indices = torch.tensor(
            [[constraint.a, constraint.b] for constraint in geometry.constraints],
            dtype=torch.long,
        )
        self.rest_lengths = torch.tensor(
            [constraint.rest_length for constraint in geometry.constraints],
            dtype=torch.float32,
        )
        compliance_by_kind = {
            "structural": structural_compliance,
            "shear": shear_compliance,
            "bend": bend_compliance,
        }
        self.compliances = torch.tensor(
            [compliance_by_kind[constraint.kind] for constraint in geometry.constraints],
            dtype=torch.float32,
        )
        self.lambdas = torch.zeros(len(geometry.constraints), dtype=torch.float32)

    def solve_constraints(self, step: float) -> None:
        positions = self.state.positions
        inverse_mass = self.state.inverse_mass
        effective_compliance = self.compliances / (step * step)
        self.lambdas.zero_()

        for _ in range(self.iterations):
            for index, constraint in enumerate(self.geometry.constraints):
                a, b = constraint.a, constraint.b
                weight = inverse_mass[a] + inverse_mass[b]
                if weight <= 0:
                    continue
                delta = positions[b] - positions[a]
                distance = torch.linalg.vector_norm(delta).clamp_min(1e-6)
                normal = delta / distance
                error = distance - self.rest_lengths[index]
                compliance = effective_compliance[index]
                delta_lambda = (
                    -error - compliance * self.lambdas[index]
                ) / (weight + compliance)
                self.lambdas[index] += delta_lambda
                positions[a] -= inverse_mass[a] * delta_lambda * normal
                positions[b] += inverse_mass[b] * delta_lambda * normal

        a = self.indices[:, 0]
        b = self.indices[:, 1]
        distances = torch.linalg.vector_norm(positions[b] - positions[a], dim=1)
        relative_error = torch.abs(distances - self.rest_lengths) / self.rest_lengths
        self.last_constraint_error = relative_error.max().item()

    def reset(self) -> None:
        super().reset()
        self.lambdas.zero_()
