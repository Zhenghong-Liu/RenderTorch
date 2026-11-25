import torch

from physics.common.simulation import Simulation
from physics.softbody.geometry import (
    SoftBodyGeometry,
    signed_tetrahedron_volume,
)


class SoftBodyXPBD(Simulation):
    def __init__(
        self,
        geometry: SoftBodyGeometry,
        iterations: int = 8,
        distance_compliance: float = 1e-7,
        volume_compliance: float = 1e-6,
        **simulation_kwargs,
    ) -> None:
        super().__init__(geometry.state, **simulation_kwargs)
        self.geometry = geometry
        self.iterations = iterations
        self.distance_compliance = distance_compliance
        self.volume_compliance = volume_compliance
        self.indices = torch.tensor(
            [
                [constraint.a, constraint.b]
                for constraint in geometry.distance_constraints
            ],
            dtype=torch.long,
        )
        self.rest_lengths = torch.tensor(
            [constraint.rest_length for constraint in geometry.distance_constraints],
            dtype=torch.float32,
        )
        self.compliances = torch.tensor(
            [
                distance_compliance
                if constraint.kind == "structural"
                else distance_compliance * 10.0
                for constraint in geometry.distance_constraints
            ],
            dtype=torch.float32,
        )
        self.volume_indices = torch.tensor(
            [constraint.indices for constraint in geometry.volume_constraints],
            dtype=torch.long,
        )
        self.rest_volumes = torch.tensor(
            [constraint.rest_volume for constraint in geometry.volume_constraints],
            dtype=torch.float32,
        )
        self.lambdas = torch.zeros(len(geometry.distance_constraints), dtype=torch.float32)
        self.volume_lambdas = torch.zeros(
            len(geometry.volume_constraints),
            dtype=torch.float32,
        )
        self.last_volume_error = 0.0

    def solve_constraints(self, step: float) -> None:
        positions = self.state.positions
        inverse_mass = self.state.inverse_mass
        effective_compliance = self.compliances / (step * step)
        effective_volume_compliance = self.volume_compliance / (step * step)
        self.lambdas.zero_()
        self.volume_lambdas.zero_()

        for _ in range(self.iterations):
            for index, constraint in enumerate(self.geometry.distance_constraints):
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

            for index, constraint in enumerate(self.geometry.volume_constraints):
                indices = self.volume_indices[index]
                a, b, c, d = indices
                edge_ab = positions[b] - positions[a]
                edge_ac = positions[c] - positions[a]
                edge_ad = positions[d] - positions[a]
                gradient_b = torch.linalg.cross(edge_ac, edge_ad) / 6.0
                gradient_c = torch.linalg.cross(edge_ad, edge_ab) / 6.0
                gradient_d = torch.linalg.cross(edge_ab, edge_ac) / 6.0
                gradients = torch.stack(
                    (
                        -(gradient_b + gradient_c + gradient_d),
                        gradient_b,
                        gradient_c,
                        gradient_d,
                    )
                )
                weights = inverse_mass[indices]
                denominator = torch.sum(weights * torch.sum(gradients * gradients, dim=1))
                compliance = effective_volume_compliance
                volume = signed_tetrahedron_volume(positions, constraint.indices)
                error = volume - self.rest_volumes[index]
                delta_lambda = (
                    -error - compliance * self.volume_lambdas[index]
                ) / (denominator + compliance)
                self.volume_lambdas[index] += delta_lambda
                positions[indices] += weights[:, None] * delta_lambda * gradients

        a = self.indices[:, 0]
        b = self.indices[:, 1]
        distances = torch.linalg.vector_norm(positions[b] - positions[a], dim=1)
        relative_error = torch.abs(distances - self.rest_lengths) / self.rest_lengths
        self.last_constraint_error = relative_error.max().item()
        rest_volume = self.rest_volumes.sum().item()
        self.last_volume_error = abs(self.volume() - rest_volume) / rest_volume

    def volume(self) -> float:
        return sum(
            signed_tetrahedron_volume(self.state.positions, constraint.indices).item()
            for constraint in self.geometry.volume_constraints
        )

    def metrics(self) -> dict[str, float | int]:
        metrics = super().metrics()
        metrics["volume_error"] = self.last_volume_error
        return metrics

    def reset(self) -> None:
        super().reset()
        self.lambdas.zero_()
        self.volume_lambdas.zero_()
        self.last_volume_error = 0.0
