import math

import torch

from physics.common.state import ParticleState


class Simulation:
    def __init__(
        self,
        state: ParticleState,
        fixed_dt: float = 1.0 / 120.0,
        substeps: int = 2,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        damping: float = 0.01,
        colliders=(),
    ) -> None:
        self.state = state
        self.initial_state = state.clone()
        self.fixed_dt = fixed_dt
        self.substeps = substeps
        self.gravity = torch.tensor(gravity, dtype=torch.float32)
        self.damping = damping
        self.colliders = list(colliders)
        self.accumulator = 0.0
        self.step_count = 0
        self.time = 0.0
        self.paused = False
        self.last_constraint_error = 0.0

    def advance(self, frame_dt: float) -> None:
        if self.paused:
            return
        self.accumulator += max(0.0, min(frame_dt, 0.25))
        while self.accumulator + 1e-9 >= self.fixed_dt:
            self.single_step()
            self.accumulator -= self.fixed_dt

    def single_step(self) -> None:
        step = self.fixed_dt / self.substeps
        for _ in range(self.substeps):
            self._substep(step)
        self.step_count += 1
        self.time += self.fixed_dt

    def _substep(self, step: float) -> None:
        state = self.state
        previous = state.positions.clone()
        free = state.inverse_mass > 0
        self.apply_forces(step)
        state.velocities[free] += self.gravity * step
        state.velocities[free] *= math.exp(-self.damping * step)
        state.positions[free] += state.velocities[free] * step
        self.solve_constraints(step)
        pinned = ~free
        state.positions[pinned] = self.initial_state.positions[pinned]
        state.velocities[pinned] = 0.0
        state.velocities[free] = (state.positions[free] - previous[free]) / step
        for collider in self.colliders:
            collider.project(state.positions, state.velocities, state.inverse_mass)

    def apply_forces(self, step: float) -> None:
        pass

    def solve_constraints(self, step: float) -> None:
        pass

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def reset(self) -> None:
        self.state.reset(self.initial_state)
        self.accumulator = 0.0
        self.step_count = 0
        self.time = 0.0
        self.paused = False
        self.last_constraint_error = 0.0

    def metrics(self) -> dict[str, float | int]:
        speeds = torch.linalg.vector_norm(self.state.velocities, dim=1)
        return {
            "steps": self.step_count,
            "time": self.time,
            "kinetic_energy": self.state.kinetic_energy().item(),
            "max_displacement": self.state.max_displacement(
                self.initial_state.positions
            ).item(),
            "max_speed": speeds.max().item(),
            "constraint_error": self.last_constraint_error,
        }
