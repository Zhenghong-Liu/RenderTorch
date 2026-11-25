from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class Material:
    color: tuple[int, int, int, int] = (220, 225, 235, 255)


@dataclass(frozen=True)
class MeshResource:
    vertices: torch.Tensor
    faces: tuple[tuple[int, ...], ...]
    lines: tuple[tuple[int, int], ...] = ()


@dataclass
class SceneObject:
    name: str
    mesh: MeshResource
    material: Material = field(default_factory=Material)
    transform: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    object_id: int = -1

    def world_vertices(self) -> torch.Tensor:
        return self.mesh.vertices @ self.transform


@dataclass
class Scene:
    objects: list[SceneObject] = field(default_factory=list)
    selected_id: Optional[int] = None
    _next_id: int = field(default=0, init=False, repr=False)

    def add(self, obj: SceneObject) -> SceneObject:
        obj.object_id = self._next_id
        self._next_id += 1
        self.objects.append(obj)
        if self.selected_id is None:
            self.selected_id = obj.object_id
        return obj

    def select(self, object_id: Optional[int]) -> None:
        if object_id is None:
            self.selected_id = None
            return
        if any(obj.object_id == object_id for obj in self.objects):
            self.selected_id = object_id

    @property
    def selected(self) -> Optional[SceneObject]:
        return next(
            (obj for obj in self.objects if obj.object_id == self.selected_id),
            None,
        )
