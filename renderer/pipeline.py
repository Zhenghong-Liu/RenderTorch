import numpy as np
import torch

from compute.camera import OrbitCamera
from compute.projection import perspective, viewport
from core.scene import Scene
from renderer.clipping import clip_line, clip_polygon
from renderer.framebuffer import FrameBuffer
from renderer.rasterizer import rasterize_line, rasterize_triangle


class SoftwareRenderer:
    def __init__(self, camera: OrbitCamera) -> None:
        self.camera = camera

    def render(
        self,
        scene: Scene,
        width: int,
        height: int,
        display_mode: str = "solid",
    ) -> FrameBuffer:
        if display_mode not in {"solid", "wireframe", "particles"}:
            raise ValueError(f"unsupported display mode: {display_mode}")
        framebuffer = FrameBuffer(width, height)
        framebuffer.clear()
        projection = perspective(
            fov_y=self.camera.fov_y,
            aspect=width / height,
            near=self.camera.near,
            far=self.camera.far,
        )
        view_projection = self.camera.view_matrix() @ projection
        screen_matrix = viewport(width, height).numpy()
        light = torch.tensor([-0.3, -0.5, -1.0])
        light = light / torch.linalg.vector_norm(light)

        for obj in scene.objects:
            world_vertices = obj.world_vertices().cpu().numpy()
            clip_vertices = world_vertices @ view_projection.numpy()
            if display_mode == "solid":
                self._render_solid(
                    framebuffer,
                    obj,
                    clip_vertices,
                    world_vertices,
                    screen_matrix,
                    light,
                )
            elif display_mode == "wireframe":
                self._render_wireframe(
                    framebuffer,
                    obj,
                    clip_vertices,
                    screen_matrix,
                )
            else:
                self._render_particles(
                    framebuffer,
                    obj,
                    clip_vertices,
                    screen_matrix,
                )
        return framebuffer

    def _render_solid(
        self,
        framebuffer,
        obj,
        clip_vertices,
        world_vertices,
        screen_matrix,
        light,
    ) -> None:
        for face in obj.mesh.faces:
            indices = list(face)
            clipped = clip_polygon(clip_vertices[indices])
            if len(clipped) < 3:
                continue
            screen, depths = self._screen_points(clipped, screen_matrix)
            color = self._shade(obj, world_vertices[indices], light)
            for index in range(1, len(screen) - 1):
                triangle_vertices = np.column_stack(
                    (screen[[0, index, index + 1]], depths[[0, index, index + 1]])
                )
                rasterize_triangle(
                    framebuffer,
                    triangle_vertices,
                    color,
                    obj.object_id,
                )

    def _render_wireframe(
        self,
        framebuffer,
        obj,
        clip_vertices,
        screen_matrix,
    ) -> None:
        edges = obj.mesh.lines or self._mesh_edges(obj.mesh)
        for a, b in edges:
            clipped = clip_line(clip_vertices[[a, b]])
            if len(clipped) < 2:
                continue
            screen, depths = self._screen_points(clipped, screen_matrix)
            rasterize_line(
                framebuffer,
                np.column_stack((screen, depths)),
                obj.material.color,
                obj.object_id,
            )

    def _render_particles(
        self,
        framebuffer,
        obj,
        clip_vertices,
        screen_matrix,
    ) -> None:
        for index in range(len(clip_vertices)):
            clipped = clip_polygon(clip_vertices[index:index + 1])
            if len(clipped) == 0:
                continue
            screen, depths = self._screen_points(clipped, screen_matrix)
            x, y = screen[0]
            depth = depths[0]
            offsets = (
                ((-2.0, 0.0), (2.0, 0.0)),
                ((0.0, -2.0), (0.0, 2.0)),
                ((-1.5, -1.5), (1.5, 1.5)),
                ((-1.5, 1.5), (1.5, -1.5)),
            )
            for start_offset, end_offset in offsets:
                rasterize_line(
                    framebuffer,
                    np.array(
                        [
                            [x + start_offset[0], y + start_offset[1], depth],
                            [x + end_offset[0], y + end_offset[1], depth],
                        ]
                    ),
                    obj.material.color,
                    obj.object_id,
                )

    @staticmethod
    def _screen_points(clipped, screen_matrix):
        ndc = clipped[:, :3] / clipped[:, 3:4]
        ndc = np.concatenate((ndc, np.ones((len(ndc), 1))), axis=1)
        screen = (ndc @ screen_matrix)[:, :2]
        return screen, clipped[:, 3]

    @staticmethod
    def _mesh_edges(mesh):
        edges = set()
        for face in mesh.faces:
            for index, a in enumerate(face):
                b = face[(index + 1) % len(face)]
                edges.add(tuple(sorted((a, b))))
        return tuple(sorted(edges))

    @staticmethod
    def _shade(obj, world_vertices, light):
        a, b, c = world_vertices[:3, :3]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            intensity = 1.0
        else:
            normal = normal / length
            intensity = 0.5 + 0.5 * abs(float(np.dot(normal, light.numpy())))
        red, green, blue, alpha = obj.material.color
        return (
            min(255, int(red * intensity)),
            min(255, int(green * intensity)),
            min(255, int(blue * intensity)),
            alpha,
        )
