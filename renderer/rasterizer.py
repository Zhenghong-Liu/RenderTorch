import numpy as np

from renderer.framebuffer import FrameBuffer


def _edge(a, b, points):
    return (points[..., 0] - a[0]) * (b[1] - a[1]) - (
        points[..., 1] - a[1]
    ) * (b[0] - a[0])


def rasterize_line(
    framebuffer: FrameBuffer,
    vertices: np.ndarray,
    color: tuple[int, int, int, int],
    object_id: int,
) -> None:
    start, end = vertices
    delta = end - start
    steps = max(1, int(max(abs(delta[0]), abs(delta[1]))))
    for index in range(steps + 1):
        t = index / steps
        point = start + t * delta
        x, y = int(np.floor(point[0])), int(np.floor(point[1]))
        if x < 0 or x >= framebuffer.width or y < 0 or y >= framebuffer.height:
            continue
        if point[2] < framebuffer.depth[y, x]:
            framebuffer.depth[y, x] = point[2]
            framebuffer.color[y, x] = color
            framebuffer.object_ids[y, x] = object_id


def rasterize_triangle(
    framebuffer: FrameBuffer,
    vertices: np.ndarray,
    color: tuple[int, int, int, int],
    object_id: int,
) -> None:
    a, b, c = vertices
    area = _edge(a, b, c)
    if abs(area) < 1e-8:
        return

    min_x = max(0, int(np.floor(min(a[0], b[0], c[0]))))
    max_x = min(framebuffer.width, int(np.ceil(max(a[0], b[0], c[0]))))
    min_y = max(0, int(np.floor(min(a[1], b[1], c[1]))))
    max_y = min(framebuffer.height, int(np.ceil(max(a[1], b[1], c[1]))))
    if min_x >= max_x or min_y >= max_y:
        return

    ys, xs = np.mgrid[min_y:max_y, min_x:max_x]
    points = np.stack((xs + 0.5, ys + 0.5), axis=-1)
    weight_a = _edge(b, c, points) / area
    weight_b = _edge(c, a, points) / area
    weight_c = 1.0 - weight_a - weight_b
    inside = (
        (weight_a >= -1e-6)
        & (weight_b >= -1e-6)
        & (weight_c >= -1e-6)
    )
    if not inside.any():
        return

    inverse_depth = (
        weight_a / a[2] + weight_b / b[2] + weight_c / c[2]
    )
    depth = 1.0 / inverse_depth
    visible = inside & (depth < framebuffer.depth[min_y:max_y, min_x:max_x])
    if not visible.any():
        return

    local_depth = framebuffer.depth[min_y:max_y, min_x:max_x]
    local_color = framebuffer.color[min_y:max_y, min_x:max_x]
    local_ids = framebuffer.object_ids[min_y:max_y, min_x:max_x]
    local_depth[visible] = depth[visible]
    local_color[visible] = color
    local_ids[visible] = object_id
