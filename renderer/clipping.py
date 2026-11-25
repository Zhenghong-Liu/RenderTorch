import numpy as np


CLIP_PLANES = (
    (1.0, 0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, -1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, -1.0, 1.0),
)


def clip_line(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 4)
    if len(points) != 2:
        raise ValueError("clip_line requires exactly two homogeneous vertices")
    start, end = points
    lower, upper = 0.0, 1.0
    for plane in CLIP_PLANES:
        plane_array = np.asarray(plane, dtype=np.float32)
        start_distance = start @ plane_array
        end_distance = end @ plane_array
        if start_distance < 0 and end_distance < 0:
            return np.empty((0, 4), dtype=np.float32)
        if start_distance < 0:
            lower = max(lower, start_distance / (start_distance - end_distance))
        if end_distance < 0:
            upper = min(upper, start_distance / (start_distance - end_distance))
        if lower > upper:
            return np.empty((0, 4), dtype=np.float32)
    direction = end - start
    return np.stack((start + lower * direction, start + upper * direction))


def clip_polygon(vertices: np.ndarray) -> np.ndarray:
    clipped = np.asarray(vertices, dtype=np.float32).reshape(-1, 4)
    for plane in CLIP_PLANES:
        if len(clipped) == 0:
            break
        plane_array = np.asarray(plane, dtype=np.float32)
        distances = clipped @ plane_array
        output = []
        for index, current in enumerate(clipped):
            previous = clipped[index - 1]
            current_distance = distances[index]
            previous_distance = distances[index - 1]
            current_inside = current_distance >= 0.0
            previous_inside = previous_distance >= 0.0
            if current_inside != previous_inside:
                ratio = previous_distance / (previous_distance - current_distance)
                output.append(previous + ratio * (current - previous))
            if current_inside:
                output.append(current)
        clipped = np.asarray(output, dtype=np.float32).reshape(-1, 4)
    return clipped
