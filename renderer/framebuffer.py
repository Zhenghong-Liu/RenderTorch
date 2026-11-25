import numpy as np


class FrameBuffer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.color = np.zeros((height, width, 4), dtype=np.uint8)
        self.depth = np.full((height, width), np.inf, dtype=np.float32)
        self.object_ids = np.full((height, width), -1, dtype=np.int32)

    def clear(self, color: tuple[int, int, int, int] = (28, 32, 43, 255)) -> None:
        self.color[:] = color
        self.depth.fill(np.inf)
        self.object_ids.fill(-1)
