from .object import OBJECT
import torch
import math

class PYRAMID(OBJECT):
    def __init__(self, position=torch.tensor([0.0, 0.0, 0.0]), size=torch.tensor([1.0, 1.0, 1.0]), step= 4) -> None:
        super().__init__()
        self.step = step
        self.build()
        self.translate(position)
        self.scale(size)

    def build(self):
        PI = torch.tensor(math.pi)
        self.vertices = []
        for i in range(self.step):
            x = torch.cos(2 * PI / self.step * i)
            y = torch.sin(2 * PI / self.step * i)
            self.vertices.append([x, y, 1, 1])
        self.vertices.append([0, 0, -1, 1])
        self.vertices = torch.tensor(self.vertices, dtype=torch.float32)

        self.faces = []
        base = []
        for i in range(self.step):
            base.append(i)
            self.faces.append([i, (i + 1) % self.step, self.step])
        self.faces.append(base)

if __name__ == '__main__':
    pyramid = PYRAMID(position=torch.tensor([200.0, 100.0, 20.0]), size=torch.tensor([1.0, 1.0, 2.0]), step=4)
    print(pyramid.vertices)
    print(pyramid.faces)
    print(pyramid.get_matrix())
