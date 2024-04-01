from .object import OBJECT
import torch
import math


class CYLINDER(OBJECT):
    def __init__(self, position=torch.tensor([0.0, 0.0, 0.0]), size=torch.tensor([1.0, 1.0, 1.0]), step=10) -> None:
        super().__init__()
        self.step = step  # 因为build要用step，所以step要先定义
        self.build()
        self.translate(position)
        self.scale(size)

    def build(self):
        # [x, y, z, w]
        PI = torch.tensor(math.pi)
        # step = 10 #圆柱体的边数
        self.vertices = []
        for i in range(self.step):
            x = torch.cos(2 * PI / self.step * i)
            y = torch.sin(2 * PI / self.step * i)
            self.vertices.append([x, y, 1, 1])
            self.vertices.append([x, y, -1, 1])
        self.vertices = torch.tensor(self.vertices, dtype=torch.float32)

        self.faces = []
        base_top = []
        base_bottom = []
        for i in range(self.step):
            base_top.append(i * 2)
            base_bottom.append(i * 2 + 1)
            self.faces.append([i * 2, (i * 2 + 2) % (self.step * 2), (i * 2 + 3) % (self.step * 2), i * 2 + 1])
        self.faces.append(base_top)
        self.faces.append(base_bottom)

        # self.faces = np.array(self.faces)

if __name__ == '__main__':
    cylinder = CYLINDER(position=torch.tensor([200.0, 100.0, 20.0]), size=torch.tensor([1.0, 1.0, 2.0]), step=10)
    print(cylinder.vertices)
    print(cylinder.faces)
    print(cylinder.get_matrix())